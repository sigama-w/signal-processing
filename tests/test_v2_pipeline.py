"""v2 全链路测试（需 PyTorch；CPU 即可，约 1 分钟）。

python -m pytest tests/test_v2_pipeline.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from models import build_model                         # noqa: E402
from models.dsp import RealDFT, SignalFrontEnd, valid_mask_from_iq  # noqa: E402
from training.common import load_mae_pretrained       # noqa: E402
from training.tasks_v2 import TASKS_V2, supcon_loss   # noqa: E402
from utils.augment import augment_iq                  # noqa: E402
from utils.optim import ModelEMA, build_optimizer     # noqa: E402
from utils.param_codec import ParamCodec              # noqa: E402

TINY = {"arch": "v2", "d_model": 48, "n_layers": 3, "nhead": 4, "ffn_hidden": 64,
        "layer_types": ["conv", "conformer", "attn"], "seq_len": 1024, "pool_heads": 4,
        "head_config": {"protocol": {"hidden": 32}, "modulation": {"hidden": 32},
                        "parameter": {"hidden": 32}}, "supcon_dim": 16, "dropout": 0.0,
        "drop_path": 0.0}


def make_codec():
    rng = np.random.default_rng(0)
    M = 2000
    p = np.zeros((M, 7))
    p[:, 0] = rng.choice([62.5, 125, 250, 500, 1000, 2000], M)
    p[:, 1] = p[:, 0] * (1 + rng.uniform(0.2, 0.75, M))
    p[:, 2] = rng.choice([-50, 0, 10, 100, 1000], M)
    p[:, 3] = rng.uniform(10e-6, 16e-6, M)
    p[:, 4] = rng.uniform(17e-6, 23e-6, M)
    p[:, 5] = rng.uniform(1e-6, 10e-6, M)
    p[:, 6] = rng.integers(2, 7, M)
    v = np.zeros((M, 7), bool)
    half = M // 2
    v[:half, :3] = True
    v[half:, 3:] = True
    return ParamCodec.fit(p, v), p, v


def tiny_model():
    codec, _, _ = make_codec()
    cfg = dict(TINY, param_head_spec=codec.head_spec())
    m = build_model(cfg)
    m.codec = codec
    return m, codec


def sig(B=3, L=1024):
    x = torch.zeros(B, 1024, 2)
    x[:, :L] = torch.randn(B, L, 2) * 0.01
    return x


def test_dft_matmul_matches_fft():
    d1, d2 = RealDFT(64, "matmul"), RealDFT(64, "fft")
    xr, xi = torch.randn(2, 64), torch.randn(2, 64)
    for inv in (False, True):
        a, b = d1(xr, xi, inv), d2(xr, xi, inv)
        assert torch.allclose(a[0], b[0], atol=1e-4) and torch.allclose(a[1], b[1], atol=1e-4)


def test_valid_length_detection():
    x = sig(2)
    x[1, 512:] = 0
    v, L = valid_mask_from_iq(x)
    assert L.tolist() == [1024, 512] and v[1, 511] == 1 and v[1, 512] == 0


def test_frontend_shapes_and_finite():
    fe = SignalFrontEnd()
    out = fe(sig(2))
    assert out["time_ch"].shape == (2, 6, 1024)
    assert out["stft"].shape == (2, 128, 32)
    assert out["spec"].shape == (2, 3, 1024)
    assert out["acf"].shape == (2, 512)
    assert out["const"].shape == (2, 32, 32)
    assert out["stats"].shape == (2, 16)
    for k, v in out.items():
        if v.is_floating_point():
            assert torch.isfinite(v).all(), k


def test_cfo_correction_recovers_offset():
    n = torch.arange(1024)
    sym = torch.randint(0, 4, (256,)).float() * (np.pi / 2) + np.pi / 4
    ph = sym.repeat_interleave(4) + 2 * np.pi * 0.0625 * n
    x = torch.stack([torch.cos(ph), torch.sin(ph)], -1)[None]
    st = SignalFrontEnd()(x)["stats"]
    cfo = torch.sign(st[0, 2]) * (torch.expm1(st[0, 2].abs())) / 10
    assert abs(float(cfo) - 0.0625) < 0.01


def test_padding_invariance_with_trim():
    m, _ = tiny_model()
    m.eval()
    x = sig(1, 512)
    with torch.no_grad():
        a = m.head_outputs(x, "modulation")
        m.tokenizer.trim_padding = False
        b = m.head_outputs(x, "modulation")
    assert torch.allclose(a, b, atol=1e-4)


def test_forward_losses_backward_and_ratio():
    m, codec = tiny_model()
    assert m.shared_ratio() > 0.5        # tiny 配置头部占比偏高；正式配置 >0.85
    B = 4
    cfg = {"supcon_weight": {"modulation": 0.1}}
    batches = {
        "protocol": {"iq": sig(B), "label": torch.randint(0, 8, (B,))},
        "modulation": {"iq": sig(B), "label": torch.randint(0, 23, (B,))},
    }
    _, p, v = make_codec()
    batches["parameter"] = {"iq": sig(B), "raw_params": torch.tensor(p[998:1002] * v[998:1002]).float(),
                            "valid": torch.tensor(v[998:1002]).float(),
                            "pattern": torch.tensor(v[998:1002, 3]).long()}
    total = 0
    for t, b in batches.items():
        loss, logs = TASKS_V2[t].loss(m, b, cfg)
        assert torch.isfinite(loss)
        total = total + loss
    total.backward()
    loss, parts = m.forward_pretrain(sig(2))
    assert torch.isfinite(loss)
    loss.backward()


def test_predict_parameter_masks_invalid():
    m, _ = tiny_model()
    m.eval()
    out = m.predict(sig(3), "parameter")
    assert out.shape == (3, 7)
    for r in out:
        assert (r[:3] == 0).all() or (r[3:] == 0).all()


def test_codec_roundtrip():
    codec, p, v = make_codec()
    assert codec.modes[6] == "cls" and codec.modes[3] == "zscore"
    raw = torch.tensor(p[:8]).float()
    z, c = codec.encode(raw)
    out = {"reg": z, "cls": [torch.nn.functional.one_hot(c[:, k], len(codec.values[j])).float() * 50
                             for k, j in enumerate(codec.cls_idx)]}
    dec = codec.decode(out)
    for j in range(7):
        assert torch.allclose(dec[:, j], raw[:, j].clamp(codec.vmin[j], codec.vmax[j]),
                                  rtol=1e-3, atol=1e-9)


def test_augment_keeps_padding_zero():
    x = sig(4, 512)
    y = augment_iq(x, "modulation")
    assert (y[:, 512:] == 0).all()


def test_supcon_finite():
    assert torch.isfinite(supcon_loss(torch.randn(8, 4), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])))


@pytest.mark.parametrize("opt", ["adamw", "muon"])
def test_optimizers_and_ema(opt):
    m, _ = tiny_model()
    o = build_optimizer(list(m.named_parameters()), {"lr": 1e-3, "optimizer": opt})
    ema = ModelEMA(m, 0.9)
    loss, _ = m.forward_pretrain(sig(2))
    loss.backward()
    o.step()
    ema.update(m)


def test_mae_weights_are_loaded(tmp_path):
    m, _ = tiny_model()
    sd = {k: v + 1.0 for k, v in m.state_dict().items() if k.startswith(("tokenizer.", "backbone."))}
    torch.save(sd, tmp_path / "mae.pt")
    m2, _ = tiny_model()
    assert load_mae_pretrained(m2, tmp_path / "mae.pt")
    k = next(iter(sd))
    assert torch.allclose(m2.state_dict()[k], sd[k])


def test_infer_end_to_end(tmp_path, monkeypatch):
    from infer import infer_file
    m, _ = tiny_model()
    m.eval()
    rows = [("parameter", "a", np.random.randn(512, 2).astype(np.float32)),
            ("modulation", "b", np.random.randn(1024, 2).astype(np.float32)),
            ("protocol", "c", np.random.randn(512, 2).astype(np.float32)),
            ("modulation", "d", np.random.randn(512, 2).astype(np.float32))]
    df = pd.DataFrame(rows, columns=["task", "sample_id", "iq"])
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: df.copy())
    out = tmp_path / "p.jsonl"
    infer_file(m, "x.parquet", str(out), 1024, None, torch.device("cpu"), batch_size=2,
               precision="fp32", arch="v2")
    res = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["sample_id"] for r in res] == ["a", "b", "c", "d"]
    assert isinstance(res[1]["prediction"], str) and len(res[0]["prediction"]) == 7
