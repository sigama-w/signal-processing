"""显存实测工具：用合成数据真实跑"预训练 step + 微调 step（三任务）"，读取峰值显存。

为什么不再用公式估算: 旧 gen_configs.vram_estimate_gb 只在 d512/27L 一个点上校准，
且未计入反向峰值（约 +25%）与 bf16 autocast 下 LayerNorm/softmax 的 fp32 激活，
对 1024/1536 宽配置系统性低估 10~15 GB（详见 docs/OPTIMIZATION_PLAN.md §6）。
本脚本无需任何数据集，几分钟内给出真实数字。

用法:
  python utils/measure_vram.py --config code/configs/default.json            # 测当前配置
  python utils/measure_vram.py --config code/configs/default.json --find_max_bs
  python utils/measure_vram.py --grid code/configs/grid.json --limit 4        # 批量测 grid
  python utils/measure_vram.py --grid code/configs/grid_v2.json --budget 76

输出: 每个配置 pretrain / finetune 两阶段的 max_allocated / max_reserved（GB），
      是否超过 --budget（默认 76GB，80G 卡留 4GB 余量），写入 logs/vram/report.json。
"""

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import torch  # noqa: E402

from models import build_model, model_arch                   # noqa: E402
from training.common import normalize_config                 # noqa: E402
from utils.optim import build_optimizer                      # noqa: E402
from utils.param_codec import ParamCodec                     # noqa: E402
from utils.train_utils import autocast_context               # noqa: E402

TASKS = ("protocol", "modulation", "parameter")


def _dev():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu"), torch.npu
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda
    return torch.device("cpu"), None


def synth_batch(task, B, N, device, codec=None):
    iq = torch.randn(B, N, 2, device=device)
    b = {"iq": iq, "sr": torch.full((B,), 1e6, device=device)}
    if task == "protocol":
        b["label"] = torch.randint(0, 8, (B,), device=device)
    elif task == "modulation":
        b["label"] = torch.randint(0, 23, (B,), device=device)
    else:
        b["pattern"] = torch.randint(0, 2, (B,), device=device)
        v = torch.zeros(B, 7, device=device)
        v[:, :3] = (b["pattern"] == 0).float()[:, None]
        v[:, 3:] = (b["pattern"] == 1).float()[:, None]
        b["valid"] = v
        lo = torch.tensor(codec.vmin, device=device).float()
        hi = torch.tensor(codec.vmax, device=device).float()
        b["raw_params"] = (lo + (hi - lo) * torch.rand(B, 7, device=device)) * v
        b["params"] = torch.rand(B, 7, device=device) * v
    return b


def measure(cfg, bs, steps=2, stage="both"):
    device, mod = _dev()
    cfg = normalize_config(json.loads(json.dumps(cfg)))
    mcfg, tcfg = cfg["model"], dict(cfg["training"])
    tcfg["batch_size"] = bs
    N = tcfg.get("seq_len", 1024)
    arch = model_arch(mcfg)
    codec = ParamCodec.default()
    if arch == "v2":
        mcfg["param_head_spec"] = codec.head_spec()
    res = {"batch_size": bs, "arch": arch}
    model = build_model(mcfg).to(device)
    model.codec = codec
    res["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6

    def peak(fn):
        if mod is not None:
            mod.empty_cache()
            mod.reset_peak_memory_stats()
        fn()
        if mod is not None:
            mod.synchronize()
            return mod.max_memory_allocated() / 1e9, mod.max_memory_reserved() / 1e9
        return float("nan"), float("nan")

    try:
        if stage in ("both", "pretrain"):
            named = [(n, p) for n, p in model.named_parameters() if not n.startswith("task_heads")]
            opt = build_optimizer(named, tcfg)

            def pre():
                for _ in range(steps):
                    iq = torch.randn(bs, N, 2, device=device)
                    with autocast_context(device, True):
                        if arch == "v2":
                            loss, _ = model.forward_pretrain(iq)
                        else:
                            r, t = model.forward_mae(iq, torch.full((bs,), 1e6, device=device))
                            loss = (r.float() - t.float()).pow(2).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            res["pretrain_alloc_gb"], res["pretrain_reserved_gb"] = peak(pre)
            del opt
            model.zero_grad(set_to_none=True)
            gc.collect()
        if stage in ("both", "finetune"):
            named = [(n, p) for n, p in model.named_parameters()
                     if not n.startswith("pretrain_head")]
            opt = build_optimizer(named, tcfg)
            if arch == "v2":
                from training.tasks_v2 import TASKS_V2 as T
            else:
                from training.tasks import TASKS as T

            def ft():
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    for t in TASKS:
                        b = synth_batch(t, bs, N, device, codec)
                        with autocast_context(device, True):
                            if arch == "v2":
                                loss, _ = T[t].loss(model, b, tcfg)
                            else:
                                loss = T[t].loss(model, model.encode(b["iq"], b["sr"]), b, tcfg)
                        loss.backward()
                    opt.step()
            res["finetune_alloc_gb"], res["finetune_reserved_gb"] = peak(ft)
            del opt
        res["status"] = "ok"
    except RuntimeError as e:
        res["status"] = "OOM" if "out of memory" in str(e).lower() else f"error: {e}"[:200]
    finally:
        del model
        gc.collect()
        if mod is not None:
            mod.empty_cache()
    return res


def find_max_bs(cfg, budget, start=64, stop=4096):
    best, bs = None, start
    while bs <= stop:
        r = measure(cfg, bs, steps=1)
        peak = max(r.get("pretrain_reserved_gb", 0), r.get("finetune_reserved_gb", 0))
        print(f"  bs={bs}: {r.get('status')} peak_reserved={peak:.1f}GB")
        if r["status"] != "ok" or peak > budget:
            break
        best, bs = bs, bs * 2
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=0, help="覆盖 batch_size（0=用配置值）")
    ap.add_argument("--budget", type=float, default=76.0)
    ap.add_argument("--find_max_bs", action="store_true")
    args = ap.parse_args()
    items = []
    if args.grid:
        g = json.load(open(args.grid, encoding="utf-8"))
        items = [(c.get("name", f"cfg{i}"), {"model": c["model"], "training": c["training"]})
                 for i, c in enumerate(g["configs"])]
        if args.limit:
            items = items[: args.limit]
    else:
        path = args.config or str(ROOT / "code" / "configs" / "default.json")
        items = [(Path(path).stem, json.load(open(path, encoding="utf-8")))]
    report = []
    for name, cfg in items:
        cfg = normalize_config(cfg)
        bs = args.bs or cfg["training"]["batch_size"]
        print(f"== {name} (bs={bs})")
        r = measure(cfg, bs)
        r["name"] = name
        peak = max(r.get("pretrain_reserved_gb", 0) or 0, r.get("finetune_reserved_gb", 0) or 0)
        r["fits_budget"] = r["status"] == "ok" and peak <= args.budget
        print("   " + json.dumps(r, ensure_ascii=False))
        if args.find_max_bs:
            r["max_bs"] = find_max_bs(cfg, args.budget)
            print(f"   max batch size within {args.budget}GB: {r['max_bs']}")
        report.append(r)
    out = ROOT / "logs" / "vram"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"report -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
