"""统一推理入口（官方评测程序直接调用）

官方要求:
  - 位于 $HOME/code/infer.py
  - 输入: test Parquet（task / sample_id / iq 3 列，混合三任务，512/1024 两种长度）
  - 输出: JSONL，每行 {task, sample_id, prediction}，顺序与输入行序一致
    分类: prediction 为官方类别名字符串
    参数: prediction 为 {参数名: 数值}（SI 原始单位，不适用参数填 0.0）

v2 推理优化（复赛"推理时延"占 30 分）:
  1. 按 (task, 原始长度) 分组成批：512 点样本只跑 64 个时域 token（trim_padding），
     批内无 padding mask -> SDPA 可走 FlashAttention 内核
  2. 默认 bf16 autocast（信号前端内部强制 fp32，精度不受影响）
  3. 可选 --compile（torch.compile, Triton 融合 LayerNorm/GELU/逐元素算子）
  4. 预热 + 计时：日志输出总耗时、每样本耗时、吞吐，以及融合算子启用状态
  5. iq 解析向量化，去掉 v1 中逐行 iterrows

用法:
  python code/infer.py --input test.parquet --output out/pred.jsonl
  python code/infer.py --input test.parquet --output out/pred.jsonl --precision fp32 --batch_size 256
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))  # code/
sys.path.insert(0, ROOT)

from models import build_model, model_arch                    # noqa: E402
from training.common import load_config                       # noqa: E402
from utils.dataset import FS, ID2LABEL, PARAM_NAMES, to_iq_array   # noqa: E402
from utils.logger import get_logger                           # noqa: E402
from utils.param_codec import ParamCodec                      # noqa: E402


def load_model(weights_dir, device):
    """加载 weights/ 下 config.json + model.safetensors（+ param_codec.json / param_stats.json）。"""
    from safetensors.torch import load_file
    cfg = load_config(os.path.join(weights_dir, "config.json"))
    arch = model_arch(cfg["model"])
    model = build_model(cfg["model"]).to(device)
    sd = load_file(os.path.join(weights_dir, "model.safetensors"))
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    if arch == "v2":
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith("pretrain_head")]
        if bad or unexpected:
            raise RuntimeError(f"权重与配置不匹配: missing={bad[:5]} unexpected={unexpected[:5]}")
        spec = cfg["model"].get("param_codec")
        cpath = os.path.join(weights_dir, "param_codec.json")
        model.codec = ParamCodec(spec) if spec else ParamCodec.load(cpath)
        param_stats = None
    else:
        model.load_state_dict(sd)
        stats_path = os.path.join(weights_dir, "param_stats.json")
        param_stats = None
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            param_stats = {"min": np.array(s["min"], dtype=np.float32),
                           "max": np.array(s["max"], dtype=np.float32)}
    model.eval()
    return model, cfg, param_stats


def _predict_v1(model, iq, sr, task, param_stats):
    out = model(iq, sr, task)
    if task != "parameter":
        return out.argmax(-1).cpu().numpy()
    pat = model(iq, sr, "parameter_pattern").argmax(-1)
    mask = torch.zeros_like(out)
    mask[:, :3] = (pat == 0).float().unsqueeze(-1)
    mask[:, 3:] = (pat == 1).float().unsqueeze(-1)
    pred = out.float().cpu().numpy()
    if param_stats is not None:
        pred = pred * (param_stats["max"] - param_stats["min"]) + param_stats["min"]
    return pred * mask.cpu().numpy()


def _predict(model, arch, iq, task, param_stats):
    if arch == "v2":
        out = model.predict(iq, task)
        return out.float().cpu().numpy() if task == "parameter" else out.cpu().numpy()
    sr = torch.full((iq.size(0),), FS, dtype=torch.float32, device=iq.device)
    return _predict_v1(model, iq, sr, task, param_stats)


def _autocast(device, precision):
    if precision == "fp32" or device.type not in ("cuda", "npu"):
        return torch.autocast(device_type=device.type, enabled=False)
    dt = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dt)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


@torch.no_grad()
def infer_file(model, parquet_path, out_path, seq_len, param_stats, device, batch_size=256,
               precision="bf16", arch=None, logger=None, warmup=True):
    """官方输入（混合三任务，行序任意）-> JSONL（严格保持输入行序）。"""
    arch = arch or ("v2" if hasattr(model, "predict") else "v1")
    df = pd.read_parquet(parquet_path, columns=["task", "sample_id", "iq"])
    df["task"] = df["task"].replace({"system": "protocol"})
    if not set(df["task"]).issubset({"protocol", "modulation", "parameter"}):
        raise ValueError(f"unknown tasks: {set(df['task'])}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    t0 = time.perf_counter()
    iqs = [to_iq_array(v) for v in df["iq"]]
    lengths = np.array([min(len(a), seq_len) for a in iqs])
    t_parse = time.perf_counter() - t0

    # (task, 长度) 分组 -> 同组内批推理
    groups = {}
    for i, (t, ln) in enumerate(zip(df["task"].to_numpy(), lengths)):
        groups.setdefault((t, int(ln)), []).append(i)

    if warmup and len(df):
        k = next(iter(groups))
        idx = groups[k][: min(4, len(groups[k]))]
        x = torch.from_numpy(np.stack([to_iq_array(iqs[i], seq_len) for i in idx])).to(device)
        with _autocast(device, precision):
            _predict(model, arch, x, k[0], param_stats)
        _sync(device)

    results = [None] * len(df)
    sids = df["sample_id"].to_numpy()
    t1 = time.perf_counter()
    for (task, _), idx_all in groups.items():
        for s in range(0, len(idx_all), batch_size):
            idx = idx_all[s:s + batch_size]
            x = np.stack([to_iq_array(iqs[i], seq_len) for i in idx])
            x = torch.from_numpy(x).to(device, non_blocking=True)
            with _autocast(device, precision):
                pred = _predict(model, arch, x, task, param_stats)
            for j, i in enumerate(idx):
                if task == "parameter":
                    p = {n: float(pred[j, k]) for k, n in enumerate(PARAM_NAMES)}
                else:
                    p = ID2LABEL[task][int(pred[j])]
                results[i] = {"task": task, "sample_id": sids[i], "prediction": p}
    _sync(device)
    t_model = time.perf_counter() - t1

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    msg = (f"[infer] {parquet_path} -> {out_path} n={len(df)} parse_s={t_parse:.3f} "
           f"model_s={t_model:.3f} per_sample_ms={1000 * t_model / max(1, len(df)):.4f} "
           f"throughput={len(df) / max(t_model, 1e-9):.1f}/s precision={precision}")
    print(msg)
    if logger:
        logger.info(msg)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="测试 Parquet 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--weights_dir", default=os.path.join(ROOT, "..", "weights"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true", help="torch.compile 加速（CUDA）")
    args = parser.parse_args()

    from utils.train_utils import get_device
    device = get_device()
    logger = get_logger("inference", os.path.join(ROOT, "..", "logs"), "inference.log")
    model, cfg, param_stats = load_model(args.weights_dir, device)
    arch = model_arch(cfg["model"])
    if args.compile and device.type == "cuda":
        # 只编译主干（DSP 前端含数据相关裁剪；predict 等方法保持 eager 可调用）
        model.backbone = torch.compile(model.backbone, dynamic=True)
    start = time.perf_counter()
    infer_file(model, args.input, args.output,
               seq_len=cfg["training"].get("seq_len", 1024), param_stats=param_stats,
               device=device, batch_size=args.batch_size, precision=args.precision,
               arch=arch, logger=logger)
    try:
        from models.ops import fused_status
        logger.info("fused ops: %s", fused_status())
    except Exception:
        pass
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("arch=%s device=%s params=%.2fM elapsed_total_s=%.3f", arch, device,
                n_params / 1e6, time.perf_counter() - start)


if __name__ == "__main__":
    main()
