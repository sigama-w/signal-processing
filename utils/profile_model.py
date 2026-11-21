"""算子级 profiling（定位推理热点，决定是否值得写融合算子）

用法:
  python utils/profile_model.py --config code/configs/default.json --batch 256 --length 1024
  python utils/profile_model.py --weights_dir weights --precision bf16

输出:
  控制台: 按设备耗时排序的 Top-30 算子表
  logs/profile/trace.json      Chrome trace（chrome://tracing 或 Perfetto 打开）
  logs/profile/top_ops.txt     算子表文本
昇腾环境自动改用 torch_npu.profiler（输出目录同上，用 MindStudio Insight 查看）。

判读建议: 若 aten::mm / aten::bmm / flash attention 已占 >70%，说明已是计算受限，
继续手写融合算子收益很小，应转向减小模型 / 量化 / 分桶裁剪；若 layer_norm、gelu、
add、mul 等逐元素算子占比高，则开 --compile（Inductor 自动生成 Triton 融合 kernel）
或使用 ops/fused.py 中的融合实现。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "utils"))

import torch  # noqa: E402

from benchmark_latency import build  # noqa: E402
from models import model_arch        # noqa: E402
from utils.train_utils import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "code" / "configs" / "default.json"))
    ap.add_argument("--weights_dir", default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--task", default="modulation")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    device = get_device()
    model, cfg, _ = build(args.config, args.weights_dir, device)
    arch = model_arch(cfg["model"])
    iq = torch.zeros(args.batch, 1024, 2, device=device)
    iq[:, :args.length] = torch.randn(args.batch, args.length, 2, device=device)
    sr = torch.full((args.batch,), 1e6, device=device)
    dt = torch.bfloat16 if args.precision == "bf16" else torch.float16

    def step():
        with torch.no_grad(), torch.autocast(device.type, dtype=dt,
                                             enabled=args.precision != "fp32" and device.type != "cpu"):
            return model.predict(iq, args.task) if arch == "v2" else model(iq, sr, args.task)

    for _ in range(3):
        step()
    out = ROOT / "logs" / "profile"
    out.mkdir(parents=True, exist_ok=True)
    if device.type == "npu":
        import torch_npu
        prof_cfg = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                experimental_config=prof_cfg,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out))) as prof:
            for _ in range(args.steps):
                step()
                prof.step()
        print(f"NPU profile -> {out}")
        return
    acts = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=acts, record_shapes=True) as prof:
        for _ in range(args.steps):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=key, row_limit=30)
    print(table)
    (out / "top_ops.txt").write_text(table, encoding="utf-8")
    prof.export_chrome_trace(str(out / "trace.json"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
