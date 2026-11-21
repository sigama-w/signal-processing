"""导出 v2 模型为 ONNX（TensorRT / 昇腾 ATC 转 .om 的入口）

v2 前端 DFT 用实数矩阵乘实现、噪底用 topk 代替 median，导出图中没有 DFT/复数算子，
ATC 与 TensorRT 都能直接转换。每个任务导出一个文件（任务提示 / 头不同），
输入固定为 [B,1024,2]，导出时关闭 trim_padding（动态裁剪需要 .item()）。

用法:
  python utils/export_onnx.py --weights_dir weights --out_dir weights/onnx
  # 昇腾: atc --model=weights/onnx/modulation.onnx --framework=5 --output=modulation \
  #           --soc_version=Ascend910B --input_shape="iq:64,1024,2"
  # TensorRT: trtexec --onnx=weights/onnx/modulation.onnx --bf16 --saveEngine=mod.plan
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import torch  # noqa: E402


class TaskWrapper(torch.nn.Module):
    def __init__(self, model, task):
        super().__init__()
        self.model, self.task = model, task

    def forward(self, iq):
        out = self.model.head_outputs(iq, self.task)
        if self.task == "parameter":
            return (out["pattern"], out["reg"], *out["cls"])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", default=str(ROOT / "weights"))
    ap.add_argument("--out_dir", default=str(ROOT / "weights" / "onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    from infer import load_model
    model, cfg, _ = load_model(args.weights_dir, torch.device("cpu"))
    model.tokenizer.trim_padding = False
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.randn(args.batch, cfg["training"].get("seq_len", 1024), 2)
    for task in ("protocol", "modulation", "parameter"):
        path = out / f"{task}.onnx"
        torch.onnx.export(TaskWrapper(model, task).eval(), (x,), str(path),
                          input_names=["iq"], opset_version=args.opset,
                          dynamic_axes={"iq": {0: "batch"}})
        print(f"exported {path}")


if __name__ == "__main__":
    main()
