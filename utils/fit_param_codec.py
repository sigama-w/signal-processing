"""拟合并查看参数编解码器（ParamCodec），同时打印每个参数的取值分布摘要。

用法:
  python utils/fit_param_codec.py --data_dir datasets --out weights/param_codec.json
  python utils/fit_param_codec.py --force number_of_pulses=cls symbol_rate=log_zscore

微调脚本在 weights/param_codec.json 不存在时会自动调用同样的拟合逻辑；
本脚本用于先看清数据（离散/连续、长尾程度）再决定是否手动 --force。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402

from utils.dataset import load_param_arrays          # noqa: E402
from utils.param_codec import PARAM_NAMES, ParamCodec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "datasets"))
    ap.add_argument("--out", default=str(ROOT / "weights" / "param_codec.json"))
    ap.add_argument("--max_classes", type=int, default=32)
    ap.add_argument("--decode", default="expectation", choices=["expectation", "argmax"])
    ap.add_argument("--force", nargs="*", default=[], help="name=mode，mode∈cls/zscore/log_zscore")
    args = ap.parse_args()
    p, v = load_param_arrays(args.data_dir, "train")
    print(f"train parameter samples: {len(p)}")
    for j, n in enumerate(PARAM_NAMES):
        x = p[v[:, j], j]
        u, c = np.unique(x, return_counts=True)
        top = ", ".join(f"{a:.4g}:{b}" for a, b in sorted(zip(u, c), key=lambda t: -t[1])[:8])
        print(f"{n:>17}: n={len(x)} unique={len(u)} min={x.min():.4g} max={x.max():.4g} "
              f"mean={x.mean():.4g} std={x.std():.4g} top=[{top}]")
    force = dict(kv.split("=") for kv in args.force)
    codec = ParamCodec.fit(p, v, args.max_classes, force=force, decode=args.decode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    codec.save(args.out)
    print("\n" + codec.describe())
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
