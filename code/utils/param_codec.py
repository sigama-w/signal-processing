"""7 个物理参数的目标编解码器（解决"量纲不同、尺度不均、离散/连续混杂"问题）

问题
----
  symbol_rate 62.5~2000 Hz（离散档位, 长尾）     bandwidth 65~3500 Hz
  frequency_shift -50~1000 Hz（大量集中在 100）   pulse_width 10~16 µs（连续均匀）
  pri 17~23 µs（连续均匀）                         time_delay 1~10 µs（连续均匀）
  number_of_pulses 2~6（整数计数）
  且每条样本只有 3 个（通信）或 4 个（雷达）参数有效。

v1 做法：min-max 到 [0,1] + Sigmoid + SmoothL1，问题：
  * 长尾参数（symbol_rate）在 min-max 后大部分样本挤在 0~0.1，梯度几乎只由少数大值主导；
  * 离散/计数参数（number_of_pulses、档位化的 symbol_rate）用回归会输出 3.6 这类"非法值"，
    而分类+期望能把 RMSE 降到接近 0；
  * Sigmoid 在边界饱和，极值样本学不动；
  * 逐元素平均让雷达 4 参数天然比通信 3 参数权重大。

v2 做法（按训练集统计自动选择每个参数的模式，结果存 weights/param_codec.json）:
  "cls"        取值个数 <= max_classes（如 number_of_pulses、档位化码速率）
               -> 分类头 + 交叉熵；解码用后验期望 Σ p_k v_k（MSE 最优）或 argmax
  "log_zscore" 正值且 max/min >= 20 的长尾连续量 -> log 后标准化
  "zscore"     其余连续量 -> 标准化（与官方 RMSE/参考尺度 线性对齐）
损失在每个参数内部按有效样本平均，再对参数取平均（7 个参数等权，不受 3/4 个数影响）。
"""

import json

import numpy as np
import torch

PARAM_NAMES = [
    "symbol_rate", "bandwidth", "frequency_shift",
    "pulse_width", "pri", "time_delay", "number_of_pulses",
]


class ParamCodec:
    def __init__(self, spec):
        self.spec = spec
        self.names = spec["names"]
        self.modes = spec["modes"]
        self.mean = np.asarray(spec["mean"], dtype=np.float64)
        self.std = np.asarray(spec["std"], dtype=np.float64)
        self.vmin = np.asarray(spec["min"], dtype=np.float64)
        self.vmax = np.asarray(spec["max"], dtype=np.float64)
        self.values = spec["values"]
        self.decode_mode = spec.get("decode", "expectation")
        self.reg_idx = [i for i, m in enumerate(self.modes) if m != "cls"]
        self.cls_idx = [i for i, m in enumerate(self.modes) if m == "cls"]
        self._t = {}

    # ------------------------------------------------------------------ fit / io
    @classmethod
    def fit(cls, params, valid, max_classes=32, log_ratio=20.0, force=None, decode="expectation"):
        """params/valid: [M,7]；force: {name: mode} 手动覆盖。"""
        params = np.asarray(params, dtype=np.float64)
        valid = np.asarray(valid).astype(bool)
        force = force or {}
        spec = {"names": PARAM_NAMES, "modes": [], "mean": [], "std": [], "min": [],
                "max": [], "values": [], "decode": decode, "n_valid": []}
        for j, name in enumerate(PARAM_NAMES):
            v = params[valid[:, j], j]
            if v.size == 0:
                raise ValueError(f"{name} 无有效样本")
            uniq = np.unique(np.round(v, 12) if np.abs(v).max() < 1 else np.round(v, 6))
            if name in force:
                mode = force[name]
            elif len(uniq) <= max_classes:
                mode = "cls"
            elif v.min() > 0 and v.max() / v.min() >= log_ratio:
                mode = "log_zscore"
            else:
                mode = "zscore"
            t = np.log(v) if mode == "log_zscore" else v
            spec["modes"].append(mode)
            spec["mean"].append(float(t.mean()))
            spec["std"].append(float(t.std() + 1e-30))
            spec["min"].append(float(v.min()))
            spec["max"].append(float(v.max()))
            spec["values"].append([float(u) for u in uniq] if mode == "cls" else None)
            spec["n_valid"].append(int(v.size))
        return cls(spec)

    @classmethod
    def default(cls):
        """无训练统计时的保守默认（全部 zscore，按官方全量范围的均匀分布近似）。"""
        mn = np.array([62.5, 65.0, -50.0, 1.0e-05, 1.7e-05, 1.0e-06, 2.0])
        mx = np.array([2000.0, 3500.0, 1000.0, 1.6e-05, 2.3e-05, 1.0e-05, 6.0])
        spec = {"names": PARAM_NAMES, "modes": ["zscore"] * 7,
                "mean": list((mn + mx) / 2), "std": list((mx - mn) / np.sqrt(12)),
                "min": list(mn), "max": list(mx), "values": [None] * 7,
                "decode": "expectation"}
        return cls(spec)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.spec, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def describe(self):
        rows = []
        for j, n in enumerate(self.names):
            extra = f"{len(self.values[j])} 类" if self.modes[j] == "cls" else \
                f"mean={self.mean[j]:.4g} std={self.std[j]:.4g}"
            rows.append(f"{n:>17}: {self.modes[j]:<10} [{self.vmin[j]:.4g}, {self.vmax[j]:.4g}] {extra}")
        return "\n".join(rows)

    # ------------------------------------------------------------------ head shape
    def head_spec(self):
        """参数头输出规格：回归维数 + 每个分类参数的类别数。"""
        return {"n_reg": len(self.reg_idx),
                "cls_sizes": [len(self.values[i]) for i in self.cls_idx]}

    # ------------------------------------------------------------------ torch helpers
    def _tensors(self, device):
        key = str(device)
        if key not in self._t:
            f = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
            self._t[key] = {
                "mean": f(self.mean), "std": f(self.std),
                "min": f(self.vmin), "max": f(self.vmax),
                "values": [f(self.values[i]) for i in self.cls_idx],
            }
        return self._t[key]

    def _fwd(self, x, j, t):
        if self.modes[j] == "log_zscore":
            x = torch.log(x.clamp_min(self.vmin[j] * 0.5))
        return (x - t["mean"][j]) / t["std"][j]

    def _inv(self, z, j, t):
        x = z * t["std"][j] + t["mean"][j]
        if self.modes[j] == "log_zscore":
            x = torch.exp(x)
        return x

    def encode(self, raw):
        """raw [B,7] SI -> (z [B,n_reg], cls_targets [B,n_cls] long)。无效位数值任意（由 valid 掩码屏蔽）。"""
        t = self._tensors(raw.device)
        raw = raw.float()
        z = torch.stack([self._fwd(raw[:, j], j, t) for j in self.reg_idx], -1) \
            if self.reg_idx else raw.new_zeros(raw.size(0), 0)
        cls_t = []
        for k, j in enumerate(self.cls_idx):
            vals = t["values"][k]
            cls_t.append((raw[:, j:j + 1] - vals[None]).abs().argmin(-1))
        cls_t = torch.stack(cls_t, -1) if cls_t else raw.new_zeros(raw.size(0), 0).long()
        return z, cls_t

    def decode(self, out):
        """头输出 dict -> SI 参数 [B,7]（裁剪到训练范围；尚未应用 pattern 掩码）。"""
        t = self._tensors(out["reg"].device)
        B = out["reg"].size(0)
        res = out["reg"].new_zeros(B, len(self.names)).float()
        for k, j in enumerate(self.reg_idx):
            res[:, j] = self._inv(out["reg"][:, k].float(), j, t)
        for k, j in enumerate(self.cls_idx):
            logits = out["cls"][k].float()
            vals = t["values"][k]
            if self.decode_mode == "argmax":
                res[:, j] = vals[logits.argmax(-1)]
            else:
                res[:, j] = (logits.softmax(-1) * vals[None]).sum(-1)
        return torch.maximum(torch.minimum(res, t["max"]), t["min"])

    def normalize_si(self, x):
        """SI -> 每参数标准化尺度（用于本地评估的无量纲 RMSE）。"""
        t = self._tensors(x.device)
        return (x - t["min"]) / (t["max"] - t["min"])
