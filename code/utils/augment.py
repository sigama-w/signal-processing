"""按任务区分的 GPU 批量数据增强（只用不改变标签的物理变换）

| 变换                    | protocol | modulation | parameter | 说明 |
|-------------------------|----------|------------|-----------|------|
| 随机相位旋转 e^{jφ}      | ✓        | ✓          | ✓         | 载波初相未知，所有标签均不变 |
| 加性高斯噪声(降 SNR)     | ✓        | ✓          | ✓(弱)     | 提升低 SNR 鲁棒性；参数标签不变 |
| 共轭(IQ 镜像)           | ✗        | 可选        | ✗         | 会翻转频偏符号/频谱方向 -> 参数任务禁用 |
| 时移/裁剪               | ✗        | ✗          | ✗         | 会改变 time_delay；且补零布局需保持 |

全部在 GPU 上对整个 batch 向量化执行，几乎零开销。补零区域保持为 0
（模型靠右侧补零推断有效长度）。
"""

import math

import torch

DEFAULT_POLICY = {
    "protocol": {"phase": 1.0, "noise": 0.3, "noise_db": [0.0, 10.0], "conj": 0.0},
    "modulation": {"phase": 1.0, "noise": 0.3, "noise_db": [0.0, 10.0], "conj": 0.0},
    "parameter": {"phase": 1.0, "noise": 0.15, "noise_db": [3.0, 15.0], "conj": 0.0},
    "pretrain": {"phase": 1.0, "noise": 0.0, "noise_db": [0.0, 10.0], "conj": 0.0},
}


def _valid(iq):
    return (iq.abs().sum(-1, keepdim=True) > 0).to(iq.dtype)


@torch.no_grad()
def augment_iq(iq, task, policy=None):
    """iq: [B,N,2]（设备上）。policy: {task: {...}}，None 用默认。"""
    pol = (policy or DEFAULT_POLICY).get(task)
    if not pol:
        return iq
    B = iq.size(0)
    dev, dt = iq.device, iq.dtype
    valid = _valid(iq)
    out = iq
    if pol.get("phase", 0) > 0:
        apply = (torch.rand(B, device=dev) < pol["phase"]).to(dt)
        phi = torch.rand(B, device=dev) * 2 * math.pi * apply
        c, s = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
        i, q = out[..., 0], out[..., 1]
        out = torch.stack([i * c - q * s, i * s + q * c], dim=-1)
    if pol.get("conj", 0) > 0:
        flip = (torch.rand(B, device=dev) < pol["conj"])[:, None]
        out = torch.stack([out[..., 0], torch.where(flip, -out[..., 1], out[..., 1])], -1)
    if pol.get("noise", 0) > 0:
        lo, hi = pol.get("noise_db", [0.0, 10.0])
        apply = (torch.rand(B, device=dev) < pol["noise"]).to(dt)
        L = valid.sum(1).clamp_min(1.0)                                   # [B,1]
        p = (out.pow(2).sum(-1, keepdim=True) * valid).sum(1) / L         # [B,1]
        snr_db = lo + (hi - lo) * torch.rand(B, 1, device=dev)
        sigma = torch.sqrt(p / (10 ** (snr_db / 10)) / 2) * apply[:, None]
        out = out + torch.randn_like(out) * sigma[:, :, None]
    return out * valid
