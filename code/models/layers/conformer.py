"""v2 主干层：SDPA 多头注意力 + Conformer 卷积模块（纯 torch，可走融合算子）

为什么用 Conformer 式 "注意力 + 深度卷积" 而不是 MLA / GatedDelta / Mamba
-------------------------------------------------------------------------
* 本任务 token 序列只有 ~130-200 个：O(L^2) 注意力本身很便宜，线性注意力 / SSM
  （GDA、Mamba）的复杂度优势完全用不上；而它们在本仓库是逐时间步 python for 循环，
  训练/推理都慢一个数量级，且对昇腾不友好。
* MLA 的核心收益是自回归解码时压缩 KV-cache；编码器分类任务没有 KV-cache，
  低秩瓶颈反而削弱表达能力并增加矩阵乘次数。
* IQ 信号兼具"局部波形结构（符号、跳变、脉冲沿）"与"全局结构（PRI、调制统计）"，
  与语音高度相似。Conformer（Gulati et al., Interspeech 2020）在语音上已证明
  卷积捕局部 + 注意力捕全局的组合优于纯 Transformer；电磁领域 SpectrumFM
  （Zhou et al., 2025）同样采用 CNN + MHSA 编码器。

层类型（layer_types 列表的取值）:
  "attn"       pre-norm MHSA + SwiGLU FFN（标准 Transformer 层）
  "conformer"  1/2 FFN -> MHSA -> ConvModule -> 1/2 FFN -> LN（Macaron 结构）
  "conv"       ConvModule + FFN（无注意力，最便宜，适合放在浅层）

ConvModule 只作用于"时域对齐 token 段"（time segment）：频谱 / 星座 / ACF token
不是时间序列，跨模态做卷积没有物理意义。padding 位置在卷积前清零，
因此 512 点与 1024 点样本的数值完全一致（不受补零影响）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.fused import attention, swiglu


class DropPath(nn.Module):
    """随机深度（Stochastic Depth），深层网络正则化。"""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = torch.rand(x.size(0), 1, 1, device=x.device, dtype=x.dtype) >= self.p
        return x * keep / (1 - self.p)


class SwiGLUFFN(nn.Module):
    def __init__(self, d, hidden, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(swiglu(self.gate(x), self.up(x))))


class MHSA(nn.Module):
    """多头自注意力（SDPA / NPU FlashAttention 分发，支持 key padding mask）。"""

    def __init__(self, d, nhead, dropout=0.0):
        super().__init__()
        assert d % nhead == 0
        self.h, self.dh = nhead, d // nhead
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, pad_mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        o = attention(q, k, v, pad_mask, self.dropout, self.training)    # [B,H,L,dh]
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


class ConvModule(nn.Module):
    """Conformer 卷积模块：LN -> PW(2d)+GLU -> DW Conv(k) -> LN -> SiLU -> PW。

    用 LayerNorm 代替原论文 BatchNorm：对 padding / 小 batch 推理 / NPU 更稳。
    """

    def __init__(self, d, kernel=15, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.ln2 = nn.LayerNorm(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, time_slice, pad_mask=None):
        s, e = time_slice
        t = x[:, s:e]
        h = F.glu(self.pw1(self.ln(t)), dim=-1)
        if pad_mask is not None:
            h = h.masked_fill(pad_mask[:, s:e, None], 0.0)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.pw2(F.silu(self.ln2(h))))
        return F.pad(h, (0, 0, s, x.size(1) - e))      # 非时域段补零（无 in-place）


class AttnBlock(nn.Module):
    def __init__(self, d, nhead, ffn, dropout=0.0, drop_path=0.0, **_):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(d), MHSA(d, nhead, dropout)
        self.n2, self.ff = nn.LayerNorm(d), SwiGLUFFN(d, ffn, dropout)
        self.dp = DropPath(drop_path)

    def forward(self, x, pad_mask, time_slice):
        x = x + self.dp(self.attn(self.n1(x), pad_mask))
        return x + self.dp(self.ff(self.n2(x)))


class ConformerBlock(nn.Module):
    def __init__(self, d, nhead, ffn, dropout=0.0, drop_path=0.0, conv_kernel=15):
        super().__init__()
        self.nf1, self.ff1 = nn.LayerNorm(d), SwiGLUFFN(d, ffn, dropout)
        self.na, self.attn = nn.LayerNorm(d), MHSA(d, nhead, dropout)
        self.conv = ConvModule(d, conv_kernel, dropout)
        self.nf2, self.ff2 = nn.LayerNorm(d), SwiGLUFFN(d, ffn, dropout)
        self.out = nn.LayerNorm(d)
        self.dp = DropPath(drop_path)

    def forward(self, x, pad_mask, time_slice):
        x = x + 0.5 * self.dp(self.ff1(self.nf1(x)))
        x = x + self.dp(self.attn(self.na(x), pad_mask))
        x = x + self.dp(self.conv(x, time_slice, pad_mask))
        x = x + 0.5 * self.dp(self.ff2(self.nf2(x)))
        return self.out(x)


class ConvBlock(nn.Module):
    def __init__(self, d, nhead, ffn, dropout=0.0, drop_path=0.0, conv_kernel=15):
        super().__init__()
        self.conv = ConvModule(d, conv_kernel, dropout)
        self.n2, self.ff = nn.LayerNorm(d), SwiGLUFFN(d, ffn, dropout)
        self.dp = DropPath(drop_path)

    def forward(self, x, pad_mask, time_slice):
        x = x + self.dp(self.conv(x, time_slice, pad_mask))
        return x + self.dp(self.ff(self.n2(x)))


V2_LAYER_TYPES = {"attn": AttnBlock, "conformer": ConformerBlock, "conv": ConvBlock}
_ALIASES = {"fa": "attn", "full_attention": "attn", "conv1d": "conv"}


def build_v2_layer(layer_type, d, nhead, ffn, dropout=0.0, drop_path=0.0, conv_kernel=15):
    t = _ALIASES.get(layer_type.lower(), layer_type.lower())
    if t not in V2_LAYER_TYPES:
        raise ValueError(f"v2 不支持层类型 {layer_type}，可选 {sorted(V2_LAYER_TYPES)}"
                         "（mla/gda/mamba 仅在 v1 hybrid backbone 中保留）")
    return V2_LAYER_TYPES[t](d, nhead, ffn, dropout=dropout, drop_path=drop_path,
                             conv_kernel=conv_kernel)


class UnifiedEncoder(nn.Module):
    """v2 主干：按 layer_types 逐层堆叠（短列表循环铺满 n_layers），末端 LayerNorm。"""

    def __init__(self, d_model=384, n_layers=12, nhead=6, ffn_hidden=1536,
                 layer_types=("conformer",), dropout=0.1, drop_path=0.1, conv_kernel=15):
        super().__init__()
        types = list(layer_types)
        if len(types) > n_layers:
            raise ValueError(f"layer_types 长度 {len(types)} > n_layers {n_layers}")
        pattern = [types[i % len(types)] for i in range(n_layers)]
        dprs = [drop_path * i / max(1, n_layers - 1) for i in range(n_layers)]
        self.layer_types = pattern
        self.layers = nn.ModuleList([
            build_v2_layer(t, d_model, nhead, ffn_hidden, dropout, dp, conv_kernel)
            for t, dp in zip(pattern, dprs)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None, time_slice=(0, 0)):
        for layer in self.layers:
            x = layer(x, pad_mask, time_slice)
        return self.norm(x)
