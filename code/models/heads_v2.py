"""v2 任务头（对应赛题任务 2：基座模型多任务适配）

统一表征 -> 任务语义映射的三层设计:
  1) 任务提示（task prompt，见 multiview.py）：CLS 叠加任务嵌入，同一共享主干
     按任务"调焦"，主干权重 100% 共享；
  2) 任务专属注意力池化 AttnPool：每个任务用自己的可学习 query 在全部 token
     （时域/频谱/星座/统计）上做注意力读出，再与 CLS 拼接——不同任务关注不同视角
     （调制看星座+高阶谱，参数看包络 ACF+时域沿，体制看频谱占用），
     而参数量只有 O(d)；
  3) 轻量 MLP 输出层；调制额外挂"调制族"辅助头（PSK/QAM/FSK/模拟/雷达 5 类，
     层次化监督）与可选 SupCon 投影头；参数头 = pattern 分类 + 回归/离散分类混合输出。
"""

import torch
import torch.nn as nn


def trunk(d_in, hidden, layers=1, dropout=0.1):
    """layers 个 (Linear -> GELU -> Dropout) 块，输出维 hidden（末端已激活）。"""
    mods, d = [], d_in
    for _ in range(max(1, layers)):
        mods += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
        d = hidden
    return nn.Sequential(*mods)


class AttnPool(nn.Module):
    """多头注意力池化：h 个头各一个可学习 query，key 为 LN(x)，value 为 x 按头切分。"""

    def __init__(self, d, nhead=4):
        super().__init__()
        assert d % nhead == 0
        self.h, self.dh = nhead, d // nhead
        self.ln = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(nhead, self.dh) * 0.02)

    def forward(self, x, pad_mask=None):
        B, L, D = x.shape
        k = self.ln(x).view(B, L, self.h, self.dh)
        score = torch.einsum("blhd,hd->bhl", k, self.q.to(k.dtype)) / self.dh ** 0.5
        if pad_mask is not None:
            score = score.masked_fill(pad_mask[:, None, :], float("-inf"))
        a = score.float().softmax(-1).to(x.dtype)                      # [B,h,L]
        v = x.view(B, L, self.h, self.dh)
        return torch.einsum("bhl,blhd->bhd", a, v).reshape(B, D)


class PooledFeature(nn.Module):
    """CLS ⊕ AttnPool -> [B, 2d]"""

    def __init__(self, d, nhead=4):
        super().__init__()
        self.pool = AttnPool(d, nhead)

    def forward(self, ctx, pad_mask=None):
        return torch.cat([ctx[:, 0], self.pool(ctx, pad_mask)], dim=-1)


class ClsHeadV2(nn.Module):
    def __init__(self, d, n_classes, hidden=512, layers=1, dropout=0.1, pool_heads=4,
                 n_aux=0, proj_dim=0):
        super().__init__()
        self.feat = PooledFeature(d, pool_heads)
        self.trunk = trunk(2 * d, hidden, layers, dropout)
        self.out = nn.Linear(hidden, n_classes)
        self.aux = nn.Linear(hidden, n_aux) if n_aux else None
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, proj_dim)) if proj_dim else None

    def forward(self, ctx, pad_mask=None, return_all=False):
        h = self.trunk(self.feat(ctx, pad_mask))
        logits = self.out(h)
        if not return_all:
            return logits
        return {"logits": logits, "hidden": h,
                "aux": self.aux(h) if self.aux is not None else None,
                "proj": self.proj(h) if self.proj is not None else None}


class ParamHeadV2(nn.Module):
    """参数头：pattern(通信/雷达) 2 类 + 回归 n_reg 维 + 每个离散参数一个分类输出。"""

    def __init__(self, d, n_reg, cls_sizes, hidden=512, layers=2, dropout=0.1, pool_heads=4):
        super().__init__()
        self.feat = PooledFeature(d, pool_heads)
        self.trunk = trunk(2 * d, hidden, layers, dropout)
        self.pattern = nn.Linear(hidden, 2)
        self.reg = nn.Linear(hidden, n_reg)
        self.cls = nn.ModuleList([nn.Linear(hidden, c) for c in cls_sizes])

    def forward(self, ctx, pad_mask=None):
        h = self.trunk(self.feat(ctx, pad_mask))
        return {"pattern": self.pattern(h), "reg": self.reg(h),
                "cls": [c(h) for c in self.cls]}
