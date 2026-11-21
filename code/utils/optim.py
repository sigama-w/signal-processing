"""优化器工厂：AdamW（默认）/ Muon（可选，2D 隐层权重用 Muon，其余用 AdamW）+ EMA

Muon（Keller Jordan 2024；Moonshot "Muon is Scalable for LLM Training", 2025）:
  对 2D 权重的动量做 Newton-Schulz 正交化后更新，相当于在谱范数意义下做"最速下降"。
  * 适用：Transformer/卷积主干中的矩阵权重（Linear、Conv 展平为 2D）
  * 不适用：embedding、位置编码、LayerNorm/偏置、输出分类层（仍用 AdamW）
  * 按 Moonshot 做法把更新 RMS 匹配到 AdamW（乘 0.2*sqrt(max(m,n))），
    因此可直接沿用 AdamW 的 lr / weight_decay，便于公平对比。
  纯 matmul 实现，对昇腾 NPU 友好（无特殊算子）。状态只有 1 份动量（AdamW 2 份），
  省 4B/参数显存。

本仓库建议：先用 AdamW 跑通基线；Muon 作为消融（预训练阶段更可能受益，
小模型/微调阶段收益通常有限）。
"""

import copy
import math

import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """五次 Newton-Schulz 迭代近似 UV^T（Keller Jordan 系数），bf16 计算。"""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda or G.device.type == "npu" else G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """单卡/手动 all-reduce 场景的 Muon（梯度已在外部同步）。"""

    def __init__(self, params, lr=1e-3, weight_decay=0.0, momentum=0.95, nesterov=True,
                 ns_steps=5):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(grad)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(grad)
                upd = grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                shape = upd.shape
                u2 = upd.reshape(shape[0], -1)
                o = zeropower_via_newtonschulz5(u2, g["ns_steps"])
                scale = 0.2 * math.sqrt(max(u2.size(0), u2.size(1)))
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o.reshape(shape), alpha=-g["lr"] * scale)
        return loss


class CombinedOptimizer:
    """把多个优化器包装成一个（统一 zero_grad/step/state_dict，兼容 LambdaLR）。"""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    @property
    def defaults(self):
        return self.optimizers[0].defaults

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["opts"]):
            o.load_state_dict(s)

    def add_param_group(self, group):            # LambdaLR 不会调用，占位
        raise NotImplementedError


_NO_DECAY_KEYS = ("pos_", "type_embed", "task_embed", "cls_token", "mask_token",
                  "pos_encoding", "sr_embed")


def _no_decay(name, p):
    """1D 参数（范数/偏置）与位置/类型/任务嵌入、特殊 token 不做权重衰减。"""
    return p.ndim < 2 or name.endswith(".bias") or ".norm" in name or \
        any(k in name for k in _NO_DECAY_KEYS)


def build_optimizer(named_params, tcfg):
    """named_params: list[(name, param)]（已过滤 requires_grad）。

    tcfg.optimizer: "adamw"(默认) | "muon"
    """
    lr, wd = tcfg["lr"], tcfg.get("weight_decay", 0.05)
    betas = tuple(tcfg.get("betas", (0.9, 0.95)))
    name = str(tcfg.get("optimizer", "adamw")).lower()
    decay = [p for n, p in named_params if not _no_decay(n, p)]
    no_decay = [p for n, p in named_params if _no_decay(n, p)]
    if name == "adamw":
        return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=lr, betas=betas)
    if name == "muon":
        # 主干/stem 中的 2D+ 矩阵用 Muon；头部输出层、嵌入、范数等用 AdamW
        muon_p, adam_d, adam_nd = [], [], []
        for n, p in named_params:
            is_head = n.startswith("task_heads") or n.startswith("pretrain_head")
            if p.ndim >= 2 and not is_head and not _no_decay(n, p):
                muon_p.append(p)
            elif _no_decay(n, p):
                adam_nd.append(p)
            else:
                adam_d.append(p)
        opts = [Muon(muon_p, lr=lr, weight_decay=wd,
                     momentum=tcfg.get("muon_momentum", 0.95))]
        opts.append(torch.optim.AdamW([{"params": adam_d, "weight_decay": wd},
                                       {"params": adam_nd, "weight_decay": 0.0}],
                                      lr=lr, betas=betas))
        return CombinedOptimizer(opts)
    raise ValueError(f"未知 optimizer: {name}")


def build_scheduler(optimizer, total_steps, warmup_steps, min_ratio=0.01):
    """线性 warmup + 余弦退火（按优化步计数）。"""
    warmup_steps = max(1, int(warmup_steps))

    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    lam = [fn] * len(optimizer.param_groups)
    if isinstance(optimizer, CombinedOptimizer):
        return _MultiScheduler([torch.optim.lr_scheduler.LambdaLR(o, fn)
                                for o in optimizer.optimizers])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lam)


class _MultiScheduler:
    def __init__(self, scheds):
        self.scheds = scheds

    def step(self):
        for s in self.scheds:
            s.step()

    def get_last_lr(self):
        return self.scheds[0].get_last_lr()

    def state_dict(self):
        return {"scheds": [s.state_dict() for s in self.scheds]}

    def load_state_dict(self, sd):
        for s, d in zip(self.scheds, sd["scheds"]):
            s.load_state_dict(d)


class ModelEMA:
    """参数指数滑动平均（评估/导出用 EMA 权重，通常 +0.3~1% 分类精度、更平滑的早停曲线）。"""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))   # 前期快速跟随
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return {"ema": self.ema.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd["ema"])
        self.updates = sd.get("updates", 0)
