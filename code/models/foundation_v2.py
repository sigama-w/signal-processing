"""EMFoundationV2：物理先验多视角 Tokenizer + Conformer 主干 + 任务提示/注意力池化多任务头

  共享（任务间共用，>95% 参数）: tokenizer（前端/stem/STFT/谱/ACF/星座/统计投影）+ backbone
  任务私有: task_embed（任务提示）+ 各任务头
  仅预训练: pretrain_head（SimMIM 线性重建头，微调/推理不导出、不计入共享率）

预训练（SimMIM 风格，Xie et al. CVPR 2022）
  * 对时域 token 做"块掩码"（默认 4 个 token=32 采样点为一块，掩码率 0.5）：
    信号过采样严重，随机单 token 掩码可由相邻 token 线性插值"作弊"恢复，块掩码迫使
    模型学习符号/脉冲级结构；
  * 被掩码 token 替换为 mask token 后**全序列**进入主干（卷积层需要连续序列，
    不能像 MAE 那样 gather 可见 token）；
  * 全局视图（频谱/星座/ACF/统计）从"掩码后的信号"计算——修复 v1 中可见 token
    通过整段 FFT（幅度+相位=无损变换）偷看被掩码内容的信息泄漏；
  * 重建目标：被掩码 patch 的归一化 IQ + 对应 STFT 帧对数幅度（时域+时频双目标）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads_v2 import ClsHeadV2, ParamHeadV2
from .layers.conformer import UnifiedEncoder
from .multiview import MultiViewTokenizer

N_PROTOCOL, N_MODULATION = 8, 23

# 调制族（层次化辅助监督）：0=PSK 1=QAM 2=FSK/CPM 3=模拟/其他 4=雷达
MOD_FAMILY = [
    3, 0, 0, 0, 1, 1, 1, 1,      # OOK BPSK QPSK 8PSK 16QAM 32QAM 64QAM 256QAM
    0, 2, 2, 2, 1, 0, 3, 2,      # OQPSK 2FSK 4FSK MSK 128QAM AQPSK CSS GFSK
    0, 3,                        # pi_4DQPSK FM
    4, 4, 4, 4, 4,               # Radar/*
]
N_FAMILY = 5


class EMFoundationV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        d = c.get("d_model", 384)
        self.patch = c.get("patch_size", 8)
        self.seq_len = c.get("seq_len", 1024)
        self.tokenizer = MultiViewTokenizer(
            d_model=d, seq_len=self.seq_len, patch_size=self.patch,
            stft_win=c.get("stft_win", 32), use_stft=c.get("use_stft", True),
            use_spec=c.get("use_spec", True), use_acf=c.get("use_acf", True),
            use_const=c.get("use_const", True), use_stats=c.get("use_stats", True),
            task_prompt=c.get("task_prompt", True), trim_padding=c.get("trim_padding", True),
            dft_impl=c.get("dft_impl", "matmul"), stem_width=c.get("stem_width", 64))
        self.backbone = UnifiedEncoder(
            d_model=d, n_layers=c.get("n_layers", 12), nhead=c.get("nhead", 6),
            ffn_hidden=c.get("ffn_hidden", 4 * d),
            layer_types=c.get("layer_types", ["conformer"]),
            dropout=c.get("dropout", 0.1), drop_path=c.get("drop_path", 0.1),
            conv_kernel=c.get("conv_kernel", 15))
        hc = c.get("head_config", {})
        ph = c.get("pool_heads", 4)

        def dims(t, h, l):
            x = hc.get(t, {}) or {}
            return x.get("hidden", h), x.get("layers", l)

        hp, lp = dims("protocol", 512, 1)
        hm, lm = dims("modulation", 512, 1)
        hq, lq = dims("parameter", 512, 2)
        pspec = c.get("param_head_spec") or {"n_reg": 7, "cls_sizes": []}
        self.task_heads = nn.ModuleDict({
            "protocol": ClsHeadV2(d, N_PROTOCOL, hp, lp, pool_heads=ph),
            "modulation": ClsHeadV2(d, N_MODULATION, hm, lm, pool_heads=ph,
                                    n_aux=N_FAMILY if c.get("mod_family_aux", True) else 0,
                                    proj_dim=c.get("supcon_dim", 0)),
            "parameter": ParamHeadV2(d, pspec["n_reg"], pspec["cls_sizes"], hq, lq,
                                     pool_heads=ph),
        })
        # 仅预训练使用的线性重建头
        stft_win = c.get("stft_win", 32)
        self.pretrain_head = nn.ModuleDict({
            "time": nn.Linear(d, 2 * self.patch),
            "stft": nn.Linear(d, stft_win),
        })
        self.mask_ratio = c.get("mask_ratio", 0.5)
        self.mask_block = c.get("mask_block", 4)
        self.stft_loss_weight = c.get("stft_loss_weight", 0.5)
        self.codec = None            # ParamCodec，由训练/推理脚本注入
        self.register_buffer("mod_family", torch.tensor(MOD_FAMILY), persistent=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 门控融合需要保持其专门初始化（偏向时域）
        if hasattr(self.tokenizer, "fuse"):
            nn.init.zeros_(self.tokenizer.fuse.gate.weight)
            nn.init.constant_(self.tokenizer.fuse.gate.bias, -1.0)

    # ------------------------------------------------------------- 参数核算
    def shared_parameter_count(self):
        priv = {id(p) for p in self.tokenizer.private_parameters()}
        n = sum(p.numel() for p in self.tokenizer.parameters() if id(p) not in priv)
        return n + sum(p.numel() for p in self.backbone.parameters())

    def head_parameter_count(self):
        n = sum(p.numel() for p in self.task_heads.parameters())
        return n + sum(p.numel() for p in self.tokenizer.private_parameters())

    def shared_ratio(self):
        s, h = self.shared_parameter_count(), self.head_parameter_count()
        return s / (s + h)

    def deploy_state_dict(self):
        """导出/推理用权重（去掉仅预训练使用的重建头）。"""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("pretrain_head")}

    # ------------------------------------------------------------- 前向
    def encode(self, iq, task=None, token_mask=None, sample_keep=None, power=None):
        tokens, pad, layout = self.tokenizer(iq, task, token_mask, sample_keep, power)
        ctx = self.backbone(tokens, pad, layout["time_slice"])
        return ctx, pad, layout

    def head_outputs(self, iq, task, return_all=False):
        ctx, pad, _ = self.encode(iq, task)
        head = self.task_heads[task]
        if task == "parameter":
            return head(ctx, pad)
        return head(ctx, pad, return_all=return_all)

    def forward(self, iq, sr=None, task="protocol"):
        """兼容 v1 接口 model(iq, sr, task)：分类返回 logits，参数返回 dict。sr 忽略。"""
        return self.head_outputs(iq, task)

    @torch.no_grad()
    def predict(self, iq, task):
        """推理：分类 -> 类别下标 [B]；参数 -> SI 原始单位 [B,7]（不适用参数已置 0）。"""
        out = self.head_outputs(iq, task)
        if task != "parameter":
            return out.argmax(-1)
        if self.codec is None:
            raise RuntimeError("parameter 推理需要 model.codec（weights/param_codec.json）")
        si = self.codec.decode(out)
        radar = out["pattern"].argmax(-1).bool()                     # 0=通信 1=雷达
        mask = torch.zeros_like(si)
        mask[:, :3] = (~radar).float()[:, None]
        mask[:, 3:] = radar.float()[:, None]
        return si * mask

    # ------------------------------------------------------------- 预训练
    def _block_mask(self, B, n_tok, device):
        b = self.mask_block
        nb = (n_tok + b - 1) // b
        k = max(1, int(round(self.mask_ratio * nb)))
        idx = torch.rand(B, nb, device=device).topk(k, dim=-1).indices
        blk = torch.zeros(B, nb, dtype=torch.bool, device=device)
        blk.scatter_(1, idx, True)
        return blk.repeat_interleave(b, dim=1)[:, :n_tok]

    def forward_pretrain(self, iq):
        """返回 (loss, {"loss_time":..., "loss_stft":...})。"""
        front = self.tokenizer.front
        patches, stft, p_raw, _ = front.pretrain_targets(iq)
        B, n_tok = patches.shape[:2]
        tmask = self._block_mask(B, n_tok, iq.device)                 # True=掩码
        keep = (~tmask).repeat_interleave(self.patch, dim=1).float()  # [B,N]
        ctx, pad, layout = self.encode(iq, None, tmask, keep, p_raw)
        s, e = layout["time_slice"]
        T = e - s
        h = ctx[:, s:e]
        w = (tmask[:, :T] & ~pad[:, s:e]).float()
        denom = w.sum().clamp_min(1.0)
        lt = (self.pretrain_head["time"](h).float() - patches[:, :T]).pow(2).mean(-1)
        ls = (self.pretrain_head["stft"](h).float() - stft[:, :T]).pow(2).mean(-1)
        loss_t = (lt * w).sum() / denom
        loss_s = (ls * w).sum() / denom
        loss = loss_t + self.stft_loss_weight * loss_s
        return loss, {"loss_time": loss_t.detach(), "loss_stft": loss_s.detach()}
