"""v2 多视角 Tokenizer：时-频双流编码 + 三级跨域融合（对应赛题任务 1）

token 布局（单流联合序列，T 为时域 token 数 = 有效长度/patch，批内按最长样本裁剪）:

  [CLS] [STAT] [TIME_1 .. TIME_T] [SPEC_1 .. SPEC_32] [ACF_1 .. ACF_16] [CONST_1 .. CONST_16]
    |      |         |                    |                   |                 |
  全局   统计量   时域卷积 stem         全局谱(|X|,          包络自相关       星座密度图
  读出   特征     ⊕(门控)STFT 帧       |X²|,|X⁴| 3 通道)    (PRI/脉冲数)     (QAM/PSK 阶数)

三级融合（对应任务书"通道、空间或注意力层面的动态对齐与融合"）:
  1) 通道级：时域 patch token 与同一时刻的 STFT 帧 token（帧移=patch，一一对齐）
     做 SE 式动态门控融合 g=σ(W[t;s])，t' = t + g ⊙ s
  2) 空间级：全局频谱 3 个非线性谱在同一频点按通道堆叠成一个 token（频点对齐）；
     时间/频率/延迟/星座 token 各有独立位置编码 + 模态类型编码
  3) 注意力级：全部 token 进入同一主干，每层自注意力都在做跨域交互
     （时域 token 可直接查询频谱线、星座簇，反之亦然）

每个视角都可以在 config 中独立开关（use_stft/use_spec/use_acf/use_const/use_stats），
直接支撑技术报告中的消融实验。
"""

import torch
import torch.nn as nn

from .dsp import N_STATS, SignalFrontEnd

TYPE_IDS = {"cls": 0, "stat": 1, "time": 2, "spec": 3, "acf": 4, "const": 5}
N_TASKS = 3            # 任务提示（task prompt）个数：protocol / modulation / parameter
TASK_IDS = {"protocol": 0, "modulation": 1, "parameter": 2}


class ConvStem(nn.Module):
    """早期卷积 stem（Xiao et al. 2021, "Early convolutions help transformers see better"）。

    [B,C,N] -> [B,N/patch,d]，总步长 = patch；比单层线性 patch embedding 保留更多
    patch 内的细粒度结构（符号跳变、脉冲前后沿——time_delay 只有几个采样点量级）。
    """

    STRIDES = {4: (2, 2), 8: (2, 4), 16: (4, 4)}

    def __init__(self, c_in, d, patch=8, width=64):
        super().__init__()
        s1, s2 = self.STRIDES[patch]
        self.net = nn.Sequential(
            nn.Conv1d(c_in, width, 7, padding=3), nn.GELU(),
            nn.Conv1d(width, 2 * width, 2 * s1 + 1, stride=s1, padding=s1), nn.GELU(),
            nn.Conv1d(2 * width, d, s2, stride=s2),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.net(x).transpose(1, 2))


class GatedChannelFusion(nn.Module):
    """通道级动态对齐融合：t' = t + σ(W[t; s]) ⊙ s（逐 token、逐通道门控）。"""

    def __init__(self, d):
        super().__init__()
        self.gate = nn.Linear(2 * d, d)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)   # 初始偏向时域，训练中逐步打开 STFT 支路
        self.norm = nn.LayerNorm(d)

    def forward(self, t, s):
        g = torch.sigmoid(self.gate(torch.cat([t, s], dim=-1)))
        return self.norm(t + g * s)


class MultiViewTokenizer(nn.Module):
    def __init__(self, d_model=384, seq_len=1024, patch_size=8, stft_win=32,
                 spec_tokens=32, acf_lags=512, acf_tokens=16, const_bins=32, const_patch=8,
                 use_stft=True, use_spec=True, use_acf=True, use_const=True, use_stats=True,
                 task_prompt=True, trim_padding=True, dft_impl="matmul", stem_width=64):
        super().__init__()
        self.d = d_model
        self.patch = patch_size
        self.n_time = seq_len // patch_size
        self.use = dict(stft=use_stft, spec=use_spec, acf=use_acf, const=use_const,
                        stats=use_stats)
        self.trim_padding = trim_padding
        self.task_prompt = task_prompt
        self.front = SignalFrontEnd(seq_len, patch_size, stft_win, acf_lags, const_bins,
                                    dft_impl=dft_impl)
        # --- 时域流 ---
        self.stem = ConvStem(self.front.n_time_channels, d_model, patch_size, stem_width)
        self.pos_time = nn.Parameter(torch.zeros(1, self.n_time, d_model))
        # --- 时频流（与时域 token 一一对齐） ---
        if use_stft:
            self.stft_proj = nn.Sequential(nn.Linear(stft_win, d_model), nn.LayerNorm(d_model))
            self.fuse = GatedChannelFusion(d_model)
        # --- 全局频谱流（3 个非线性谱按频点堆叠） ---
        if use_spec:
            assert seq_len % spec_tokens == 0
            self.spec_tokens = spec_tokens
            self.spec_proj = nn.Sequential(nn.Linear(3 * seq_len // spec_tokens, d_model),
                                           nn.LayerNorm(d_model))
            self.pos_spec = nn.Parameter(torch.zeros(1, spec_tokens, d_model))
        # --- 包络自相关流 ---
        if use_acf:
            assert acf_lags % acf_tokens == 0
            self.acf_tokens = acf_tokens
            self.acf_proj = nn.Sequential(nn.Linear(acf_lags // acf_tokens, d_model),
                                          nn.LayerNorm(d_model))
            self.pos_acf = nn.Parameter(torch.zeros(1, acf_tokens, d_model))
        # --- 星座图流（2D patch） ---
        if use_const:
            assert const_bins % const_patch == 0
            self.cp = const_patch
            self.n_const = (const_bins // const_patch) ** 2
            self.const_proj = nn.Sequential(nn.Linear(const_patch * const_patch, d_model),
                                            nn.LayerNorm(d_model))
            self.pos_const = nn.Parameter(torch.zeros(1, self.n_const, d_model))
        # --- 统计量 token ---
        if use_stats:
            self.stat_proj = nn.Sequential(nn.Linear(N_STATS, d_model), nn.GELU(),
                                           nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_embed = nn.Embedding(len(TYPE_IDS), d_model)
        if task_prompt:
            # 任务提示：加到 CLS 上（任务私有参数，计入 head 参数量）
            self.task_embed = nn.Embedding(N_TASKS, d_model)
            nn.init.normal_(self.task_embed.weight, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        for p in (self.pos_time, self.cls_token, self.mask_token):
            nn.init.trunc_normal_(p, std=0.02)
        for name in ("pos_spec", "pos_acf", "pos_const"):
            if hasattr(self, name):
                nn.init.trunc_normal_(getattr(self, name), std=0.02)

    # ------------------------------------------------------------------
    def private_parameters(self):
        """任务私有参数（task prompt）。"""
        return list(self.task_embed.parameters()) if self.task_prompt else []

    def _typed(self, x, kind):
        return x + self.type_embed.weight[TYPE_IDS[kind]]

    def forward(self, iq, task=None, token_mask=None, sample_keep=None, power=None):
        """
        Args:
            iq: [B,N,2]
            task: 任务名（str）或 None（预训练）
            token_mask: [B,n_time] bool，True=该时域 token 被掩码（SimMIM 预训练）
            sample_keep / power: 透传给前端（预训练防泄漏，见 dsp.SignalFrontEnd）
        Returns:
            tokens [B,Ltot,d], pad_mask [B,Ltot] (True=忽略), layout dict
        """
        fe = self.front(iq, sample_keep=sample_keep, power=power)
        B = iq.size(0)
        t = self.stem(fe["time_ch"].to(self.cls_token.dtype))               # [B,T,d]
        if self.use["stft"]:
            t = self.fuse(t, self.stft_proj(fe["stft"]))
        if token_mask is not None:      # 先替换为 mask token，再加位置编码（保留被掩码位置信息）
            t = torch.where(token_mask[..., None], self.mask_token.expand_as(t).to(t.dtype), t)
        t = t + self.pos_time[:, : t.size(1)]
        tok_valid = fe["length"] // self.patch                               # [B]
        T = t.size(1)
        if self.trim_padding:
            T = int(tok_valid.max().item())
            t = t[:, :T]
        time_pad = torch.arange(T, device=iq.device)[None, :] >= tok_valid[:, None]

        cls = self.cls_token.expand(B, -1, -1)
        if self.task_prompt and task is not None:
            cls = cls + self.task_embed.weight[TASK_IDS[task]]
        parts = [self._typed(cls, "cls")]
        if self.use["stats"]:
            parts.append(self._typed(self.stat_proj(fe["stats"]).unsqueeze(1), "stat"))
        n_head = len(parts)
        parts.append(self._typed(t, "time"))
        if self.use["spec"]:
            sp = fe["spec"].view(B, 3, self.spec_tokens, -1).permute(0, 2, 1, 3).reshape(
                B, self.spec_tokens, -1)
            parts.append(self._typed(self.spec_proj(sp) + self.pos_spec, "spec"))
        if self.use["acf"]:
            ac = fe["acf"].view(B, self.acf_tokens, -1)
            parts.append(self._typed(self.acf_proj(ac) + self.pos_acf, "acf"))
        if self.use["const"]:
            G, p = fe["const"].size(-1), self.cp
            c = fe["const"].view(B, G // p, p, G // p, p).permute(0, 1, 3, 2, 4).reshape(
                B, self.n_const, p * p)
            parts.append(self._typed(self.const_proj(c) + self.pos_const, "const"))
        tokens = torch.cat(parts, dim=1)
        n_tail = tokens.size(1) - n_head - T
        pad = torch.cat([torch.zeros(B, n_head, dtype=torch.bool, device=iq.device),
                         time_pad,
                         torch.zeros(B, n_tail, dtype=torch.bool, device=iq.device)], dim=1)
        layout = {"time_slice": (n_head, n_head + T), "T": T, "frontend": fe}
        return tokens, pad, layout
