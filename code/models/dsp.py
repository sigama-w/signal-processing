"""物理先验信号前端（v2，无可学习参数，纯 torch 实数运算）

设计目标
--------
1. 把雷达/通信信号处理里"专家知识"做成确定性的前端，给 Transformer 提供
   多视角输入（时域 / 时频 STFT / 全局频谱 / 非线性谱 / 包络自相关 / 星座图 / 统计量）。
2. **全部用实数张量 + 矩阵乘法实现 DFT**（不依赖 torch.fft 与复数 dtype）：
     - 昇腾 NPU 的 Cube 单元对 GEMM 最友好，复数/FFT 算子支持不完整；
     - ONNX / ATC / TensorRT 导出时只有 MatMul，没有 DFT 算子兼容问题；
     - 1024 点 DFT = [B,1024]x[1024,1024] 矩阵乘，相对 Transformer 主干开销 <3%。
   通过 impl="fft" 可切换回 torch.fft 做数值对照（tests/test_v2_pipeline.py）。
3. 前端强制 fp32（autocast 关闭），避免 bf16 下频谱精度崩坏。

主要输出（SignalFrontEnd.forward 返回 dict）
------------------------------------------
  valid      [B,N]  有效采样点掩码（512 点样本右侧补零部分为 0）
  length     [B]    有效长度（已向上取整到 patch 整数倍）
  x          [B,N,2] 单位功率归一化原始信号（保留载频偏移 -> frequency_shift 可估计）
  xc         [B,N,2] 载频粗校正 + 带外噪声抑制后的信号（星座/高阶统计量更干净）
  time_ch    [B,C,N] 时域多通道（I,Q,Ic,Qc,|x|,瞬时频率）
  stft       [B,T,W] 对数 STFT 幅度（帧与 patch 一一对齐，T=N/patch）
  spec       [B,3,N] 对数谱: |X|, |FFT(xc^2)|, |FFT(xc^4)|（fftshift 后）
  acf        [B,A]   包络自相关（lag 0..A-1，归一化）——PRI/脉冲数的强线索
  const      [B,G,G] 星座密度图（校正后信号的 2D 直方图，log1p）
  stats      [B,F]   统计特征（功率、CFO、占用带宽、SNR、高阶矩/累积量、占空比…）

参考: Swami & Sadler, "Hierarchical digital modulation classification using
cumulants", IEEE TCOM 2000（高阶累积量）；经典 M 次方谱线法估计 PSK/QAM 阶数与频偏。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_STATS = 16


def _symlog(x):
    """sign(x)*log1p(|x|)：把跨数量级的统计量压到 O(1) 区间。"""
    return torch.sign(x) * torch.log1p(x.abs())


def valid_mask_from_iq(iq, patch_size=8):
    """由右侧补零推断有效长度（官方 512 点样本被补零到 1024）。

    Returns:
        valid [B,N] float, length [B] long（向上取整到 patch 倍数, >= patch）
    """
    B, N, _ = iq.shape
    nz = (iq.abs().sum(-1) > 0).float()                         # [B,N]
    idx = torch.arange(1, N + 1, device=iq.device, dtype=torch.float32)
    last = (nz * idx).amax(-1)                                  # 最后一个非零点下标+1
    length = torch.ceil(last / patch_size) * patch_size
    length = length.clamp(min=patch_size, max=N).long()
    valid = (torch.arange(N, device=iq.device)[None, :] < length[:, None]).float()
    return valid, length


class RealDFT(nn.Module):
    """实数矩阵乘实现的 DFT / IDFT（NPU / ONNX 友好）。

    X = sum_n x_n e^{-j2πkn/N}:
      Xr = xr@C + xi@S ;  Xi = xi@C - xr@S   (C=cos, S=sin, 对称矩阵)
    impl="fft" 时改用 torch.fft（仅用于 CUDA 对照/调试）。
    """

    def __init__(self, n, impl="matmul"):
        super().__init__()
        self.n = n
        self.impl = impl
        k = torch.arange(n, dtype=torch.float64)
        ang = 2 * math.pi * torch.outer(k, k) / n
        self.register_buffer("C", torch.cos(ang).float(), persistent=False)
        self.register_buffer("S", torch.sin(ang).float(), persistent=False)

    def forward(self, xr, xi, inverse=False):
        """xr/xi: [..., n] -> (Xr, Xi) [..., n]；inverse=True 时含 1/n 归一化。"""
        if self.impl == "fft":
            z = torch.complex(xr.float(), xi.float())
            Z = torch.fft.ifft(z, dim=-1) if inverse else torch.fft.fft(z, dim=-1)
            return Z.real, Z.imag
        C, S = self.C, self.S
        if inverse:   # e^{+j}: Xr = xr C - xi S ; Xi = xi C + xr S
            return (xr @ C - xi @ S) / self.n, (xi @ C + xr @ S) / self.n
        return xr @ C + xi @ S, xi @ C - xr @ S


def _fftshift(x):
    n = x.shape[-1]
    return torch.roll(x, shifts=n // 2, dims=-1)


def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


class SignalFrontEnd(nn.Module):
    """多视角确定性信号前端（见模块文档）。全部运算在 fp32 下执行。"""

    def __init__(self, seq_len=1024, patch_size=8, stft_win=32, acf_lags=512,
                 const_bins=32, const_range=2.5, floor_ratio=4.0, dft_impl="matmul"):
        super().__init__()
        assert seq_len % patch_size == 0
        assert stft_win >= patch_size and (stft_win - patch_size) % 2 == 0
        self.N = seq_len
        self.patch = patch_size
        self.win = stft_win
        self.acf_lags = acf_lags
        self.G = const_bins
        self.R = const_range
        self.floor_ratio = floor_ratio
        self.dft = RealDFT(seq_len, dft_impl)
        self.dft_acf = RealDFT(2 * seq_len, dft_impl)    # 线性自相关需 2N 点
        self.dft_win = RealDFT(stft_win, dft_impl)
        n = torch.arange(stft_win, dtype=torch.float64)
        self.register_buffer("hann", (0.5 - 0.5 * torch.cos(2 * math.pi * n / stft_win)).float(),
                             persistent=False)
        f = (torch.arange(seq_len, dtype=torch.float32) - seq_len // 2) / seq_len
        self.register_buffer("freq_axis", f, persistent=False)   # fftshift 后的归一化频率

    # ------------------------------------------------------------------
    @property
    def n_time_channels(self):
        return 6

    def n_tokens(self):
        return self.N // self.patch

    # ------------------------------------------------------------------
    def _normalize(self, iq, valid, power=None):
        """单位功率归一化（按有效长度），返回 x [B,N,2]、原始功率 p [B]。"""
        L = valid.sum(-1).clamp_min(1.0)
        p_raw = ((iq ** 2).sum(-1) * valid).sum(-1) / L
        p = p_raw if power is None else power
        x = iq / torch.sqrt(p + 1e-20)[:, None, None]
        return x * valid[..., None], p_raw

    def _spectrum(self, xr, xi, L):
        """|X|^2/L 的 fftshift 功率谱 + 平滑 + 噪底 + 占用掩码。"""
        Xr, Xi = self.dft(xr, xi)
        psd = (Xr ** 2 + Xi ** 2) / L[:, None]
        psd_s = _fftshift(psd)                                   # [B,N] 频率 -N/2..N/2-1
        k = 9
        sm = F.avg_pool1d(F.pad(psd_s[:, None], (k // 2, k // 2), mode="circular"),
                          k, stride=1)[:, 0]
        # 噪底 = 最低 50% 频点的均值（topk 可导出 ONNX，避免 median）
        low = torch.topk(sm, self.N // 2, dim=-1, largest=False).values
        floor = low.mean(-1).clamp_min(1e-12)                    # [B]
        occ = (sm > floor[:, None] * self.floor_ratio).float()
        # 膨胀一下避免截断带边
        occ = F.max_pool1d(F.pad(occ[:, None], (4, 4), mode="circular"), 9, stride=1)[:, 0]
        none = (occ.sum(-1, keepdim=True) == 0).float()
        occ = occ + none                                          # 全空时退化为全通
        occ = occ.clamp(max=1.0)
        return Xr, Xi, psd_s, sm, floor, occ

    def _carrier_correct(self, Xr, Xi, sm, floor, occ, valid):
        """频域加权质心估计 CFO -> 频域整数 bin 平移 + 带外置零 -> IDFT。"""
        w = F.relu(sm - floor[:, None]) * occ
        cfo = (w * self.freq_axis[None]).sum(-1) / w.sum(-1).clamp_min(1e-12)   # cycles/sample
        shift = torch.round(cfo * self.N).long()                                  # [B]
        Xr_s, Xi_s = _fftshift(Xr) * occ, _fftshift(Xi) * occ
        idx = (torch.arange(self.N, device=Xr.device)[None, :] + shift[:, None]) % self.N
        Xr_s = torch.gather(Xr_s, 1, idx)
        Xi_s = torch.gather(Xi_s, 1, idx)
        # 反 fftshift（N 偶数时 shift N/2 自逆）后 IDFT
        xr, xi = self.dft(_fftshift(Xr_s), _fftshift(Xi_s), inverse=True)
        xr, xi = xr * valid, xi * valid
        L = valid.sum(-1).clamp_min(1.0)
        p = ((xr ** 2 + xi ** 2) * valid).sum(-1) / L
        s = torch.rsqrt(p + 1e-12)[:, None]
        return xr * s, xi * s, cfo

    def _stft(self, xr, xi):
        """帧与 patch 对齐的对数 STFT：[B, N/patch, win]（fftshift）。"""
        pad = (self.win - self.patch) // 2
        fr = F.pad(xr, (pad, pad)).unfold(-1, self.win, self.patch) * self.hann
        fi = F.pad(xi, (pad, pad)).unfold(-1, self.win, self.patch) * self.hann
        Sr, Si = self.dft_win(fr, fi)
        return _fftshift(torch.log1p(Sr ** 2 + Si ** 2))

    def _logspec(self, yr, yi, valid, L):
        Yr, Yi = self.dft(yr * valid, yi * valid)
        pw = ((yr ** 2 + yi ** 2) * valid).sum(-1) / L
        P = (Yr ** 2 + Yi ** 2) / (L * pw + 1e-12)[:, None]
        return _fftshift(torch.log1p(P))

    def _envelope_acf(self, amp, valid, L):
        e = (amp - (amp * valid).sum(-1, keepdim=True) / L[:, None]) * valid
        e2 = F.pad(e, (0, self.N))                                # 2N 零填充 -> 线性相关
        Er, Ei = self.dft_acf(e2, torch.zeros_like(e2))
        P = Er ** 2 + Ei ** 2
        acf, _ = self.dft_acf(P, torch.zeros_like(P), inverse=True)
        acf = acf[:, : self.acf_lags]
        # 恒包络信号（FSK/PSK）包络方差≈0，加相对下限避免除零放大噪声
        return acf / (acf[:, :1] + 1e-3 * L[:, None])

    def _constellation(self, xr, xi, valid, L):
        G, R = self.G, self.R
        ix = ((xr.clamp(-R, R - 1e-4) + R) / (2 * R) * G).floor().long()
        iy = ((xi.clamp(-R, R - 1e-4) + R) / (2 * R) * G).floor().long()
        flat = (iy * G + ix).clamp(0, G * G - 1)
        dens = torch.zeros(xr.size(0), G * G, device=xr.device, dtype=xr.dtype)
        dens.scatter_add_(1, flat, valid)
        dens = dens / L[:, None] * (G * G)
        return torch.log1p(dens).view(-1, G, G)

    def _stats(self, p_raw, length, cfo, occ, sm, floor, xr, xi, xcr, xci, ifreq, valid, L):
        mean = lambda t: (t * valid).sum(-1) / L
        # 二阶/高阶矩（校正后信号，单位功率）
        x2r, x2i = _cmul(xcr, xci, xcr, xci)
        a2 = xcr ** 2 + xci ** 2
        M20 = torch.sqrt(mean(x2r) ** 2 + mean(x2i) ** 2)
        x4r, x4i = _cmul(x2r, x2i, x2r, x2i)
        m40r, m40i = mean(x4r), mean(x4i)
        m20r, m20i = mean(x2r), mean(x2i)
        s2r, s2i = _cmul(m20r, m20i, m20r, m20i)
        C40 = torch.sqrt((m40r - 3 * s2r) ** 2 + (m40i - 3 * s2i) ** 2)
        M41r, M41i = mean(x2r * a2), mean(x2i * a2)
        C41 = torch.sqrt((M41r - 3 * m20r) ** 2 + (M41i - 3 * m20i) ** 2)  # M21=1
        M42 = mean(a2 ** 2)
        C42 = M42 - (m20r ** 2 + m20i ** 2) - 2.0
        M63 = mean(a2 ** 3)
        amp = torch.sqrt(xr ** 2 + xi ** 2 + 1e-12)
        pw = xr ** 2 + xi ** 2
        papr = (pw * valid).amax(-1) / mean(pw).clamp_min(1e-12)
        duty = mean((pw > 0.5 * mean(pw)[:, None]).float())
        a_mu = mean(amp)
        a_var = mean((amp - a_mu[:, None]) ** 2)
        kurt = mean((amp - a_mu[:, None]) ** 4) / (a_var ** 2 + 1e-12)
        if_std = torch.sqrt(mean((ifreq - mean(ifreq)[:, None]) ** 2))
        bw = occ.mean(-1)
        sig = ((sm - floor[:, None]).clamp_min(0) * occ).sum(-1)
        snr = 10 * torch.log10(sig / (floor * self.N) + 1e-6)
        feats = torch.stack([
            torch.log10(p_raw + 1e-20), length.float() / self.N, cfo * 10, bw,
            snr / 10, torch.log10(floor), M20, C40, C41, C42, M63 / 10,
            papr / 10, duty, kurt / 3, if_std, a_var,
        ], dim=-1)
        return _symlog(feats)

    # ------------------------------------------------------------------
    def forward(self, iq, sample_keep=None, power=None):
        """
        Args:
            iq: [B,N,2] 原始 IQ（右侧补零到 N）
            sample_keep: [B,N] 可选，预训练掩码（0=被掩码采样点），全局视图从掩码后信号计算，
                         防止 MAE 通过全局频谱"偷看"被掩码内容（v1 的信息泄漏问题）
            power: [B] 可选，归一化功率覆盖（预训练用整段信号功率，保持重建目标尺度一致）
        """
        with torch.autocast(device_type=iq.device.type, enabled=False):
            iq = iq.float()
            valid, length = valid_mask_from_iq(iq, self.patch)
            if sample_keep is not None:
                iq = iq * sample_keep[..., None]
            L = valid.sum(-1).clamp_min(1.0)
            x, p_raw = self._normalize(iq, valid, power)
            if power is not None:
                p_raw = power
            xr, xi = x[..., 0], x[..., 1]
            Xr, Xi, psd_s, sm, floor, occ = self._spectrum(xr, xi, L)
            xcr, xci, cfo = self._carrier_correct(Xr, Xi, sm, floor, occ, valid)
            amp = torch.sqrt(xr ** 2 + xi ** 2 + 1e-12) * valid
            # 瞬时频率 angle(x_n conj(x_{n-1}))/π
            pr = xr[:, 1:] * xr[:, :-1] + xi[:, 1:] * xi[:, :-1]
            pi_ = xi[:, 1:] * xr[:, :-1] - xr[:, 1:] * xi[:, :-1]
            ifreq = F.pad(torch.atan2(pi_, pr + 1e-12) / math.pi, (1, 0)) * valid
            time_ch = torch.stack([xr, xi, xcr, xci, amp, ifreq], dim=1)       # [B,6,N]
            stft = self._stft(xr, xi)                                           # [B,T,W]
            x2r, x2i = _cmul(xcr, xci, xcr, xci)
            x4r, x4i = _cmul(x2r, x2i, x2r, x2i)
            spec = torch.stack([torch.log1p(psd_s),
                                self._logspec(x2r, x2i, valid, L),
                                self._logspec(x4r, x4i, valid, L)], dim=1)       # [B,3,N]
            acf = self._envelope_acf(amp, valid, L)
            const = self._constellation(xcr, xci, valid, L)
            stats = self._stats(p_raw, length, cfo, occ, sm, floor, xr, xi, xcr, xci,
                                ifreq, valid, L)
        return {
            "valid": valid, "length": length, "x": x,
            "xc": torch.stack([xcr, xci], -1), "time_ch": time_ch, "stft": stft,
            "spec": spec, "acf": acf, "const": const, "stats": stats,
        }

    @torch.no_grad()
    def pretrain_targets(self, iq):
        """MAE/SimMIM 重建目标（整段未掩码信号）：
        patch 目标 [B,T,2*patch]（单位功率归一化 IQ）与对数 STFT [B,T,W]，以及整段功率。"""
        with torch.autocast(device_type=iq.device.type, enabled=False):
            iq = iq.float()
            valid, _ = valid_mask_from_iq(iq, self.patch)
            x, p_raw = self._normalize(iq, valid)
            B, N, _ = x.shape
            patches = x.reshape(B, N // self.patch, 2 * self.patch)
            stft = self._stft(x[..., 0], x[..., 1])
        return patches, stft, p_raw, valid
