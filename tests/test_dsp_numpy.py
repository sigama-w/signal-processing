"""不依赖 PyTorch 的 DSP 公式校验（实数矩阵 DFT、频偏估计与校正、STFT 帧对齐）。"""
import numpy as np


def test_real_dft_and_cfo():
    N = 1024
    k = np.arange(N)
    ang = 2 * np.pi * np.outer(k, k) / N
    C, S = np.cos(ang), np.sin(ang)
    rng = np.random.default_rng(0)
    sym = (rng.integers(0, 2, 256) * 2 - 1 + 1j * (rng.integers(0, 2, 256) * 2 - 1)) / np.sqrt(2)
    x = np.convolve(np.repeat(sym, 4)[:N], np.ones(4) / 4, "same")
    x = x * np.exp(2j * np.pi * 0.0625 * k) + 0.05 * (rng.normal(size=N) + 1j * rng.normal(size=N))
    Xr = x.real @ C + x.imag @ S
    Xi = x.imag @ C - x.real @ S
    F = np.fft.fft(x)
    assert np.abs(Xr - F.real).max() < 1e-8 and np.abs(Xi - F.imag).max() < 1e-8
    ps = np.roll((Xr ** 2 + Xi ** 2) / N, N // 2)
    sm = np.convolve(np.concatenate([ps[-4:], ps, ps[:4]]), np.ones(9) / 9, "valid")
    floor = np.sort(sm)[: N // 2].mean()
    occ = sm > 4 * floor
    f = (k - N // 2) / N
    w = np.maximum(sm - floor, 0) * occ
    cfo = (w * f).sum() / w.sum()
    assert abs(cfo - 0.0625) < 0.002
    shift = int(round(cfo * N))
    xc = np.fft.ifft(np.roll(np.roll(F, N // 2)[(k + shift) % N], N // 2))
    assert np.argmax(np.abs(np.fft.fft(xc ** 4))) == 0


def test_stft_frames_align_with_patches():
    x = np.zeros(1024)
    fr = np.lib.stride_tricks.sliding_window_view(np.pad(x, (12, 12)), 32)[::8]
    assert fr.shape == (128, 32)
