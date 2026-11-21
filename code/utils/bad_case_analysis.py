"""分类任务 bad case 分析（重点：任务 2 调制识别准确率低的根因定位）

用法（单卡，v1/v2 权重均可）:
  python code/utils/bad_case_analysis.py --task modulation --weights_dir weights \
      --data_dir datasets --split val --n_max 20000 --n_fig 6

输出到 logs/bad_case/<task>/:
  confusion.png / confusion.csv     行归一化混淆矩阵（行=真值，列=预测）
  per_class.csv                     每类召回、精确率、样本数、平均置信度
  top_confusions.csv                最易混淆的 (真值, 预测) 对及占比
  acc_vs_snr.png / acc_vs_snr.csv   准确率随估计 SNR 分桶的变化（判断是否"低 SNR 天花板"）
  acc_vs_length.csv                 512 vs 1024 点准确率
  acc_vs_confidence.png             置信度校准曲线
  bad_cases.csv                     全部错分样本（sample_id、真值、预测、置信度、SNR、长度）
  figs/<真值>__as__<预测>_*.png      每个高频混淆对抽 n_fig 个样本画 4 联图：
                                   时域 / 校正后星座 / 频谱(含 x^4 谱) / STFT
  summary.md                        自动生成的结论摘要（含针对性改进建议）

诊断逻辑（读 summary.md 时对照）:
  * 错误集中在低 SNR 桶、高 SNR 桶接近 100%  -> 数据本身的可分性天花板，重点在降噪/
    带外抑制、更长有效观测、测试时增强；
  * 高 SNR 桶仍有成块混淆（如 64QAM<->128QAM）-> 特征表达不足：看频偏校正后的星座/
    高阶累积量是否可分，检查 dsp 前端的 CFO 估计是否失效；
  * 512 点样本显著更差 -> 观测长度不足，可考虑对 512 点样本加强增强或单独阈值；
  * 置信度高却错（过度自信）-> label smoothing / 温度校准 / 数据标签噪声排查。
"""

import argparse
import csv
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models import model_arch
from training.common import ROOT
from utils.dataset import ID2LABEL, collate_fn, make_task_dataset
from utils.train_utils import autocast_context, get_device


def estimate_snr_db(iq):
    """频域噪底法估计 SNR（与 v2 前端一致的思路，numpy 版，用于分桶统计）。"""
    x = iq[:, 0] + 1j * iq[:, 1]
    nz = np.nonzero(np.abs(iq).sum(-1))[0]
    L = (nz[-1] + 1) if len(nz) else len(x)
    x = x[:L]
    P = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2 / L
    k = 9
    sm = np.convolve(np.concatenate([P[-(k // 2):], P, P[:k // 2]]), np.ones(k) / k, "valid")
    floor = np.sort(sm)[: len(sm) // 2].mean() + 1e-20
    occ = sm > 4 * floor
    sig = np.clip(sm - floor, 0, None)[occ].sum()
    return float(10 * np.log10(sig / (floor * len(sm)) + 1e-6)), int(L)


def run(model, ds, task, device, arch, bs=512):
    loader = DataLoader(ds, batch_size=bs, collate_fn=collate_fn, num_workers=2)
    ys, ps, conf, ids = [], [], [], []
    model.eval()
    with torch.inference_mode(), autocast_context(device, True):
        for b in loader:
            iq = b["iq"].to(device)
            if arch == "v2":
                logits = model.head_outputs(iq, task)
            else:
                sr = b["sr"].to(device)
                logits = model.task_heads[task](model.encode(iq, sr))
            pr = logits.float().softmax(-1)
            c, p = pr.max(-1)
            ys.append(b["label"].numpy())
            ps.append(p.cpu().numpy())
            conf.append(c.cpu().numpy())
            ids += list(b["sample_id"])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(conf), ids


def plot_case(iq, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nz = np.nonzero(np.abs(iq).sum(-1))[0]
    L = (nz[-1] + 1) if len(nz) else len(iq)
    x = iq[:L, 0] + 1j * iq[:L, 1]
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    X = np.fft.fftshift(np.fft.fft(x))
    P = np.abs(X) ** 2
    f = (np.arange(L) - L // 2) / L
    w = np.clip(P - np.median(P), 0, None)
    cfo = (w * f).sum() / (w.sum() + 1e-12)
    xc = x * np.exp(-2j * np.pi * cfo * np.arange(L))
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(x.real, lw=.6, label="I")
    ax[0, 0].plot(x.imag, lw=.6, label="Q")
    ax[0, 0].set_title("time (normalized)")
    ax[0, 0].legend()
    ax[0, 1].scatter(xc.real, xc.imag, s=2, alpha=.5)
    ax[0, 1].set_title(f"constellation after CFO corr (cfo={cfo:.4f})")
    ax[0, 1].set_aspect("equal")
    ax[1, 0].plot(f, 10 * np.log10(P + 1e-12), lw=.6, label="|X|")
    P4 = np.abs(np.fft.fftshift(np.fft.fft(xc ** 4))) ** 2
    ax[1, 0].plot(f, 10 * np.log10(P4 / P4.max() * P.max() + 1e-12), lw=.6, alpha=.7, label="|FFT(x^4)|")
    ax[1, 0].legend()
    ax[1, 0].set_title("spectrum")
    win = 32
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[::8] * np.hanning(win)
    S = np.abs(np.fft.fftshift(np.fft.fft(fr, axis=-1), axes=-1)) ** 2
    ax[1, 1].imshow(10 * np.log10(S.T + 1e-12), aspect="auto", origin="lower")
    ax[1, 1].set_title("STFT")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("bad case 分析")
    ap.add_argument("--task", default="modulation", choices=["modulation", "protocol"])
    ap.add_argument("--weights_dir", default=str(ROOT / "weights"))
    ap.add_argument("--data_dir", default=str(ROOT / "datasets"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_max", type=int, default=20000)
    ap.add_argument("--n_fig", type=int, default=6)
    ap.add_argument("--top_pairs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from infer import load_model
    device = get_device()
    model, cfg, _ = load_model(args.weights_dir, device)
    arch = model_arch(cfg["model"])
    ds = make_task_dataset(args.data_dir, args.task, args.split,
                           seq_len=cfg["training"].get("seq_len", 1024))
    if args.n_max and len(ds) > args.n_max:
        ds = Subset(ds, np.random.default_rng(0).choice(len(ds), args.n_max, replace=False).tolist())
    names = ID2LABEL[args.task]
    C = len(names)
    out = Path(args.out or ROOT / "logs" / "bad_case" / args.task)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    y, p, conf, ids = run(model, ds, args.task, device, arch)
    acc = float((y == p).mean())
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    cmn = cm / cm.sum(1, keepdims=True).clip(1)

    # SNR / 长度（逐样本 numpy 计算）
    snr, lens = np.zeros(len(ds)), np.zeros(len(ds), dtype=int)
    for i in range(len(ds)):
        snr[i], lens[i] = estimate_snr_db(ds[i]["iq"].numpy())

    # --- 表格输出 ---
    with open(out / "confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + [names[i] for i in range(C)])
        for i in range(C):
            w.writerow([names[i]] + [f"{v:.4f}" for v in cmn[i]])
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "n", "recall", "precision", "mean_conf"])
        for i in range(C):
            m = y == i
            prec = cm[i, i] / max(1, cm[:, i].sum())
            w.writerow([names[i], int(m.sum()), f"{cmn[i, i]:.4f}", f"{prec:.4f}",
                        f"{conf[m].mean() if m.any() else 0:.4f}"])
    off = [(cm[i, j], i, j) for i in range(C) for j in range(C) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    n_err = max(1, int((y != p).sum()))
    with open(out / "top_confusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true", "pred", "count", "frac_of_true_class", "frac_of_all_errors"])
        for c, i, j in off[:30]:
            w.writerow([names[i], names[j], c, f"{cmn[i, j]:.4f}", f"{c / n_err:.4f}"])
    bins = [-np.inf, -5, 0, 5, 10, 15, 20, np.inf]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (snr >= lo) & (snr < hi)
        if m.any():
            rows.append((f"[{lo},{hi})", int(m.sum()), float((y[m] == p[m]).mean())))
    with open(out / "acc_vs_snr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snr_bin_db", "n", "acc"])
        w.writerows(rows)
    with open(out / "acc_vs_length.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["length", "n", "acc"])
        for ln in np.unique(lens):
            m = lens == ln
            w.writerow([int(ln), int(m.sum()), f"{(y[m] == p[m]).mean():.4f}"])
    with open(out / "bad_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "true", "pred", "conf", "snr_db", "length"])
        for k in np.where(y != p)[0]:
            w.writerow([ids[k], names[y[k]], names[p[k]], f"{conf[k]:.4f}", f"{snr[k]:.2f}",
                        int(lens[k])])

    # --- 图 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, C * 0.45), max(5, C * 0.4)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C))
    ax.set_xticklabels([names[i] for i in range(C)], rotation=90, fontsize=7)
    ax.set_yticks(range(C))
    ax.set_yticklabels([names[i] for i in range(C)], fontsize=7)
    for i in range(C):
        for j in range(C):
            if cmn[i, j] >= 0.05:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if cmn[i, j] > .5 else "black")
    ax.set_title(f"{args.task} confusion ({args.split}, acc={acc:.4f})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=150)
    plt.close(fig)
    if rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([r[0] for r in rows], [r[2] for r in rows])
        for k, r in enumerate(rows):
            ax.text(k, r[2], f"n={r[1]}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("acc")
        ax.set_xlabel("estimated SNR (dB)")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(out / "acc_vs_snr.png", dpi=130)
        plt.close(fig)
    qs = np.quantile(conf, np.linspace(0, 1, 11))
    cx, cy = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (conf >= lo) & (conf <= hi)
        if m.any():
            cx.append(conf[m].mean())
            cy.append((y[m] == p[m]).mean())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(cx, cy, "o-")
    ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(out / "acc_vs_confidence.png", dpi=130)
    plt.close(fig)
    for c, i, j in off[: args.top_pairs]:
        ks = np.where((y == i) & (p == j))[0][: args.n_fig]
        for k in ks:
            tag = f"{names[i]}__as__{names[j]}".replace("/", "-").replace(" ", "_")
            plot_case(ds[int(k)]["iq"].numpy(),
                      f"true={names[i]} pred={names[j]} conf={conf[k]:.2f} snr={snr[k]:.1f}dB",
                      out / "figs" / f"{tag}_{ids[k][:8]}.png")

    # --- 摘要 ---
    hi = snr >= 10
    lo = snr < 0
    lines = [f"# {args.task} bad case 摘要（{args.split}, n={len(y)}）", "",
             f"- 总体 acc = {acc:.4f}，错分 {n_err} 条",
             f"- 估计 SNR>=10dB 子集 acc = {(y[hi] == p[hi]).mean() if hi.any() else float('nan'):.4f}"
             f"（n={int(hi.sum())}）；SNR<0dB 子集 acc = "
             f"{(y[lo] == p[lo]).mean() if lo.any() else float('nan'):.4f}（n={int(lo.sum())}）",
             "", "## Top 混淆对", ""]
    for c, i, j in off[:10]:
        lines.append(f"- {names[i]} → {names[j]}: {c} 条（占该类 {cmn[i, j]:.1%}，占全部错误 {c / n_err:.1%}）")
    lines += ["", "## 判读", ""]
    if hi.any() and (y[hi] == p[hi]).mean() > 0.95 and lo.any() and (y[lo] == p[lo]).mean() < 0.6:
        lines.append("- 错误主要来自低 SNR 样本：属于可分性天花板，优先降噪（带外抑制）、增强、TTA。")
    else:
        lines.append("- 高 SNR 样本仍有明显混淆：属于特征/模型问题，检查 figs/ 中校正后星座与 x^4 谱"
                     "是否可分；必要时提高该族的损失权重或增加族内专门特征。")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[bad_case] 输出目录: {out}")


if __name__ == "__main__":
    main()
