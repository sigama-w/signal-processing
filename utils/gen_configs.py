"""【v2 注意】本脚本面向 v1 hybrid 大模型（600M~1B），其显存公式未计入反向峰值，
对 1024/1536 宽配置低估 10~15GB（见 docs/OPTIMIZATION_PLAN.md §6）。v2 请改用
code/configs/grid_v2.json + utils/measure_vram.py 实测。

调参配置生成器：按网格/指定参数批量生成合法 config

用法:
    # 默认网格: d_model 1024/1536 × 层数 18~39(每3层一档) × batch 128/256/512 × lr auto,
    #           参数量下限 600M(--min_params), 直接落盘 code/configs/grid.json
    python utils/gen_configs.py

    # 只预览不落盘
    python utils/gen_configs.py --dry

    # 自定义网格（覆盖默认, 同样直接落盘）
    python utils/gen_configs.py --d_model 512 1024 1472 --n_layers 24 27 --write

    # 完全指定（单点配置）
    python utils/gen_configs.py --d_model 1472 --n_layers 27 --ffn_hidden 6400 \
        --batch_size 16 --lr 1e-4

    # ffn_hidden 自动撑满 1B 参数预算（二分搜索 64 对齐）
    python utils/gen_configs.py --d_model 1024 --ffn_hidden max

    # 显存预算（默认 74GB, 80G 卡留 ~6G 物理余量）
    python utils/gen_configs.py --vram 64

输出格式（默认落盘, 单一 JSON 文件）:
    {"vram_budget_gb": ..., "configs": [
        {"name": "...", "model": {...}, "training": {...},
         "params_M": ..., "shared_ratio": ..., "vram_est_gb": ...}, ...]}
    每项 model/training 均为完整可训练配置（无 base/override 拼接），
    搜参 runner 逐项取出、落临时 json 即可直接 --config 训练。

硬性约束（与 finetune_multitask 入口校验一致, 违者直接剔除）:
  1. 总参数量 < 1B (MAX_TOTAL_PARAMS = 1e9, 参数量由真实实例化 EMFoundation 精确统计)
  2. 权重共用率 > 80% (tokenizer+backbone / 任务头, 实测 >99%, 自动校验)
  3. d_model % nhead == 0 且 head_dim 偶数（RoPE 旋转要求）
  4. 显存估算不超 --vram × --headroom 预算（估算模型见 vram_estimate 文档字符串）
  5. 参数量 ≥ --min_params（默认 600M; 赛方上限 1B, 尽量顶格做大规模,
     低于下限视为劣质配置剔除——小容量模型学不动百万级大样本）

自动推导规则:
  nhead      = d_model // 64（head_dim 固定 64, 与现有配置一致）
  ffn_hidden 默认 = 64 对齐的 4×d_model; --ffn_hidden max 则二分撑满 1B 预算
  batch_size 默认 auto = 预算内可容纳的最大档（--min_batch_size 起步翻倍, 上限 1024）
  lr         默认 auto = 1e-4 × sqrt(bs/16)（以 bs16@1e-4 基线 sqrt 缩放, 2 位有效数字）
  任务头     两种模式:
    共用率模式(默认): --shared_ratio 默认 [0.85 0.90 0.95] 三档——按 backbone
      共享量反解各任务头宽度（protocol 0.2 / modulation 0.2 /
      parameter+pattern 0.3+0.3 按难度分配, 深度固定 2 层, 64 对齐）,
      命名后缀 _shared85/_shared90/_shared95
    显式模式: 传任一 --head_hidden_* / --head_layers_* 即切换六维扫参,
      命名后缀 _protoHead512x1 / _modHead256x2 / _paramHead1024x2
  显存       默认预算 74GB, 估算峰值 ≤ 74GB（80G 卡留 ~6G 物理余量; probe_vram
             实测校准系数 vram_calibration.json 存在时自动相乘）
  截断       默认按参数量降序只保留最大的 20 组（--max_configs 可调, 0 = 不截断）

冻结字段（backbone/patch_size/mask_ratio/epochs/patience/task_weights 等）取
code/configs/default.json 模板, 与现有配置完全一致, 保证生成配置可直接训练。
"""

import os
import sys
import json
import math
import argparse
import itertools

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

MAX_TOTAL_PARAMS = 1_000_000_000   # 赛题红线: 总参数量 < 1B
MIN_SHARED_RATIO = 0.80            # 赛题红线: 权重共用率 > 80%
SEQ_LEN = 1024                     # 官方 IQ 长度对齐
PATCH = 8                          # 与全部现有配置一致
N_TOKENS = SEQ_LEN // PATCH + 2    # 130（含 CLS/SR）
HEAD_DIM = 64                      # nhead 推导基准
LR_REF_BS = 16                     # lr sqrt 缩放基线: bs16 @ 1e-4
# 实测校准系数（utils/probe_vram.py --apply 生成 vram_calibration.json;
# 缺省 1.0 = 未校准, 估算公式原样生效）
_CALIBRATION_PATH = os.path.join(ROOT, "code", "configs", "vram_calibration.json")
try:
    VRAM_CALIB = float(json.load(open(_CALIBRATION_PATH, encoding="utf-8"))["factor"])
    if not (0.5 <= VRAM_CALIB <= 3.0):
        VRAM_CALIB = 1.0
except Exception:
    VRAM_CALIB = 1.0
# 基线头配置（default.json head_config 同款; 共用率模式下仅用于测共享绝对量）
BASE_HEAD_CONFIG = {
    "protocol": {"hidden": 256, "layers": 1},
    "modulation": {"hidden": 256, "layers": 1},
    "parameter": {"hidden": 512, "layers": 2},
    "parameter_pattern": {"hidden": 512, "layers": 2},
}

# ---------------------------------------------------------------------------
# 参数量精确统计（实例化真实模型, 与 finetune 入口校验同一来源）
# ---------------------------------------------------------------------------

def measure(d_model, n_layers, ffn_hidden, nhead, head_config):
    """实例化 EMFoundation, 返回 (total, shared, heads) 或抛 ValueError。

    total 与 finetune_multitask.assert_competition_limits 完全同源:
    sum(p.numel()) 全模型（含 mae_decoder/mask_token），不是 shared+head——
    后者会漏掉 decoder（512 宽约 8.4M、1472 宽约 70M，足以把 990M 误判合规）。

    head_config: 按任务独立配置 {task: {hidden, layers}}——任务难度不同
    允许不同参数量（parameter 7 参数联合回归计算量最大）。
    """
    from models import EMFoundation
    cfg = {
        "backbone": "hybrid", "d_model": d_model, "n_layers": n_layers,
        "nhead": nhead, "ffn_hidden": ffn_hidden, "patch_size": PATCH,
        "sr_bins": 64, "pos_encoding": "rope", "mask_ratio": 0.75,
        "mae_decoder_layers": 2, "use_frequency": True,
        "head_hidden": 256, "head_layers": 1,
        "head_config": head_config,
        # 短模式串自动循环铺满 n_layers（backbone.HybridEncoder 语义）
        "layer_types": ["full_attention", "mla", "mla"],
        "mla_kv_lora_rank": 128, "mla_q_lora_rank": 128,
    }
    m = EMFoundation(cfg)
    try:
        return (sum(p.numel() for p in m.parameters()),
                m.shared_parameter_count(), m.head_parameter_count())
    finally:
        del m


def solve_head_config(d_model, shared_params, target_ratio):
    """按目标共用率反解各任务头宽度（深度固定 2 层, 64 对齐）。

    头部预算 H = shared × (1-r)/r; 按任务难度分配:
      protocol 0.2 / modulation 0.2 / parameter 0.3 / parameter_pattern 0.3
    头参数近似 P(h) = (L-1)h² + (D+C+L)h（L=2 时解二次方程取正根）。
    """
    H = shared_params * (1.0 - target_ratio) / target_ratio
    L = 2

    def width(share, n_classes):
        a, b = L - 1, d_model + n_classes + L
        h = (-b + math.sqrt(b * b + 4 * a * share)) / (2 * a)
        return round64(h)

    hp = width(0.2 * H, 8)       # protocol
    hm = width(0.2 * H, 23)      # modulation
    hpa = width(0.3 * H, 7)      # parameter（pattern 同款跟随）
    return {
        "protocol": {"hidden": hp, "layers": L},
        "modulation": {"hidden": hm, "layers": L},
        "parameter": {"hidden": hpa, "layers": L},
        "parameter_pattern": {"hidden": hpa, "layers": L},
    }

# ---------------------------------------------------------------------------
# 显存估算（按 docs/gpu_memory_analysis.md 的分桶账目校准）
# ---------------------------------------------------------------------------
def vram_estimate_gb(total_params, d_model, n_layers, ffn_hidden, nhead, bs):
    """两阶段训练峰值显存估算（allocated 口径, 偏保守）。

    校准锚点（docs/gpu_memory_analysis.md, d=512/n=27/F=2048/H=8）:
      静态 16B/param（fp32 权重+梯度+AdamW m/v）= 文档 1.88GB ✓
      前向保存/样本 ≈ 8FL+20DL+8HL² 每层, B=1024 预训练 39.8GB vs 文档 41GB ✓
      backward 峰值 +25% ≈ 文档 +9/41GB=22% ✓
      workspace 常备 4GB（cuBLAS/cuFFT/碎片, 文档 5-10GB 口径偏 reserved）

    两阶段取最大者（同一 batch_size 配置对预训练/微调都生效）:
      MAE 预训练: backbone 只编码 keep 25% 可见 token（L=34）,
                  但 mae_decoder 与频域分支跑全序列 L=130（文档易漏算项）
      微调:       backbone 全序列 L=130, 无 decoder
    """
    L_vis, L_full = 34, N_TOKENS

    def per_layer(L):
        # SwiGLU 4份@2B + norm/proj/rope/残差@2B + attention scores+probs fp32
        return 8 * ffn_hidden * L + 20 * d_model * L + 8 * nhead * L * L

    freq_misc = 20 * d_model * 128 + 2 ** 20          # 频域分支+杂项
    fwd_pre = (n_layers * per_layer(L_vis)            # backbone(可见 25%)
               + 2 * per_layer(L_full)                # mae_decoder(全序列)
               + freq_misc) * bs
    fwd_ft = (n_layers * per_layer(L_full) + freq_misc) * bs
    static = 16 * total_params
    peak = max(static + 1.25 * fwd_pre, static + 1.25 * fwd_ft) + 4e9
    return VRAM_CALIB * peak / 1e9

def max_batch_size(total_params, d_model, n_layers, ffn_hidden, nhead,
                   vram_gb, headroom, min_bs=256):
    """预算内可容纳的最大 batch（min_bs 起逐级翻倍, 上限 1024）。"""
    budget = vram_gb * headroom
    best = None
    bs = min_bs
    while bs <= 1024:
        if vram_estimate_gb(total_params, d_model, n_layers, ffn_hidden,
                           nhead, bs) <= budget:
            best = bs
        bs *= 2
    return best

# ---------------------------------------------------------------------------
# 推导工具
# ---------------------------------------------------------------------------
def round64(x):
    return max(64, int(round(x / 64)) * 64)

def derive_nhead(d_model):
    if d_model % HEAD_DIM != 0:
        raise ValueError(f"d_model={d_model} 不是 head_dim={HEAD_DIM} 的整数倍, "
                         "RoPE 要求 d_model % nhead == 0 且 head_dim 为偶数")
    return d_model // HEAD_DIM

def derive_lr(bs, ref_lr=1e-4):
    """sqrt 缩放: lr = ref_lr × sqrt(bs / LR_REF_BS), 2 位有效数字。"""
    lr = ref_lr * math.sqrt(bs / LR_REF_BS)
    mag = 10 ** math.floor(math.log10(lr))
    return round(lr / mag, 2) * mag

def fill_1b_ffn(d_model, n_layers, nhead):
    """二分搜索 64 对齐的最大 ffn_hidden 使总参数 < 1B。"""
    lo, hi = round64(2 * d_model), round64(8 * d_model)
    if measure(d_model, n_layers, hi, nhead)[0] < MAX_TOTAL_PARAMS:
        return hi
    if measure(d_model, n_layers, lo, nhead)[0] >= MAX_TOTAL_PARAMS:
        raise ValueError(f"d_model={d_model} n={n_layers} 最小 ffn 已超 1B, 放弃")
    while hi - lo > 64:
        mid = round64((lo + hi) / 2)
        if mid == lo or mid == hi:
            break
        if measure(d_model, n_layers, mid, nhead)[0] < MAX_TOTAL_PARAMS:
            lo = mid
        else:
            hi = mid
    return lo

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_config(base, d_model, n_layers, ffn_hidden, bs, lr, head_config):
    cfg = json.loads(json.dumps(base))  # 深拷贝模板
    cfg["model"].update({
        "d_model": d_model, "n_layers": n_layers,
        "nhead": derive_nhead(d_model), "ffn_hidden": ffn_hidden,
        # 短模式串, backbone 自动循环铺满 n_layers（等价于现有 27 项显式列表）
        "layer_types": ["full_attention", "mla", "mla"],
        # 按任务独立写入 head_config（结构同 default.json 基线）
        "head_config": head_config,
    })
    cfg["training"]["batch_size"] = bs
    cfg["training"]["lr"] = lr
    return cfg

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d_model", nargs="+", type=int, default=[1024, 1536],
                    help="宽度网格（默认聚焦 1024/1536 大宽度: 256/512/768 在 "
                         "40 层上限内够不到 600M 下限, 默认不扫, 显式传入仍支持）")
    ap.add_argument("--n_layers", nargs="+", type=int,
                    default=[18, 21, 24, 27, 30, 33, 36, 39],
                    help="层数网格（默认 18~39 每 3 层一档: 1536 宽度用浅档 "
                         "18/21/24, 1024 宽度需加深到 33/36/39 才够 600M 下限）")
    ap.add_argument("--ffn_hidden", nargs="+", type=str, default=None,
                    help="如 '2048 6400'；'max'=自动撑满 1B；缺省=4×d_model 对齐 64")
    ap.add_argument("--batch_size", nargs="+", type=str, default=["128", "256", "512"],
                    help="如 '128 256 512'；'auto'=预算内最大档")
    ap.add_argument("--lr", nargs="+", type=str, default=["auto"],
                    help="如 '1e-4 3e-4'；'auto'=sqrt(bs/16) 缩放（与显式值混用）")
    ap.add_argument("--head_hidden_protocol", nargs="+", type=int, default=None,
                    help="protocol 头宽度（显式模式: 传任一头参数即切换六维扫参）")
    ap.add_argument("--head_hidden_modulation", nargs="+", type=int, default=None,
                    help="modulation 头宽度（显式模式）")
    ap.add_argument("--head_hidden_parameter", nargs="+", type=int, default=None,
                    help="parameter 头宽度（显式模式）")
    ap.add_argument("--head_layers_protocol", nargs="+", type=int, default=None,
                    help="protocol 头深度（显式模式）")
    ap.add_argument("--head_layers_modulation", nargs="+", type=int, default=None,
                    help="modulation 头深度（显式模式）")
    ap.add_argument("--head_layers_parameter", nargs="+", type=int, default=None,
                    help="parameter 头深度（显式模式）")
    ap.add_argument("--shared_ratio", nargs="+", type=float,
                    default=[0.85, 0.90, 0.95],
                    help="目标权重共用率三档（默认 0.85/0.90/0.95, 默认模式; "
                         "按 backbone 共享量反解各任务头宽度, 难度权重 "
                         "protocol 0.2 / modulation 0.2 / parameter 0.3+pattern 0.3, "
                         "深度固定 2 层）")
    ap.add_argument("--vram", type=float, default=74.0,
                    help="显存预算 GB（默认 74, 80G 卡留 ~6G 物理余量; "
                         "1536 宽度需 ≥74, 1472 宽度需 ≥72, 否则被预算剔除）")
    ap.add_argument("--headroom", type=float, default=1.0,
                    help="显存安全余量比例（默认 1.0, 即估算峰值 ≤ 预算内）")
    ap.add_argument("--min_params", type=float, default=600.0, metavar="M",
                    help="参数量下限, 单位 M（默认 600: 赛方限 1B 上限, 尽量 "
                         "顶格做大规模; 低于下限视为劣质配置直接剔除）")
    ap.add_argument("--min_batch_size", type=int, default=128,
                    help="batch 下限（数据集大, 默认 128 起步；显存吃紧可调低）")
    ap.add_argument("--max_configs", type=int, default=20,
                    help="最终保留的配置数上限（默认 20, 按参数量降序保留最大的一批）")
    ap.add_argument("--out", default=os.path.join(ROOT, "code", "configs", "grid.json"),
                   help="输出单一文件路径（默认 code/configs/grid.json）")
    ap.add_argument("--dry", action="store_true", help="只打印不落盘（默认直接写 grid.json）")
    args = ap.parse_args()

    base = json.load(open(os.path.join(ROOT, "code", "configs", "default.json")))
    if VRAM_CALIB != 1.0:
        print(f"[calib] 显存估算 × 实测校准系数 {VRAM_CALIB:.2f} "
              f"(vram_calibration.json, probe_vram --apply 生成)")

    # 头部网格两模式:
    #   共用率模式(默认): --shared_ratio 三档, 按 backbone 共享量反解头宽度
    #   显式模式: 传任一 --head_hidden_* / --head_layers_* 时启用六维笛卡尔积
    explicit_head = any([args.head_hidden_protocol, args.head_hidden_modulation,
                         args.head_hidden_parameter,
                         args.head_layers_protocol, args.head_layers_modulation,
                         args.head_layers_parameter])
    if explicit_head:
        head_grid = [(hp, hm, hpa, lp, lm, lpa)
                     for hp in (args.head_hidden_protocol or [256])
                     for hm in (args.head_hidden_modulation or [256])
                     for hpa in (args.head_hidden_parameter or [512])
                     for lp in (args.head_layers_protocol or [1])
                     for lm in (args.head_layers_modulation or [1])
                     for lpa in (args.head_layers_parameter or [2])]
        ratio_list = [None]
    else:
        head_grid = [None]
        ratio_list = args.shared_ratio

    rows = []
    for d, n in itertools.product(args.d_model, args.n_layers):
        nhead = derive_nhead(d)
        if not (12 <= n <= 40):
            print(f"[skip] d={d} n={n}: n_layers 超出推荐范围 [12,40]")
            continue
        if args.ffn_hidden is None:
            ffns = [round64(4 * d)]
        else:
            ffns = []
            for f in args.ffn_hidden:
                if f.lower() == "max":
                    ffns.append(fill_1b_ffn(d, n, nhead))
                else:
                    ffns.append(int(float(f)))
        for f in ffns:
            # 共用率模式: 先测一次拿 backbone 共享绝对量（与头大小无关）
            shared0 = None
            if ratio_list[0] is not None:
                _, shared0, _ = measure(d, n, f, nhead, BASE_HEAD_CONFIG)
            for head_variant in head_grid:
                for r_target in ratio_list:
                    if head_variant is None:
                        # 共用率模式: 反解头宽度（protocol/modulation/parameter
                        # 按难度 0.2/0.2/0.3+0.3 分配头部预算）
                        head_config = solve_head_config(d, shared0, r_target)
                        head_tag = f"_shared{int(round(r_target * 100))}"
                    else:
                        hp, hm, hpa, lp, lm, lpa = head_variant
                        head_tag = ""
                        if (hp, lp) != (256, 1):
                            head_tag += f"_protoHead{hp}x{lp}"
                        if (hm, lm) != (256, 1):
                            head_tag += f"_modHead{hm}x{lm}"
                        if (hpa, lpa) != (512, 2):
                            head_tag += f"_paramHead{hpa}x{lpa}"
                        head_config = {
                            "protocol": {"hidden": hp, "layers": lp},
                            "modulation": {"hidden": hm, "layers": lm},
                            "parameter": {"hidden": hpa, "layers": lpa},
                            "parameter_pattern": {"hidden": hpa, "layers": lpa},
                        }
                    total, shared_n, heads_n = measure(d, n, f, nhead, head_config)
                    ratio = shared_n / (shared_n + heads_n)
                    if total < args.min_params * 1e6:
                        print(f"[skip] d={d} n={n} ffn={f}{head_tag}: 参数 "
                              f"{total/1e6:.0f}M < {args.min_params:.0f}M 下限"
                              f"（赛方限 1B 尽量顶格, --min_params 可调）")
                        continue
                    if total >= MAX_TOTAL_PARAMS:
                        print(f"[skip] d={d} n={n} ffn={f}{head_tag}: 参数 "
                              f"{total/1e6:.0f}M ≥ 1B 红线")
                        continue
                    if ratio <= MIN_SHARED_RATIO:
                        print(f"[skip] d={d} n={n} ffn={f}{head_tag}: 共用率 "
                              f"{ratio:.1%} ≤ 80% 红线")
                        continue
                    # batch_size 解析: 'auto' 可与显式值混用（如 --batch_size auto 512）
                    bs_list = []
                    for b in args.batch_size:
                        if b.lower() == "auto":
                            mb = max_batch_size(total, d, n, f, nhead, args.vram,
                                                args.headroom, args.min_batch_size)
                            if mb is None:
                                print(f"[skip] d={d} n={n} ffn={f}{head_tag}: 预算内连 "
                                      f"bs={args.min_batch_size} 都放不下（--vram {args.vram}GB "
                                      f"× {args.headroom}）——需减 d_model/层数或上梯度检查点")
                            else:
                                bs_list.append(mb)
                        else:
                            bs_list.append(int(float(b)))
                    for bs in bs_list:
                        if bs < args.min_batch_size:
                            print(f"[skip] d={d} n={n} ffn={f}{head_tag} bs={bs}: 低于 "
                                  f"min_batch_size={args.min_batch_size} 下限")
                            continue
                        vram = vram_estimate_gb(total, d, n, f, nhead, bs)
                        if vram > args.vram * args.headroom:
                            print(f"[skip] d={d} n={n} ffn={f}{head_tag} bs={bs}: "
                                  f"估算 {vram:.1f}GB > 预算 {args.vram*args.headroom:.0f}GB")
                            continue
                        # lr 解析: 'auto' 可与显式值混用（如 --lr auto 1e-4 3e-4）
                        lrs = []
                        for x in args.lr:
                            if x.lower() == "auto":
                                lrs.append(derive_lr(bs))
                            else:
                                lrs.append(float(x))
                        for lr_val in lrs:
                            rows.append({
                                "d": d, "n": n, "ffn": f, "bs": bs, "lr": lr_val,
                                "head_config": head_config, "head_tag": head_tag,
                                "total": total, "shared": ratio, "vram": vram,
                            })

    if not rows:
        print("[done] 无合法配置")
        return

    # 截断: 按参数量降序保留最大的一批（同参数时优先大 batch）
    if args.max_configs and len(rows) > args.max_configs:
        rows.sort(key=lambda r: (-r["total"], -r["bs"]))
        dropped = len(rows) - args.max_configs
        del rows[args.max_configs:]
        print(f"[cut] 按参数量降序保留前 {args.max_configs} 组, "
              f"裁掉 {dropped} 个较小配置")

    hdr = f'{"file":<52}{"params":>9}{"shared":>9}{"vram(est)":>11}'
    print(hdr)
    print("-" * len(hdr))
    names = []
    for r in rows:
        name = f"gen_{r['d']}d_{r['n']}l_{r['ffn']}ffn_{r['bs']}b_lr{r['lr']:.0e}" \
               f"{r['head_tag']}"
        names.append(name)
        print(f"{name:<52}{r['total']/1e6:>8.0f}M{r['shared']:>9.2%}"
              f"{r['vram']:>10.1f}G")

    if not args.dry:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        entries = []
        for name, r in zip(names, rows):
            cfg = build_config(base, r["d"], r["n"], r["ffn"], r["bs"], r["lr"],
                               r["head_config"])
            entries.append({
                "name": name,
                "model": cfg["model"],           # 完整配置, 无需拼接
                "training": cfg["training"],      # 完整配置, 无需拼接
                "params_M": r["total"] / 1e6, "shared_ratio": r["shared"],
                "vram_est_gb": r["vram"],
            })
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump({"vram_budget_gb": args.vram, "headroom": args.headroom,
                       "n_tokens": N_TOKENS, "configs": entries}, fp, indent=2)
        print(f"\n[write] {len(entries)} 个配置 -> {args.out}（单文件）")
    else:
        print(f"\n[dry] 共 {len(rows)} 个合法配置（默认直接落盘, --dry 只预览）")

if __name__ == "__main__":
    main()
