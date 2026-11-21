"""两阶段训练脚本（pretrain_mae / finetune_multitask）共享的脚手架：

  - base_arg_parser:   统一命令行参数
  - normalize_config:  新旧配置格式归一化 -> {"model": ..., "training": ...}
  - load_config:       读取 JSON 配置（自动归一化）
  - load_mae_pretrained: 加载 MAE 预训练共享权重（微调用）
  - ROOT / CODE_ROOT: 目录常量
"""

import argparse
import json
from pathlib import Path

import torch

# code/training/common.py -> parents[1]=code/, parents[2]=仓库根
CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent

# 归属 model 部分的键（其余键一律归 training，兼容旧扁平配置的未知训练参数）
_MODEL_KEYS = {
    "backbone", "d_model", "n_layers", "nhead", "ffn_hidden", "patch_size",
    "sr_bins", "pos_encoding", "mask_ratio", "mae_decoder_layers", "head_hidden",
    "use_frequency", "layer_types", "hybrid_layers",
    "mla_kv_lora_rank", "mla_q_lora_rank", "arch",
}


def normalize_config(cfg):
    """配置格式归一化：输出恒为 {"model": {...}, "training": {...}}。

    - 新格式（含 "model"/"training" 顶层键）: 直接透传，缺失部分补空 dict
    - 旧版扁平格式: 按 _MODEL_KEYS 拆分为 model / training 两部分
    """
    if "model" in cfg or "training" in cfg:
        return {"model": cfg.get("model", {}), "training": cfg.get("training", {})}
    return {"model": {k: v for k, v in cfg.items() if k in _MODEL_KEYS},
            "training": {k: v for k, v in cfg.items() if k not in _MODEL_KEYS}}


def base_arg_parser(description):
    """预训练/微调共用的命令行参数（脚本可在此基础上追加专属参数）。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(CODE_ROOT / "configs" / "default.json"),
                        help="模型/训练 JSON 配置路径")
    parser.add_argument("--data_dir", default=str(ROOT / "datasets"),
                        help="数据目录（官方预处理 memmap 或模拟 parquet 布局）")
    parser.add_argument("--out_dir", default=str(ROOT / "weights"),
                        help="权重与导出文件输出目录")
    parser.add_argument("--resume", default=None,
                        help="断点文件（mae_resume.pt / finetune_resume.pt），续训用")
    parser.add_argument("--smoke_steps", type=int, default=0,
                        help=">0 时每 epoch 只跑指定步数（冒烟测试，跳过评估）")
    return parser


def load_config(path):
    """读取 JSON 配置并归一化为 {"model": ..., "training": ...}。"""
    with open(path, "r", encoding="utf-8") as f:
        return normalize_config(json.load(f))


def load_mae_pretrained(model, path, logger=None):
    """将 MAE 预训练的共享权重加载进模型。

    【v1 遗留 bug 修复】export_pretrained 保存的是"扁平" state_dict
    （键形如 tokenizer.cls_token / backbone.layers.0...），而旧实现按
    state["tokenizer"] 这种"模块名"键查找——永远找不到，日志里显示
    loaded [...] 为空列表，即**预训练权重从未真正进入微调**。
    现按前缀筛选扁平键 + strict=False 加载，并打印实际加载/缺失的数量。

    任务头与重建头不加载（保持随机初始化 / 丢弃）。
    Returns:
        bool: 是否成功加载了至少一个张量。
    """
    if not Path(path).exists():
        if logger:
            logger.info("no MAE checkpoint at %s, training from scratch", path)
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    prefixes = ("tokenizer.", "backbone.", "frequency_tokenizer.", "cross_domain_fusion.",
                "mask_token")
    # 兼容两种格式：扁平键（当前导出格式）与 {模块名: 子 state_dict}（更早的格式）
    flat = {}
    for k, v in state.items():
        if isinstance(v, dict):
            flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        else:
            flat[k] = v
    own = model.state_dict()
    picked = {k: v for k, v in flat.items()
              if k.startswith(prefixes) and k in own and own[k].shape == v.shape}
    skipped = [k for k in flat if k.startswith(prefixes) and k not in picked]
    model.load_state_dict(picked, strict=False)
    if logger:
        logger.info("loaded MAE pretrained weights from %s: %d tensors (%d shape/key mismatch skipped)",
                    path, len(picked), len(skipped))
        if skipped:
            logger.info("  skipped examples: %s", skipped[:5])
    return len(picked) > 0
