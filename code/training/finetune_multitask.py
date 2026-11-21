"""阶段二：单 backbone 多任务微调（protocol / modulation / parameter）。

用法:
  python training/finetune_multitask.py --data_dir <dir> [--from_mae ...]      # 单卡
                                       [--config ...] [--resume ...] [--eval_test]
  torchrun --nproc_per_node=N training/finetune_multitask.py --data_dir <dir> [...]  # 多卡

本文件只负责：参数/配置/数据组装 -> 多任务微调 -> fit()。
训练生命周期（调度/断点/早停/评估/导出）见 training/trainer.py，
任务损失/评估的多态实现见 training/tasks.py。
"""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from models import EMFoundation, build_model, model_arch
from training.common import ROOT, base_arg_parser, load_config, load_mae_pretrained
from training.tasks import TASK_ORDER, TASKS
from training.tasks_v2 import TASKS_V2
from training.trainer import MultitaskFinetuner, MultitaskFinetunerV2
from utils.dataset import build_task_dataloaders, build_task_dataloaders_v2, load_param_arrays
from utils.param_codec import ParamCodec
from utils.logger import get_logger
from utils.train_utils import get_device, init_distributed, seed_all, teardown_distributed

MAX_TOTAL_PARAMS = 1_000_000_000   # 赛题红线：总参数量 < 1B
MIN_SHARED_RATIO = 0.80            # 赛题红线：权重共用率 > 80%


def assert_competition_limits(model):
    """赛题红线校验，返回 (shared_ratio, total_params)。"""
    shared_ratio, total = model.shared_ratio(), \
        sum(p.numel() for p in model.parameters())
    if total >= MAX_TOTAL_PARAMS:
        raise ValueError(f"总参数量 {total} 超过 1B 红线")
    if shared_ratio <= MIN_SHARED_RATIO:
        raise ValueError(f"权重共用率 {shared_ratio:.2%} 未达 80% 红线")
    return shared_ratio, total


def evaluate_on_test(model, data_dir, train_cfg, param_stats, device, logger, out_dir):
    """在 test 划分上做无偏评估（加载导出的 best 权重）。

    test 从未参与训练/早停/导出决策，其指标是官方隐藏测试集出分前
    最可信的本地水位参考；频繁查看 test 会引入选择偏差，只在此处用一次。
    """
    from safetensors.torch import load_file
    best_path = Path(out_dir) / "model.safetensors"
    model.load_state_dict(load_file(str(best_path)))
    logger.info("[test] loaded best weights: %s", best_path)
    loaders, _ = build_task_dataloaders(
        data_dir, batch_size=train_cfg["batch_size"], seq_len=train_cfg["seq_len"],
        splits=("test",), num_workers=train_cfg.get("num_workers", 0))
    results = {t: TASKS[t].evaluate(model, loaders["test"][t], device, param_stats,
                                    train_cfg.get("ref_scale", 0.5))
               for t in TASK_ORDER if t in loaders["test"]}
    for t, m in results.items():
        logger.info("[test] %s: %s", t, {k: round(v, 5) for k, v in m.items()})
    score = (0.3 * results["protocol"]["acc"]
             + 0.3 * results["modulation"]["acc"]
             + 0.4 * results["parameter"]["precision"])
    logger.info("[test] proxy score=%.5f (unbiased; val best was the model-selection "
                "criterion, test was never seen during training)", score)
    return score


def prepare_codec(args, train_cfg, logger):
    """v2 参数编解码器：优先复用 out_dir/param_codec.json，否则用 train 划分拟合并保存。"""
    path = Path(args.out_dir) / "param_codec.json"
    if path.exists() and not args.refit_codec:
        codec = ParamCodec.load(path)
        logger.info("loaded param codec: %s", path)
    else:
        try:
            p, v = load_param_arrays(args.data_dir, "train")
            ccfg = train_cfg.get("codec", {})
            codec = ParamCodec.fit(p, v, max_classes=ccfg.get("max_classes", 32),
                                   log_ratio=ccfg.get("log_ratio", 20.0),
                                   force=ccfg.get("force"), decode=ccfg.get("decode", "expectation"))
            logger.info("fitted param codec on %d train samples", len(p))
        except FileNotFoundError:
            codec = ParamCodec.default()
            logger.info("no parameter train data found; using default codec")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        codec.save(path)
    logger.info("param codec:\n%s", codec.describe())
    return codec


def evaluate_on_test_v2(model, data_dir, train_cfg, device, logger, out_dir, codec):
    from safetensors.torch import load_file
    from utils.train_utils import autocast_context
    sd = load_file(str(Path(out_dir) / "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    model.codec = codec
    loaders = build_task_dataloaders_v2(
        data_dir, batch_size=train_cfg["batch_size"], seq_len=train_cfg.get("seq_len", 1024),
        splits=("test",), num_workers=train_cfg.get("num_workers", 4),
        eval_batch_size=train_cfg.get("eval_batch_size", 1024))
    res = {}
    with autocast_context(device, train_cfg.get("amp", True)):
        for t in TASK_ORDER:
            if t in loaders["test"]:
                res[t] = TASKS_V2[t].evaluate(model, loaders["test"][t], device, train_cfg)
                logger.info("[test] %s: %s", t, {k: round(v, 6) for k, v in res[t].items()})
    if len(res) == 3:
        score = 0.3 * res["protocol"]["acc"] + 0.3 * res["modulation"]["acc"] + \
            0.4 * res["parameter"]["precision"]
        logger.info("[test] proxy score=%.5f", score)


def main_v2(args, cfg, device, rank, world_size, is_main, logger):
    train_cfg = cfg["training"]
    # 阶段专属覆盖：lr_finetune / batch_size_finetune（缺省沿用 lr / batch_size）
    train_cfg["lr"] = train_cfg.get("lr_finetune", train_cfg["lr"])
    train_cfg["batch_size"] = train_cfg.get("batch_size_finetune", train_cfg["batch_size"])
    codec = prepare_codec(args, train_cfg, logger)
    cfg["model"]["param_head_spec"] = codec.head_spec()
    cfg["model"]["param_codec"] = codec.spec
    loaders = build_task_dataloaders_v2(
        args.data_dir, batch_size=train_cfg["batch_size"], seq_len=train_cfg.get("seq_len", 1024),
        num_workers=train_cfg.get("num_workers", 4),
        bucket=train_cfg.get("length_bucketing", True) and world_size == 1,
        eval_batch_size=train_cfg.get("eval_batch_size", 1024),
        rank=rank, world_size=world_size, seed=train_cfg.get("seed", 42))
    missing = [t for t in TASK_ORDER if t not in loaders["train"]]
    if missing:
        raise RuntimeError(f"missing required train tasks: {missing}")
    model = build_model(cfg["model"]).to(device)
    model.codec = codec
    shared, total = assert_competition_limits(model)
    logger.info("arch=v2 device=%s shared_ratio=%.2f%% params=%.2fM (deploy %.2fM)", device,
                shared * 100, total / 1e6,
                sum(v.numel() for v in model.deploy_state_dict().values()) / 1e6)
    if not args.resume:
        mae_path = args.from_mae or str(Path(args.out_dir) / "mae_pretrained.pt")
        load_mae_pretrained(model, mae_path, logger)
    MultitaskFinetunerV2(cfg, model, device, logger, out_dir=args.out_dir, loaders=loaders,
                         codec=codec, resume=args.resume, smoke_steps=args.smoke_steps,
                         rank=rank, world_size=world_size).fit()
    if args.eval_test and is_main:
        evaluate_on_test_v2(model, args.data_dir, train_cfg, device, logger, args.out_dir, codec)


def main():
    parser = base_arg_parser("Single-backbone multi-task fine-tuning")
    parser.add_argument("--from_mae", default=None,
                        help="MAE 预训练权重路径（默认: out_dir/mae_pretrained.pt）")
    parser.add_argument("--eval_test", action="store_true",
                        help="训练结束后在 test 划分上无偏评估（加载导出的 best 权重）")
    parser.add_argument("--refit_codec", action="store_true",
                        help="v2：强制重新拟合 param_codec.json（数据划分变化后使用）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    seed_all(train_cfg.get("seed", 42))  # 各卡同 seed：初始化一致 + 梯度平均 => 权重同步
    device, rank, world_size, is_main = init_distributed()
    if device is None:
        device = get_device()
    logger = get_logger("finetune", ROOT / "logs", quiet=not is_main)
    if model_arch(cfg["model"]) == "v2":
        main_v2(args, cfg, device, rank, world_size, is_main, logger)
        teardown_distributed()
        return

    # ---- 数据（多卡时 train/val 均分片：train 训练、val 并行评估汇总；test 仅主卡） ----
    loaders, param_stats = build_task_dataloaders(
        args.data_dir, batch_size=train_cfg["batch_size"], seq_len=train_cfg["seq_len"],
        num_workers=train_cfg.get("num_workers", 0), rank=rank, world_size=world_size)
    missing = [t for t in TASK_ORDER if t not in loaders["train"]]
    if missing:
        raise RuntimeError(f"missing required train tasks: {missing}")

    # ---- 模型 ----
    model = EMFoundation(cfg["model"]).to(device)
    shared, total = assert_competition_limits(model)
    logger.info("device=%s shared_ratio=%.2f%% params=%d", device, shared * 100, total)

    if not args.resume:  # resume 时继续原权重，不重复加载 MAE 初始化
        mae_path = args.from_mae or str(Path(args.out_dir) / "mae_pretrained.pt")
        load_mae_pretrained(model, mae_path, logger)

    MultitaskFinetuner(cfg, model, device, logger, out_dir=args.out_dir,
                       loaders=loaders, param_stats=param_stats,
                       resume=args.resume, smoke_steps=args.smoke_steps,
                       rank=rank, world_size=world_size).fit()

    # test 无偏评估单进程足够（主卡做，各卡权重同步结果一致）
    if args.eval_test and is_main:
        evaluate_on_test(model, args.data_dir, train_cfg, param_stats,
                         device, logger, args.out_dir)
    teardown_distributed()


if __name__ == "__main__":
    main()
