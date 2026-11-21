"""训练器：生命周期基类 + 两阶段派生实现。

BaseTrainer               fit() 封装完整生命周期（optimizer/scheduler/scaler 构建、
                          resume、epoch 循环、断点保存、指标记录），子类按钩子填充差异
├── MAEPretrainer         掩码重建预训练；train_one_epoch 唯一必填钩子，结束导出 mae_pretrained.pt
└── MultitaskFinetuner    三任务轮转微调；on_epoch_end 评估+代理得分+best 导出，should_stop 早停

入口脚本只负责：参数/配置/数据组装 -> 实例化对应 Trainer -> fit()。
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter

from training.common import ROOT
from training.tasks import TASKS, TASK_ORDER
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.dataset import save_param_stats
from utils.logger import (plot_metrics, plot_step_metrics, record_metrics,
                          record_step, reset_metrics)
from utils.optim import build_optimizer, build_scheduler
from utils.train_utils import all_reduce_mean, autocast_context, broadcast_flag, \
    optimizer_step, tqdm


# ================================================================ 导出
def export_pretrained(model, out_dir):
    """导出 MAE 预训练共享权重（tokenizer/backbone/频域分支）。"""
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith("task_heads")}
    torch.save(state, Path(out_dir) / "mae_pretrained.pt")


def export_model(model, cfg, param_stats, out_dir):
    """导出官方推理格式（config.json + model.safetensors + param_stats.json）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()},
              str(out_dir / "model.safetensors"))
    save_param_stats(param_stats, out_dir / "param_stats.json")


# ================================================================ 基类
class BaseTrainer(ABC):
    """训练生命周期骨架。

    子类钩子（除 train_one_epoch / scheduler_total_steps / n_epochs 必填外均可选）:
      train_parameters       优化器参数（预训练排除任务头）
      on_epoch_end           epoch 末回调（评估/导出），返回追加进 metrics 的 dict
      checkpoint_extra       断点附加状态（如 finetune 的 best/bad）
      on_resume              从断点恢复附加状态
      should_stop            早停条件
      on_finish              fit 结束回调（导出等）
    """

    tag = "train"          # metrics jsonl 命名
    resume_name = None     # 断点文件名（子类必填）

    def __init__(self, cfg, model, device, logger, out_dir,
                 resume=None, smoke_steps=0, rank=0, world_size=1):
        self.cfg = cfg                      # 完整配置（导出 config.json 用）
        self.tcfg = cfg["training"]         # 训练参数（optimizer/scheduler/epoch 等）
        self.model = model
        self.device = device
        self.logger = logger
        self.out_dir = Path(out_dir)
        self.smoke_steps = smoke_steps
        self.rank = rank                    # 多卡：本进程卡号（0 为主卡）
        self.world_size = world_size        # 多卡：总卡数（单卡为 1）
        self.is_main = rank == 0
        self.start_epoch = 0
        # 优化器：AdamW（默认，范数/偏置/嵌入不衰减）或 Muon（tcfg.optimizer="muon"）
        self.optimizer = build_optimizer(self.train_named_parameters(), self.tcfg)
        total = self.scheduler_total_steps()
        # warmup 优先按步数配置（warmup_steps），否则按比例；比例默认 0.03（v1 的 0.1×200epoch
        # 会导致 20 个 epoch 都在 warmup，早停在学习率退火前就触发）
        warm = self.tcfg.get("warmup_steps") or int(total * self.tcfg.get("warmup_ratio", 0.03))
        self.scheduler = build_scheduler(self.optimizer, total, warm,
                                         self.tcfg.get("min_lr_ratio", 0.01))
        # bf16 混合精度无需 loss scaling（GradScaler 仅 fp16 需要）；
        # 保留 scaler 对象以维持 optimizer_step / 断点保存接口不变（enabled=False 时透传）
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(device.type == "cuda" and self.tcfg.get("amp_dtype") == "fp16"))
        # TensorBoard 可视化（train/loss、train/grad_norm、train/lr 等；tag 区分阶段）
        # 多卡时仅主卡写（事件文件/日志/断点均由主卡独占，避免多进程写坏文件）
        self.global_step = 0
        self._epoch = 0
        self.writer = (SummaryWriter(log_dir=str(ROOT / "logs" / "tb" / self.tag))
                       if self.is_main else None)
        if resume:
            self.start_epoch, extra = load_checkpoint(
                resume, model=model, optimizer=self.optimizer,
                scheduler=self.scheduler, scaler=self.scaler)
            self.on_resume(extra)
            logger.info("resumed from %s at epoch %d", resume, self.start_epoch)

    # ---------------- 钩子 ----------------
    def train_named_parameters(self):
        """参与训练的 (name, param) 列表（子类可过滤）。"""
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    @abstractmethod
    def train_one_epoch(self):
        """一个 epoch 的训练，返回指标 dict。"""

    @abstractmethod
    def scheduler_total_steps(self):
        """调度器总步数（决定 warmup + cosine 形状）。"""

    @abstractmethod
    def n_epochs(self):
        """总 epoch 数。"""

    def on_epoch_end(self, epoch):
        return {}

    def checkpoint_extra(self):
        return {}

    def on_resume(self, extra):
        pass

    def should_stop(self):
        return False

    def on_finish(self):
        pass

    # ---------------- 可视化（仅主卡生效） ----------------
    def _log_scalars(self, values):
        """step 级指标写 TensorBoard + step jsonl（含 epoch，供 step 级出图）。"""
        if not self.is_main:
            return
        for k, v in values.items():
            self.writer.add_scalar(k, v, self.global_step)
        record_step(ROOT / "logs", self.tag, self.global_step,
                    self._epoch + 1, values)

    def _log_epoch(self, values, epoch):
        """epoch 级指标写 TensorBoard（epoch/ 前缀；jsonl 由 record_metrics 落盘）。"""
        if not self.is_main:
            return
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"epoch/{k}", v, epoch)
        self.writer.flush()

    # ---------------- 多卡 ----------------
    def _train_loaders(self):
        """本阶段的训练 loader 列表（子类按需覆写）。"""
        return []

    def _set_sampler_epoch(self, epoch):
        """多卡时同步 DistributedSampler 的打乱种子（各 epoch 数据划分不同）。"""
        if self.world_size < 2:
            return
        for loader in self._train_loaders():
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    # ---------------- 生命周期 ----------------
    def fit(self):
        # 从头训练时刷新指标 jsonl（新运行覆盖旧记录）；断点续训保留历史继续追加
        if self.is_main and self.start_epoch == 0:
            reset_metrics(ROOT / "logs", self.tag)
        try:
            for epoch in range(self.start_epoch, self.n_epochs()):
                self._epoch = epoch
                self._set_sampler_epoch(epoch)
                values = self.train_one_epoch()
                values["lr"] = self.scheduler.get_last_lr()[0]
                values.update(self.on_epoch_end(epoch))
                if self.is_main:
                    record_metrics(ROOT / "logs", self.tag, epoch + 1, values)
                    self.logger.info("epoch=%d %s", epoch + 1,
                                     " ".join(f"{k}={v:.5f}" for k, v in values.items()))
                    self._log_epoch(values, epoch + 1)
                    save_checkpoint(self.out_dir / self.resume_name, epoch=epoch,
                                    model=self.model, optimizer=self.optimizer,
                                    scheduler=self.scheduler, scaler=self.scaler,
                                    **self.checkpoint_extra())
                if self.smoke_steps:
                    break
                # 早停判定：val 指标经 all-reduce 后各卡一致，各卡独立判定结果相同；
                # 广播兜底保证各卡 epoch 循环步调严格一致（主卡写断点略慢）
                stop = self.should_stop()
                if broadcast_flag(stop, self.device, self.world_size):
                    self.logger.info("early stopping at epoch %d", epoch + 1)
                    break
        finally:
            if self.writer is not None:
                self.writer.close()
        # 训练结束后自动绘制指标曲线图（仅主卡）
        # epoch 级：logs/plots/<tag>_<group>.png；step 级（含 epoch 分界标注）：
        # logs/plots/<tag>_step_<group>.png
        if self.is_main:
            for png in plot_metrics(ROOT / "logs" / f"{self.tag}_metrics.jsonl",
                                    ROOT / "logs" / "plots", self.tag):
                self.logger.info("plot saved: %s", png)
            for png in plot_step_metrics(ROOT / "logs" / f"{self.tag}_steps.jsonl",
                                         ROOT / "logs" / "plots", self.tag,
                                         ROOT / "logs" / f"{self.tag}_metrics.jsonl"):
                self.logger.info("step plot saved: %s", png)
            self.on_finish()


# ================================================================ 子类
class MAEPretrainer(BaseTrainer):
    """阶段一：MAE 掩码重建自监督预训练（任务头不参与）。"""

    tag = "pretrain"
    resume_name = "mae_resume.pt"

    def __init__(self, cfg, model, device, logger, out_dir, loader,
                 resume=None, smoke_steps=0, rank=0, world_size=1):
        self.loader = loader  # 须在 super().__init__ 前就位（调度器构建依赖）
        self.best_loss, self.bad = float("inf"), 0  # 须在 super().__init__ 前（resume 依赖）
        super().__init__(cfg, model, device, logger, out_dir, resume, smoke_steps,
                         rank, world_size)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("device=%s samples=%d world_size=%d shared_ratio=%.3f",
                    device, len(loader.dataset), world_size, model.shared_ratio())

    def _train_loaders(self):
        return [self.loader]

    def train_named_parameters(self):
        # 任务头无梯度，不参与预训练
        return [(name, p) for name, p in self.model.named_parameters()
                if not name.startswith("task_heads") and p.requires_grad]

    def n_epochs(self):
        return self.tcfg["epochs_pretrain"]

    def scheduler_total_steps(self):
        steps = len(self.loader)
        if self.smoke_steps:
            steps = min(steps, self.smoke_steps)
        return max(1, steps * self.tcfg["epochs_pretrain"])

    def train_one_epoch(self):
        self.model.train()
        total, count = 0.0, 0
        norms = []
        desc = f"pretrain epoch {self._epoch + 1}/{self.n_epochs()}"
        for step, (iq, sr) in enumerate(tqdm(self.loader, desc=desc,
                                             leave=False, dynamic_ncols=True,
                                             disable=not self.is_main)):
            if self.smoke_steps and step >= self.smoke_steps:
                break
            iq, sr = iq.to(self.device), sr.to(self.device)
            with autocast_context(self.device, self.tcfg.get("amp", True)):
                recon, target = self.model.forward_mae(iq, sr)
                loss = (recon.float() - target.float()).square().mean()
            grad_norm = optimizer_step(loss, self.optimizer, self.model, self.scaler,
                                       self.tcfg.get("grad_clip", 1.0),
                                       world_size=self.world_size)
            self.scheduler.step()
            loss_v = loss.item()
            total += loss_v * iq.size(0)
            count += iq.size(0)
            norms.append(grad_norm)
            self.global_step += 1
            self._log_scalars({"train/loss": loss_v,
                               "train/grad_norm": grad_norm,
                               "train/lr": self.scheduler.get_last_lr()[0]})
        self._epoch_loss = total / max(1, count)
        return {"recon_mse": self._epoch_loss,
                "grad_norm": sum(norms) / max(1, len(norms))}

    def on_epoch_end(self, epoch):
        """预训练无验证集, 以训练 loss（recon_mse）作早停依据。"""
        if self.smoke_steps:
            return {}
        if self._epoch_loss < self.best_loss:
            self.best_loss, self.bad = self._epoch_loss, 0
            self.logger.info("pretrain best recon_mse=%.5f", self.best_loss)
        else:
            self.bad += 1
        return {}

    def checkpoint_extra(self):
        return {"best_loss": self.best_loss, "bad": self.bad}

    def on_resume(self, extra):
        self.best_loss = extra.get("best_loss", float("inf"))
        self.bad = extra.get("bad", 0)

    def should_stop(self):
        # 训练 loss 连续 patience_pretrain 个 epoch 不下降即停
        return self.bad >= self.tcfg.get("patience_pretrain", 10)

    def on_finish(self):
        export_pretrained(self.model, self.out_dir)
        self.logger.info("saved MAE pretrained weights -> %s",
                         self.out_dir / "mae_pretrained.pt")


class MultitaskFinetuner(BaseTrainer):
    """阶段二：单 backbone 多任务微调（protocol/modulation/parameter 轮转）。"""

    tag = "finetune"
    resume_name = "finetune_resume.pt"

    def __init__(self, cfg, model, device, logger, out_dir, loaders, param_stats,
                 resume=None, smoke_steps=0, rank=0, world_size=1):
        self.loaders = loaders
        self.param_stats = param_stats
        # tcfg 在 super().__init__ 才就位，此处直接读 training 子配置
        self.weights = cfg["training"].get("task_weights",
                                           {"protocol": 0.3, "modulation": 0.3, "parameter": 0.4})
        self.best, self.bad = -float("inf"), 0  # 须在 super().__init__ 前（resume 依赖）
        super().__init__(cfg, model, device, logger, out_dir, resume, smoke_steps,
                         rank, world_size)

    def n_epochs(self):
        return self.tcfg["epochs_finetune"]

    def _train_loaders(self):
        return list(self.loaders["train"].values())

    def steps_per_epoch(self):
        steps = max(len(self.loaders["train"][t]) for t in TASK_ORDER)
        return min(steps, self.smoke_steps) if self.smoke_steps else steps

    def scheduler_total_steps(self):
        # 每步依次更新三个任务 -> 每 epoch 优化步数 = steps * len(TASK_ORDER)
        return max(1, self.steps_per_epoch() * len(TASK_ORDER)
                   * self.tcfg["epochs_finetune"])

    def train_one_epoch(self):
        self.model.train()
        n_steps = self.steps_per_epoch()
        iters = {t: iter(self.loaders["train"][t]) for t in TASK_ORDER}
        losses = {t: 0.0 for t in TASK_ORDER}
        norms = []
        desc = f"finetune epoch {self._epoch + 1}/{self.n_epochs()}"
        for _ in tqdm(range(n_steps), desc=desc, leave=False, dynamic_ncols=True,
                      disable=not self.is_main):
            for task in TASK_ORDER:
                batch = next_batch(iters, self.loaders["train"], task)
                # task/sample_id 等字符串字段 collate 后是 list，只搬移 tensor
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with autocast_context(self.device, self.tcfg.get("amp", True)):
                    feat = self.model.encode(batch["iq"], batch["sr"])
                    loss = TASKS[task].loss(self.model, feat, batch, self.tcfg)
                    loss = loss * float(self.weights[task])
                grad_norm = optimizer_step(loss, self.optimizer, self.model, self.scaler,
                                           self.tcfg.get("grad_clip", 1.0),
                                           world_size=self.world_size)
                self.scheduler.step()
                loss_v = loss.item()
                losses[task] += loss_v
                norms.append(grad_norm)
                self.global_step += 1
                self._log_scalars({f"train/loss_{task}": loss_v,
                                   "train/grad_norm": grad_norm,
                                   "train/lr": self.scheduler.get_last_lr()[0]})
        values = {f"train_{t}_weighted_loss": losses[t] / max(1, n_steps)
                  for t in TASK_ORDER}
        values["grad_norm"] = sum(norms) / max(1, len(norms))
        return values

    def on_epoch_end(self, epoch):
        # 多卡时 val 分片并行评估（各卡权重同步，分片结果等价），指标 all-reduce
        # 汇总——若只有主卡独自评估全量 val（~5 分钟），其余卡会在早停广播上
        # 空等导致 HCCL 链路超时（EI0006），分片后评估秒级完成且各卡同时到达广播
        if self.smoke_steps or (epoch + 1) % self.tcfg.get("eval_every", 1) != 0:
            return {}
        if any(t not in self.loaders.get("val", {}) for t in TASK_ORDER):
            raise RuntimeError("missing validation split for one or more tasks")
        results = {}
        for t in TASK_ORDER:
            loader = self.loaders["val"][t]
            m = TASKS[t].evaluate(self.model, loader, self.device,
                                  self.param_stats, self.tcfg.get("ref_scale", 0.5))
            # len(sampler) = 本卡分片样本数（多卡）/ 全量（单卡 SequentialSampler）
            results[t] = all_reduce_mean(m, len(loader.sampler),
                                         self.device, self.world_size)
        values = {f"val_{t}_{k}": v for t, m in results.items() for k, v in m.items()}
        score = (0.3 * results["protocol"]["acc"]
                 + 0.3 * results["modulation"]["acc"]
                 + 0.4 * results["parameter"]["precision"])
        values["val_proxy_score"] = score
        # best/bad 各卡一致更新（all-reduce 结果比特级一致）；导出与日志仅主卡
        if score > self.best:
            self.best, self.bad = score, 0
            if self.is_main:
                export_model(self.model, self.cfg, self.param_stats, self.out_dir)
                self.logger.info("new best proxy score=%.5f", self.best)
        else:
            self.bad += 1
        return values

    def checkpoint_extra(self):
        return {"best": self.best, "bad": self.bad}

    def on_resume(self, extra):
        self.best = extra.get("best", -float("inf"))
        self.bad = extra.get("bad", 0)

    def should_stop(self):
        return self.bad >= self.tcfg.get("patience", 7)

    def on_finish(self):
        if self.smoke_steps:
            export_model(self.model, self.cfg, self.param_stats, self.out_dir)
        self.logger.info("finished; official inference weights: %s",
                         self.out_dir / "model.safetensors")


# ================================================================ 工具
def next_batch(iters, loader, task):
    """循环取 batch：迭代器耗尽时自动重开。"""
    try:
        return next(iters[task])
    except StopIteration:
        iters[task] = iter(loader[task])
        return next(iters[task])


# ================================================================ v2 训练器
from training.tasks_v2 import TASKS_V2, TASK_ORDER as TASK_ORDER_V2   # noqa: E402
from utils.augment import augment_iq                                  # noqa: E402
from utils.optim import ModelEMA                                      # noqa: E402
from utils.train_utils import all_reduce_grads                        # noqa: E402


class _Cycle:
    """无限循环迭代器（steps_per_epoch 与 loader 长度解耦）。"""

    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def export_model_v2(model, cfg, codec, out_dir):
    """导出官方推理格式：config.json（内嵌 param_codec）+ model.safetensors + param_codec.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(cfg))
    cfg["model"]["param_codec"] = codec.spec
    cfg["model"]["param_head_spec"] = codec.head_spec()
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    sd = model.deploy_state_dict()
    save_file({k: v.detach().contiguous().clone().cpu() for k, v in sd.items()},
              str(out_dir / "model.safetensors"))
    codec.save(out_dir / "param_codec.json")


class MAEPretrainerV2(MAEPretrainer):
    """v2 预训练：SimMIM 块掩码 + 防泄漏全局视图 + val 重建损失早停。

    epoch 定义为 steps_per_epoch_pretrain 个优化步（0 = 遍历一次 loader），
    每个 epoch 末在 val 子集上用固定随机种子的掩码计算重建损失，
    连续 patience_pretrain 次没有改善 min_delta(相对) 即早停。
    """

    def __init__(self, cfg, model, device, logger, out_dir, loader, val_loader=None,
                 resume=None, smoke_steps=0, rank=0, world_size=1):
        self.val_loader = val_loader
        self._spe = cfg["training"].get("steps_per_epoch_pretrain", 0) or len(loader)
        super().__init__(cfg, model, device, logger, out_dir, loader, resume, smoke_steps,
                         rank, world_size)
        self.cycle = _Cycle(loader)

    def train_named_parameters(self):
        return [(n, p) for n, p in self.model.named_parameters()
                if not n.startswith("task_heads") and p.requires_grad]

    def scheduler_total_steps(self):
        spe = min(self._spe, self.smoke_steps) if self.smoke_steps else self._spe
        return max(1, spe * self.tcfg["epochs_pretrain"])

    def train_one_epoch(self):
        self.model.train()
        n = min(self._spe, self.smoke_steps) if self.smoke_steps else self._spe
        tot, cnt, norms = 0.0, 0, []
        acc = {"loss_time": 0.0, "loss_stft": 0.0}
        log_every = self.tcfg.get("log_every", 50)
        for step in tqdm(range(n), desc=f"pretrain-v2 epoch {self._epoch + 1}/{self.n_epochs()}",
                         leave=False, dynamic_ncols=True, disable=not self.is_main):
            iq, _ = self.cycle.next()
            iq = augment_iq(iq.to(self.device, non_blocking=True), "pretrain",
                            self.tcfg.get("augment_policy"))
            with autocast_context(self.device, self.tcfg.get("amp", True)):
                loss, parts = self.model.forward_pretrain(iq)
            gn = optimizer_step(loss, self.optimizer, self.model, self.scaler,
                                self.tcfg.get("grad_clip", 1.0), world_size=self.world_size)
            self.scheduler.step()
            self.global_step += 1
            if step % log_every == 0 or step == n - 1:
                lv = float(loss)
                tot += lv
                cnt += 1
                norms.append(gn)
                for k in acc:
                    acc[k] += float(parts[k])
                self._log_scalars({"train/loss": lv, "train/grad_norm": gn,
                                   "train/lr": self.scheduler.get_last_lr()[0]})
        self._epoch_loss = tot / max(1, cnt)
        out = {"recon_loss": self._epoch_loss,
               "grad_norm": sum(norms) / max(1, len(norms))}
        out.update({k: v / max(1, cnt) for k, v in acc.items()})
        return out

    @torch.no_grad()
    def _val_loss(self):
        if self.val_loader is None:
            return None
        self.model.eval()
        n_max = self.tcfg.get("val_batches_pretrain", 50)
        tot, cnt = 0.0, 0
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(1234)
            for i, (iq, _) in enumerate(self.val_loader):
                if i >= n_max:
                    break
                with autocast_context(self.device, self.tcfg.get("amp", True)):
                    loss, _ = self.model.forward_pretrain(iq.to(self.device))
                tot += float(loss)
                cnt += 1
        return tot / max(1, cnt)

    def on_epoch_end(self, epoch):
        if self.smoke_steps:
            return {}
        v = self._val_loss()
        metric = v if v is not None else self._epoch_loss
        rel = self.tcfg.get("min_delta_pretrain", 1e-3)
        if metric < self.best_loss * (1 - rel):
            self.best_loss, self.bad = metric, 0
            if self.is_main:
                export_pretrained(self.model, self.out_dir)   # 随时保留最佳预训练权重
                self.logger.info("pretrain best %s=%.5f", "val_recon" if v is not None
                                 else "train_recon", metric)
        else:
            self.bad += 1
        return {"val_recon_loss": v} if v is not None else {}

    def should_stop(self):
        return self.bad >= self.tcfg.get("patience_pretrain", 5)

    def on_finish(self):
        if self.smoke_steps or not (self.out_dir / "mae_pretrained.pt").exists():
            export_pretrained(self.model, self.out_dir)
        self.logger.info("best pretrained weights -> %s", self.out_dir / "mae_pretrained.pt")


class MultitaskFinetunerV2(BaseTrainer):
    """v2 多任务微调：一次迭代 = 三任务各一个 batch 的梯度累加 -> 一次优化步。

    相比 v1 "每个任务各自 optimizer.step()"：
      * 共享参数每步看到的是三任务的联合梯度（真正的多任务优化，而非交替拉扯）；
      * 可选 PCGrad（Yu et al., NeurIPS 2020）消解任务间冲突梯度；
      * 可选不确定性加权（Kendall et al., CVPR 2018）或固定权重；
      * EMA 权重评估/导出；评估指标带 min_delta 的早停（按评估次数计）。
    """

    tag = "finetune"
    resume_name = "finetune_resume.pt"

    def __init__(self, cfg, model, device, logger, out_dir, loaders, codec,
                 resume=None, smoke_steps=0, rank=0, world_size=1):
        self.loaders = loaders
        self.codec = codec
        tc = cfg["training"]
        self.weights = tc.get("task_weights", {"protocol": 0.5, "modulation": 1.0,
                                               "parameter": 1.0})
        self.weighting = tc.get("task_weighting", "fixed")
        self.log_vars = None
        if self.weighting == "uncertainty":
            self.log_vars = torch.nn.Parameter(torch.zeros(len(TASK_ORDER_V2), device=device))
        self._spe = tc.get("steps_per_epoch", 0) or max(len(loaders["train"][t])
                                                        for t in TASK_ORDER_V2)
        self.best, self.bad = -float("inf"), 0
        self.ema = None
        super().__init__(cfg, model, device, logger, out_dir, resume, smoke_steps,
                         rank, world_size)
        if tc.get("ema_decay", 0) > 0:
            self.ema = ModelEMA(model, tc["ema_decay"])
            if resume:
                st = torch.load(resume, map_location="cpu", weights_only=False)
                if "ema" in st:
                    self.ema.load_state_dict(st["ema"])
        self.cycles = {t: _Cycle(loaders["train"][t]) for t in TASK_ORDER_V2}
        self.pcgrad = bool(tc.get("pcgrad", False))
        self.shared_params = [p for p in list(model.tokenizer.parameters())
                              + list(model.backbone.parameters())
                              if p.requires_grad and all(p is not q for q in
                                                         model.tokenizer.private_parameters())]

    # ---------------- 钩子 ----------------
    def train_named_parameters(self):
        named = [(n, p) for n, p in self.model.named_parameters()
                 if p.requires_grad and not n.startswith("pretrain_head")]
        if self.log_vars is not None:
            named.append(("log_vars.bias", self.log_vars))      # ".bias" -> 不做权重衰减
        return named

    def n_epochs(self):
        return self.tcfg["epochs_finetune"]

    def _train_loaders(self):
        return list(self.loaders["train"].values())

    def scheduler_total_steps(self):
        spe = min(self._spe, self.smoke_steps) if self.smoke_steps else self._spe
        return max(1, spe * self.tcfg["epochs_finetune"])

    def _weighted(self, i, task, loss):
        if self.log_vars is not None:
            s = self.log_vars[i]
            return 0.5 * torch.exp(-s) * loss + 0.5 * s
        return float(self.weights.get(task, 1.0)) * loss

    def _flat_shared_grad(self):
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in self.shared_params])

    def _apply_pcgrad(self, grads):
        out = []
        for i, g in enumerate(grads):
            g = g.clone()
            for j in torch.randperm(len(grads)).tolist():
                if j == i:
                    continue
                dot = torch.dot(g, grads[j])
                if dot < 0:
                    g -= dot / (grads[j].pow(2).sum() + 1e-12) * grads[j]
            out.append(g)
        total = torch.stack(out).sum(0)
        off = 0
        for p in self.shared_params:
            n = p.numel()
            p.grad = total[off:off + n].view_as(p).to(p.dtype)
            off += n

    def train_one_epoch(self):
        self.model.train()
        n = min(self._spe, self.smoke_steps) if self.smoke_steps else self._spe
        sums = {t: 0.0 for t in TASK_ORDER_V2}
        cnt, norms = 0, []
        log_every = self.tcfg.get("log_every", 50)
        for step in tqdm(range(n), desc=f"finetune-v2 epoch {self._epoch + 1}/{self.n_epochs()}",
                         leave=False, dynamic_ncols=True, disable=not self.is_main):
            self.optimizer.zero_grad(set_to_none=True)
            grads, losses = [], {}
            for i, task in enumerate(TASK_ORDER_V2):
                batch = self.cycles[task].next()
                batch = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with autocast_context(self.device, self.tcfg.get("amp", True)):
                    loss, _ = TASKS_V2[task].loss(self.model, batch, self.tcfg)
                    wl = self._weighted(i, task, loss)
                wl.backward()
                losses[task] = loss.detach()
                if self.pcgrad:
                    grads.append(self._flat_shared_grad())
                    for p in self.shared_params:
                        p.grad = None
            if self.pcgrad:
                self._apply_pcgrad(grads)
            all_reduce_grads(self.model, self.world_size)
            gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                      self.tcfg.get("grad_clip", 1.0)))
            self.optimizer.step()
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)
            self.global_step += 1
            if step % log_every == 0 or step == n - 1:
                cnt += 1
                norms.append(gn)
                vals = {f"train/loss_{t}": float(v) for t, v in losses.items()}
                for t, v in losses.items():
                    sums[t] += float(v)
                vals.update({"train/grad_norm": gn, "train/lr": self.scheduler.get_last_lr()[0]})
                if self.log_vars is not None:
                    vals.update({f"train/logvar_{t}": float(self.log_vars[i])
                                 for i, t in enumerate(TASK_ORDER_V2)})
                self._log_scalars(vals)
        out = {f"train_{t}_loss": sums[t] / max(1, cnt) for t in TASK_ORDER_V2}
        out["grad_norm"] = sum(norms) / max(1, len(norms))
        return out

    def eval_model(self):
        m = self.ema.ema if self.ema is not None else self.model
        m.codec = self.codec
        return m

    def on_epoch_end(self, epoch):
        if self.smoke_steps or (epoch + 1) % self.tcfg.get("eval_every", 1) != 0:
            return {}
        model = self.eval_model()
        results = {}
        with autocast_context(self.device, self.tcfg.get("amp", True)):
            for t in TASK_ORDER_V2:
                loader = self.loaders["val"][t]
                m = TASKS_V2[t].evaluate(model, loader, self.device, self.tcfg)
                results[t] = all_reduce_mean(m, len(loader.dataset) // max(1, self.world_size),
                                             self.device, self.world_size)
        values = {f"val_{t}_{k}": v for t, m in results.items() for k, v in m.items()}
        w = self.tcfg.get("score_weights", {"protocol": 0.3, "modulation": 0.3,
                                            "parameter": 0.4})
        score = (w["protocol"] * results["protocol"]["acc"]
                 + w["modulation"] * results["modulation"]["acc"]
                 + w["parameter"] * results["parameter"]["precision"])
        values["val_proxy_score"] = score
        if score > self.best + self.tcfg.get("min_delta", 1e-4):
            self.best, self.bad = score, 0
            if self.is_main:
                export_model_v2(model, self.cfg, self.codec, self.out_dir)
                self.logger.info("new best proxy score=%.5f (exported %s weights)", score,
                                 "EMA" if self.ema is not None else "raw")
        else:
            self.bad += 1
        self.model.train()
        return values

    def checkpoint_extra(self):
        extra = {"best": self.best, "bad": self.bad}
        if self.ema is not None:
            extra["ema"] = self.ema.state_dict()
        if self.log_vars is not None:
            extra["log_vars"] = self.log_vars.detach().cpu()
        return extra

    def on_resume(self, extra):
        self.best = extra.get("best", -float("inf"))
        self.bad = extra.get("bad", 0)
        if self.log_vars is not None and "log_vars" in extra:
            with torch.no_grad():
                self.log_vars.copy_(extra["log_vars"].to(self.log_vars.device))

    def should_stop(self):
        return self.bad >= self.tcfg.get("patience", 6)

    def on_finish(self):
        if self.smoke_steps or not (self.out_dir / "model.safetensors").exists():
            export_model_v2(self.eval_model(), self.cfg, self.codec, self.out_dir)
        self.logger.info("finished; official inference weights: %s",
                         self.out_dir / "model.safetensors")
