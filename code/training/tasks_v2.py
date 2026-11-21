"""v2 任务模块：损失 + 评估（与 v1 tasks.py 同契约风格，调用方零分支）

分类（protocol / modulation）
  L = CE(label_smoothing) + λ_fam·CE(调制族, 仅 modulation) + λ_sup·SupCon(可选)
参数（parameter）
  L = mean_j L_j + λ_pat·CE(pattern)
      L_j: 连续参数 -> 标准化空间 MSE/Huber（按有效样本平均）
           离散参数 -> CE（按有效样本平均）
      7 个参数等权（不受通信 3 / 雷达 4 个数影响）

评估（只做监控与模型选择；官方分数以主办方评测为准）
  分类: acc, macro_recall
  参数: pattern_acc；每参数 SI 空间 RMSE；按训练集取值范围归一化的 nRMSE；
        precision = 1/(1+mean nRMSE/ref_scale)（ref_scale 默认 0.1，仅作代理）
        ——使用"预测的" pattern 硬掩码（与官方推理一致：模式判错的参数输出 0 会被计入误差）
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from utils.augment import augment_iq
from utils.metrics import accuracy, macro_recall

TASK_ORDER = ("protocol", "modulation", "parameter")


def supcon_loss(z, y, temperature=0.1):
    """监督对比损失（Khosla et al., NeurIPS 2020），单视角 batch 内同类为正样本。"""
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.t() / temperature
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    npos = pos.sum(1)
    valid = npos > 0
    loss = -(log_prob * pos).sum(1)[valid] / npos[valid]
    return loss.mean() if valid.any() else z.sum() * 0


class BaseTaskV2(ABC):
    name = ""

    @abstractmethod
    def loss(self, model, batch, cfg):
        """返回 (loss, logs: dict[str,float])。batch 已在设备上。"""

    @abstractmethod
    def evaluate(self, model, loader, device, cfg):
        """返回指标 dict。"""

    @staticmethod
    def _aug(batch, name, cfg):
        if cfg.get("augment", True):
            return augment_iq(batch["iq"], name, cfg.get("augment_policy"))
        return batch["iq"]


class ClassificationTaskV2(BaseTaskV2):
    def __init__(self, name, n_classes):
        self.name = name
        self.n_classes = n_classes

    def loss(self, model, batch, cfg):
        iq = self._aug(batch, self.name, cfg)
        out = model.head_outputs(iq, self.name, return_all=True)
        y = batch["label"]
        ls = cfg.get("label_smoothing", {}).get(self.name, 0.05)
        loss = F.cross_entropy(out["logits"].float(), y, label_smoothing=ls)
        logs = {"ce": loss.detach()}
        if out.get("aux") is not None:
            fam = model.mod_family[y]
            l_fam = F.cross_entropy(out["aux"].float(), fam)
            loss = loss + cfg.get("family_loss_weight", 0.2) * l_fam
            logs["fam"] = l_fam.detach()
        w_sup = cfg.get("supcon_weight", {}).get(self.name, 0.0)
        if w_sup > 0 and out.get("proj") is not None:
            l_sup = supcon_loss(out["proj"], y, cfg.get("supcon_temperature", 0.1))
            loss = loss + w_sup * l_sup
            logs["supcon"] = l_sup.detach()
        return loss, logs

    @torch.no_grad()
    def predict_loader(self, model, loader, device, return_logits=False):
        model.eval()
        ys, ps, lg, ids = [], [], [], []
        for batch in loader:
            logits = model.head_outputs(batch["iq"].to(device, non_blocking=True), self.name)
            ys.append(batch["label"].numpy())
            ps.append(logits.argmax(-1).cpu().numpy())
            ids += list(batch["sample_id"])
            if return_logits:
                lg.append(logits.float().cpu().numpy())
        out = {"y": np.concatenate(ys), "p": np.concatenate(ps), "sample_id": ids}
        if return_logits:
            out["logits"] = np.concatenate(lg)
        return out

    def evaluate(self, model, loader, device, cfg):
        r = self.predict_loader(model, loader, device)
        return {"acc": accuracy(r["y"], r["p"]),
                "macro_recall": macro_recall(r["y"], r["p"], self.n_classes)}


class ParameterTaskV2(BaseTaskV2):
    name = "parameter"

    def _reg_loss(self, pred, target, kind):
        if kind == "mse":
            return (pred - target) ** 2
        if kind == "huber":
            return F.huber_loss(pred, target, reduction="none", delta=1.0)
        return F.smooth_l1_loss(pred, target, reduction="none", beta=0.5)

    def loss(self, model, batch, cfg):
        codec = model.codec
        iq = self._aug(batch, "parameter", cfg)
        out = model.head_outputs(iq, "parameter")
        valid = batch["valid"].float()
        z, cls_t = codec.encode(batch["raw_params"])
        kind = cfg.get("param_loss", "huber")
        per_param, logs = [], {}
        for k, j in enumerate(codec.reg_idx):
            v = valid[:, j]
            l = self._reg_loss(out["reg"][:, k].float(), z[:, k], kind)
            per_param.append((l * v).sum() / v.sum().clamp_min(1.0))
        for k, j in enumerate(codec.cls_idx):
            v = valid[:, j]
            l = F.cross_entropy(out["cls"][k].float(), cls_t[:, k], reduction="none")
            per_param.append((l * v).sum() / v.sum().clamp_min(1.0))
        reg = torch.stack(per_param).mean()
        pat = F.cross_entropy(out["pattern"].float(), batch["pattern"])
        loss = reg + cfg.get("pattern_loss_weight", 0.3) * pat
        logs.update({"param": reg.detach(), "pattern": pat.detach()})
        return loss, logs

    @torch.no_grad()
    def predict_loader(self, model, loader, device):
        model.eval()
        P, Y, V, PT, PP = [], [], [], [], []
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            out = model.head_outputs(iq, "parameter")
            si = model.codec.decode(out)
            radar = out["pattern"].argmax(-1)
            mask = torch.zeros_like(si)
            mask[:, :3] = (radar == 0).float()[:, None]
            mask[:, 3:] = (radar == 1).float()[:, None]
            P.append((si * mask).cpu().numpy())
            Y.append(batch["raw_params"].numpy())
            V.append(batch["valid"].numpy())
            PT.append(batch["pattern"].numpy())
            PP.append(radar.cpu().numpy())
        return {"pred": np.concatenate(P).astype(np.float64),
                "true": np.concatenate(Y).astype(np.float64),
                "valid": np.concatenate(V).astype(bool),
                "pattern": np.concatenate(PT), "pattern_pred": np.concatenate(PP)}

    def evaluate(self, model, loader, device, cfg):
        r = self.predict_loader(model, loader, device)
        codec = model.codec
        rng = codec.vmax - codec.vmin
        names = codec.names
        res = {"pattern_acc": float((r["pattern"] == r["pattern_pred"]).mean())}
        nrmse = []
        for j, n in enumerate(names):
            m = r["valid"][:, j]
            if not m.any():
                continue
            e = r["pred"][m, j] - r["true"][m, j]
            rm = float(np.sqrt(np.mean(e ** 2)))
            res[f"rmse_{n}"] = rm
            nrmse.append(rm / max(rng[j], 1e-30))
        mean_n = float(np.mean(nrmse)) if nrmse else 0.0
        ref = cfg.get("ref_scale_v2", 0.1)
        res["nrmse"] = mean_n
        res["precision"] = 1.0 / (1.0 + mean_n / ref)
        return res


TASKS_V2 = {
    "protocol": ClassificationTaskV2("protocol", 8),
    "modulation": ClassificationTaskV2("modulation", 23),
    "parameter": ParameterTaskV2(),
}
