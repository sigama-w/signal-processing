from .foundation import EMFoundation
from .foundation_v2 import EMFoundationV2
from .tokenizer import IQTokenizer
from .backbone import build_backbone
from .heads import ClassificationHead, RegressionHead


def model_arch(model_cfg):
    """配置里 arch 缺省为 v1（兼容旧 config.json / 旧权重）。"""
    return str(model_cfg.get("arch", "v1")).lower()


def build_model(model_cfg):
    """统一模型工厂：arch='v2' -> EMFoundationV2；否则 v1 EMFoundation。"""
    if model_arch(model_cfg) == "v2":
        return EMFoundationV2(model_cfg)
    return EMFoundation(model_cfg)


__all__ = [
    "EMFoundation", "EMFoundationV2", "build_model", "model_arch",
    "IQTokenizer", "build_backbone", "ClassificationHead", "RegressionHead",
]
