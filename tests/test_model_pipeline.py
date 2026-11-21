"""v1 回归测试（修复：旧版引用了不存在的 training.finetune_multitask.parameter_loss）。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from models import EMFoundation                      # noqa: E402
from training.tasks import TASKS                     # noqa: E402

CFG = {"backbone": "transformer", "d_model": 32, "n_layers": 1, "nhead": 4, "patch_size": 8,
       "mae_decoder_layers": 1, "head_hidden": 32, "use_frequency": True, "mask_ratio": 0.75}


def test_masked_mae_and_fusion():
    torch.manual_seed(5)
    m = EMFoundation(CFG)
    x = torch.randn(2, 64, 2)
    sr = torch.ones(2) * 1e6
    y, target = m.forward_mae(x, sr)
    assert y.shape == target.shape == (2, 6, 16)
    for task, classes in (("protocol", 8), ("modulation", 23), ("parameter", 7)):
        assert m(x, sr, task).shape == (2, classes)
    assert m.shared_ratio() > .8


def test_masked_loss_excludes_invalid_entries():
    m = EMFoundation(CFG)
    x = torch.randn(2, 64, 2)
    feat = m.encode(x, torch.ones(2) * 1e6)
    b = {"params": torch.zeros(2, 7), "valid": torch.zeros(2, 7), "pattern": torch.zeros(2).long()}
    l0 = TASKS["parameter"].loss(m, feat, b, {"pattern_loss_weight": 0.0})
    assert float(l0) == 0.0
