# 教学注释：验证学习率调度（warmup+余弦）与参数量格式化。
import json
from src.config import TrainConfig
from src.utils import set_seed, get_lr, human_params


def test_get_lr_warmup_and_decay():
    cfg = TrainConfig(warmup_steps=10, max_steps=110, lr=1.0, min_lr=0.0)
    # warmup 阶段线性上升
    assert get_lr(0, cfg) == 0.1
    assert abs(get_lr(9, cfg) - 1.0) < 1e-9
    # 训练结束后固定在 min_lr
    assert get_lr(110, cfg) == 0.0
    # 中点在 warmup 与 min 之间
    mid = get_lr(60, cfg)
    assert 0.0 <= mid <= 1.0


def test_human_params():
    assert human_params(1_234) == "1.2K"
    assert human_params(2_500_000) == "2.5M"
    assert human_params(3_100_000_000) == "3.1B"


def test_set_seed_reproducible():
    import torch
    set_seed(0)
    a = torch.randn(3)
    set_seed(0)
    b = torch.randn(3)
    assert torch.equal(a, b)
