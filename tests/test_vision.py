# 教学注释：只用**双向**注意力编码 patch 序列（无因果 mask），
# 输出每个 patch 的特征，供后续投影到语言模型维度。
import torch
from src.vision import VisionEncoder


def test_vit_output_shape_and_patch_count():
    vit = VisionEncoder(d_vision=64, depth=2, n_head=4, img_size=32, patch_size=16)
    x = torch.randn(2, 3, 32, 32)
    out = vit(x)
    assert vit.num_patches == 4          # (32/16)^2
    assert out.shape == (2, 4, 64)


def test_vit_is_bidirectional():
    torch.manual_seed(0)
    vit = VisionEncoder(d_vision=32, depth=1, n_head=4, img_size=32, patch_size=16)
    vit.eval()
    x = torch.randn(1, 3, 32, 32)
    o1 = vit(x)
    x2 = x.clone()
    x2[:, :, :16, :16] = torch.randn(3, 16, 16)  # 只改左上 patch
    o2 = vit(x2)
    # 双向注意力：改一个 patch 会影响**所有** patch 的输出
    assert not torch.allclose(o1[0, 0], o2[0, 0], atol=1e-5)
    assert not torch.allclose(o1[0, 3], o2[0, 3], atol=1e-5)


def test_vit_params_positive():
    vit = VisionEncoder(d_vision=64, depth=2, n_head=4, img_size=32, patch_size=16)
    assert vit.num_params() > 0
