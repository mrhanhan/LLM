# 教学注释：核心正确性——因果性（未来 token 不能影响过去位置的输出）。
import torch
from src.attention import build_rope_cache, apply_rope, MultiHeadAttention


def test_rope_preserves_shape_and_first_position():
    cos, sin = build_rope_cache(8, 16, 10000.0, "cpu", torch.float32)
    x = torch.randn(2, 3, 5, 8)
    out = apply_rope(x, cos, sin, offset=0)
    assert out.shape == x.shape
    # 位置 0 的旋转角为 0，数值应保持不变
    assert torch.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6)


def test_causal_masking():
    torch.manual_seed(0)
    attn = MultiHeadAttention(32, 4, 4, dropout=0.0, causal=True)
    cos, sin = build_rope_cache(8, 32, 10000.0, "cpu", torch.float32)
    x = torch.randn(1, 10, 32)
    out1, _ = attn(x, cos, sin)
    x2 = x.clone()
    x2[:, 9] = torch.randn(32)  # 篡改最后一个位置
    out2, _ = attn(x2, cos, sin)
    # 前 9 个位置输出必须完全一致
    assert torch.allclose(out1[:, :9], out2[:, :9], atol=1e-6)


def test_gqa_shapes_and_cache():
    attn = MultiHeadAttention(32, 8, 2, dropout=0.0, causal=True)
    cos, sin = build_rope_cache(32 // 8, 32, 10000.0, "cpu", torch.float32)
    x = torch.randn(2, 6, 32)
    out, (k, v) = attn(x, cos, sin)
    assert out.shape == x.shape
    assert k.shape[1] == 2  # KV 头数为 2
    # 用缓存增量解码一个 token，输出形状正确
    x1 = torch.randn(2, 1, 32)
    out1, (k1, _) = attn(x1, cos, sin, past_kv=(k, v), offset=6)
    assert out1.shape == (2, 1, 32)
    assert k1.shape[2] == 7
