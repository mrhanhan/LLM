"""手写视觉编码器 ViT：patch embedding + 位置编码 + 双向 Transformer 编码。

教学注释：语言模型按"词"处理序列；视觉编码器把图片切成固定大小的 patch，每个
patch 当作一个"视觉词"，这样图片也能变成一条序列交给 Transformer。
与语言模型不同，图片内部没有先后顺序，所以注意力是**双向**的（causal=False）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention, build_rope_cache


class PatchEmbed(nn.Module):
    """用 stride=patch_size 的卷积把图片切成 patch 并映射到 d_vision 维。"""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, d_vision: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size 必须能被 patch_size 整除"
        self.grid = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, d_vision, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                       # (B, d_vision, grid, grid)
        x = x.flatten(2).transpose(1, 2)       # (B, N, d_vision)
        return x


class ViTBlock(nn.Module):
    """双向 Transformer 块（复用 MultiHeadAttention，causal=False）。"""

    def __init__(self, d_vision: int, n_head: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_vision)
        self.attn = MultiHeadAttention(d_vision, n_head, n_head, dropout=dropout, causal=False)
        self.n2 = nn.LayerNorm(d_vision)
        hidden = int(d_vision * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_vision, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_vision), nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        h, _ = self.attn(self.n1(x), cos, sin)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, d_vision: int = 384, depth: int = 6, n_head: int = 6,
                 img_size: int = 128, patch_size: int = 16, in_chans: int = 3, dropout: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, d_vision)
        self.num_patches = self.patch_embed.grid ** 2
        # 可学习的位置编码（ViT 原论文做法）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_vision))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(d_vision, n_head, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_vision)
        head_dim = d_vision // n_head
        cos, sin = build_rope_cache(head_dim, self.num_patches, 10000.0, "cpu", torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, pixel_values):
        x = self.patch_embed(pixel_values) + self.pos_embed
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        return self.norm(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
