"""手写注意力：旋转位置编码 RoPE + 因果多头注意力（支持 GQA 与 KV 缓存）。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_rope_cache(head_dim: int, max_seq: int, theta: float, device, dtype):
    """预计算 RoPE 的 cos/sin 表。

    RoPE 把位置编码成"旋转角度"：第 i 对维度用频率 1/theta^(2i/d)。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (max_seq, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """对 (B, H, T, D) 施加旋转位置编码。offset 用于增量解码时的位置偏移。"""
    t = x.shape[-2]
    c = cos[offset: offset + t].view(1, 1, t, -1)
    s = sin[offset: offset + t].view(1, 1, t, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class MultiHeadAttention(nn.Module):
    """多头自注意力：QKV 投影 -> RoPE -> （可选）拼接缓存 -> 缩放点积 -> 因果 mask -> 输出投影。

    GQA（分组查询注意力）：KV 头数 n_kv_head 少于 Q 头数 n_head，多个 Q 头共享一组 KV，
    可显著减小 KV 缓存的显存占用（推理时关键）。
    """

    def __init__(self, d_model: int, n_head: int, n_kv_head: int, dropout: float = 0.0,
                 causal: bool = True, bias: bool = False):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        assert n_head % n_kv_head == 0, "n_head 必须能被 n_kv_head 整除"
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.causal = causal
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_head * self.head_dim, d_model, bias=bias)

    def forward(self, x, cos, sin, past_kv=None, offset=0):
        B, T, _ = x.shape
        hd = self.head_dim
        q = self.q_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, hd).transpose(1, 2)

        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)

        # GQA：把 KV 头复制到与 Q 头数相同
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        if self.causal:
            tq, tk = q.shape[-2], k.shape[-2]
            mask = torch.triu(
                torch.ones(tq, tk, device=x.device, dtype=torch.bool),
                diagonal=tk - tq + 1,
            )
            att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = torch.dropout(att, self.dropout_p, train=self.training)
        out = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), new_kv
