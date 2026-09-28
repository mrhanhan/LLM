"""GPT 模型：RMSNorm + SwiGLU + 残差 Block，权重共享的 decoder-only 语言模型。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import MultiHeadAttention, build_rope_cache
from src.config import ModelConfig, TrainConfig


class RMSNorm(nn.Module):
    """均方根归一化：比 LayerNorm 少一个均值项，更快且效果好。"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    """门控前馈：w3(silu(w1(x)) * w2(x))，比普通 MLP 更强。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    """Pre-Norm 残差块：x + Attn(Norm(x)) 再 x + MLP(Norm(x))。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_head, cfg.n_kv_head,
                                       dropout=cfg.dropout, causal=True)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)

    def forward(self, x, cos, sin, past_kv=None, offset=0):
        h, new_kv = self.attn(self.n1(x), cos, sin, past_kv=past_kv, offset=offset)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        head_dim = cfg.d_model // cfg.n_head
        cos, sin = build_rope_cache(head_dim, cfg.ctx_len, cfg.rope_theta, "cpu", torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed(self, idx):
        return self.drop(self.tok_emb(idx))

    def forward(self, idx=None, inputs_embeds=None, targets=None, past_kvs=None, use_cache=False):
        x = inputs_embeds if inputs_embeds is not None else self.embed(idx)
        offset = past_kvs[0][0].shape[-2] if past_kvs is not None else 0
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pk = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, self.cos, self.sin, past_kv=pk, offset=offset)
            if use_cache:
                new_kvs.append(new_kv)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss, new_kvs

    @torch.no_grad()
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def configure_optimizers(self, train_cfg: TrainConfig) -> torch.optim.Optimizer:
        """把参数分成"要 weight decay"和"不要"两组（Norm 与 bias 不衰减）。"""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = torch.cuda.is_available()
        try:
            return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                     betas=(train_cfg.beta1, train_cfg.beta2), fused=fused)
        except TypeError:
            return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                     betas=(train_cfg.beta1, train_cfg.beta2))
