"""可视化运行时：一个很小的字符级 GPT，用于"实时训练 / 实时推理"演示。

用它而不是加载 192M 的 `out/*/latest.pt`，是为了在浏览器里真的能看到
权重逐步变化，而不必等几分钟一步。想看真实 checkpoint 的权重分布，用
`--mode weights`（见 server.py）。
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable

import torch
import torch.nn.functional as F

from src.config import ModelConfig, TrainConfig
from src.model import GPT
from src.attention import apply_rope

# 内置一小段中文语料（诗词/常识句），字符级建模，几十秒就能看到 loss 下降。
CORPUS = (
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"
    "人工智能正在改变世界，语言模型可以理解并生成文字。"
    "模型由很多层组成，每一层都在学习不同的规律。"
    "训练就是不断调整权重，让预测的损失越来越小。"
    "注意力机制让模型知道该关注句子里的哪些词。"
    "神经网络是一层一层地提取特征，从简单到复杂。"
    "这是一段用来演示的训练语料，反复出现让模型记住它。"
) * 4


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)


def build_tiny(vocab_size: int, device: str) -> GPT:
    cfg = ModelConfig(
        vocab_size=vocab_size, d_model=64, n_layer=4, n_head=4, n_kv_head=2,
        d_ff=128, ctx_len=64, dropout=0.0, rope_theta=10000.0, tie_embeddings=True,
    )
    return GPT(cfg).to(device)


class LiveTrainer:
    """后台线程训练小模型，每个 log_interval 通过 on_step 回调推送状态。"""

    def __init__(self, tokenizer: CharTokenizer, device: str,
                 on_step: Callable[[int, float, float], None],
                 lr: float = 3e-3, batch_size: int = 16, block_size: int = 48,
                 model: GPT | None = None):
        self.tok = tokenizer
        self.device = device
        self.on_step = on_step
        self.lr = lr
        self.batch = batch_size
        self.block = block_size
        self.model: GPT | None = model
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.step = 0
        self.loss = float("nan")
        data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
        self.data = data.to(device)
        self.optim: torch.optim.Optimizer | None = None

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = build_tiny(self.tok.vocab_size, self.device)
        if self.optim is None:
            tcfg = TrainConfig(lr=self.lr, weight_decay=0.01, grad_clip=1.0)
            self.optim = self.model.configure_optimizers(tcfg)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ensure_model()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        assert self.model is not None and self.optim is not None
        self.model.train()
        n = self.data.numel()
        while not self._stop.is_set():
            ix = torch.randint(0, max(n - self.block - 1, 1), (self.batch,), device=self.device)
            x = torch.stack([self.data[i:i + self.block] for i in ix])
            y = torch.stack([self.data[i + 1:i + 1 + self.block] for i in ix])
            _, loss, _ = self.model(x, targets=y)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.step += 1
            self.loss = float(loss.detach())
            if self.step % 5 == 0:
                self.on_step(self.step, self.loss, self.lr)
            time.sleep(0.001)


# ----------------------------------------------------------------------------
# 推理：逐 token 生成，同时暴露内部状态
# ----------------------------------------------------------------------------
def attention_probs(attn, x: torch.Tensor, cos, sin, offset: int = 0):
    """复刻 MultiHeadAttention 的前半段，返回注意力概率 [B, H, Tq, Tk]。"""
    B, T, _ = x.shape
    hd = attn.head_dim
    q = attn.q_proj(x).view(B, T, attn.n_head, hd).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, attn.n_kv_head, hd).transpose(1, 2)
    q = apply_rope(q, cos, sin, offset)
    k = apply_rope(k, cos, sin, offset)
    if attn.n_kv_head != attn.n_head:
        rep = attn.n_head // attn.n_kv_head
        k = k.repeat_interleave(rep, dim=1)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    Tq, Tk = q.shape[-2], k.shape[-2]
    mask = torch.triu(torch.ones(Tq, Tk, device=x.device, dtype=torch.bool),
                      diagonal=Tk - Tq + 1)
    att = att.masked_fill(mask, float("-inf"))
    return torch.softmax(att, dim=-1)


@torch.no_grad()
def stream_generate(model: GPT, tok: CharTokenizer, prompt: str, on_token: Callable,
                    max_new_tokens: int = 40, temperature: float = 0.9,
                    top_k: int = 20, attn_layer: int = 0):
    """自回归生成，每生成一个 token 调一次 on_token(text, probs)。

    on_token 收到该步的注意力概率（最后 24 个位置），供前端画热力图。
    """
    model.eval()
    ids = tok.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    past = None
    out_text = prompt
    for _ in range(max_new_tokens):
        T = idx.shape[1]
        offset = past[0][0].shape[-2] if past is not None else 0
        cur = idx[:, offset:] if past is not None else idx
        logits, _, new_kvs = model(cur, past_kvs=past, use_cache=True)
        past = new_kvs
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        nxt_id = int(nxt.item())
        out_text += tok.decode([nxt_id])

        # 注意力概率：用缓存里的 K 和当前 token 的 q，取这一行
        attn_pack = None
        if past is not None and offset >= 0:
            with torch.no_grad():
                attn = model.blocks[attn_layer].attn
                x_last = model.blocks[attn_layer].n1(model.embed(idx[:, -1:]))
                hd = attn.head_dim
                q = attn.q_proj(x_last).view(1, 1, attn.n_head, hd).transpose(1, 2)
                q = apply_rope(q, model.cos, model.sin, offset)
                k = past[attn_layer][0]
                if attn.n_kv_head != attn.n_head:
                    k = k.repeat_interleave(attn.n_head // attn.n_kv_head, dim=1)
                a = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
                a = torch.softmax(a, dim=-1)[0].mean(dim=0)[-1]  # 平均头、最后一行
                attn_pack = [round(float(v), 4) for v in a[-24:]]
        on_token(out_text, nxt_id, attn_pack)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return out_text
