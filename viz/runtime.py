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
                 model: GPT | None = None, data: torch.Tensor | None = None,
                 weight_decay: float = 0.01, grad_clip: float = 1.0):
        self.tok = tokenizer
        self.device = device
        self.on_step = on_step
        self.lr = lr
        self.batch = batch_size
        self.block = block_size
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.model: GPT | None = model
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.step = 0
        self.loss = float("nan")
        if data is None:
            data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
        self.data = data.to(device)
        self.optim: torch.optim.Optimizer | None = None

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = build_tiny(self.tok.vocab_size, self.device)
        if self.optim is None:
            tcfg = TrainConfig(lr=self.lr, weight_decay=self.weight_decay,
                               grad_clip=self.grad_clip)
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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


def _pool2d(mat: torch.Tensor, max_rows: int, max_cols: int) -> dict:
    if mat.numel() == 0:
        return {"rows": 0, "cols": 0, "values": []}
    r, c = mat.shape
    pr = max(1, min(r, int(max_rows)))
    pc = max(1, min(c, int(max_cols)))
    m = mat.float().reshape(1, 1, r, c)
    m = F.adaptive_avg_pool2d(m, (pr, pc))[0, 0]
    return {"rows": pr, "cols": pc,
            "values": [[round(float(x), 4) for x in row] for row in m.tolist()]}


def _kv_snapshot(past, max_rows: int = 12, max_cols: int = 24) -> dict:
    """把逐层 KV cache 压成小网格，供前端热力图；同时统计长度与字节数。"""
    layers = []
    total_len = 0
    total_bytes = 0
    for i, (k, v) in enumerate(past):
        _, H, T, hd = k.shape
        total_len = max(total_len, int(T))
        nbytes = int((k.numel() + v.numel()) * k.element_size())
        total_bytes += nbytes
        kk = k.reshape(H, T, hd).permute(1, 0, 2).reshape(T, H * hd)
        vv = v.reshape(H, T, hd).permute(1, 0, 2).reshape(T, H * hd)
        layers.append({
            "layer": i, "len": int(T), "bytes": nbytes,
            "heads": int(H), "head_dim": int(hd),
            "k": _pool2d(kk, max_rows, max_cols),
            "v": _pool2d(vv, max_rows, max_cols),
        })
    return {"layers": layers, "total_len": total_len, "total_bytes": total_bytes,
            "n_layer": len(past)}


def _topn(probs: torch.Tensor, tok, n: int) -> list[dict]:
    p = probs[0]
    k = max(1, min(int(n), int(p.numel())))
    vals, idxs = torch.topk(p, k)
    out = []
    for val, idx in zip(vals.tolist(), idxs.tolist()):
        out.append({"id": int(idx), "token": tok.decode([int(idx)]),
                    "prob": round(float(val), 4)})
    return out


@torch.no_grad()
def stream_generate(model: GPT, tok: CharTokenizer, prompt: str, on_token: Callable,
                    max_new_tokens: int = 40, temperature: float = 0.9,
                    top_k: int = 20, top_p: float = 0.95, top_n: int = 8,
                    attn_layer: int = 0, seed=None):
    """自回归生成，每生成一个 token 调一次 on_token。

    on_token(text, token_id, attn_pack, topn, kv)：topn 为该步 topN
    候选词（id/词/概率），kv 为逐层 KV cache 快照。
    """
    if seed is not None:
        try:
            torch.manual_seed(int(seed))
        except (TypeError, ValueError):
            pass
    model.eval()
    ids = tok.encode(prompt) or [0]
    device = next(model.parameters()).device
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    out_text = prompt
    for _ in range(max_new_tokens):
        offset = past[0][0].shape[-2] if past is not None else 0
        cur = idx[:, offset:] if past is not None else idx
        logits, _, new_kvs = model(cur, past_kvs=past, use_cache=True)
        past = new_kvs
        logits = logits[:, -1, :] / max(float(temperature), 1e-6)
        if top_k and int(top_k) > 0:
            v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        if top_p and 0.0 < float(top_p) < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = (cum - sorted_probs) <= float(top_p)
            keep[..., 0] = True
            sorted_probs = sorted_probs * keep
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topn = _topn(probs, tok, top_n)
        nxt = torch.multinomial(probs, 1)
        nxt_id = int(nxt.item())
        out_text += tok.decode([nxt_id])

        attn_pack = None
        if past is not None:
            attn = model.blocks[attn_layer].attn
            x_last = model.blocks[attn_layer].n1(model.embed(idx[:, -1:]))
            hd = attn.head_dim
            q = attn.q_proj(x_last).view(1, 1, attn.n_head, hd).transpose(1, 2)
            q = apply_rope(q, model.cos, model.sin, offset)
            k = past[attn_layer][0]
            if attn.n_kv_head != attn.n_head:
                k = k.repeat_interleave(attn.n_head // attn.n_kv_head, dim=1)
            a = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            a = torch.softmax(a, dim=-1)[0].mean(dim=0)[-1]
            attn_pack = [round(float(x), 4) for x in a[-24:]]

        kv = _kv_snapshot(past) if past is not None else None
        on_token(out_text, nxt_id, attn_pack, topn, kv)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return out_text
