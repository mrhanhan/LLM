"""自回归采样生成（带 KV 缓存）：temperature / top-k / top-p / 重复惩罚。"""
from __future__ import annotations

import torch


def sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_k=None,
                  top_p: float | None = None, generator=None) -> torch.Tensor:
    """从一维 logits 中采样一个 token。temperature<=0 时退化为贪心。"""
    if temperature <= 0:
        return torch.argmax(logits).unsqueeze(0)

    logits = logits / max(temperature, 1e-8)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < vals[-1], float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # 保留累积概率达到 top_p 的最小集合
        cutoff = cum > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _apply_repetition_penalty(logits, prev_ids, penalty):
    if penalty and penalty != 1.0 and prev_ids:
        for tid in set(int(i) for i in prev_ids):
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_k=None, top_p: float | None = 0.9,
             repetition_penalty: float = 1.0, device: str = "cuda",
             generator=None) -> str:
    """给定提示词，自回归生成文本。使用 KV 缓存避免重复计算。"""
    model.eval()
    model.to(device)
    ids = tokenizer.encode(prompt, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.special_id("eos")
    generated = []

    past = None
    for _ in range(max_new_tokens):
        cur = idx if past is None else idx[:, -1:]
        logits, _, past = model(cur, past_kvs=past, use_cache=True)
        step_logits = logits[0, -1].float().clone()
        _apply_repetition_penalty(step_logits, ids + generated, repetition_penalty)
        nxt = sample_logits(step_logits, temperature, top_k, top_p, generator)
        tid = int(nxt.item())
        if tid == eos_id:
            break
        generated.append(tid)
        idx = torch.cat([idx, nxt.view(1, 1).to(device)], dim=1)

    return tokenizer.decode(generated)
