"""VLM 推理：图片描述与 VQA（复用训练 prompt 模板 + KV 缓存）。

教学注释：推理时的输入构造必须和训练时**完全一致**——同样是
<bos> <image>*N ...，否则模型看到的分布变了，输出会退化。
"""
from __future__ import annotations

import torch

from src.generate import sample_logits


@torch.no_grad()
def _decode(model, tokenizer, first_embeds, max_new_tokens,
            temperature, top_k, top_p, device):
    """从给定的首步 embedding 起，自回归解码。"""
    eos_id = tokenizer.special_id("eos")
    past = None
    # 第一步：整段 prompt（含视觉特征）作为输入
    logits, _, past = model.gpt(inputs_embeds=first_embeds.to(device), use_cache=True)
    out = []
    next_logits = logits[0, -1].float()
    for _ in range(max_new_tokens):
        tid = int(sample_logits(next_logits, temperature, top_k, top_p).item())
        if tid == eos_id:
            break
        out.append(tid)
        tok_t = torch.tensor([[tid]], device=device, dtype=torch.long)
        logits, _, past = model.gpt(inputs_embeds=model.gpt.embed(tok_t), past_kvs=past, use_cache=True)
        next_logits = logits[0, -1].float()
    return tokenizer.decode(out)


@torch.no_grad()
def generate_caption(model, tokenizer, image_tensor, max_new_tokens: int = 64,
                     temperature: float = 0.0, top_k=None, top_p: float | None = 0.9,
                     device: str = "cuda") -> str:
    """图片 -> 中文描述。"""
    model.eval().to(device)
    n_img = model.vision.num_patches
    prompt_ids = [tokenizer.special_id("bos")] + [model.image_token_id] * n_img
    input_ids = torch.tensor([prompt_ids], device=device)
    pix = image_tensor.unsqueeze(0).to(device)
    embeds = model.replace_image_embeds(input_ids, pix)
    return _decode(model, tokenizer, embeds, max_new_tokens,
                   temperature, top_k, top_p, device)


@torch.no_grad()
def answer_question(model, tokenizer, image_tensor, question: str, max_new_tokens: int = 32,
                    temperature: float = 0.0, top_k=None, top_p: float | None = 0.9,
                    device: str = "cuda") -> str:
    """VQA：图片 + 问题 -> 答案。"""
    model.eval().to(device)
    n_img = model.vision.num_patches
    prompt_ids = ([tokenizer.special_id("bos")] + [model.image_token_id] * n_img
                  + tokenizer.encode(question + "答："))
    input_ids = torch.tensor([prompt_ids], device=device)
    pix = image_tensor.unsqueeze(0).to(device)
    embeds = model.replace_image_embeds(input_ids, pix)
    return _decode(model, tokenizer, embeds, max_new_tokens,
                   temperature, top_k, top_p, device)
