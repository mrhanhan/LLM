"""统一模型图：把 GPT 拆成 layers / modules / matrices / connections 供 3D 前端使用。"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

CUBE_THRESHOLD = 4096

# 每个 block 内的矩阵（module 相对路径 -> 中文标签、role）
_BLOCK_MATRICES = [
    ("n1.weight", "Norm1", "norm"),
    ("attn.q_proj.weight", "Q 投影", "attn"),
    ("attn.k_proj.weight", "K 投影", "attn"),
    ("attn.v_proj.weight", "V 投影", "attn"),
    ("attn.o_proj.weight", "O 投影", "attn"),
    ("n2.weight", "Norm2", "norm"),
    ("mlp.w1.weight", "W1 (gate)", "mlp"),
    ("mlp.w2.weight", "W2 (up)", "mlp"),
    ("mlp.w3.weight", "W3 (down)", "mlp"),
]
_HEAD_MATRICES = [("norm_f.weight", "最终 Norm", "norm"), ("lm_head.weight", "输出头", "head")]


def _params(model: nn.Module) -> dict[str, Any]:
    return {n: p for n, p in model.named_parameters(remove_duplicate=False)}


def _shape(p) -> list[int]:
    s = list(p.shape)
    return [1, s[0]] if len(s) == 1 else s


def _matrix(name: str, layer: str, module: str, label: str, role: str, p) -> dict:
    shape = _shape(p)
    n = shape[0] * shape[1]
    return {"name": name, "label": label, "role": role, "layer": layer,
            "module": module, "shape": shape, "ndim": p.dim(),
            "elements": n, "small": n <= CUBE_THRESHOLD}


def build_graph(model: nn.Module, source: str = "live") -> dict:
    params = _params(model)
    cfg = model.cfg
    n_layer = len(model.blocks)
    layers = [{"id": "emb", "index": -1, "label": "词嵌入", "kind": "emb"}]
    matrices: list[dict] = []
    if "tok_emb.weight" in params:
        matrices.append(_matrix("tok_emb.weight", "emb", "emb", "词嵌入", "embedding",
                                params["tok_emb.weight"]))
    for i in range(n_layer):
        layers.append({"id": f"L{i}", "index": i, "label": f"层 {i}", "kind": "block"})
        for rel, label, role in _BLOCK_MATRICES:
            name = f"blocks.{i}.{rel}"
            if name in params:
                matrices.append(_matrix(name, f"L{i}", rel, label, role, params[name]))
    layers.append({"id": "final", "index": n_layer, "label": "最终 Norm + 输出头", "kind": "final"})
    for rel, label, role in _HEAD_MATRICES:
        if rel in params:
            matrices.append(_matrix(rel, "final", rel, label, role, params[rel]))

    # 连线
    connections: list[dict] = []
    rep = lambda i: f"blocks.{i}.mlp.w3.weight"          # 该层代表矩阵

    def edge(src, dst, kind, weight_from):
        connections.append({"id": f"{kind}::{src}->{dst}", "src": src, "dst": dst,
                            "kind": kind, "weight_from": weight_from})

    if n_layer and "tok_emb.weight" in params:
        edge("tok_emb.weight", "blocks.0.n1.weight", "spine", rep(0))
    for i in range(n_layer - 1):
        edge(rep(i), f"blocks.{i+1}.n1.weight", "spine", rep(i + 1))
    if n_layer:
        edge(rep(n_layer - 1), "norm_f.weight", "spine", "norm_f.weight")
        edge("norm_f.weight", "lm_head.weight", "spine", "lm_head.weight")
    for i in range(n_layer):
        b = f"blocks.{i}"
        e = lambda dst: edge(f"{b}.n1.weight", dst, "inner", dst)
        e(f"{b}.attn.q_proj.weight"); e(f"{b}.attn.k_proj.weight"); e(f"{b}.attn.v_proj.weight")
        for k in ("q_proj", "k_proj", "v_proj"):
            edge(f"{b}.attn.{k}.weight", f"{b}.attn.o_proj.weight", "inner",
                 f"{b}.attn.o_proj.weight")
        edge(f"{b}.attn.o_proj.weight", f"{b}.n2.weight", "inner", f"{b}.n2.weight")
        edge(f"{b}.n2.weight", f"{b}.mlp.w1.weight", "inner", f"{b}.mlp.w1.weight")
        edge(f"{b}.n2.weight", f"{b}.mlp.w2.weight", "inner", f"{b}.mlp.w2.weight")
        edge(f"{b}.mlp.w1.weight", f"{b}.mlp.w3.weight", "inner", f"{b}.mlp.w3.weight")
        edge(f"{b}.mlp.w2.weight", f"{b}.mlp.w3.weight", "inner", f"{b}.mlp.w3.weight")

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "source": source,
        "cube_threshold": CUBE_THRESHOLD,
        "model": {"name": getattr(cfg, "name", source), "d_model": cfg.d_model,
                  "n_layer": n_layer, "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head,
                  "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size, "params": n_params},
        "layers": layers, "matrices": matrices, "connections": connections,
    }
