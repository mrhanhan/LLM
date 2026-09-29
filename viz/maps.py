"""梯度/ΔW/权重 地图张量与色阶。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SEQUENTIAL_LO = (0.10, 0.20, 0.55)
SEQUENTIAL_HI = (0.98, 0.62, 0.10)


def sequential_rgb(v: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, float(v)))
    return tuple(lo + (hi - lo) * t
                 for lo, hi in zip(SEQUENTIAL_LO, SEQUENTIAL_HI))  # type: ignore[return-value]


def pool_map(t: torch.Tensor, tiles: int) -> torch.Tensor:
    x = t.detach().float()
    if x.dim() == 1:
        x = x.reshape(1, -1)
    o, i = x.shape
    h, k = max(1, min(o, int(tiles))), max(1, min(i, int(tiles)))
    if (h, k) == (o, i):
        return x
    return F.adaptive_avg_pool2d(x[None, None], (h, k))[0, 0]


def matrix_map(model, name: str, mode: str,
               prev: torch.Tensor | None = None) -> torch.Tensor:
    params = {n: p for n, p in model.named_parameters(remove_duplicate=False)}
    p = params[name].detach().float()
    if mode == "weight":
        return p.reshape(1, -1) if p.dim() == 1 else p
    if mode == "delta":
        if prev is None:
            return torch.zeros_like(p)
        return p - prev.reshape(p.shape).to(p.device).float()
    raise ValueError(f"matrix_map 不支持 mode={mode!r}（grad 由缓存提供）")
