"""梯度/ΔW/权重 地图张量与色阶。"""
from __future__ import annotations

import threading

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


class GradCache:
    """训练线程每 tick 把梯度池化成小网格缓存，HTTP 线程只读，避免与反传竞争。"""

    def __init__(self, tiles: int = 128):
        self.tiles = int(tiles)
        self._lock = threading.Lock()
        self._grids: dict[str, torch.Tensor] = {}
        self._stats: dict[str, dict] = {}
        self._absmax = 0.0
        self.step = -1

    def capture(self, model, matrices: list[dict], step: int) -> None:
        from viz.stats import snapshot_gradients
        grads = {n: p.grad for n, p in model.named_parameters(remove_duplicate=False)}
        grids: dict[str, torch.Tensor] = {}
        stats = snapshot_gradients(model, matrices)
        amax = 0.0
        for m in matrices:
            name = m["name"]
            g = grads.get(name)
            grids[name] = (pool_map(g, self.tiles) if g is not None
                           else torch.zeros(1, 1))
            amax = max(amax, stats[name]["grad_absmax"])
        with self._lock:
            self._grids = {k: v.cpu() for k, v in grids.items()}
            self._stats = stats
            self._absmax = amax
            self.step = int(step)

    def grid(self, name: str):
        with self._lock:
            t = self._grids.get(name)
        return t.tolist() if t is not None else None

    def stats(self, name: str):
        with self._lock:
            s = self._stats.get(name)
        return dict(s) if s is not None else None

    def all_stats(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._stats.items()}

    @property
    def absmax(self) -> float:
        with self._lock:
            return self._absmax
