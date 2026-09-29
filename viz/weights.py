"""真实/小模型权重数据层：分块网格、发散色阶纹理、元素级查询与统计。"""
from __future__ import annotations

import io

import torch
import torch.nn.functional as F

from src.config import load_config
from src.model import GPT


def diverging_rgb(v: float) -> tuple[float, float, float]:
    t = max(-1.0, min(1.0, float(v)))
    if t >= 0:
        return (0.16 + 0.84 * t, 0.08 + 0.16 * t, 0.08 + 0.16 * t)
    a = abs(t)
    return (0.08 + 0.16 * a, 0.16 + 0.28 * a, 0.16 + 0.84 * a)


def load_checkpoint(path: str, config_path: str):
    cfg = load_config(config_path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw)
    cfg.model.vocab_size = int(sd["tok_emb.weight"].shape[0])
    model = GPT(cfg.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[viz] 权重差异 missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, cfg


class MatrixStore:
    def __init__(self, model) -> None:
        self.model = model
        self._params = {n: p for n, p in model.named_parameters(remove_duplicate=False)}
        self._global_absmax: float | None = None

    # ---- 取值 ----
    def _weight(self, name: str) -> torch.Tensor:
        p = self._params[name].detach().float()
        return p.reshape(1, -1) if p.dim() == 1 else p

    def _grid(self, w: torch.Tensor, tiles: int) -> torch.Tensor:
        o, i = w.shape
        h, k = min(o, tiles), min(i, tiles)
        if h == o and k == i:
            return w
        return F.adaptive_avg_pool2d(w[None, None], (h, k))[0, 0]

    def grid(self, name: str, tiles: int = 64) -> dict:
        w = self._weight(name)
        g = self._grid(w, max(1, min(int(tiles), 256)))
        return {"shape": list(w.shape), "grid": list(g.shape), "values": g.tolist()}

    def stats(self, name: str) -> dict:
        w = self._weight(name)
        return {"shape": list(w.shape), "vmin": float(w.min()), "vmax": float(w.max()),
                "absmax": float(w.abs().max()), "mean": float(w.mean()),
                "std": float(w.std())}

    def global_stats(self) -> dict:
        if self._global_absmax is None:
            self._global_absmax = max((float(p.detach().float().abs().max())
                                       for p in self._params.values()), default=1.0)
        return {"absmax": self._global_absmax}

    def cell(self, name: str, i: int, j: int) -> float:
        w = self._weight(name)
        i = max(0, min(int(i), w.shape[0] - 1))
        j = max(0, min(int(j), w.shape[1] - 1))
        return float(w[i, j])

    def patch(self, name: str, r: int, c: int, h: int, w: int) -> dict:
        t = self._weight(name)
        r = max(0, min(int(r), t.shape[0] - 1)); c = max(0, min(int(c), t.shape[1] - 1))
        sub = t[r:r + int(h), c:c + int(w)]
        return {"r": r, "c": c, "h": int(sub.shape[0]), "w": int(sub.shape[1]),
                "values": sub.tolist()}

    def png(self, name: str, tiles: int = 64, norm: str = "global") -> bytes:
        from PIL import Image
        w = self._weight(name)
        g = self._grid(w, max(1, min(int(tiles), 256)))
        absmax = (self.stats(name)["absmax"] if norm == "matrix"
                  else self.global_stats()["absmax"]) or 1.0
        t = (g / absmax).clamp(-1, 1)
        rgb = torch.empty(t.shape + (3,), dtype=torch.uint8)
        pos = t >= 0
        a = t.abs()
        rgb[..., 0] = torch.where(pos, 0.16 + 0.84 * a, 0.08 + 0.16 * a).mul(255).to(torch.uint8)
        rgb[..., 1] = torch.where(pos, 0.08 + 0.16 * a, 0.16 + 0.28 * a).mul(255).to(torch.uint8)
        rgb[..., 2] = torch.where(pos, 0.08 + 0.16 * a, 0.16 + 0.84 * a).mul(255).to(torch.uint8)
        img = Image.fromarray(rgb.numpy(), mode="RGB")
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return buf.getvalue()
