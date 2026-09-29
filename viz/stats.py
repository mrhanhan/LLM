"""可视化后端：权重统计与激活统计（按矩阵聚合）。

逻辑自 `viz/topology.py` 搬入，供 3D 前端渲染（颜色/连线粗细 = 权重强弱）。
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from src.model import RMSNorm


def _finite(x: float) -> float:
    return x if math.isfinite(x) else 0.0


# ----------------------------------------------------------------------------
# 权重统计
# ----------------------------------------------------------------------------
def summarize(tensors: Iterable[torch.Tensor]) -> dict:
    """对一组参数张量做聚合统计：L2 范数、平均绝对值、最大绝对值、元素数。"""
    total_sq = 0.0
    abs_sum = 0.0
    n = 0
    mx = 0.0
    for t in tensors:
        tf = t.detach().float()
        total_sq += float(tf.pow(2).sum())
        abs_sum += float(tf.abs().sum())
        n += tf.numel()
        if tf.numel():
            mx = max(mx, float(tf.abs().max()))
    return {
        "norm": _finite(total_sq ** 0.5),
        "mean": _finite(abs_sum / max(n, 1)),
        "max": _finite(mx),
        "n": n,
    }


class WeightTracker:
    """记录上一份权重快照，用来衡量"这一步权重改了多少"（相对更新幅度）。

    大模型（>800 万参数）自动跳过，避免翻倍显存。
    """

    MAX_PARAMS = 8_000_000

    def __init__(self) -> None:
        self.prev: dict[str, torch.Tensor] = {}
        self.enabled = True

    def capture(self, model: nn.Module) -> None:
        total = sum(p.numel() for p in model.parameters())
        if total > self.MAX_PARAMS:
            self.enabled = False
            return
        self.prev = {k: v.detach().float().clone().reshape(-1)
                     for k, v in model.named_parameters()}

    def delta(self, names: list[str], params: dict) -> float:
        """‖W - W_prev‖ / (‖W_prev‖ + eps)，对所有参数求和后取比值。"""
        if not self.enabled or not self.prev:
            return 0.0
        num = 0.0
        den = 0.0
        for name in names:
            if name not in self.prev or name not in params:
                continue
            cur = params[name].detach().float().reshape(-1)
            old = self.prev[name]
            if cur.shape != old.shape:
                continue
            num += float((cur - old).pow(2).sum())
            den += float(old.pow(2).sum())
        return _finite((num ** 0.5) / (den ** 0.5 + 1e-9))


# ----------------------------------------------------------------------------
# 激活统计：forward hook 读每个模块输出的平均绝对值
# ----------------------------------------------------------------------------
class ActivationRecorder:
    """注册 forward hook，记录每个模块输出的平均绝对值（推理时给节点上色）。"""

    def __init__(self) -> None:
        self.acts: dict[str, float] = {}
        self._handles: list = []
        self._last_inputs: dict[str, torch.Tensor] = {}

    def attach(self, model: nn.Module) -> None:
        types = (nn.Linear, nn.Embedding, RMSNorm)
        for name, mod in model.named_modules():
            if isinstance(mod, types):
                self._handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def fn(module, inputs, output):
            with torch.no_grad():
                t = output.detach().float()
                self.acts[name] = float(t.abs().mean())
                if inputs and torch.is_tensor(inputs[0]):
                    self._last_inputs[name] = inputs[0].detach()
                else:
                    self._last_inputs[name] = None
        return fn

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.acts.clear()
        self._last_inputs.clear()

    def for_matrix(self, name: str) -> float:
        mod = name.rsplit(".", 1)[0] if "." in name else name
        return _finite(self.acts.get(mod, 0.0))


def snapshot_matrices(model: nn.Module, matrices: list[dict],
                      tracker: "WeightTracker | None" = None) -> dict:
    """按矩阵全名聚合当前权重统计（以及相对上一步的更新幅度）。"""
    params = {n: p for n, p in model.named_parameters(remove_duplicate=False)}
    out: dict[str, dict] = {}
    for m in matrices:
        name = m["name"]
        p = params.get(name)
        s = summarize([p]) if p is not None else {"norm": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
        if tracker is not None:
            s["delta"] = tracker.delta([name], params)
        out[name] = s
    return out
