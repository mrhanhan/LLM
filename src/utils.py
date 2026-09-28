"""通用工具：随机种子、学习率调度、参数量格式化、日志、loss 曲线。"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch

from src.config import TrainConfig


def set_seed(seed: int) -> None:
    """固定 python / numpy / torch 随机种子，保证可复现。"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, cfg: TrainConfig) -> float:
    """warmup 线性上升 + 余弦退火到 min_lr。"""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * ratio))


def human_params(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


def gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    used = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{used:.2f}/{total:.1f}GB"


class JsonlLogger:
    """把每步指标追加写入 metrics.jsonl，便于之后画曲线。"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def plot_loss(jsonl_path: str, out_png: str) -> None:
    """读取 metrics.jsonl，画出 train/val loss 曲线（教学：观察是否收敛/过拟合）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses, val_steps, val_losses = [], [], [], []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        # 教学注释：训练中途崩溃会留下半行 JSON，画曲线时应跳过坏行而不是让 train() 抛异常。
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if "loss" in rec and rec.get("split") == "train":
            steps.append(rec["step"])
            losses.append(rec["loss"])
        if "val_loss" in rec:
            val_steps.append(rec["step"])
            val_losses.append(rec["val_loss"])

    plt.figure(figsize=(8, 5))
    if steps:
        plt.plot(steps, losses, label="train loss")
    if val_steps:
        plt.plot(val_steps, val_losses, label="val loss", marker="o", ms=3)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
