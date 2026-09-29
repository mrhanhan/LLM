"""训练引擎：simple（轻量可中断）与 hifi（复用 src.trainer）。"""
from __future__ import annotations

import threading
import time
from typing import Callable

import torch

from src.config import TrainConfig
from src.trainer import SFTTrainer, Trainer
from src.utils import get_lr


class SimpleEngine:
    """朴素训练循环：fp32、可随时停、逐 step 回调。batch_fn() -> (inputs, targets)。"""

    def __init__(self, model, batch_fn: Callable, device: str, on_step: Callable,
                 *, lr: float = 3e-4, max_steps: int = 200, grad_accum: int = 1,
                 grad_clip: float = 1.0, weight_decay: float = 0.05,
                 warmup_steps: int = 0, min_lr: float | None = None,
                 log_interval: int = 5):
        self.model = model
        self.batch_fn = batch_fn
        self.device = device
        self.on_step = on_step
        self.cfg = TrainConfig(
            lr=lr, min_lr=min_lr if min_lr is not None else lr * 0.1,
            warmup_steps=warmup_steps, max_steps=max_steps, grad_accum=grad_accum,
            grad_clip=grad_clip, weight_decay=weight_decay, log_interval=log_interval)
        self.opt = model.configure_optimizers(self.cfg)
        self.step = 0
        self.loss = float("nan")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        self.model.train()
        cfg = self.cfg
        for step in range(cfg.max_steps):
            if self._stop.is_set():
                break
            lr = get_lr(step, cfg)
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(cfg.grad_accum):
                x, y = self.batch_fn()
                _, loss, _ = self.model(x, targets=y)
                total += float(loss.detach())
                (loss / cfg.grad_accum).backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self.step = step + 1
            self.loss = total / cfg.grad_accum
            if self.step % cfg.log_interval == 0 or self.step == 1:
                self.on_step(self.step, self.loss, lr)
            time.sleep(0)
        self.model.eval()

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(), "step": self.step,
                    "cfg": getattr(self.model, "cfg", None)}, path)


class _VizLoop:
    """给 src.Trainer/SFTTrainer 注入 on_step 回调与可中断循环。"""

    on_step: Callable
    _stop: threading.Event

    def viz_run(self) -> None:
        cfg = self.cfg
        self.model.train()
        self._current_step = getattr(self, "start_step", 0)
        for step in range(self._current_step, cfg.max_steps):
            if self._stop.is_set():
                break
            lr = get_lr(step, cfg)
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(cfg.grad_accum):
                loss = self._forward_loss(self._next_batch())
                total += float(loss.detach())
                (loss / cfg.grad_accum).backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self._current_step = step + 1
            self.on_step(self._current_step, total / cfg.grad_accum, lr)
            time.sleep(0)
        self.model.eval()

    def _launch(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self.viz_run, daemon=True)
        self._thread.start()

    def _halt(self) -> None:
        self._stop.set()
        if getattr(self, "_thread", None):
            self._thread.join(timeout=10)


class HifiTrainer(_VizLoop, Trainer):
    def __init__(self, model, train_cfg, train_data, *, tokenizer=None,
                 device=None, ctx_len=None, on_step=None):
        self.on_step = on_step or (lambda *a: None)
        self._stop = threading.Event()
        self._thread = None
        super().__init__(model, train_cfg, train_data, None,
                         tokenizer=tokenizer, device=device, ctx_len=ctx_len)

    def start(self) -> None:
        self._launch()

    def stop(self) -> None:
        self._halt()


class HifiSFTTrainer(_VizLoop, SFTTrainer):
    def __init__(self, model, train_cfg, train_ds, tokenizer, *, max_len=256,
                 val_ds=None, device=None, on_step=None):
        self.on_step = on_step or (lambda *a: None)
        self._stop = threading.Event()
        self._thread = None
        super().__init__(model, train_cfg, train_ds, tokenizer,
                         max_len=max_len, val_ds=val_ds, device=device)

    def start(self) -> None:
        self._launch()

    def stop(self) -> None:
        self._halt()
