"""训练器：bf16 混合精度 + 梯度累积 + 裁剪 + 学习率调度 + 断点续训 + 日志/曲线。"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.config import TrainConfig
from src.utils import JsonlLogger, get_lr, gpu_mem_str, human_params, plot_loss


def _autocast_ctx(train_cfg: TrainConfig, device: str):
    """bf16 训练不需要 GradScaler，直接 autocast 即可。

    教学注释：是否启用 autocast 必须看"实际选用的设备"，而不是机器上是否装了 CUDA。
    否则在 CUDA 主机上强制 device="cpu" 时，会对 CPU 张量请求 autocast("cuda") 而报错。
    """
    if train_cfg.dtype == "bf16" and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class Trainer:
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, train_data, val_data,
                 tokenizer=None, device: str | None = None, ctx_len: int | None = None):
        self.cfg = train_cfg
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 教学注释：上下文长度不强制依赖 model.cfg（如 MiniVLM 没有 .cfg）。
        # 优先用显式入参，其次回退到 model.cfg.ctx_len，都没有则为 None。
        self.ctx_len = ctx_len if ctx_len is not None else getattr(
            getattr(model, "cfg", None), "ctx_len", None)
        self.model.to(self.device)
        self.opt = model.configure_optimizers(train_cfg)
        self.raw_model = model  # 保存未 compile 的原始模型引用

        if train_cfg.compile:
            # 教学注释：torch.compile 在 Windows 上需要 triton，可能失败；失败自动回退。
            # 注意 torch.compile 是"惰性编译"：真正编译发生在第一次前向，所以这里先拿一个
            # 极小 dummy batch 做一次 warmup 前向，把编译错误提前暴露，才能被 try 捕获并回退。
            try:
                compiled = torch.compile(model)
                with torch.no_grad():
                    n = min(4, self.ctx_len) if self.ctx_len else 4
                    dummy = torch.randint(0, 2, (1, n), device=self.device)
                    compiled(dummy, targets=dummy)
                self.model = compiled
                print("[compile] torch.compile 已启用")
            except Exception as e:  # noqa: BLE001
                print(f"[compile] 失败，回退 eager：{e}")
                self.model = model

        self.out_dir = Path(train_cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = JsonlLogger(str(self.out_dir / "metrics.jsonl"))
        self.start_step = 0

    def _forward_loss(self, batch):
        """batch 是 _next_batch() 的返回值：纯文本为 (x, y)，多模态为 collate 后的字典。"""
        x, y = batch
        with _autocast_ctx(self.cfg, self.device):
            _, loss, _ = self.model(x, targets=y)
        return loss

    @torch.no_grad()
    def _estimate_val(self) -> float:
        self.model.eval()
        losses = []
        for _ in range(self.cfg.eval_iters):
            losses.append(self._forward_loss(self._get_batch(self.val_data)).item())
        self.model.train()
        return sum(losses) / len(losses)

    def _get_batch(self, data):
        from src.data import get_batch
        return get_batch(data, self.cfg.batch_size, self.ctx_len, self.device)

    def _next_batch(self):
        """取一个训练 batch（子类可覆写以支持多模态等不同数据形态）。"""
        return self._get_batch(self.train_data)

    def train(self):
        cfg = self.cfg
        self.model.train()
        # 教学注释：先同步 _current_step，保证 max_steps<=start_step（已训完）时
        # 循环不执行、末尾 save() 也能记录正确的步数，而不是退回 0。
        self._current_step = self.start_step
        t0 = time.time()
        for step in range(self.start_step, cfg.max_steps):
            self._current_step = step + 1
            lr = get_lr(step, cfg)
            for g in self.opt.param_groups:
                g["lr"] = lr

            # 梯度累积：多个 micro-batch 的梯度平均后再更新一次
            self.opt.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum):
                batch = self._next_batch()
                loss = self._forward_loss(batch) / cfg.grad_accum
                loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()

            if step % cfg.log_interval == 0:
                dt = time.time() - t0
                print(f"step {step:>6}/{cfg.max_steps} | loss {loss.item() * cfg.grad_accum:.4f} "
                      f"| lr {lr:.2e} | {dt:.1f}s | mem {gpu_mem_str()}")
                self.logger.log({"step": step, "split": "train",
                                 "loss": loss.item() * cfg.grad_accum, "lr": lr})
            if step > 0 and step % cfg.eval_interval == 0:
                v = self._estimate_val()
                print(f"  [eval] step {step} val_loss {v:.4f}")
                self.logger.log({"step": step, "val_loss": v})
            if step > 0 and step % cfg.save_interval == 0:
                self.save(str(self.out_dir / "latest.pt"))
                plot_loss(str(self.out_dir / "metrics.jsonl"), str(self.out_dir / "loss.png"))

        self.save(str(self.out_dir / "latest.pt"))
        plot_loss(str(self.out_dir / "metrics.jsonl"), str(self.out_dir / "loss.png"))
        print(f"训练完成，总耗时 {time.time() - t0:.1f}s，参数量 {human_params(self.raw_model.num_params())}")

    def save(self, path: str):
        ckpt = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "step": getattr(self, "_current_step", 0),
            "cfg": getattr(self.raw_model, "cfg", None),
            "train_cfg": self.cfg,
        }
        torch.save(ckpt, path)

    def load(self, path: str, resume: bool = True) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model"])
        if resume:
            try:
                self.opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:  # noqa: BLE001
                print(f"优化器状态恢复失败（可忽略）：{e}")
            self.start_step = int(ckpt.get("step", 0))
        # 教学注释：把已训练步数同步到 _current_step，避免未进入 train() 循环的
        # save() 把 checkpoint 步数写回 0（否则续训会从头开始）。
        self._current_step = self.start_step
        return self.start_step


class VLMTrainer(Trainer):
    """多模态训练器：批数据由 collate_vlm 生成，loss 在图像位置上被屏蔽。

    教学注释：它复用 Trainer 的优化/日志/续训逻辑，只覆写"如何取一批数据"
    和"如何算 loss"这两个接缝，这就是把训练循环和数据形态解耦的好处。
    """

    def __init__(self, model, train_cfg, dataset, tokenizer, num_image_tokens,
                 device: str | None = None, use_qa: bool = False):
        from torch.utils.data import DataLoader

        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.use_qa = use_qa
        self.loader = DataLoader(
            dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True,
            collate_fn=self._collate, num_workers=0,
        )
        self._iter = None
        super().__init__(model, train_cfg, None, None, tokenizer=tokenizer, device=device)

    def _collate(self, batch):
        from src.data import collate_vlm
        for b in batch:
            b["use_qa"] = self.use_qa
        return collate_vlm(batch, self.tokenizer, self.num_image_tokens)

    @torch.no_grad()
    def _estimate_val(self) -> float:
        # 简化：用训练分布上的若干 batch 估计，避免额外验证集管线
        self.model.eval()
        losses = []
        for _ in range(max(1, min(5, self.cfg.eval_iters))):
            b = self._next_batch()
            with _autocast_ctx(self.cfg, self.device):
                _, loss, _ = self.model(b["input_ids"].to(self.device),
                                        b["pixel_values"].to(self.device),
                                        b["labels"].to(self.device))
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def _next_batch(self):
        """从 DataLoader 取一批（自动循环 epoch）。"""
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            b = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            b = next(self._iter)
        return b

    def _forward_loss(self, batch):
        """基类以单个 batch 对象调用；这里 batch 是 collate 后的字典。"""
        _, loss, _ = self.model(batch["input_ids"].to(self.device),
                                batch["pixel_values"].to(self.device),
                                batch["labels"].to(self.device))
        return loss
