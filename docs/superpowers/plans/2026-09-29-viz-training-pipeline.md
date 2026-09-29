# viz 训练配置与实时刷新 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `viz` 左栏可选「模型 / 训练方式(pretrain|sft) / 数据集 / 超参(步数/学习率/batch/梯度累积) / 高保真开关」并在训练中让 3D 板材与右侧抽屉实时反映权重变化。

**Architecture:** 后端新增 `viz/datasets.py`（写死数据集注册表 + 批次提供者）与 `viz/train_engine.py`（`simple` 轻量循环、`hifi` 复用 `src.trainer` 子类）；`viz/server.py` 的 `State` 按 `(target, mode, dataset, engine, params)` 组装训练并保存到 `out/viz/`。前端在左栏加控件，`tick` 驱动分级节流刷新（3D 每 tick；抽屉 ~500ms；展开层纹理每 20 tick）。

**Tech Stack:** Python 3.12 · FastAPI/uvicorn · PyTorch · numpy · Three.js(vendored) · pytest

**Spec:** `docs/superpowers/specs/2026-09-29-viz-training-pipeline-design.md`

## Global Constraints

- 运行环境：Windows，shell 为 pwsh；Python 用 `& ".venv\Scripts\python.exe"`；`node` 可直接调用。
- 分词器统一 **Qwen**（`data/tokenizer/qwen2.5-0.5b`）；live 也用 Qwen 词表（`vocab_size = QwenTokenizer.vocab_size`）。
- 数据集**写死注册表** `viz/datasets.py`，`kind ∈ {"pretrain","sft"}`。
- 超参只暴露 `max_steps`、`lr`、`batch_size`、`grad_accum`（+ 左栏高保真勾选）。
- 保存路径：`out/viz/<target>-<mode>-<dataset>-<YYYYmmdd-HHMMSS>.pt`（`out/` 已在 .gitignore）。
- `tie_embeddings=True` 时 `lm_head.weight` 与 `tok_emb.weight` 共享张量；遍历参数一律用 `model.named_parameters(remove_duplicate=False)`。
- 矩阵名一律带 `.weight`；发散色阶 Python/JS 必须一致（沿用现状）。
- 提交信息用中文，风格 `feat:` / `fix:` / `docs:`；每个任务结束跑对应 pytest，全绿再进入下一个。
- 不加注释除非必要；不改动 `src/` 训练语义（hifi 只做子类扩展）。

---

## File Structure

**创建**
- `viz/datasets.py` — 数据集注册表、`list_datasets/resolve_spec/load_pretrain`、`PretrainBatches/SFTBatches`
- `viz/train_engine.py` — `SimpleEngine`、`HifiTrainer`、`HifiSFTTrainer`
- `tests/test_viz_datasets.py` — 注册表与批次测试
- `tests/test_viz_train_engine.py` — 两套引擎测试

**修改**
- `viz/server.py` — `State.start_train(target, mode, dataset, engine, params)`；`/api/datasets`；`tick` 增字段；`out/viz` 保存；live 改用 Qwen 词表；`smoke()` 不再依赖字符分词器
- `viz/runtime.py` — 删除 `LiveTrainer`（迁至 `train_engine.py`），保留 `CharTokenizer/CORPUS/stream_generate/build_tiny`
- `viz/static/index.html`、`viz/static/style.css` — 左栏四段式
- `viz/static/js/main.js` — 取数据集、组装 payload、训练中禁用输入、分级节流刷新
- `viz/static/js/panel.js` — `setLive()` + `refreshLive()`
- `viz/static/js/shelf.js` — `refreshTextures()`
- `viz/static/js/matrix.js` — `refreshMatrixTexture()`
- `tests/test_viz.py` — 适配 `smoke()`；新增 `resolve_spec` 测试
- `tests/test_viz_frontend.py` — 新增 index.html 控件断言
- `docs/07-visualization.md` — 训练配置与实时刷新说明

---

## Task 1: 数据集注册表 `viz/datasets.py`

**Files:**
- Create: `viz/datasets.py`
- Test: `tests/test_viz_datasets.py`

**Interfaces:**
- Consumes: `viz.runtime.CORPUS`
- Produces:
  - `DATASETS: dict[str, dict]`（`id -> {id,label,kind,tokenizer,path?,builtin?,note?}`）
  - `list_datasets() -> list[dict]`（每项含 `available: bool`）
  - `resolve_spec(dataset_id: str, mode: str) -> dict`（kind 不匹配抛 `ValueError`；未知 id 抛 `KeyError`）
  - `load_pretrain(spec: dict, tok) -> np.ndarray`（`int64`，`builtin` 用 `tok.encode(CORPUS)`，否则 `np.asarray(load_bin(path), dtype=np.int64)`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_viz_datasets.py
import numpy as np
import pytest

from viz.datasets import DATASETS, list_datasets, load_pretrain, resolve_spec


class _FakeTok:
    def encode(self, text):
        return [1, 2, 3]


def test_registry_shape_and_kinds():
    rows = list_datasets()
    ids = {r["id"] for r in rows}
    assert {"corpus_qwen", "tiny_stories", "poetry", "sft_chat"} <= ids
    for r in rows:
        assert r["kind"] in ("pretrain", "sft")
        assert r["tokenizer"] == "qwen"
        assert isinstance(r["available"], bool)


def test_resolve_spec_enforces_kind():
    assert resolve_spec("poetry", "pretrain")["id"] == "poetry"
    assert resolve_spec("sft_chat", "sft")["id"] == "sft_chat"
    with pytest.raises(ValueError):
        resolve_spec("poetry", "sft")
    with pytest.raises(ValueError):
        resolve_spec("sft_chat", "pretrain")
    with pytest.raises(KeyError):
        resolve_spec("nope", "pretrain")


def test_load_pretrain_builtin_uses_tokenizer():
    arr = load_pretrain(DATASETS["corpus_qwen"], _FakeTok())
    assert arr.dtype == np.int64 and arr.ndim == 1
    assert arr.tolist()[:3] == [1, 2, 3]
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_datasets.py -q`
Expected: FAIL（`ModuleNotFoundError: viz.datasets`）

- [ ] **Step 3: 实现 `viz/datasets.py`**

```python
"""viz 训练用数据集注册表与批次提供者（写死，不扫描目录）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATASETS: dict[str, dict] = {
    "corpus_qwen": {
        "id": "corpus_qwen", "label": "内置中文语料（最快）",
        "kind": "pretrain", "tokenizer": "qwen", "builtin": True,
        "note": "viz.runtime.CORPUS 用 Qwen 分词",
    },
    "tiny_stories": {
        "id": "tiny_stories", "label": "TinyStories",
        "kind": "pretrain", "tokenizer": "qwen",
        "path": "data/processed/text/train.bin",
    },
    "poetry": {
        "id": "poetry", "label": "中文诗词",
        "kind": "pretrain", "tokenizer": "qwen",
        "path": "data/processed/text/poetry.train.bin",
    },
    "advertise": {
        "id": "advertise", "label": "AdvertiseGen",
        "kind": "pretrain", "tokenizer": "qwen",
        "path": "data/processed/text/advertise.train.bin",
    },
    "fineweb_cmn": {
        "id": "fineweb_cmn", "label": "FineWeb 中文（大文件）",
        "kind": "pretrain", "tokenizer": "qwen",
        "path": "data/processed/text/fineweb_cmn.train.bin",
    },
    "sft_chat": {
        "id": "sft_chat", "label": "Alpaca+Firefly 多轮",
        "kind": "sft", "tokenizer": "qwen",
        "path": "data/processed/sft/train.npz",
    },
    "sft_poetry": {
        "id": "sft_poetry", "label": "诗词续写",
        "kind": "sft", "tokenizer": "qwen",
        "path": "data/processed/sft/poetry.train.npz",
    },
    "sft_advertise": {
        "id": "sft_advertise", "label": "广告文案",
        "kind": "sft", "tokenizer": "qwen",
        "path": "data/processed/sft/advertise.train.npz",
    },
}


def _available(spec: dict) -> bool:
    if spec.get("builtin"):
        return True
    return Path(spec["path"]).exists()


def list_datasets() -> list[dict]:
    out = []
    for spec in DATASETS.values():
        out.append({
            "id": spec["id"], "label": spec["label"], "kind": spec["kind"],
            "tokenizer": spec["tokenizer"], "available": _available(spec),
            "note": spec.get("note", ""),
        })
    return out


def resolve_spec(dataset_id: str, mode: str) -> dict:
    spec = DATASETS[dataset_id]  # 未知 id -> KeyError
    if spec["kind"] != mode:
        raise ValueError(f"数据集 {dataset_id} 属于 {spec['kind']}，不能用于 {mode}")
    return spec


def load_pretrain(spec: dict, tok) -> np.ndarray:
    if spec.get("builtin"):
        from viz.runtime import CORPUS
        return np.asarray(tok.encode(CORPUS), dtype=np.int64)
    from src.data import load_bin
    return np.asarray(load_bin(spec["path"]), dtype=np.int64)
```

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_datasets.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add viz/datasets.py tests/test_viz_datasets.py
git commit -m "feat: viz 数据集注册表"
```

---

## Task 2: 批次提供者 `PretrainBatches` / `SFTBatches`

**Files:**
- Modify: `viz/datasets.py`
- Test: `tests/test_viz_datasets.py`

**Interfaces:**
- Produces:
  - `PretrainBatches(data: np.ndarray, batch_size: int, block: int, device: str)`；`.next() -> (LongTensor[B,T], LongTensor[B,T])`
  - `SFTBatches(npz_path: str, batch_size: int, max_len: int, pad_id: int, device: str)`；`.next() -> (LongTensor[B,L], LongTensor[B,L])`（labels 含 `-100`，pad 为 `pad_id`/`-100`）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_viz_datasets.py`）**

```python
import torch


def test_pretrain_batches_shapes():
    data = np.arange(1000, dtype=np.int64)
    b = PretrainBatches(data, batch_size=4, block=8, device="cpu")
    x, y = b.next()
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_sft_batches_masks_padding(tmp_path):
    import numpy as np
    npz = tmp_path / "toy.npz"
    np.savez(npz,
             ids=np.array([5, 6, 7, 8, 9, 10], dtype=np.int32),
             labels=np.array([-100, 6, 7, -100, 9, 10], dtype=np.int32),
             offsets=np.array([0, 3, 6], dtype=np.int64))
    b = SFTBatches(str(npz), batch_size=2, max_len=16, pad_id=0, device="cpu")
    inputs, labels = b.next()
    assert inputs.shape == labels.shape
    assert labels.dtype == torch.long and inputs.dtype == torch.long
    assert int((labels == -100).sum()) >= 2
```

并在文件顶部导入：`from viz.datasets import PretrainBatches, SFTBatches`。

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_datasets.py -q`
Expected: FAIL（`ImportError: cannot import name 'PretrainBatches'`）

- [ ] **Step 3: 实现（追加到 `viz/datasets.py`）**

```python
class PretrainBatches:
    """随机取样 (x, y=next-token) 的预训练批次。"""

    def __init__(self, data: np.ndarray, batch_size: int, block: int, device: str):
        self.data = data
        self.batch_size = int(batch_size)
        self.block = int(block)
        self.device = device

    def next(self):
        import torch
        n = len(self.data)
        hi = max(n - self.block - 1, 1)
        ix = np.random.randint(0, hi, size=self.batch_size)
        x = np.stack([self.data[i:i + self.block] for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.block] for i in ix])
        return (torch.from_numpy(x.astype("int64")).to(self.device),
                torch.from_numpy(y.astype("int64")).to(self.device))


class SFTBatches:
    """从 npz 随机取样本，按批内最大长度 padding。"""

    def __init__(self, npz_path: str, batch_size: int, max_len: int,
                 pad_id: int, device: str):
        from src.sft_data import SFTDataset
        self.ds = SFTDataset(npz_path)
        self.batch_size = int(batch_size)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.device = device

    def next(self):
        import torch
        from src.sft_data import collate_sft
        n = len(self.ds)
        idx = np.random.randint(0, max(n, 1), size=self.batch_size)
        batch = [self.ds[int(k)] for k in idx]
        out = collate_sft(batch, self.pad_id, self.max_len)
        return (out["input_ids"].to(self.device), out["labels"].to(self.device))
```

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_datasets.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add viz/datasets.py tests/test_viz_datasets.py
git commit -m "feat: viz 预训练/SFT 批次提供者"
```

---

## Task 3: 轻量引擎 `SimpleEngine`

**Files:**
- Create: `viz/train_engine.py`
- Test: `tests/test_viz_train_engine.py`

**Interfaces:**
- Consumes: `src.config.TrainConfig`、`src.utils.get_lr`
- Produces:
  - `SimpleEngine(model, batch_fn, device, on_step, *, lr=3e-4, max_steps=200, grad_accum=1, grad_clip=1.0, weight_decay=0.05, warmup_steps=0, min_lr=None, log_interval=5)`
  - `.start()` / `.stop()` / `.save(path)`；属性 `.step`、`.loss`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_viz_train_engine.py
import torch

from src.config import ModelConfig
from src.model import GPT
from viz.train_engine import SimpleEngine


def _model(vocab=64, d=32, layers=2, ctx=16):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, d_model=d, n_layer=layers, n_head=4,
                           n_kv_head=2, d_ff=64, ctx_len=ctx))


def test_simple_engine_runs_and_calls_on_step():
    model = _model()
    seen = []

    def batch_fn():
        x = torch.randint(0, 64, (4, 8))
        return x, x.clone()

    eng = SimpleEngine(model, batch_fn, "cpu", lambda s, l, lr: seen.append(s),
                       lr=3e-3, max_steps=6, grad_accum=2, log_interval=2)
    eng.start()
    for _ in range(200):
        if eng.step >= 6:
            break
        import time; time.sleep(0.02)
    eng.stop()
    assert eng.step == 6
    assert seen and seen[-1] == 6
    assert eng.loss == eng.loss  # not NaN


def test_simple_engine_stop_before_max_steps():
    model = _model()
    eng = SimpleEngine(model, lambda: (torch.zeros(2, 8, dtype=torch.long),
                                        torch.zeros(2, 8, dtype=torch.long)),
                       "cpu", lambda *a: None, max_steps=100000, log_interval=100000)
    eng.start()
    import time; time.sleep(0.05)
    eng.stop()
    assert eng.step < 100000
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_train_engine.py -q`
Expected: FAIL（`ModuleNotFoundError: viz.train_engine`）

- [ ] **Step 3: 实现**

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_train_engine.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add viz/train_engine.py tests/test_viz_train_engine.py
git commit -m "feat: viz 轻量训练引擎"
```

---

## Task 4: 高保真引擎 `HifiTrainer` / `HifiSFTTrainer`

**Files:**
- Modify: `viz/train_engine.py`
- Test: `tests/test_viz_train_engine.py`

**Interfaces:**
- Produces:
  - `HifiTrainer(model, train_cfg, train_data, *, tokenizer=None, device=None, ctx_len=None, on_step=None)`；`train_data` 为 `np.ndarray`（走 `src.data.get_batch`）
  - `HifiSFTTrainer(model, train_cfg, train_ds, tokenizer, *, max_len=256, val_ds=None, device=None, on_step=None)`
  - 两者：`.start()` / `.stop()` / `.save(path)`（`save` 继承自 `Trainer`），内部调用 `.viz_run()` 执行循环

- [ ] **Step 1: 写失败测试（追加到 `tests/test_viz_train_engine.py`）**

```python
import numpy as np
from src.config import TrainConfig
from src.sft_data import SFTDataset
from viz.train_engine import HifiTrainer, HifiSFTTrainer


def test_hifi_pretrain_runs_steps():
    model = _model()
    data = np.arange(2000, dtype=np.int64) % 64
    cfg = TrainConfig(batch_size=2, grad_accum=1, max_steps=3, lr=1e-3,
                      warmup_steps=0, log_interval=1, weight_decay=0.0)
    seen = []
    tr = HifiTrainer(model, cfg, data, device="cpu", ctx_len=8,
                     on_step=lambda s, l, lr: seen.append((s, l)))
    tr.viz_run()
    assert tr._current_step == 3 and seen and seen[-1][0] == 3
    assert all(np.isfinite(l) for _, l in seen)


def test_hifi_sft_runs_and_interrupts(tmp_path):
    npz = tmp_path / "toy.npz"
    np.savez(npz,
             ids=np.array([5, 6, 7, 8, 9, 10], dtype=np.int32),
             labels=np.array([-100, 6, 7, -100, 9, 10], dtype=np.int32),
             offsets=np.array([0, 3, 6], dtype=np.int64))
    model = _model(vocab=64, ctx=16)
    cfg = TrainConfig(batch_size=2, grad_accum=1, max_steps=2, lr=1e-3,
                      warmup_steps=0, log_interval=1, weight_decay=0.0)

    class _Tok:
        def special_id(self, name):
            return 0

    seen = []
    tr = HifiSFTTrainer(model, cfg, SFTDataset(str(npz)), _Tok(), max_len=8,
                        device="cpu", on_step=lambda s, l, lr: seen.append(s))
    tr.viz_run()
    assert tr._current_step == 2 and seen and seen[-1] == 2

    tr2 = HifiSFTTrainer(model, cfg, SFTDataset(str(npz)), _Tok(), max_len=8, device="cpu")
    tr2._stop.set()
    tr2.viz_run()
    assert tr2._current_step == tr2.start_step  # 已中断，未前进
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_train_engine.py -q`
Expected: FAIL（`ImportError: cannot import name 'HifiTrainer'`）

- [ ] **Step 3: 实现（追加到 `viz/train_engine.py`）**

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_train_engine.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add viz/train_engine.py tests/test_viz_train_engine.py
git commit -m "feat: viz 高保真训练引擎"
```

---

## Task 5: 服务端接线（State / API / live 词表 / 保存）

**Files:**
- Modify: `viz/server.py`、`viz/runtime.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `viz.datasets.{list_datasets,resolve_spec,load_pretrain,PretrainBatches,SFTBatches}`、`viz.train_engine.{SimpleEngine,HifiTrainer,HifiSFTTrainer}`
- Produces:
  - `State.start_train(target="live", mode="pretrain", dataset="corpus_qwen", engine="simple", params=None)`
  - `State.stop_train()`（停止并保存）
  - `GET /api/datasets`；`POST /api/train/start`（新 body）；`tick` 增 `total_steps/mode/dataset/engine`
  - `smoke()` 直接 `build_tiny(256, "cpu")`，不再用字符分词器

- [ ] **Step 1: 更新测试（`tests/test_viz.py`）**

```python
def test_server_smoke_builds_live_graph():
    info = server.smoke()
    assert info["source"] == "live"
    assert info["n_matrices"] > 0 and info["n_connections"] > 0
    assert "blocks.0.attn.q_proj.weight" in info["sample_matrix"]


def test_train_spec_validation():
    from viz.datasets import resolve_spec
    assert resolve_spec("poetry", "pretrain")["kind"] == "pretrain"
    import pytest
    with pytest.raises(ValueError):
        resolve_spec("poetry", "sft")
```

（`smoke` 断言保持不变，因为 `smoke()` 仍返回同样的键。）

- [ ] **Step 2: 运行确认当前状态**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py -q`
Expected: PASS（此时未改实现也应通过；作为接线前的基线）

- [ ] **Step 3: 删除 `viz/runtime.py` 的 `LiveTrainer`**

删除 `runtime.py` 中 `class LiveTrainer` 整个类（约 58–120 行），保留 `CharTokenizer`、`CORPUS`、`build_tiny`、`stream_generate` 及 KV 辅助函数。

- [ ] **Step 4: 改写 `viz/server.py`**

导入改为：

```python
from viz.runtime import CharTokenizer, build_tiny, stream_generate
from viz.datasets import (list_datasets, resolve_spec, load_pretrain,
                          PretrainBatches, SFTBatches)
from viz.train_engine import SimpleEngine, HifiTrainer, HifiSFTTrainer
```

`build_live` / `smoke` 改为：

```python
def build_live():
    """tiny 结构 + Qwen 词表；数据集只影响训练数据，不影响模型结构。"""
    from viz.datasets import DATASETS
    from viz.runtime import CORPUS
    tok = _qwen_tokenizer()
    model = build_tiny(tok.vocab_size, "cpu")
    graph = build_graph(model, source="live")
    return model, tok, graph, (DATASETS["corpus_qwen"],)
```

新增模块级缓存 tokenizer：

```python
_QWEN = None

def _qwen_tokenizer():
    global _QWEN
    if _QWEN is None:
        _QWEN = QwenTokenizer.load("data/tokenizer/qwen2.5-0.5b",
                                   add_image_token=False)
    return _QWEN
```

`smoke()`：

```python
def smoke() -> dict:
    model = build_tiny(256, "cpu")
    store = MatrixStore(model)
    graph = build_graph(model, source="live")
    attn = next(m for m in graph["matrices"] if m["role"] == "attn")
    return {"source": "live", "n_matrices": len(graph["matrices"]),
            "n_connections": len(graph["connections"]),
            "sample_matrix": attn["name"],
            "global_absmax": store.global_stats()["absmax"]}
```

`State.__init__` 增：`self.train_engine = None`、`self._train_meta = {}`、`self._train_ctx = None`。

`State.start_train` 重写：

```python
def start_train(self, target="live", mode="pretrain", dataset="corpus_qwen",
                engine="simple", params=None):
    params = params or {}
    if target not in ("live", "ckpt", "scratch"):
        target = "live"
    if target == "ckpt" and not self.has_ckpt:
        raise CkptUnavailable("服务端未加载 checkpoint，请用 --ckpt <path> 启动后重试。")
    spec = resolve_spec(dataset, mode)
    if engine not in ("simple", "hifi"):
        engine = "simple"
    if self.trainer:
        self.trainer.stop()
    model, tok, graph = self.source(target)

    max_steps = max(1, int(params.get("max_steps", 200)))
    lr = float(params.get("lr", getattr(self.args, "lr", 3e-4)))
    batch_size = max(1, int(params.get("batch_size", 8)))
    grad_accum = max(1, int(params.get("grad_accum", 1)))
    block = int(self.args.block)

    if mode == "pretrain":
        data = load_pretrain(spec, tok)
        make_batches = lambda: PretrainBatches(data, batch_size, block, self.device)
        hifi_data = data
    else:
        path = spec["path"]
        pad_id = tok.special_id("pad")
        make_batches = lambda: SFTBatches(path, batch_size, self.args.sft_max_len,
                                          pad_id, self.device)
        hifi_data = path

    self.tracker.capture(model)
    self._train_ctx = (model, graph)
    self._train_meta = {"target": target, "mode": mode, "dataset": dataset,
                        "engine": engine, "total_steps": max_steps}
    model.train()
    if engine == "hifi":
        from src.config import TrainConfig
        cfg = TrainConfig(batch_size=batch_size, grad_accum=grad_accum, lr=lr,
                          min_lr=lr * 0.1, warmup_steps=max(1, max_steps // 20),
                          max_steps=max_steps, weight_decay=0.05, grad_clip=1.0,
                          dtype="bf16", out_dir="out/viz", log_interval=5)
        if mode == "sft":
            from src.sft_data import SFTDataset
            ds = SFTDataset(hifi_data)
            self.trainer = HifiSFTTrainer(model, cfg, ds, tok,
                                          max_len=self.args.sft_max_len,
                                          device=self.device, on_step=self._on_step)
        else:
            self.trainer = HifiTrainer(model, cfg, hifi_data, tokenizer=tok,
                                       device=self.device, ctx_len=block,
                                       on_step=self._on_step)
    else:
        batches = make_batches()
        self.trainer = SimpleEngine(model, batches.next, self.device, self._on_step,
                                    lr=lr, max_steps=max_steps, grad_accum=grad_accum,
                                    weight_decay=0.05, warmup_steps=0)
    self.train_target = target
    self.train_engine = engine
    self.trainer.start()
```

`_on_step` 增字段：

```python
def _on_step(self, step, loss, lr):
    model, graph = self._train_ctx
    vals = snapshot_matrices(model, graph["matrices"], tracker=self.tracker)
    msg = {"type": "tick", "step": step,
           "loss": loss if math.isfinite(loss) else 0.0, "lr": lr,
           "target": self.train_target, "values": vals}
    msg.update(self._train_meta)
    self.hub.publish(msg)
    self.tracker.capture(model)
```

`stop_train`（保存）：

```python
def stop_train(self):
    if not self.trainer:
        return None
    self.trainer.stop()
    path = self._save_path()
    try:
        self.trainer.save(path)
    except Exception as e:  # noqa: BLE001
        print(f"[viz] 保存失败：{e}")
        path = None
    self.trainer = None
    return path

def _save_path(self):
    import datetime
    meta = self._train_meta
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "out" / "viz"
    out.mkdir(parents=True, exist_ok=True)
    return str(out / f"{meta.get('target','live')}-{meta.get('mode','pretrain')}"
                      f"-{meta.get('dataset','corpus_qwen')}-{ts}.pt")
```

`api_model` 中 `source in ("ckpt","scratch")` 保持不变（live 现在也要加载 Qwen 分词器，稍慢，改走线程）：

```python
@app.get("/api/model")
async def api_model(source: str = "live"):
    graph = await asyncio.to_thread(state.graph, source)
    return JSONResponse(graph)
```

`/api/datasets`：

```python
@app.get("/api/datasets")
async def api_datasets():
    return JSONResponse(list_datasets())
```

`api_train_start` / `api_train_stop`：

```python
@app.post("/api/train/start")
async def api_train_start(payload: dict | None = None):
    body = payload or {}
    target = str(body.get("target", "live"))
    mode = str(body.get("mode", "pretrain"))
    dataset = str(body.get("dataset", "corpus_qwen"))
    engine = str(body.get("engine", "simple"))
    params = body.get("params") or {}
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: state.start_train(target, mode, dataset, engine, params))
    return {"ok": True, "target": target, "mode": mode, "dataset": dataset,
            "engine": engine, "total_steps": int(params.get("max_steps", 200))}


@app.post("/api/train/stop")
async def api_train_stop():
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, state.stop_train)
    state.hub.publish({"type": "status", "training": False, "saved": path})
    return {"ok": True, "saved": path}
```

CLI 增参：

```python
ap.add_argument("--sft-max-len", type=int, default=256)
```

- [ ] **Step 5: 运行全部 viz 测试 + 冒烟**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py tests/test_viz_datasets.py tests/test_viz_train_engine.py tests/test_viz_frontend.py -q`
Expected: PASS
Run: `& ".venv\Scripts\python.exe" viz/server.py --check`
Expected: 打印字典且退出码 0

- [ ] **Step 6: 提交**

```bash
git add viz/server.py viz/runtime.py tests/test_viz.py
git commit -m "feat: viz 训练配置服务端接线（数据集/方式/引擎/保存）"
```

---

## Task 6: 左栏四段式 UI（index.html + style.css）

**Files:**
- Modify: `viz/static/index.html`、`viz/static/style.css`
- Test: `tests/test_viz_frontend.py`

**Interfaces:**
- Produces（DOM id）：`#train-target`、`#train-mode`、`#train-dataset`、`#hp-max-steps`、`#hp-lr`、`#hp-batch`、`#hp-accum`、`#hp-hifi`、`#btn-train-start`、`#btn-train-stop`

- [ ] **Step 1: 追加测试（`tests/test_viz_frontend.py`）**

```python
def test_index_has_training_controls():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for dom_id in ["train-target", "train-mode", "train-dataset",
                   "hp-max-steps", "hp-lr", "hp-batch", "hp-accum", "hp-hifi"]:
        assert f'id="{dom_id}"' in html, f"缺少 #{dom_id}"
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py::test_index_has_training_controls -q`
Expected: FAIL（缺少 #train-mode）

- [ ] **Step 3: 改写 `index.html` 的 `#train-panel`**

```html
    <div class="train-panel" id="train-panel">
      <label class="arch-label" for="train-target">模型</label>
      <select id="train-target">
        <option value="live">实时小模型（d_model=64 / 4 层）</option>
        <option value="ckpt">192M 微调（--ckpt 权重）</option>
        <option value="scratch">192M 从头训练</option>
      </select>
      <label class="arch-label" for="train-mode">训练方式</label>
      <select id="train-mode">
        <option value="pretrain">预训练（next-token）</option>
        <option value="sft">SFT（只监督助手）</option>
      </select>
      <label class="arch-label" for="train-dataset">训练数据集</label>
      <select id="train-dataset"></select>
      <div class="train-grid">
        <label>最大步数<input id="hp-max-steps" type="number" min="1" step="1" value="200"></label>
        <label>学习率<input id="hp-lr" type="number" min="0" step="0.0001" value="0.0003"></label>
        <label>batch<input id="hp-batch" type="number" min="1" step="1" value="8"></label>
        <label>梯度累积<input id="hp-accum" type="number" min="1" step="1" value="1"></label>
      </div>
      <label class="train-check"><input id="hp-hifi" type="checkbox"> 高保真模式（bf16 + warmup）</label>
      <div class="train-row">
        <button id="btn-train-start">开始训练</button>
        <button id="btn-train-stop" disabled>停止</button>
      </div>
      <div id="train-stats">未开始训练</div>
    </div>
```

- [ ] **Step 4: `style.css` 追加样式**

```css
#train-panel .train-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 8px 0;
}
#train-panel .train-grid label {
  display: flex; flex-direction: column; font-size: 11px; color: #9fb2d8; gap: 3px;
}
#train-panel .train-grid input {
  width: 100%; padding: 4px 6px; border-radius: 7px; font-size: 12px;
  background: #0b1226; color: #dfe6f3; border: 1px solid #243454;
}
#train-panel .train-check {
  display: flex; align-items: center; gap: 6px; font-size: 12px;
  color: #9fb2d8; margin: 6px 0;
}
```

- [ ] **Step 5: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py -q`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add viz/static/index.html viz/static/style.css tests/test_viz_frontend.py
git commit -m "feat: viz 左栏训练配置控件"
```

---

## Task 7: 前端接线（main.js：数据集/超参/payload/禁用）

**Files:**
- Modify: `viz/static/js/main.js`
- Test: `tests/test_viz_frontend.py`（`node --check` 已覆盖）

**Interfaces:**
- Consumes: `GET /api/datasets`、`POST /api/train/start`（新 body）
- Produces: `loadDatasets()`、`filterDatasets()`、`readTrainPayload()`

- [ ] **Step 1: 写失败测试（追加）**

```python
def test_main_has_train_config_wiring():
    src = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    assert "/api/datasets" in src
    assert "train-dataset" in src and "hp-max-steps" in src and "hp-hifi" in src
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py::test_main_has_train_config_wiring -q`
Expected: FAIL

- [ ] **Step 3: 实现（`main.js` 替换训练配置相关片段）**

在训练区块（`const trainTargetSel = ...` 附近）新增：

```js
const trainModeSel = document.getElementById('train-mode');
const trainDatasetSel = document.getElementById('train-dataset');
const hpInputs = {
  max_steps: document.getElementById('hp-max-steps'),
  lr: document.getElementById('hp-lr'),
  batch_size: document.getElementById('hp-batch'),
  grad_accum: document.getElementById('hp-accum'),
};
const hifiCheck = document.getElementById('hp-hifi');
let datasetList = [];

async function loadDatasets() {
  try {
    const res = await fetch('/api/datasets');
    datasetList = res.ok ? await res.json() : [];
  } catch (e) {
    console.error('加载数据集失败', e);
    datasetList = [];
  }
  filterDatasets();
}

function filterDatasets() {
  if (!trainDatasetSel) return;
  const mode = (trainModeSel && trainModeSel.value) || 'pretrain';
  const prev = trainDatasetSel.value;
  trainDatasetSel.replaceChildren();
  for (const d of datasetList) {
    if (d.kind !== mode) continue;
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = d.label + (d.available ? '' : '（文件缺失）');
    opt.disabled = !d.available;
    trainDatasetSel.appendChild(opt);
  }
  if (prev && [...trainDatasetSel.options].some((o) => o.value === prev)) {
    trainDatasetSel.value = prev;
  }
}

function readHyper() {
  const read = (node, fallback) => {
    const v = node ? Number(node.value) : NaN;
    return Number.isFinite(v) ? v : fallback;
  };
  return {
    max_steps: Math.max(1, Math.round(read(hpInputs.max_steps, 200))),
    lr: read(hpInputs.lr, 3e-4),
    batch_size: Math.max(1, Math.round(read(hpInputs.batch_size, 8))),
    grad_accum: Math.max(1, Math.round(read(hpInputs.grad_accum, 1))),
  };
}

function readTrainPayload() {
  return {
    target: (trainTargetSel && trainTargetSel.value) || 'live',
    mode: (trainModeSel && trainModeSel.value) || 'pretrain',
    dataset: (trainDatasetSel && trainDatasetSel.value) || 'corpus_qwen',
    engine: hifiCheck && hifiCheck.checked ? 'hifi' : 'simple',
    params: readHyper(),
  };
}
```

把 `trainStart()` 改为：

```js
async function trainStart() {
  if (training) return;
  const payload = readTrainPayload();
  if (!(await ensureSourceForTraining(payload.target))) return;
  markModeButtons(payload.target);
  const prev = training;
  setTraining(true);
  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (e) {
    console.error('开始训练失败', e);
    setTraining(prev);
  }
}
```

`setTraining` 增禁用：

```js
function setTraining(on) {
  training = !!on;
  if (startBtn) {
    startBtn.disabled = training;
    startBtn.textContent = training ? '训练中…' : '开始训练';
  }
  if (stopBtn) stopBtn.disabled = !training;
  if (trainTargetSel) trainTargetSel.disabled = training;
  if (trainModeSel) trainModeSel.disabled = training;
  if (trainDatasetSel) trainDatasetSel.disabled = training;
  if (hifiCheck) hifiCheck.disabled = training;
  for (const node of Object.values(hpInputs)) if (node) node.disabled = training;
}
```

事件与初始化：

```js
trainModeSel?.addEventListener('change', filterDatasets);
await loadDatasets();
```

`tick` 分支进度显示：

```js
if (m.type === 'tick') {
  shelf?.updateValues(m.values);
  links?.update(m.values);
  maybeAutoExpand(m.values);
  pushLoss(m.loss);
  updateTrainStats(m.step, m.loss, m.lr, m.total_steps);
  liveRefresh(m.step);
  return;
}
```

`updateTrainStats` 支持总量：

```js
function updateTrainStats(step, loss, lr, total) {
  if (!trainStatsEl) return;
  const parts = [];
  if (step != null) parts.push(total ? `step ${step}/${total}` : `step ${step}`);
  if (loss != null && Number.isFinite(Number(loss))) parts.push(`loss ${Number(loss).toFixed(4)}`);
  if (lr != null && Number.isFinite(Number(lr))) parts.push(`lr ${Number(lr).toExponential(1)}`);
  trainStatsEl.textContent = parts.length ? parts.join(' · ') : '未开始训练';
}
```

（`liveRefresh` 在 Task 8 定义；本任务先加一个占位 `function liveRefresh() {}`，Task 8 替换。）

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/main.js tests/test_viz_frontend.py
git commit -m "feat: viz 训练配置前端接线"
```

---

## Task 8: 训练中实时刷新（分级节流）

**Files:**
- Modify: `viz/static/js/matrix.js`、`viz/static/js/shelf.js`、`viz/static/js/panel.js`、`viz/static/js/main.js`
- Test: `tests/test_viz_frontend.py`

**Interfaces:**
- Produces:
  - `matrix.js`: `refreshMatrixTexture(mesh, source, nonce) -> Promise<void>`（plane 换 `map`；InstancedMesh 重取 grid 重设色）
  - `shelf.js`: `Shelf.refreshTextures(step) -> Promise<void>`（仅刷新 `expanded` 层的矩阵）
  - `panel.js`: `Panel.setLive(on)`、`Panel.refreshLive()`
  - `main.js`: `liveRefresh(step)`

- [ ] **Step 1: 写失败测试（追加）**

```python
def test_live_refresh_hooks_exist():
    for f, needle in [("matrix.js", "refreshMatrixTexture"),
                      ("shelf.js", "refreshTextures"),
                      ("panel.js", "refreshLive"),
                      ("main.js", "liveRefresh")]:
        src = (STATIC / "js" / f).read_text(encoding="utf-8")
        assert needle in src, f"{f} 缺少 {needle}"
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py::test_live_refresh_hooks_exist -q`
Expected: FAIL

- [ ] **Step 3: `matrix.js` 增刷新函数**

在 `buildPlane` 里记录 tiles：

```js
  mesh.userData = { kind: 'matrix', name: spec.name, spec, shape: spec.shape, tiles };
```

追加：

```js
export async function refreshMatrixTexture(mesh, source = 'live', nonce = 0) {
  const ud = mesh.userData || {};
  const name = ud.name;
  if (!name) return;
  if (mesh.isInstancedMesh) {
    const { values } = await getGrid(name, source, 256);
    const cols = values[0] ? values[0].length : 1;
    let absmax = 1e-6;
    for (const row of values) for (const v of row) absmax = Math.max(absmax, Math.abs(v));
    const col = new THREE.Color();
    for (let r = 0; r < values.length; r++) {
      for (let c = 0; c < cols; c++) {
        const [rr, gg, bb] = divergingRGB(values[r][c] / absmax);
        mesh.setColorAt(r * cols + c, col.setRGB(rr, gg, bb));
      }
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    return;
  }
  const tiles = ud.tiles || 64;
  const url = `${matrixPngUrl(name, source, tiles, 'global')}&t=${nonce}`;
  await new Promise((resolve, reject) => {
    loader.load(url, (tex) => {
      tex.magFilter = THREE.NearestFilter;
      tex.minFilter = THREE.LinearFilter;
      const old = mesh.material.map;
      mesh.material.map = tex;
      mesh.material.needsUpdate = true;
      if (old) old.dispose();
      resolve();
    }, undefined, reject);
  });
}
```

- [ ] **Step 4: `shelf.js` 增 `refreshTextures`**

```js
  async refreshTextures(step) {
    if (this.expanded == null || this._refreshing) return;
    const tray = this.trays.get(this.expanded);
    const list = tray && tray.userData.matrixMeshes;
    if (!list || !list.length) return;
    this._refreshing = true;
    try {
      const { refreshMatrixTexture } = await import('./matrix.js');
      await Promise.all(list.map((m) =>
        refreshMatrixTexture(m, this.source, step).catch(() => {})));
    } finally {
      this._refreshing = false;
    }
  }
```

（`matrix.js` 也能静态 `import`；若顶部已 `import { buildPlane, buildCubes }`，直接改成
`import { buildPlane, buildCubes, refreshMatrixTexture }` 并去掉动态 import。）

- [ ] **Step 5: `panel.js` 增实时刷新**

构造函数加 `this.live = false;`；追加方法：

```js
  setLive(on) {
    this.live = !!on;
  }

  refreshLive() {
    if (!this.live || this.tab !== 'matrix' || !this.spec) return;
    this._draw(this.tiles);
  }
```

- [ ] **Step 6: `main.js` 定义 `liveRefresh` 并在启动/停止时切换 `panel.setLive`**

删除占位，改为：

```js
let lastPanelRefresh = 0;

function liveRefresh(step) {
  const now = performance.now();
  if (now - lastPanelRefresh >= 500) {
    lastPanelRefresh = now;
    panel.refreshLive?.();
  }
  if (step % 20 === 0) shelf?.refreshTextures?.(step);
}
```

`setTraining` 内追加：

```js
  panel.setLive(training);
```

- [ ] **Step 7: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py -q`
Expected: PASS（含 `node --check`）

- [ ] **Step 8: 提交**

```bash
git add viz/static/js/matrix.js viz/static/js/shelf.js viz/static/js/panel.js viz/static/js/main.js tests/test_viz_frontend.py
git commit -m "feat: viz 训练中分级节流实时刷新"
```

---

## Task 9: 文档与收尾

**Files:**
- Modify: `docs/07-visualization.md`

**Interfaces:** 无代码接口；只更新文档。

- [ ] **Step 1: 更新 `docs/07-visualization.md`**

新增「训练配置」小节，包含：
- 左栏四段：模型 / 训练方式 / 数据集 / 超参 / 高保真勾选。
- 数据集注册表（列出 `viz/datasets.py` 的 id 与文件）。
- 引擎 `simple` 与 `hifi` 的区别（fp32 轻量 vs bf16+warmup）。
- `POST /api/train/start` 新 body 示例与 `GET /api/datasets`。
- 保存路径 `out/viz/<target>-<mode>-<dataset>-<时间>.pt`。
- 实时刷新策略（3D 每 tick、抽屉 500ms、展开层纹理每 20 tick）。

- [ ] **Step 2: 运行全部相关测试**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py tests/test_viz_datasets.py tests/test_viz_train_engine.py tests/test_viz_frontend.py tests/test_ms_datasets.py -q`
Expected: 全 PASS

- [ ] **Step 3: 服务端冒烟**

Run: `& ".venv\Scripts\python.exe" viz/server.py --check`
Expected: 打印字典，退出码 0

- [ ] **Step 4: 提交**

```bash
git add docs/07-visualization.md docs/superpowers/specs/2026-09-29-viz-training-pipeline-design.md docs/superpowers/plans/2026-09-29-viz-training-pipeline.md
git commit -m "docs: viz 训练配置与实时刷新说明"
```

---

## Self-Review

**Spec coverage**
- 三种目标 + 数据集选择 → Task 5（`start_train` 对 live/ckpt/scratch 通用）+ Task 7（UI）。
- tiny 用 Qwen → Task 5（`_qwen_tokenizer` + `build_tiny(tok.vocab_size)`）。
- 双引擎可切换 → Task 3/4/5/7（`hp-hifi`）。
- 左栏四段式 → Task 6。
- 超参 4 项 → Task 6/7。
- 分级节流刷新 → Task 8。
- 写死注册表 → Task 1。
- 不限制 SFT → Task 5（无前置校验）。
- 保存到 out/viz → Task 5。

**风险/备注**
- `QwenTokenizer.load(..., add_image_token=False)` 避免写回 tokenizer 目录；live 词表以 tokenizer 为准。
- 192M + `hifi` + 大 SFT 序列可能吃显存：`--sft-max-len` 默认 256，可下调。
- 无头浏览器不可用：前端交互（控件禁用、实时刷新视觉）需人工冒烟。

**Type consistency**
- `batch_fn() -> (inputs, targets)` 在 `PretrainBatches.next`、`SFTBatches.next`、`SimpleEngine` 三处一致。
- `on_step(step, loss, lr)` 在 `SimpleEngine`、`_VizLoop`、`State._on_step` 三处一致。
- `tick` 字段 `total_steps/mode/dataset/engine` 在 `State._on_step`（`msg.update(self._train_meta)`）与 main.js 一致。
