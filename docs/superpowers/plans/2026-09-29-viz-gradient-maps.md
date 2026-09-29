# viz 梯度地图 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训练中把梯度作为与权重并列的可视化对象：抽屉元素级梯度热力图、3D 板材按梯度上色、底栏逐层梯度流，并提供「权重/梯度/ΔW」与「全局/局部」与「有符号/幅值」切换。

**Architecture:** 后端新增 `viz/maps.py`（地图张量池化 + 顺序色阶 + 梯度缓存），`viz/stats.py` 增 `snapshot_gradients`，`viz/weights.py` 的 PNG 渲染泛化为任意张量/色阶；`viz/server.py` 在现有 `grid/png/stats` 端点加 `mode/norm/signed`，`tick` 增每矩阵梯度标量。前端在顶栏加显示开关，`matrix/shelf/panel/api/main` 全面 mode-aware，底栏加梯度流 canvas。

**Tech Stack:** Python 3.12 · FastAPI · PyTorch · numpy · Pillow · Three.js(vendored) · pytest

**Spec:** `docs/superpowers/specs/2026-09-29-viz-gradient-maps-design.md`

## Global Constraints

- 运行环境：Windows，shell 为 pwsh；Python 用 `& ".venv\Scripts\python.exe"`；`node` 可直接调用。
- 三态：`mode ∈ {"weight","grad","delta"}`；`norm ∈ {"global","local"}`；色阶 `signed ∈ {true,false}`。
- 发散色阶沿用 `viz/weights.diverging_rgb` / JS `divergingRGB`（**不得改动**）。
- 新增顺序色阶必须 Python/JS 一致：`lo=(0.10,0.20,0.55)`，`hi=(0.98,0.62,0.10)`，`sequential_rgb(t)=lerp(lo,hi,clamp(t,0,1))`。
- 梯度仅在训练中：非训练态 `grad=0`；`tiles` 默认 64、上限 256；梯度缓存网格 `tiles=128`。
- ΔW 元素地图 = `W − W_prev`（`WeightTracker.prev`，`named_parameters(remove_duplicate=False)`）；标量 `delta` 保留。
- 参数名一律带 `.weight`；遍历参数用 `model.named_parameters(remove_duplicate=False)`。
- 不改动 `src/` 训练语义；提交信息中文 `feat:/fix:/docs:`；每任务跑对应 pytest 全绿。
- 不加无关注释；用 YAGNI。

---

## File Structure

**创建**
- `viz/maps.py` — `pool_map`、`matrix_map`、`sequential_rgb`、`GradCache`
- `tests/test_viz_maps.py` — 地图/缓存测试

**修改**
- `viz/stats.py` — `snapshot_gradients`
- `viz/weights.py` — `render_png` 泛化（任意张量、`signed`）
- `viz/server.py` — 端点 `mode/norm/signed`；`State` 的梯度缓存与地图服务；`tick` 增 `grad`
- `viz/static/js/api.js` — `getGrid/getStats/matrixPngUrl` 支持 `mode/norm/signed`
- `viz/static/index.html`、`viz/static/style.css` — 顶栏显示开关 + 底栏 `#gradflow`
- `viz/static/js/panel.js` — mode-aware 热力图/图例
- `viz/static/js/matrix.js`、`shelf.js` — 3D 按 mode 着色/刷新
- `viz/static/js/main.js` — 开关联动、tick 梯度、梯度流 HUD
- `tests/test_viz.py`、`tests/test_viz_frontend.py` — 端点/静态断言
- `docs/07-visualization.md` — 梯度地图说明

---

## Task 1: `viz/maps.py` — 池化、地图张量与顺序色阶

**Files:**
- Create: `viz/maps.py`
- Test: `tests/test_viz_maps.py`

**Interfaces:**
- Produces:
  - `SEQUENTIAL_LO = (0.10, 0.20, 0.55)`，`SEQUENTIAL_HI = (0.98, 0.62, 0.10)`
  - `sequential_rgb(v: float) -> tuple[float,float,float]`（`v` 先 `clamp(0,1)`，线性 `lerp`）
  - `pool_map(t: torch.Tensor, tiles: int) -> torch.Tensor`（一维 reshape 为 `[1,-1]`；`adaptive_avg_pool2d`；返回二维 `[h,k]`）
  - `matrix_map(model, name: str, mode: str, prev: torch.Tensor | None = None) -> torch.Tensor`：
    `weight → p.detach().float()`；`delta → (p - prev.reshape(p.shape)).float()`（`prev is None` 时返回 `zeros_like`）；`grad` 不支持（由缓存提供，抛 `ValueError`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_viz_maps.py
import torch

from src.config import ModelConfig
from src.model import GPT
from viz.maps import SEQUENTIAL_HI, SEQUENTIAL_LO, matrix_map, pool_map, sequential_rgb


def _model(vocab=64, d=32, layers=2, ctx=16):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, d_model=d, n_layer=layers, n_head=4,
                           n_kv_head=2, d_ff=64, ctx_len=ctx))


def test_pool_map_shapes_and_1d():
    t = torch.arange(60, dtype=torch.float32).reshape(5, 12)
    g = pool_map(t, 4)
    assert g.shape == (4, 4)
    one = pool_map(torch.arange(10, dtype=torch.float32), 4)  # 1D -> [1,10]
    assert one.shape[0] == 1 and one.shape[1] <= 10


def test_matrix_map_weight_and_delta():
    m = _model()
    w = matrix_map(m, "blocks.0.attn.q_proj.weight", "weight")
    assert w.shape == (32, 32)
    import torch as T
    prev = T.zeros_like(w).reshape(-1)
    d = matrix_map(m, "blocks.0.attn.q_proj.weight", "delta", prev=prev)
    assert T.allclose(d, w)
    assert T.allclose(matrix_map(m, "tok_emb.weight", "delta", prev=None),
                      T.zeros_like(matrix_map(m, "tok_emb.weight", "weight")))


def test_sequential_rgb_monotonic_and_bounds():
    lo, hi = sequential_rgb(0.0), sequential_rgb(1.0)
    assert abs(lo[0] - SEQUENTIAL_LO[0]) < 1e-6
    assert abs(hi[0] - SEQUENTIAL_HI[0]) < 1e-6
    assert sequential_rgb(0.5)[0] > sequential_rgb(0.2)[0]
    for c in (*sequential_rgb(-3.0), *sequential_rgb(9.0)):
        assert 0.0 <= c <= 1.0
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_maps.py -q`
Expected: FAIL（`ModuleNotFoundError: viz.maps`）

- [ ] **Step 3: 实现**

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_maps.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add viz/maps.py tests/test_viz_maps.py
git commit -m "feat: viz 地图张量与顺序色阶"
```

---

## Task 2: 梯度统计 `snapshot_gradients` 与 `GradCache`

**Files:**
- Modify: `viz/maps.py`、`viz/stats.py`
- Test: `tests/test_viz_maps.py`

**Interfaces:**
- Produces:
  - `viz.stats.snapshot_gradients(model, matrices: list[dict]) -> dict[str, dict]`：每矩阵 `{"grad_norm","grad_absmean","grad_absmax"}`（`grad is None` 时为 0）
  - `viz.maps.GradCache(tiles: int = 128)`：`.capture(model, matrices, step: int) -> None`；`.grid(name) -> list | None`；`.stats(name) -> dict | None`；`.absmax: float`；`.step: int`（线程安全，读写在 `threading.Lock` 内）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_viz_maps.py`）**

```python
from viz.maps import GradCache
from viz.stats import snapshot_gradients
from viz.graph import build_graph


def _with_grads(m):
    x = torch.randint(0, 64, (2, 8))
    _, loss, _ = m(x, targets=x)
    loss.backward()
    return m


def test_snapshot_gradients_keys_and_nonneg():
    m = _with_grads(_model())
    g = build_graph(m)
    snap = snapshot_gradients(m, g["matrices"])
    q = snap["blocks.0.attn.q_proj.weight"]
    assert set(q) == {"grad_norm", "grad_absmean", "grad_absmax"}
    assert q["grad_norm"] >= 0 and q["grad_absmax"] >= 0


def test_grad_cache_grid_and_absmax():
    m = _with_grads(_model())
    g = build_graph(m)
    cache = GradCache(tiles=8)
    cache.capture(m, g["matrices"], step=3)
    assert cache.step == 3 and cache.absmax > 0
    grid = cache.grid("blocks.0.attn.q_proj.weight")
    assert grid is not None and len(grid) <= 8 and len(grid[0]) <= 8
    assert cache.stats("blocks.0.attn.q_proj.weight")["grad_norm"] >= 0
    assert cache.grid("nope") is None
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_maps.py -q`
Expected: FAIL（`ImportError: cannot import name 'GradCache'`）

- [ ] **Step 3: 实现**

`viz/stats.py` 追加：

```python
def snapshot_gradients(model: nn.Module, matrices: list[dict]) -> dict:
    """按矩阵聚合梯度标量（无梯度记 0）。梯度必须在 backward 之后、zero_grad 之前取。"""
    grads = {n: p.grad for n, p in model.named_parameters(remove_duplicate=False)}
    out: dict[str, dict] = {}
    for m in matrices:
        g = grads.get(m["name"])
        if g is None:
            out[m["name"]] = {"grad_norm": 0.0, "grad_absmean": 0.0, "grad_absmax": 0.0}
            continue
        gf = g.detach().float()
        out[m["name"]] = {
            "grad_norm": _finite(float(gf.pow(2).sum()) ** 0.5),
            "grad_absmean": _finite(float(gf.abs().mean())),
            "grad_absmax": _finite(float(gf.abs().max())),
        }
    return out
```

`viz/maps.py` 追加：

```python
import threading


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
```

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_maps.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add viz/maps.py viz/stats.py tests/test_viz_maps.py
git commit -m "feat: viz 梯度统计与逐 tick 缓存"
```

---

## Task 3: `viz/weights.py` PNG 渲染泛化（任意张量 + signed/magnitude）

**Files:**
- Modify: `viz/weights.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Produces: `render_png(t: torch.Tensor, tiles: int, absmax: float, signed: bool = True) -> bytes`
  （`signed=True` 用发散色阶；`False` 用 `maps.sequential_rgb` 的 `t/absmax ∈ [0,1]`）
- `MatrixStore.png(name, tiles, norm)` 保持原签名，内部改为调用 `render_png`（行为不变）。

- [ ] **Step 1: 写失败测试（追加到 `tests/test_viz.py`）**

```python
def test_render_png_signed_vs_magnitude():
    from viz.weights import render_png
    t = torch.randn(16, 16)
    a = render_png(t, tiles=8, absmax=float(t.abs().max()), signed=True)
    b = render_png(t, tiles=8, absmax=float(t.abs().max()), signed=False)
    assert a[:8] == b"\x89PNG\r\n\x1a\n" and b[:8] == b"\x89PNG\r\n\x1a\n"
    assert a != b
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_render_png_signed_vs_magnitude -q`
Expected: FAIL（`ImportError: cannot import name 'render_png'`）

- [ ] **Step 3: 实现（`viz/weights.py`）**

```python
def render_png(t: torch.Tensor, tiles: int, absmax: float, signed: bool = True) -> bytes:
    from PIL import Image
    from viz.maps import sequential_rgb
    g = _pool(t, tiles)
    absmax = absmax or 1.0
    if signed:
        v = (g / absmax).clamp(-1, 1)
        rgb = _diverging_image(v)
    else:
        v = (g.abs() / absmax).clamp(0, 1)
        rgb = _sequential_image(v)
    img = Image.fromarray(rgb.numpy(), mode="RGB")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()
```

其中把原 `png` 里的分块/上色拆成模块级 `_pool(t, tiles)`、`_diverging_image(v)`、`_sequential_image(v)`（`_sequential_image` 用 `sequential_rgb` 生成 `uint8` 张量），`MatrixStore.png` 改为：

```python
    def png(self, name, tiles=64, norm="global"):
        w = self._weight(name)
        absmax = (self.stats(name)["absmax"] if norm == "matrix"
                  else self.global_stats()["absmax"]) or 1.0
        return render_png(w, tiles, absmax, signed=True)
```

（保持 `norm="matrix"` 与既有调用兼容。）

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py -q`
Expected: PASS（既有 `test_cell_patch_stats_png` 不回归）

- [ ] **Step 5: 提交**

```bash
git add viz/weights.py tests/test_viz.py
git commit -m "refactor: viz PNG 渲染泛化为任意张量/色阶"
```

---

## Task 4: 服务端接线（mode/norm/signed、梯度缓存、tick 增 grad）

**Files:**
- Modify: `viz/server.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `viz.maps.{matrix_map, GradCache, sequential_rgb}`、`viz.stats.snapshot_gradients`、`viz.weights.render_png`
- Produces:
  - `State.grad_cache: GradCache`
  - `State.map_tensor(source, name, mode) -> torch.Tensor`
  - `State.map_grid(source, name, mode, tiles, norm) -> dict`（形状如 `MatrixStore.grid`：`{shape, grid, values}`）
  - `State.map_png(source, name, mode, tiles, norm, signed) -> bytes`
  - `State.map_stats(source, name, mode) -> dict`（`{shape, vmin, vmax, absmax, mean, std}`）
  - 端点加 `mode: str = "weight"`、`norm: str = "global"`（`/grid`、`/png`）、`mode`（`/stats`）、`signed: bool = True`（`/png`）
  - `/api/model` 返回 `global`（`{weight, grad, delta}`），保留旧 `global_absmax`
  - `tick.values[name]` 增 `grad`（`{grad_norm, grad_absmean, grad_absmax}`）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_viz.py`）**

```python
def test_map_service_modes():
    from viz.server import State
    import argparse
    st = State(argparse.Namespace(ckpt="", config="configs/gpt_tinystories.yaml",
                                  lr=3e-3, batch=16, block=48, sft_max_len=256))
    # 用 tiny 模型替换 live，避免加载 Qwen 分词器
    from viz.graph import build_graph
    from viz.runtime import build_tiny
    model = build_tiny(64, "cpu")
    st._live = (model, None, build_graph(model, source="live"))
    name = "blocks.0.attn.q_proj.weight"
    g = st.map_grid("live", name, "weight", 16, "global")
    assert g["grid"] == [16, 16]
    assert st.map_png("live", name, "weight", 8, "global", True)[:8] == b"\x89PNG\r\n\x1a\n"
    st_val = st.map_stats("live", name, "weight")
    assert st_val["shape"] == [32, 32]
    # 未训练时梯度为 0
    zg = st.map_grid("live", name, "grad", 8, "global")
    assert all(v == 0 for row in zg["values"] for v in row)
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_map_service_modes -q`
Expected: FAIL（`AttributeError: 'State' object has no attribute 'map_grid'`）

- [ ] **Step 3: 实现（`viz/server.py`）**

导入：`from viz.maps import GradCache, matrix_map, pool_map`、`from viz.weights import MatrixStore, load_checkpoint, render_png`。

`State.__init__` 增：`self.grad_cache = GradCache(tiles=128)`、`self._delta_global = 0.0`、`self._delta_step = -1`。

新增方法：

```python
    def _mode_model(self, source: str):
        return self.source(source)[0]

    def map_tensor(self, source, name, mode):
        model = self._mode_model(source)
        if mode == "weight":
            return matrix_map(model, name, "weight")
        if mode == "grad":
            if source == self.train_target:
                g = self.grad_cache.grid(name)
                if g is not None:
                    return torch.tensor(g, dtype=torch.float32)
            w = matrix_map(model, name, "weight")
            return torch.zeros_like(w)
        if mode == "delta":
            prev = self.tracker.prev.get(name)
            return matrix_map(model, name, "delta", prev=prev)
        raise ValueError(f"未知 mode：{mode}")

    def _map_absmax(self, source, name, mode, norm):
        if mode == "weight":
            store = self.store(source)
            return (store.stats(name)["absmax"] if norm == "local"
                    else store.global_stats()["absmax"]) or 1.0
        if mode == "grad":
            if norm == "local":
                return float(self.map_tensor(source, name, "grad").abs().max()) or 1.0
            return self.grad_cache.absmax or 1.0
        # delta
        if norm == "local":
            return float(self.map_tensor(source, name, "delta").abs().max()) or 1.0
        return self._delta_absmax()

    def _delta_absmax(self):
        step = self.grad_cache.step
        if step != self._delta_step:
            mx = 0.0
            for n, prev in self.tracker.prev.items():
                cur = self.tracker.prev
                p = {k: v for k, v in self._mode_model(self.train_target)
                     .named_parameters(remove_duplicate=False)}.get(n)
                if p is None:
                    continue
                mx = max(mx, float((p.detach().float().reshape(-1) - prev).abs().max()))
            self._delta_global = mx
            self._delta_step = step
        return self._delta_global or 1.0

    def map_grid(self, source, name, mode, tiles=64, norm="global"):
        t = self.map_tensor(source, name, mode)
        g = pool_map(t, tiles)
        return {"shape": list(t.shape), "grid": list(g.shape), "values": g.tolist()}

    def map_png(self, source, name, mode, tiles=64, norm="global", signed=True):
        t = self.map_tensor(source, name, mode)
        return render_png(t, tiles, self._map_absmax(source, name, mode, norm), signed)

    def map_stats(self, source, name, mode):
        t = self.map_tensor(source, name, mode)
        return {"shape": list(t.shape), "vmin": float(t.min()), "vmax": float(t.max()),
                "absmax": float(t.abs().max()), "mean": float(t.mean()),
                "std": float(t.std())}
```

（`_pool_map` 用 `viz.maps.pool_map`；上面 `_delta_absmax` 里对 `p` 的取法请写成一次性 `params = {n: p for ...}` 再 `params.get(n)`，避免重复构造。）

`_on_step` 在 publish 前采集：

```python
        self.grad_cache.capture(model, graph["matrices"], step)
        for name, s in self.grad_cache.all_stats().items():
            if name in vals:
                vals[name]["grad"] = s
```

端点：

```python
@app.get("/api/matrix/{name}/grid")
async def api_grid(name: str, source: str = "live", tiles: int = 64,
                   mode: str = "weight", norm: str = "global"):
    if mode == "weight":
        return JSONResponse(state.store(source).grid(name, tiles))
    return JSONResponse(await asyncio.to_thread(state.map_grid, source, name, mode, tiles, norm))
```

`/png`、`/stats` 同理（`mode != "weight"` 走 `to_thread`；`png` 加 `signed: bool = True`）。
`api_model` 增：

```python
    g = await asyncio.to_thread(state.graph, source)
    g["global"] = {m: await asyncio.to_thread(state._map_absmax, source, sample, m, "global")
                   for m in ("weight", "grad", "delta")}
```

其中 `sample` 取 `graph["matrices"][0]["name"]`（weight 用 store，grad/delta 用缓存全局）。

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py tests/test_viz_maps.py -q`
Expected: PASS；再 `& ".venv\Scripts\python.exe" viz/server.py --check`（exit 0）

- [ ] **Step 5: 提交**

```bash
git add viz/server.py tests/test_viz.py
git commit -m "feat: viz 梯度/ΔW 地图服务端与 tick 梯度"
```

---

## Task 5: 前端服务与顶栏显示开关 + 抽屉 mode-aware

**Files:**
- Modify: `viz/static/js/api.js`、`viz/static/index.html`、`viz/static/style.css`、`viz/static/js/panel.js`
- Test: `tests/test_viz_frontend.py`

**Interfaces:**
- Produces（api.js）:
  - `getGrid(name, source, tiles, mode='weight', norm='global')`
  - `getStats(name, source, mode='weight')`
  - `matrixPngUrl(name, source, tiles, norm='global', mode='weight', signed=true)`
- Produces（DOM）：`#map-mode`（`weight|grad|delta`）、`#map-norm`（`global|local`）、`#map-signed`（`signed|magnitude`）
- Produces（panel.js）：`Panel.setMap(mode, norm, signed)`；`_draw` 用当前 map 参数请求并绘制；图例文案含模式/色标

- [ ] **Step 1: 写失败测试（追加到 `tests/test_viz_frontend.py`）**

```python
def test_map_controls_and_api():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for dom_id in ["map-mode", "map-norm", "map-signed", "gradflow"]:
        assert f'id="{dom_id}"' in html, f"缺少 #{dom_id}"
    api = (STATIC / "js" / "api.js").read_text(encoding="utf-8")
    assert "mode" in api and "norm" in api and "signed" in api
    panel = (STATIC / "js" / "panel.js").read_text(encoding="utf-8")
    assert "setMap" in panel
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py::test_map_controls_and_api -q`
Expected: FAIL

- [ ] **Step 3: 实现**

`api.js`：

```js
export const getGrid = (name, source = 'live', tiles = 64, mode = 'weight', norm = 'global') =>
  j(`/api/matrix/${encodeURIComponent(name)}/grid?${q({ source, tiles, mode, norm })}`);
export const getStats = (name, source = 'live', mode = 'weight') =>
  j(`/api/matrix/${encodeURIComponent(name)}/stats?${q({ source, mode })}`);
export const matrixPngUrl = (name, source = 'live', tiles = 64, norm = 'global',
                             mode = 'weight', signed = true) =>
  `/api/matrix/${encodeURIComponent(name)}/png?${q({ source, tiles, norm, mode, signed })}`;
```

`index.html` 顶栏在 `.viewswitch` 后加：

```html
    <div class="mapswitch">
      <select id="map-mode" title="显示量">
        <option value="weight" selected>权重</option>
        <option value="grad">梯度</option>
        <option value="delta">更新量 ΔW</option>
      </select>
      <select id="map-norm" title="色标范围">
        <option value="global" selected>全局色标</option>
        <option value="local">局部色标</option>
      </select>
      <select id="map-signed" title="颜色语义">
        <option value="signed" selected>有符号</option>
        <option value="magnitude">幅值</option>
      </select>
    </div>
```

底栏 `<canvas id="gradflow" width="240" height="48"></canvas>`（放在 `#attn` 后）。

`style.css` 加 `.mapswitch { display:flex; gap:6px; } .mapswitch select { ... }`（与顶栏下拉一致）。

`panel.js`：构造加 `this.mapMode='weight'; this.mapNorm='global'; this.mapSigned=true;`；新增

```js
  setMap(mode, norm, signed) {
    this.mapMode = mode || 'weight';
    this.mapNorm = norm || 'global';
    this.mapSigned = signed !== false;
    if (this.built && this.spec) this._draw(this.tiles);
  }
```

`_draw` 改：

```js
      const [stats, grid] = await Promise.all([
        getStats(name, this.source, this.mapMode),
        getGrid(name, this.source, tiles, this.mapMode, this.mapNorm),
      ]);
```

`_renderHeat` 上色按 `this.mapSigned` 选 `divergingRGB` 或新增 JS `sequentialRGB`（在 `colors.js` 加，公式与 Python 一致：`lo/hi` 线性插值）；`_drawLegend` 文案加 `（${modeLabel}·${normLabel}）`。`colors.js` 新增：

```js
export const SEQUENTIAL_LO = [0.10, 0.20, 0.55];
export const SEQUENTIAL_HI = [0.98, 0.62, 0.10];
export function sequentialRGB(t) {
  const v = Math.max(0, Math.min(1, +t));
  return SEQUENTIAL_LO.map((lo, i) => lo + (SEQUENTIAL_HI[i] - lo) * v);
}
```

`matrix.js` 的 canvas/cube 上色也改用 `sequentialRGB`（Task 6 一并处理）。

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py -q`
Expected: PASS（含 `node --check`）

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/api.js viz/static/js/colors.js viz/static/js/panel.js viz/static/index.html viz/static/style.css tests/test_viz_frontend.py
git commit -m "feat: viz 显示开关与抽屉梯度热力图"
```

---

## Task 6: 3D 按 mode 着色（板材/立方块/纹理刷新）

**Files:**
- Modify: `viz/static/js/matrix.js`、`viz/static/js/shelf.js`、`viz/static/js/main.js`

**Interfaces:**
- Consumes: `matrixPngUrl(...mode,signed)`、`getGrid(...mode,norm)`、`sequentialRGB`
- Produces:
  - `matrix.js`: `buildPlane(spec, source, tiles, height, opts={ mode, norm, signed })`、`buildCubes(spec, source, height, opts)`、`refreshMatrixTexture(mesh, source, nonce, opts)`
  - `shelf.js`: `updateValues(values, mode)`、`refreshTextures(step, opts)`、`_buildMatrices` 传 opts；`this.mapOpts`
  - `main.js`: 全局 `mapMode/mapNorm/mapSigned`，开关 change 时 `shelf.setMap(...)`→刷新纹理并 `updateValues(lastValues, mode)`；tick 时 `shelf.updateValues(m.values, mapMode)`

- [ ] **Step 1: 写失败测试（追加到 `tests/test_viz_frontend.py`）**

```python
def test_shelf_and_matrix_are_mode_aware():
    shelf = (STATIC / "js" / "shelf.js").read_text(encoding="utf-8")
    matrix = (STATIC / "js" / "matrix.js").read_text(encoding="utf-8")
    main = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    assert "sequentialRGB" in matrix or "sequentialRGB" in shelf
    assert "mode" in matrix and "mode" in shelf
    assert "map-mode" in main and "setMap" in main
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py::test_shelf_and_matrix_are_mode_aware -q`
Expected: FAIL

- [ ] **Step 3: 实现**

- `matrix.js`：`buildPlane(spec, source, tiles, height, opts = {})` 用 `opts.mode`/`opts.norm`/`opts.signed` 调 `matrixPngUrl`，并把 `opts` 存进 `mesh.userData.mapOpts`；`buildCubes(..., opts)` 用 `getGrid(name, source, 256, opts.mode, opts.norm)`，背景色按 `opts.signed ? divergingRGB : sequentialRGB`；`refreshMatrixTexture(mesh, source, nonce, opts)` 同样带上 `mode/norm/signed`。
- `shelf.js`：`constructor` 加 `this.mapOpts = { mode: 'weight', norm: 'global', signed: true }`；`setMap(opts)` 合并并清空已建矩阵网格以强制重建（`delete tray.userData.matrixMeshes` 后重新 `expand`），或直接 `refreshTextures`；`_buildMatrices` 用 `this.mapOpts`；`updateValues(values, mode='weight')` 取色值：`weight→raw.norm`、`grad→raw.grad?.grad_norm ?? 0`、`delta→raw.delta`；薄板聚合同理。
- `main.js`：读 `#map-mode/#map-norm/#map-signed`，暴露

```js
let lastValues = null;
function currentMapOpts() {
  return {
    mode: document.getElementById('map-mode')?.value || 'weight',
    norm: document.getElementById('map-norm')?.value || 'global',
    signed: (document.getElementById('map-signed')?.value || 'signed') === 'signed',
  };
}
function onMapChange() {
  const opts = currentMapOpts();
  shelf?.setMap(opts);
  if (lastValues) shelf?.updateValues(lastValues, opts.mode);
}
```

`tick` 分支存 `lastValues = m.values;` 并 `shelf?.updateValues(m.values, currentMapOpts().mode)`。三个 select 都 `addEventListener('change', onMapChange)`。

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py -q`
Expected: PASS（含 `node --check`）

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/matrix.js viz/static/js/shelf.js viz/static/js/main.js tests/test_viz_frontend.py
git commit -m "feat: viz 3D 板材按梯度/ΔW 着色"
```

---

## Task 7: 底栏逐层梯度流 HUD

**Files:**
- Modify: `viz/static/js/main.js`、`viz/static/style.css`
- Test: `tests/test_viz_frontend.py`

**Interfaces:**
- Consumes: `tick.values[name].grad.grad_norm`、`shelf.graph.matrices`（`name→layer`）
- Produces: `drawGradFlow(values)`（按层聚合 `grad_norm` 均值画柱）；`#gradflow` canvas

- [ ] **Step 1: 写失败测试（追加到 `tests/test_viz_frontend.py`）**

```python
def test_gradflow_hud_hook():
    main = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    assert "drawGradFlow" in main and "gradflow" in main
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py::test_gradflow_hud_hook -q`
Expected: FAIL

- [ ] **Step 3: 实现（`main.js`）**

```js
const gradCanvas = document.getElementById('gradflow');
let nameLayer = null;

function layerOfName(name) {
  if (!nameLayer && shelf) {
    nameLayer = new Map();
    for (const m of shelf.graph.matrices || []) nameLayer.set(m.name, m.layer);
  }
  return nameLayer ? nameLayer.get(name) : null;
}

function drawGradFlow(values) {
  if (!gradCanvas) return;
  const ctx = gradCanvas.getContext('2d');
  const w = gradCanvas.width, h = gradCanvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a1020'; ctx.fillRect(0, 0, w, h);
  if (!values) return;
  const agg = new Map();
  for (const [name, v] of Object.entries(values)) {
    const g = v && v.grad ? Number(v.grad.grad_norm) : 0;
    const layer = layerOfName(name);
    if (!layer || !Number.isFinite(g)) continue;
    const cur = agg.get(layer) || { s: 0, n: 0 };
    cur.s += g; cur.n += 1; agg.set(layer, cur);
  }
  const ids = (shelf && shelf.graph.layers || []).map((l) => l.id);
  const vals = ids.map((id) => { const a = agg.get(id); return a ? a.s / a.n : 0; });
  const mx = Math.max(1e-9, ...vals);
  const bw = w / Math.max(vals.length, 1);
  for (let i = 0; i < vals.length; i++) {
    const bh = (vals[i] / mx) * (h - 4);
    ctx.fillStyle = '#e0b23c';
    ctx.fillRect(i * bw + 0.5, h - bh, Math.max(1, bw - 1), bh);
  }
}
```

`tick` 分支加 `drawGradFlow(m.values);`。`nameLayer` 在 `reload()` 里重置为 `null`。

`style.css` 给 `#gradflow` 一个标题提示（可选，放 `title` 属性即可，无需额外样式）。

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/main.js viz/static/style.css tests/test_viz_frontend.py
git commit -m "feat: viz 底栏逐层梯度流 HUD"
```

---

## Task 8: 文档与全量验证

**Files:**
- Modify: `docs/07-visualization.md`

- [ ] **Step 1: 更新 `docs/07-visualization.md`**

新增「梯度地图」小节：三态开关（权重/梯度/ΔW）、全局/局部、有符号/幅值；梯度仅训练中；每 tick 全量采集梯度网格 + 标量；ΔW 元素地图与标量比值；底栏梯度流；端点 `mode/norm/signed` 参数。

- [ ] **Step 2: 全量测试**

Run: `& ".venv\Scripts\python.exe" -m pytest tests -q`
Expected: 全 PASS

Run: `& ".venv\Scripts\python.exe" viz/server.py --check`
Expected: 打印字典，exit 0

- [ ] **Step 3: 提交**

```bash
git add docs/07-visualization.md docs/superpowers/specs/2026-09-29-viz-gradient-maps-design.md docs/superpowers/plans/2026-09-29-viz-gradient-maps.md
git commit -m "docs: viz 梯度地图说明"
```

---

## Self-Review

**Spec coverage**
- 抽屉热力图 → Task 5；3D 着色 → Task 6；梯度流 HUD → Task 7。
- 有符号/幅值可切 → Task 1（顺序色阶）+ Task 3（PNG）+ Task 5（JS 色阶）+ Task 6。
- 仅训练中 → Task 4（非训练 grad=0）。
- 全部矩阵都算 / 每 tick 全量采集 → Task 4 `_on_step` + Task 2 `GradCache.capture`。
- 权重/梯度/ΔW 三态 → Task 5/6。
- ΔW 元素 + 标量比值 → Task 1 `matrix_map("delta")` + Task 4 tick 保留 `delta`。
- 全局/局部 → Task 4 `_map_absmax` + Task 5/6。
- 底栏 HUD → Task 7。

**风险/备注**
- 192M 每 tick 全量池化梯度偏重；`GradCache.tiles`（128）与 `log_interval`（5）可调。
- `_delta_absmax` 初次计算会扫一遍全部参数；按 `grad_cache.step` 失效，避免每请求重算。
- 无头浏览器不可用：三态切换/梯度流的视觉需人工冒烟。

**Type consistency**
- `mode/norm/signed` 在 server、api.js、panel、shelf、main 五处命名一致。
- `GradCache.grid/stats/absmax/step` 在 server 与测试两处一致；`snapshot_gradients` 的键 `grad_norm/grad_absmean/grad_absmax` 在 stats、server、main 三处一致。
