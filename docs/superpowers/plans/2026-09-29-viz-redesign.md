# viz 3D 可视化重构 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `viz/` 重构成 bbycroft 式的层叠书架：竖直 12 层 → 点层展开为权重矩阵板（方块/热力图、颜色=数值、悬停取值、连接、推理流动），并保留实时训练、神经元点云、架构浏览器。

**Architecture:** 后端新增 `graph.py`（统一模型图）、`weights.py`（checkpoint 纹理/元素/统计）、`stats.py`、`neurons.py`，重写 `server.py` 路由；前端重写为 ES 模块（`shelf/matrix/links/panel/cloud/arch/colors/api/main`），用 Three.js 渲染层托盘与矩阵平面。

**Tech Stack:** Python 3.12 · FastAPI/uvicorn · PyTorch · Pillow（PNG） · Three.js（vendored） · KaTeX（vendored） · pytest

**Spec:** `docs/superpowers/specs/2026-09-29-viz-redesign-design.md`

## Global Constraints

- 运行环境：Windows，shell 为 pwsh；Python 用 `& ".venv\Scripts\python.exe"`；Node 在 `S:\Users\Mrhan\AppData\Local\nodejs\node.exe`（用 `node` 即可）。
- 只使用**本地** checkpoint（`out/gpt/latest.pt`，config `configs/gpt_tinystories.yaml`）；**不下载**任何 HF 权重。
- 小矩阵"立方块"判定：矩阵**元素数 ≤ 4096**（含一维 norm，reshape 为 `[1, d]`）。
- 纹理/网格 `tiles` 默认 **64**，上限 **256**。
- 发散色阶（Python 与 JS 必须一致）：`t=clip(v/absmax,-1,1)`；
  `t≥0`（正）→ `(0.16+0.84t, 0.08+0.16t, 0.08+0.16t)`；
  `t<0`（负）→ `(0.08+0.16|t|, 0.16+0.28|t|, 0.16+0.84|t|)`；输出 RGB ∈ [0,1]。
- 参数名约定（`src/model.py`）：`tok_emb.weight`、`blocks.{i}.n1.weight`、
  `blocks.{i}.attn.{q_proj,k_proj,v_proj,o_proj}.weight`、`blocks.{i}.n2.weight`、
  `blocks.{i}.mlp.{w1,w2,w3}.weight`、`norm_f.weight`、`lm_head.weight`。
- 矩阵名一律用带 `.weight` 的全名（如 `blocks.0.attn.q_proj.weight`）。
- `tie_embeddings=True` 时 `lm_head.weight` 与 `tok_emb.weight` 共享张量；
  遍历参数一律用 `model.named_parameters(remove_duplicate=False)`。
- 提交信息用中文，风格对齐仓库现有提交（`feat:` / `fix:` / `docs:`）。
- 每个任务结束跑对应 pytest，必须全绿再进入下一个。

---

## File Structure

**创建**
- `viz/graph.py` — 统一模型图：layers / modules / matrices / connections
- `viz/stats.py` — `summarize`、`WeightTracker`、`ActivationRecorder`、`snapshot_matrices`
- `viz/neurons.py` — `neuron_cloud`、`_pca3`
- `viz/weights.py` — `load_checkpoint`、`diverging_rgb`、`MatrixStore`（grid/png/cell/patch/stats）
- `viz/static/js/{api,colors,shelf,matrix,links,panel,cloud,arch,main}.js`
- `tests/test_viz_frontend.py` — 静态资源存在 + `node --check` 语法门

**修改**
- `viz/runtime.py` — `LiveTrainer` 回调用矩阵名；其余保持
- `viz/server.py` — 重写路由/State/WS
- `tests/test_viz.py` — 迁移到新模块
- `viz/static/index.html`、`viz/static/style.css` — 重写
- `docs/07-visualization.md` — 重写

**删除（最后清理）**
- `viz/topology.py`、`viz/static/app.js`

---

## 里程碑 V1 · 后端数据层

### Task 1: 模型图 `viz/graph.py`

**Files:**
- Create: `viz/graph.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `src.model.GPT`（`blocks`、`named_parameters(remove_duplicate=False)`）
- Produces: `CUBE_THRESHOLD: int`；`build_graph(model, source: str = "live") -> dict`，
  返回 `{source, model, layers, matrices, connections, cube_threshold}`；
  matrix 元素 `{name,label,role,layer,module,shape,ndim,elements,small}`；
  connection 元素 `{id,src,dst,kind,weight_from}`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_viz.py（在文件顶部替换导入后追加）
from viz.graph import build_graph, CUBE_THRESHOLD


def test_build_graph_matrices_and_connections():
    model = tiny_model()
    g = build_graph(model, source="live")
    names = {m["name"] for m in g["matrices"]}
    assert {"tok_emb.weight", "norm_f.weight",
            "blocks.0.attn.q_proj.weight", "blocks.1.mlp.w3.weight"} <= names
    layers = {l["id"] for l in g["layers"]}
    assert {"emb", "L0", "L1", "final"} <= layers
    q = next(m for m in g["matrices"] if m["name"] == "blocks.0.attn.q_proj.weight")
    assert q["shape"] == [32, 32] and q["layer"] == "L0" and q["role"] == "attn"
    for e in g["connections"]:
        assert e["src"] in names and e["dst"] in names and e["weight_from"] in names
    assert g["cube_threshold"] == CUBE_THRESHOLD
    assert g["model"]["n_layer"] == 2 and g["model"]["d_model"] == 32
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_build_graph_matrices_and_connections -v`
Expected: FAIL（`ModuleNotFoundError: viz.graph`）

- [ ] **Step 3: 实现 `viz/graph.py`**

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_build_graph_matrices_and_connections -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add viz/graph.py tests/test_viz.py
git commit -m "feat: viz 统一模型图 graph.py"
```

---

### Task 2: 统计模块 `viz/stats.py`

**Files:**
- Create: `viz/stats.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Produces: `summarize(tensors) -> {norm,mean,max,n}`；`WeightTracker`（`capture(model)`、`delta(names, params)`）；
  `ActivationRecorder`（`attach/detach/clear/acts`、`for_matrix(name)`）；
  `snapshot_matrices(model, matrices, tracker=None) -> {matrixName: {norm,mean,max,delta?}}`。
- 逻辑照搬 `viz/topology.py` 中同名实现，`for_params` 改名 `for_matrix`（参数为矩阵全名）。

- [ ] **Step 1: 写失败测试**

```python
from viz.stats import (ActivationRecorder, WeightTracker, snapshot_matrices, summarize)


def test_summarize_and_snapshot_by_matrix():
    model = tiny_model()
    g = build_graph(model)
    s = summarize([__import__("torch").randn(8, 4)])
    assert s["n"] == 32 and s["norm"] > 0
    tracker = WeightTracker(); tracker.capture(model)
    with __import__("torch").no_grad():
        for p in model.parameters():
            p.add_(0.01)
    snap = snapshot_matrices(model, g["matrices"], tracker=tracker)
    assert snap["tok_emb.weight"]["delta"] > 0
    assert snap["tok_emb.weight"]["norm"] > 0
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_summarize_and_snapshot_by_matrix -v`
Expected: FAIL（`ModuleNotFoundError: viz.stats`）

- [ ] **Step 3: 实现 `viz/stats.py`**

把 `viz/topology.py` 的 `summarize`、`WeightTracker`、`ActivationRecorder` 原样搬入，
并新增：

```python
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.model import RMSNorm


def _finite(x: float) -> float:
    return x if math.isfinite(x) else 0.0


# summarize / WeightTracker / ActivationRecorder：从 viz/topology.py 原样搬入。
# ActivationRecorder 增加：
#     def for_matrix(self, name: str) -> float:
#         mod = name.rsplit(".", 1)[0] if "." in name else name
#         return _finite(self.acts.get(mod, 0.0))


def snapshot_matrices(model: nn.Module, matrices: list[dict],
                      tracker: "WeightTracker | None" = None) -> dict:
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
```

实现者：从 `viz/topology.py` 复制 `summarize`、`WeightTracker`（含 `MAX_PARAMS=8_000_000`、`capture`、`delta`）、
`ActivationRecorder`（`attach` 用 `(nn.Linear, nn.Embedding, RMSNorm)`），按下划线要求补 `for_matrix`。
本任务**不删除** `viz/topology.py`（留到 Task 17 清理）。

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_summarize_and_snapshot_by_matrix -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add viz/stats.py tests/test_viz.py
git commit -m "feat: viz 统计模块 stats.py（按矩阵聚合）"
```

---

### Task 3: 神经元点云 `viz/neurons.py`

**Files:**
- Create: `viz/neurons.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Produces: `neuron_cloud(weight, max_nodes=192, top_edges=2500, seed=0) -> dict`（与旧 `topology.neuron_cloud` 完全一致）。

- [ ] **Step 1: 写失败测试**

```python
from viz.neurons import neuron_cloud


def test_neuron_cloud_shapes():
    w = __import__("torch").randn(40, 24)
    c = neuron_cloud(w, max_nodes=16, top_edges=50)
    assert len(c["out_coords"]) == 16 and len(c["in_coords"]) == 16
    assert all(len(p) == 3 for p in c["out_coords"])
    assert len(c["edges"]) == 50
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_neuron_cloud_shapes -v`
Expected: FAIL（`ModuleNotFoundError: viz.neurons`）

- [ ] **Step 3: 实现 `viz/neurons.py`**

把 `viz/topology.py` 的 `neuron_cloud` 与 `_pca3` 原样搬到 `viz/neurons.py`。

- [ ] **Step 4: 清掉 `test_viz.py` 里旧的 `test_neuron_cloud_shapes` 重复定义**（保留 Task 3 新增的那个），并运行：

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add viz/neurons.py tests/test_viz.py
git commit -m "feat: viz 神经元点云 neurons.py"
```

---

### Task 4: 权重数据层 `viz/weights.py`

**Files:**
- Create: `viz/weights.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Produces:
  - `diverging_rgb(v_norm: float) -> tuple[float,float,float]`
  - `load_checkpoint(path: str, config_path: str) -> (GPT, Any)`
  - `MatrixStore(model)`：`grid(name, tiles=64) -> {"shape":[o,i],"grid":[h,w],"values":[[...]]}`；
    `stats(name) -> {shape, vmin, vmax, absmax, mean, std}`；
    `global_stats() -> {absmax}`；
    `cell(name,i,j) -> float`；`patch(name,r,c,h,w) -> {"r","c","h","w","values"}`；
    `png(name, tiles=64, norm="global") -> bytes`。
- Consumes: `viz.graph` 的矩阵名；`src.config.load_config`；`src.model.GPT`。

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

import pytest
import torch

from viz.weights import MatrixStore, diverging_rgb, load_checkpoint


def _tensor(model, name):
    return {n: p for n, p in model.named_parameters(remove_duplicate=False)}[name]


def test_diverging_rgb_signs():
    assert diverging_rgb(1.0)[0] > diverging_rgb(1.0)[2]      # 正 -> 偏红
    assert diverging_rgb(-1.0)[2] > diverging_rgb(-1.0)[0]    # 负 -> 偏蓝
    for c in (*diverging_rgb(1.0), *diverging_rgb(-1.0)):
        assert 0.0 <= c <= 1.0


def test_grid_element_level_and_pooling():
    model = tiny_model()
    store = MatrixStore(model)
    name = "blocks.0.attn.q_proj.weight"
    g = store.grid(name, tiles=64)
    assert g["grid"] == [32, 32]
    assert abs(g["values"][0][0] - float(_tensor(model, name)[0, 0])) < 1e-6
    p = store.grid(name, tiles=8)
    assert p["grid"] == [8, 8]


def test_cell_patch_stats_png():
    model = tiny_model()
    store = MatrixStore(model)
    name = "blocks.0.mlp.w1.weight"     # [d_ff, d_model] = [64, 32]
    w = _tensor(model, name)
    assert abs(store.cell(name, 1, 2) - float(w[1, 2])) < 1e-6
    patch = store.patch(name, 0, 0, 4, 5)
    assert patch["h"] == 4 and patch["w"] == 5 and len(patch["values"]) == 4
    st = store.stats(name)
    assert st["shape"] == [64, 32] and st["absmax"] > 0
    png = store.png(name, tiles=8, norm="matrix")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_load_checkpoint_optional():
    if not Path("out/gpt/latest.pt").exists():
        pytest.skip("无本地 checkpoint")
    model, cfg = load_checkpoint("out/gpt/latest.pt", "configs/gpt_tinystories.yaml")
    assert model.cfg.d_model == 768 and len(model.blocks) == 12
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_diverging_rgb_signs -v`
Expected: FAIL（`ModuleNotFoundError: viz.weights`）

- [ ] **Step 3: 实现 `viz/weights.py`**

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py -k "diverging or grid_element or cell_patch or load_checkpoint" -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add viz/weights.py tests/test_viz.py
git commit -m "feat: viz 权重数据层 weights.py（网格/纹理/元素查询）"
```

---

### Task 5: 重写后端 `viz/server.py`

**Files:**
- Modify: `viz/server.py`（整体重写）
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `viz.graph.build_graph`、`viz.stats.*`、`viz.weights.MatrixStore/load_checkpoint`、`viz.neurons.neuron_cloud`、`viz.runtime.*`、`viz.arch.*`。
- Produces: `State`、`smoke() -> dict`、FastAPI `app`；端点见 spec 4.2；WS `init|tick|token|status`。

- [ ] **Step 1: 写失败测试**

```python
import viz.server as server


def test_server_smoke_builds_live_graph():
    info = server.smoke()
    assert info["source"] == "live"
    assert info["n_matrices"] > 0 and info["n_connections"] > 0
    assert "blocks.0.attn.q_proj.weight" in info["sample_matrix"]
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_server_smoke_builds_live_graph -v`
Expected: FAIL（`AttributeError: module 'viz.server' has no attribute 'smoke'`）

- [ ] **Step 3: 重写 `viz/server.py`**

要点（实现时按此结构写全）：

```python
"""mini-llm-lab 3D 可视化后端（重写版）。"""
from __future__ import annotations

import argparse, asyncio, math, sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch, uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.config import load_config
from src.model import GPT
from viz import arch as arch_mod
from viz.graph import build_graph
from viz.neurons import neuron_cloud
from viz.runtime import CharTokenizer, LiveTrainer, build_tiny, stream_generate
from viz.stats import ActivationRecorder, WeightTracker, snapshot_matrices
from viz.weights import MatrixStore, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"


class Hub:  # 与旧实现相同（queues/bind/register/unregister/publish/_offer）
    ...


def build_live(*, lr=3e-3, batch=16, block=48):
    """返回 (model, tokenizer, graph, trainer_factory)。"""
    from viz.runtime import CORPUS
    tok = CharTokenizer(CORPUS)
    model = build_tiny(tok.vocab_size, "cpu")
    graph = build_graph(model, source="live")
    return model, tok, graph, (lr, batch, block)


class State:
    def __init__(self, args) -> None:
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hub = Hub()
        self.tracker = WeightTracker()
        self.recorder = ActivationRecorder()
        self.mode = "weights" if args.ckpt else "live"
        self._live = None   # (model, tok, graph)
        self._ckpt = None   # (model, cfg, graph)
        self.trainer = None

    def ensure_live(self):
        if self._live is None:
            model, tok, graph, (lr, batch, block) = build_live(
                lr=self.args.lr, batch=self.args.batch, block=self.args.block)
            self._live = (model, tok, graph)
            self.recorder.attach(model)
            self.trainer = LiveTrainer(tok, self.device, on_step=self._on_step,
                                       lr=lr, batch_size=batch, block_size=block, model=model)
        return self._live

    def ensure_ckpt(self):
        if self._ckpt is None:
            model, cfg = load_checkpoint(self.args.ckpt, self.args.config)
            model.to(self.device)
            self._ckpt = (model, cfg, build_graph(model, source="ckpt"))
            self.recorder.attach(model)
        return self._ckpt

    def source(self, which: str):
        return self.ensure_live() if which == "live" else self.ensure_ckpt()

    def store(self, which: str) -> MatrixStore:
        return MatrixStore(self.source(which)[0])

    def graph(self, which: str) -> dict:
        model, _, graph = self.source(which)
        g = dict(graph)
        g["device"] = self.device
        g["global_absmax"] = self.store(which).global_stats()["absmax"]
        return g

    def _on_step(self, step, loss, lr):
        model, _, graph = self.ensure_live()
        vals = snapshot_matrices(model, graph["matrices"], tracker=self.tracker)
        self.hub.publish({"type": "tick", "step": step,
                          "loss": loss if math.isfinite(loss) else 0.0,
                          "lr": lr, "values": vals})
        self.tracker.capture(model)

    def infer(self, which, prompt, max_new_tokens):
        model, tok, graph = self.source(which)
        self.recorder.clear()

        def on_token(text, token_id, attn_pack):
            vals = {m["name"]: {"act": self.recorder.for_matrix(m["name"])}
                    for m in graph["matrices"]}
            self.hub.publish({"type": "token", "token": (text[-1] if text else ""),
                              "id": token_id, "text": text, "values": vals, "attn": attn_pack})

        stream_generate(model, tok, prompt, on_token, max_new_tokens=max_new_tokens, attn_layer=0)


app = FastAPI(title="mini-llm-lab 3D 可视化", lifespan=lifespan)
state: State | None = None


@asynccontextmanager
async def lifespan(_app):
    state.hub.bind(asyncio.get_running_loop()); yield


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/model")
async def api_model(source: str = "live"):
    return JSONResponse(state.graph(source))


@app.get("/api/matrix/{name}/grid")
async def api_grid(name: str, source: str = "live", tiles: int = 64):
    return JSONResponse(state.store(source).grid(name, tiles))


@app.get("/api/matrix/{name}/png")
async def api_png(name: str, source: str = "live", tiles: int = 64, norm: str = "global"):
    return Response(content=state.store(source).png(name, tiles, norm), media_type="image/png")


@app.get("/api/matrix/{name}/stats")
async def api_stats(name: str, source: str = "live"):
    return JSONResponse(state.store(source).stats(name))


@app.get("/api/matrix/{name}/cell")
async def api_cell(name: str, i: int, j: int, source: str = "live"):
    return JSONResponse({"i": i, "j": j, "value": state.store(source).cell(name, i, j)})


@app.get("/api/matrix/{name}/patch")
async def api_patch(name: str, r: int, c: int, h: int, w: int, source: str = "live"):
    return JSONResponse(state.store(source).patch(name, r, c, h, w))


@app.get("/api/neurons")
async def api_neurons(matrix: str, source: str = "live", n: int = 160, top: int = 2500):
    model = state.source(source)[0]
    mod = dict(model.named_modules()).get(matrix)
    if mod is None or not isinstance(mod, torch.nn.Linear):
        return JSONResponse({"error": f"未找到 Linear 模块：{matrix}"}, status_code=404)
    data = neuron_cloud(mod.weight.detach().cpu(), max_nodes=int(n), top_edges=int(top))
    data["matrix"] = matrix; data["shape"] = list(mod.weight.shape)
    return JSONResponse(data)


@app.get("/api/arch/list")
async def api_arch_list():
    return JSONResponse(arch_mod.list_models())


@app.get("/api/arch/{model_id}")
async def api_arch(model_id: str):
    spec = arch_mod.load(model_id)
    if spec is None:
        return JSONResponse({"error": f"未找到架构：{model_id}"}, status_code=404)
    return JSONResponse(arch_mod.public(spec))


@app.post("/api/train/start")
async def api_train_start():
    state.ensure_live(); state.tracker.capture(state._live[0]); state.trainer.start()
    state.hub.publish({"type": "status", "training": True}); return {"ok": True}


@app.post("/api/train/stop")
async def api_train_stop():
    if state.trainer: state.trainer.stop()
    state.hub.publish({"type": "status", "training": False}); return {"ok": True}


@app.post("/api/infer")
async def api_infer(payload: dict):
    which = str(payload.get("source", "live"))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, state.infer, which,
                               str(payload.get("prompt", "人工智能")),
                               int(payload.get("max_new_tokens", 40)))
    return {"ok": True}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept(); q = state.hub.register()
    try:
        await websocket.send_json({"type": "init", **state.graph("live")})
        while True:
            await websocket.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    finally:
        state.hub.unregister(q)


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def smoke() -> dict:
    model, tok, graph, _ = build_live()
    store = MatrixStore(model)
    return {"source": "live", "n_matrices": len(graph["matrices"]),
            "n_connections": len(graph["connections"]),
            "sample_matrix": graph["matrices"][1]["name"],
            "global_absmax": store.global_stats()["absmax"]}


def main() -> None:
    global state
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--config", default="configs/gpt_tinystories.yaml")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    state = State(args)
    if args.check:
        print(smoke()); return
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
```

`Hub`/`_offer` 从旧 `server.py` 原样保留。`viz/runtime.py` 的 `build_tiny` 需能从模块导入（已存在）。

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py::test_server_smoke_builds_live_graph -v`
Expected: PASS

- [ ] **Step 5: 冒烟 CLI**

Run: `& ".venv\Scripts\python.exe" viz/server.py --check`
Expected: 打印含 `n_matrices` 的 dict，退出码 0。

- [ ] **Step 6: 提交**

```bash
git add viz/server.py tests/test_viz.py
git commit -m "feat: 重写 viz 后端 server.py（多源模型图/矩阵纹理/元素查询）"
```

---

## 里程碑 V2 · 前端场景

### Task 6: 前端外壳与 API 层

**Files:**
- Create: `viz/static/js/api.js`、`viz/static/js/colors.js`、`viz/static/js/main.js`
- Modify: `viz/static/index.html`、`viz/static/style.css`
- Test: `tests/test_viz_frontend.py`

**Interfaces:**
- Produces: `api.js` 导出 `getModel(source)`、`getGrid(name, source, tiles)`、`matrixPngUrl(name, source, tiles, norm)`、`getCell(name,i,j,source)`、`getPatch(name,r,c,h,w,source)`、`connectWS(onmsg)`。
- `colors.js` 导出 `divergingRGB(t)`（与 Python 同公式）、`colorFor(t)`（返回 `THREE.Color`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_viz_frontend.py
import json
import subprocess
from pathlib import Path

STATIC = Path("viz/static")
JS = ["api.js", "colors.js", "shelf.js", "matrix.js", "links.js", "panel.js", "cloud.js", "arch.js", "main.js"]


def test_static_files_exist():
    for f in JS:
        assert (STATIC / "js" / f).exists(), f"缺少 viz/static/js/{f}"
    assert (STATIC / "index.html").exists()
    assert not (STATIC / "app.js").exists(), "旧 app.js 应已删除"


def test_js_syntax_ok():
    for f in JS:
        p = STATIC / "js" / f
        if not p.exists():
            continue
        r = subprocess.run(["node", "--check", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f} 语法错误：{r.stderr}"
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py -v`
Expected: FAIL（缺少 js 文件 / app.js 仍在）

- [ ] **Step 3: 实现外壳**

`viz/static/index.html`（结构）：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>mini-llm-lab · 3D 模型浏览器</title>
  <link rel="stylesheet" href="/static/style.css" />
  <link rel="stylesheet" href="/static/vendor/katex/katex.min.css" />
  <script src="/static/vendor/katex/katex.min.js" defer></script>
  <script type="importmap">
    {"imports":{"three":"/static/vendor/three.module.js","three/addons/":"/static/vendor/jsm/"}}
  </script>
</head>
<body>
  <canvas id="scene"></canvas>
  <header id="topbar">
    <div class="brand">mini-llm-lab<span>3D 模型浏览器</span></div>
    <div class="viewswitch">
      <button data-mode="ckpt" class="active">真实模型检视</button>
      <button data-mode="live">实时训练</button>
      <button data-mode="arch">架构浏览器</button>
    </div>
    <div class="spacer"></div>
    <span class="badge" id="badge-model">–</span>
    <span class="badge" id="badge-device">–</span>
  </header>
  <aside id="left"></aside>
  <aside id="drawer" class="hidden"><div id="drawerBody"></div></aside>
  <footer id="bottom">
    <div id="player"><button id="btn-play">▶</button><button id="btn-step">⏭</button><span id="prompt"></span></div>
    <canvas id="loss" width="240" height="48"></canvas>
    <div id="hover"></div>
  </footer>
  <script type="module" src="/static/js/main.js"></script>
</body>
</html>
```

`viz/static/js/api.js`：

```js
const j = async (u) => (await fetch(u)).json();
const q = (o) => new URLSearchParams(o).toString();

export const getModel = (source = 'live') => j(`/api/model?${q({ source })}`);
export const getGrid = (name, source = 'live', tiles = 64) =>
  j(`/api/matrix/${encodeURIComponent(name)}/grid?${q({ source, tiles })}`);
export const getStats = (name, source = 'live') =>
  j(`/api/matrix/${encodeURIComponent(name)}/stats?${q({ source })}`);
export const getCell = (name, i, k, source = 'live') =>
  j(`/api/matrix/${encodeURIComponent(name)}/cell?${q({ source, i, j: k })}`);
export const getPatch = (name, r, c, h, w, source = 'live') =>
  j(`/api/matrix/${encodeURIComponent(name)}/patch?${q({ source, r, c, h, w })}`);
export const matrixPngUrl = (name, source = 'live', tiles = 64, norm = 'global') =>
  `/api/matrix/${encodeURIComponent(name)}/png?${q({ source, tiles, norm })}`;

export function connectWS(onmsg) {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onmessage = (e) => onmsg(JSON.parse(e.data));
  ws.onclose = () => setTimeout(() => connectWS(onmsg), 1200);
  return ws;
}
```

`viz/static/js/colors.js`：

```js
export function divergingRGB(t) {
  t = Math.max(-1, Math.min(1, t));
  if (t >= 0) return [0.16 + 0.84 * t, 0.08 + 0.16 * t, 0.08 + 0.16 * t];
  const a = Math.abs(t);
  return [0.08 + 0.16 * a, 0.16 + 0.28 * a, 0.16 + 0.84 * a];
}
export function colorFor(t) {
  const [r, g, b] = divergingRGB(t);
  return { r, g, b };
}
```

`viz/static/js/main.js`（本任务只建场景与模式切换骨架）：

```js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a12);
const canvas = document.getElementById('scene');
export const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
export const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
camera.position.set(0, 0, 18);
export const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const key = new THREE.DirectionalLight(0xbcd2ff, 1.1); key.position.set(6, 10, 8); scene.add(key);

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

function tick() { requestAnimationFrame(tick); controls.update(); renderer.render(scene, camera); }
tick();
```

`viz/static/style.css`：深色主题；`#drawer` 右抽屉（宽 420px，`.hidden{display:none}`）；
`#left` 左栏（宽 240px）；`#bottom` 底栏。给出与旧版一致的配色变量。

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz_frontend.py -v`
Expected: PASS（此任务需先把 `shelf/matrix/links/panel/cloud/arch.js` 建为空导出占位，
例如 `export {}`，以满足文件存在性；后续任务填充内容。）

- [ ] **Step 5: 手动冒烟**

Run: `& ".venv\Scripts\python.exe" viz/server.py`，浏览器开 `http://127.0.0.1:7861`
Expected: 深色 3D 空白场景可旋转缩放；顶栏三按钮可切换高亮。

- [ ] **Step 6: 提交**

```bash
git add viz/static/index.html viz/static/style.css viz/static/js tests/test_viz_frontend.py
git commit -m "feat: viz 前端外壳与 API/颜色模块"
```

---

### Task 7: 书架布局 `shelf.js`

**Files:**
- Create: `viz/static/js/shelf.js`
- Modify: `viz/static/js/main.js`
- Test: 手动冒烟 + `node --check`

**Interfaces:**
- Consumes: `graph`（Task 1）、`api.getModel`。
- Produces: `class Shelf { constructor(group, graph); layout(); expand(layerId|null); pickables(); }`
  - `layout()` 计算每个 layer 的 Y 位置与每块矩阵的本地变换，存到 `matrix.position` 思路的数据里。
  - 折叠态：每个 layer 一个薄板 `BoxGeometry`（宽 = 由矩阵数决定，高 0.35）。
  - 展开态：该层 9 个矩阵平面按 M1 单排沿 X 排列（间距 0.15），其余层保持薄板。

- [ ] **Step 1: 实现 `shelf.js`**

```js
import * as THREE from 'three';

export class Shelf {
  constructor(group, graph) {
    this.group = group;
    this.graph = graph;
    this.byLayer = new Map();
    this.trays = new Map();
    this.expanded = null;
    this._build();
  }
  _build() {
    const layers = this.graph.layers;
    const n = layers.length;
    const gap = 1.25;
    layers.forEach((L, idx) => {
      const mats = this.graph.matrices.filter((m) => m.layer === L.id);
      const tray = new THREE.Group();
      tray.position.y = ((n - 1) / 2 - idx) * gap;
      this.group.add(tray);
      this.trays.set(L.id, tray);
      this.byLayer.set(L.id, mats);
    });
  }
  expand(layerId) {
    this.expanded = this.expanded === layerId ? null : layerId;
    for (const [id, tray] of this.trays) {
      tray.userData.targetScale = id === this.expanded ? 1 : (this.expanded ? 0.0 : 1);
    }
  }
  pickables() { return []; }
}
```

实现者补：矩阵平面/薄板的实际 mesh 创建放在 Task 8（`matrix.js`），本任务先把
`tray`、Y 布局、`expand` 状态机与透明度/缩放插值写出来（在 `main.js` 的 `tick` 里对
`tray.userData.targetScale` 做 `lerp`）。

- [ ] **Step 2: 接入 `main.js`**

```js
import { getModel } from './api.js';
import { Shelf } from './shelf.js';
const modelGroup = new THREE.Group(); scene.add(modelGroup);
let shelf = null;
try {
  const graph = await getModel('live');
  shelf = new Shelf(modelGroup, graph);
} catch (e) { console.error(e); }
```

- [ ] **Step 3: 语法门**

Run: `node --check viz/static/js/shelf.js && node --check viz/static/js/main.js`
Expected: 无输出（退出码 0）

- [ ] **Step 4: 手动冒烟**：占位后可见 6 个（tiny 4 层 + emb + final）沿 Y 排列的托盘。

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/shelf.js viz/static/js/main.js
git commit -m "feat: viz 书架布局 shelf.js"
```

---

### Task 8: 矩阵板/立方块 `matrix.js`

**Files:**
- Create: `viz/static/js/matrix.js`
- Modify: `viz/static/js/shelf.js`、`viz/static/js/main.js`
- Test: 手动冒烟 + `node --check`

**Interfaces:**
- Produces:
  - `async function buildPlane(spec, source, tiles=64)`：`THREE.Mesh(PlaneGeometry, MeshBasicMaterial{map: TextureLoader(png)})`，`userData={kind:'matrix', name, spec}`。
  - `async function buildCubes(spec, source)`：用 `getGrid(name, source, 256)` 取元素值，`InstancedMesh` 立方块，每块颜色 = `divergingRGB(v/absmax)`。
  - 两者返回对象带 `spec`，供 `links/panel` 使用。
- `Shelf._build` 对每个矩阵：`spec.small ? buildCubes : buildPlane`；展开层显示，折叠层隐藏。

- [ ] **Step 1: 实现 `matrix.js`**

```js
import * as THREE from 'three';
import { getGrid, matrixPngUrl } from './api.js';
import { divergingRGB } from './colors.js';

const loader = new THREE.TextureLoader();

export function buildPlane(spec, source = 'live', tiles = 64, height = 0.9) {
  const aspect = spec.shape[0] / spec.shape[1];
  const geo = new THREE.PlaneGeometry(height / aspect, height);
  const tex = loader.load(matrixPngUrl(spec.name, source, tiles, 'global'));
  tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.LinearFilter;
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide }));
  mesh.userData = { kind: 'matrix', name: spec.name, spec, shape: spec.shape };
  return mesh;
}

export async function buildCubes(spec, source = 'live', height = 0.9) {
  const { values } = await getGrid(spec.name, source, 256);
  const [o, i] = spec.shape;
  const absmax = Math.max(...values.flat().map(Math.abs), 1e-6);
  const cell = height / Math.max(o, i);
  const geo = new THREE.BoxGeometry(cell * 0.92, cell * 0.92, cell * 0.92);
  const mat = new THREE.MeshBasicMaterial({ vertexColors: false });
  const mesh = new THREE.InstancedMesh(geo, mat, o * i);
  const m = new THREE.Matrix4(); const col = new THREE.Color();
  for (let r = 0; r < o; r++) for (let c = 0; c < i; c++) {
    const k = r * i + c;
    m.makeTranslation((c - (i - 1) / 2) * cell, ((o - 1) / 2 - r) * cell, 0);
    mesh.setMatrixAt(k, m);
    const [rr, gg, bb] = divergingRGB(values[r][c] / absmax);
    mesh.setColorAt(k, col.setRGB(rr, gg, bb));
  }
  mesh.userData = { kind: 'matrix', name: spec.name, spec, shape: spec.shape };
  return mesh;
}
```

- [ ] **Step 2: `Shelf` 使用它**

在 `Shelf._build` 里对展开层为每个矩阵创建（`small` 用 `buildCubes`，否则 `buildPlane`），
沿 X 排列：`x = (i - (n-1)/2) * (height + 0.15)`。薄板折叠态用一个半透明 `BoxGeometry`。

- [ ] **Step 3: 语法门**

Run: `node --check viz/static/js/matrix.js && node --check viz/static/js/shelf.js`
Expected: 退出码 0

- [ ] **Step 4: 手动冒烟**：点某层展开，看到 9 个矩阵板（norm 为细长条/立方块，q/k/v/o/w1/w2/w3 为热力图纹理）。

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/matrix.js viz/static/js/shelf.js
git commit -m "feat: viz 矩阵板与立方块 matrix.js"
```

---

## 里程碑 V3 · 拾取 / 抽屉 / 点云

### Task 9: 拾取与悬停

**Files:**
- Modify: `viz/static/js/main.js`、`viz/static/js/matrix.js`
- Test: 手动冒烟

**Interfaces:**
- Produces: `uvToCell(uv, shape) -> {i, j}`；`main.js` 注册 pointermove 做 raycast，
  命中 `userData.kind==='matrix'` 时显示 `#hover`（`(i,j) value`）；点击打开抽屉（Task 10）。

- [ ] **Step 1: 实现 `uvToCell`**（`matrix.js`）

```js
export function uvToCell(uv, shape) {
  const [o, i] = shape;
  const u = Math.min(0.999999, Math.max(0, uv.x));
  const v = Math.min(0.999999, Math.max(0, uv.y));
  const c = Math.floor(u * i);
  const r = Math.floor((1 - v) * o);
  return { i: r, j: c };
}
```

- [ ] **Step 2: `main.js` 拾取**

用 `Raycaster`，在 `pointermove` 上取 `intersectObjects(shelf.meshes, true)[0].uv`，
调用 `getCell(name, i, j, source)`（防抖 120ms）更新 `#hover` 文本。
`pointerdown/up` 位移 < 5px 视为点击 → 打开抽屉。

- [ ] **Step 3: 语法门**：`node --check viz/static/js/main.js`

- [ ] **Step 4: 手动冒烟**：悬停矩阵板底栏显示行列与数值；点击打开抽屉。

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/main.js viz/static/js/matrix.js
git commit -m "feat: viz 矩阵拾取与悬停取值"
```

---

### Task 10: 右抽屉 `panel.js`

**Files:**
- Create: `viz/static/js/panel.js`
- Modify: `viz/static/js/main.js`、`viz/static/index.html`（抽屉内加 `<canvas id="matrix2d">`）
- Test: 手动冒烟 + `node --check`

**Interfaces:**
- Produces: `class Panel { constructor(el); async show(spec, source); close(); }`
  - 2D canvas 画 `getGrid(name, source, 192)` 的发散色阶大图。
  - 块↔元素切换：按钮调用 `getGrid(name, source, 256)` 或 `getPatch`。
  - 图例：`divergingRGB` 渐变条 + vmin/vmax（`getStats`）。
  - 公式/源码：若 `arch_spec` 提供则 KaTeX 渲染（架构模式）；否则显示矩阵形状/role。

- [ ] **Step 1: 实现 `panel.js`**：canvas 逐格 `fillRect`，颜色用 `divergingRGB`；标题、shape、role、图例、`关闭` 按钮。

- [ ] **Step 2: `main.js` 接线**：点击矩阵 → `panel.show(spec, source)`；`#drawer` 去掉 `.hidden`。

- [ ] **Step 3: 语法门**：`node --check viz/static/js/panel.js`

- [ ] **Step 4: 手动冒烟**：抽屉显示大热力图、图例与形状信息；关闭按钮工作。

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/panel.js viz/static/index.html viz/static/js/main.js
git commit -m "feat: viz 右抽屉 2D 矩阵详情 panel.js"
```

---

### Task 11: 神经元点云 `cloud.js`

**Files:**
- Create: `viz/static/js/cloud.js`
- Modify: `viz/static/js/main.js`
- Test: 手动冒烟 + `node --check`

**Interfaces:**
- Consumes: `GET /api/neurons`（已有）。
- Produces: `async function showCloud(group, matrix, source, n=160)`：清空 group，
  用 `THREE.Points` 画输入/输出神经元，`LineSegments2` 按 |w| 分桶画连线（沿用旧 app.js 逻辑）。

- [ ] **Step 1: 实现 `cloud.js`**（把旧 `app.js` 的 `buildNeurons/cluster` 移植为模块函数）。

- [ ] **Step 2: `main.js` 接线**：抽屉里加「点云」切换按钮，调用 `showCloud`。

- [ ] **Step 3: 语法门**：`node --check viz/static/js/cloud.js`

- [ ] **Step 4: 手动冒烟**：切到点云可见神经元点与连线。

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/cloud.js viz/static/js/main.js
git commit -m "feat: viz 神经元点云 cloud.js"
```

---

## 里程碑 V4 · 连接与动画

### Task 12: 结构连线 `links.js`（L1）

**Files:**
- Create: `viz/static/js/links.js`
- Modify: `viz/static/js/main.js`
- Test: 手动冒烟 + `node --check`

**Interfaces:**
- Produces: `class Links { constructor(group, graph); rebuild(shelf); update(values); }`
  - `kind==='inner'`：同一托盘内矩阵中心连线（细，绿）。
  - `kind==='spine'`：相邻托盘中心连线（粗，蓝），线粗 = `values[weight_from].norm` 归一化。

- [ ] **Step 1: 实现 `links.js`**：`LineSegments2`/`Line2`（vendored），每帧根据 `values` 调 `linewidth`。

- [ ] **Step 2: `main.js` 接线**：`new Links(group, graph).rebuild(shelf)`；`tick/token` 时 `links.update(values)`。

- [ ] **Step 3: 语法门**：`node --check viz/static/js/links.js`

- [ ] **Step 4: 手动冒烟**：层间蓝色主干与层内绿色连线随训练变粗变细。

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/links.js viz/static/js/main.js
git commit -m "feat: viz 结构连线 links.js（L1）"
```

---

### Task 13: 数据流带与推理播放器（L3）

**Files:**
- Modify: `viz/static/js/links.js`、`viz/static/js/main.js`、`viz/static/index.html`（`#player` 已有）
- Test: 手动冒烟 + `node --check`

**Interfaces:**
- Produces: `class Flow { constructor(group, graph); setToken(values); reset(); }`
  - 相邻托盘间画半透明四边形"带子"，颜色/透明度由 `values[rep].act` 决定。
- `main.js` 播放器：`▶/⏭` 调 `POST /api/infer`；WS `token` 消息驱动 `flow.setToken`、`panel` 注意力与底栏文本。

- [ ] **Step 1: 实现 `Flow` 与播放器接线**：`token` 消息为 `{token,text,values,attn}`；
  底栏显示生成文本，`flow.setToken(values)` 点亮流带；`attn` 画到底栏小 canvas。

- [ ] **Step 2: 语法门**：`node --check viz/static/js/main.js && node --check viz/static/js/links.js`

- [ ] **Step 3: 手动冒烟**：点 ▶ 后逐 token 生成，流带随之点亮；⏭ 单步。

- [ ] **Step 4: 提交**

```bash
git add viz/static/js/links.js viz/static/js/main.js viz/static/index.html
git commit -m "feat: viz 数据流带与推理播放器（L3）"
```

---

### Task 14: 神经元连线（L2，仅下钻）

**Files:**
- Modify: `viz/static/js/cloud.js`、`viz/static/js/panel.js`
- Test: 手动冒烟 + `node --check`

**Interfaces:**
- 在 `cloud.js` 增加 `showNeuronLinks(group, matrix, source, top=800)`：从 `/api/neurons`
  取 edges，用红/蓝区分正负、线粗 ∝ |w|，仅当矩阵 `small===true` 时允许调用。

- [ ] **Step 1: 实现 `showNeuronLinks`** 与抽屉按钮（`spec.small` 才显示）。

- [ ] **Step 2: 语法门**：`node --check viz/static/js/cloud.js`

- [ ] **Step 3: 手动冒烟**：展开 norm（小矩阵）后可开神经元连线。

- [ ] **Step 4: 提交**

```bash
git add viz/static/js/cloud.js viz/static/js/panel.js
git commit -m "feat: viz 神经元连线（L2，仅小矩阵下钻）"
```

---

## 里程碑 V5 · 训练联动与架构浏览器

### Task 15: 实时训练联动

**Files:**
- Modify: `viz/static/js/main.js`、`viz/static/js/shelf.js`、`viz/static/index.html`（底栏加训练按钮/统计）
- Test: 手动冒烟 + `node --check`

**Interfaces:**
- 顶栏「实时训练」模式：`getModel('live')` 重建 `Shelf`，切到 `source='live'`。
- WS `tick` → `shelf.updateValues(values)` 按 `norm` 给矩阵板上色/缩放；loss 画到 `#loss`。
- 左栏「开始/停止」调 `POST /api/train/start|stop`。

- [ ] **Step 1: 实现 `Shelf.updateValues(values)`**：对每个矩阵 mesh，把 `values[name].norm`
  归一化后调制 `material.color` 或 `opacity`；`main.js` 处理 `tick`。

- [ ] **Step 2: 语法门**：`node --check viz/static/js/shelf.js && node --check viz/static/js/main.js`

- [ ] **Step 3: 手动冒烟**：切「实时训练」→ 开始训练 → loss 下降、矩阵板颜色随权重变化。

- [ ] **Step 4: 提交**

```bash
git add viz/static/js/shelf.js viz/static/js/main.js viz/static/index.html
git commit -m "feat: viz 实时训练联动"
```

---

### Task 16: 架构浏览器 `arch.js`

**Files:**
- Create: `viz/static/js/arch.js`
- Modify: `viz/static/js/main.js`、`viz/static/js/panel.js`
- Test: 手动冒烟 + `node --check`

**Interfaces:**
- 移植旧 `app.js` 的 `loadArchList/loadArch/renderArch/showArchLayer`，用 `Shelf` 的简化版
  （每行一个层板）渲染；点击层在抽屉里用 KaTeX 显示公式/源码（沿用 `appendModule` 逻辑）。

- [ ] **Step 1: 实现 `arch.js`**：`export async function showArch(group, panelEl, modelId)`。

- [ ] **Step 2: `main.js` 接线**：`data-mode="arch"` 时调用；隐藏权重书架。

- [ ] **Step 3: 语法门**：`node --check viz/static/js/arch.js`

- [ ] **Step 4: 手动冒烟**：选模型 → 逐层板 → 点层抽屉出公式/源码。

- [ ] **Step 5: 提交**

```bash
git add viz/static/js/arch.js viz/static/js/main.js viz/static/js/panel.js
git commit -m "feat: viz 架构浏览器 arch.js"
```

---

## 里程碑 V6 · 文档与清理

### Task 17: 重写文档 + 删除旧文件 + 终验

**Files:**
- Modify: `docs/07-visualization.md`
- Delete: `viz/topology.py`、`viz/static/app.js`
- Test: 全量 pytest

- [ ] **Step 1: 重写 `docs/07-visualization.md`**：按 spec 的界面/端点/协议/选型表描述新 viz，
  更新启动命令（`--ckpt` 检视真实模型）。

- [ ] **Step 2: 删除旧文件**

Run: `git rm viz/topology.py viz/static/app.js`
Expected: 删除成功

- [ ] **Step 3: 全量测试**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_viz.py tests/test_viz_frontend.py -v`
Expected: 全 PASS

- [ ] **Step 4: 冒烟**

Run: `& ".venv\Scripts\python.exe" viz/server.py --check` 再 `... viz/server.py`
Expected: 检视/训练/架构三模式可用，展开/抽屉/悬停/推理/训练均工作。

- [ ] **Step 5: 提交**

```bash
git add docs/07-visualization.md viz/tests
git add -A
git commit -m "docs: 重写 viz 可视化文档并清理旧文件"
```

---

## Self-Review 记录

- **Spec 覆盖**：界面外壳（T6）、书架展开 C1+C2（T7）、矩阵 M1+M2（T8/T10）、
  颜色与图例（T4/T6/T10）、悬停取值（T9）、L1/L2/L3（T12/T14/T13）、
  实时训练（T15）、神经元点云（T11）、架构浏览器（T16）、真实 checkpoint（T4/T5）、
  文档（T17）——均有对应任务。
- **占位扫描**：无 TBD/TODO；前端视觉任务给出可执行代码与手动冒烟判据。
- **类型一致性**：矩阵名统一带 `.weight`；`Shelf.updateValues`、`Links.update`、
  `Flow.setToken` 形参均为 `{matrixName: {norm|act|delta}}`；`getGrid` 返回 `{shape,grid,values}` 前后端一致。
- **已知取舍**：前端无 JS 单测框架，用 `node --check` + 手动冒烟作为门；真实 checkpoint
  单测默认 `skip`，避免 2.2GB 加载拖慢 CI。
