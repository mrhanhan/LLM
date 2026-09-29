"""mini-llm-lab 3D 可视化后端（重写版）。

一个进程同时负责：
  1. 实时训练字符级小模型，把每步权重变化推给浏览器；
  2. 加载真实 checkpoint 做静态权重检视；
  3. 逐 token 推理，把激活/注意力变化推给浏览器；
  4. 暴露神经元点云数据（权重矩阵 PCA 到 3D）。

启动：
    & ".venv\\Scripts\\python.exe" viz/server.py
然后浏览器打开 http://127.0.0.1:7861
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from viz import arch as arch_mod
from viz.graph import build_graph
from viz.neurons import neuron_cloud
from viz.runtime import CharTokenizer, LiveTrainer, build_tiny, stream_generate
from viz.stats import ActivationRecorder, WeightTracker, snapshot_matrices
from viz.weights import MatrixStore, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"


# ----------------------------------------------------------------------------
# 广播中心：后台训练线程 -> 所有 WebSocket 客户端
# ----------------------------------------------------------------------------
class Hub:
    def __init__(self) -> None:
        self.queues: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last_step: dict | None = None
        self.last_token: dict | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.queues.add(q)
        return q

    def unregister(self, q: asyncio.Queue) -> None:
        self.queues.discard(q)

    def publish(self, msg: dict) -> None:
        """线程安全：可从训练线程调用。"""
        if msg.get("type") == "tick":
            self.last_step = msg
        elif msg.get("type") == "token":
            self.last_token = msg
        if self.loop is None:
            return
        for q in list(self.queues):
            try:
                self.loop.call_soon_threadsafe(_offer, q, msg)
            except RuntimeError:
                pass


def _offer(q: asyncio.Queue, msg: dict) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(msg)


# ----------------------------------------------------------------------------
# 数据源构建：实时小模型 / 真实 checkpoint
# ----------------------------------------------------------------------------
def build_live(*, lr: float = 3e-3, batch: int = 16, block: int = 48):
    """返回 (model, tokenizer, graph, 训练超参)。"""
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
        self._ckpt = None   # (model, tok, graph)
        self._stores: dict[str, MatrixStore] = {}
        self.trainer: LiveTrainer | None = None

    def ensure_live(self):
        if self._live is None:
            model, tok, graph, (lr, batch, block) = build_live(
                lr=self.args.lr, batch=self.args.batch, block=self.args.block)
            model = model.to(self.device)
            self._live = (model, tok, graph)
            self.recorder.attach(model)
            self.trainer = LiveTrainer(tok, self.device, on_step=self._on_step,
                                       lr=lr, batch_size=batch, block_size=block, model=model)
        return self._live

    def ensure_ckpt(self):
        if self._ckpt is None:
            model, cfg = load_checkpoint(self.args.ckpt, self.args.config)
            model.to(self.device)
            graph = build_graph(model, source="ckpt")
            from src.tokenizer import QwenTokenizer
            tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
            self._ckpt = (model, tok, graph)
            self.recorder.attach(model)
        return self._ckpt

    def source(self, which: str):
        return self.ensure_live() if which == "live" else self.ensure_ckpt()

    def store(self, which: str) -> MatrixStore:
        if which not in self._stores:
            self._stores[which] = MatrixStore(self.source(which)[0])
        return self._stores[which]

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


state: State | None = None


@asynccontextmanager
async def lifespan(_app):
    state.hub.bind(asyncio.get_running_loop())
    yield


app = FastAPI(title="mini-llm-lab 3D 可视化", lifespan=lifespan)


# ----------------------------------------------------------------------------
# HTTP 端点
# ----------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/model")
async def api_model(source: str = "live"):
    if source == "ckpt":
        graph = await asyncio.to_thread(state.graph, source)
    else:
        graph = state.graph(source)
    return JSONResponse(graph)


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
    data["matrix"] = matrix
    data["shape"] = list(mod.weight.shape)
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


@app.get("/compare")
async def compare_page():
    f = ROOT / "docs" / "models" / "compare.html"
    if not f.exists():
        return JSONResponse({"error": "compare.html 尚未生成，请运行 scripts/gen_compare_html.py"},
                            status_code=404)
    return FileResponse(f)


@app.post("/api/train/start")
async def api_train_start():
    state.ensure_live()
    state.tracker.capture(state._live[0])
    state.trainer.start()
    state.hub.publish({"type": "status", "training": True})
    return {"ok": True}


@app.post("/api/train/stop")
async def api_train_stop():
    if state.trainer:
        state.trainer.stop()
    state.hub.publish({"type": "status", "training": False})
    return {"ok": True}


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
    await websocket.accept()
    q = state.hub.register()
    try:
        await websocket.send_json(
            {"type": "init", "source": "ckpt" if state.mode == "weights" else "live"})
        while True:
            await websocket.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    finally:
        state.hub.unregister(q)


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ----------------------------------------------------------------------------
# 冒烟与入口
# ----------------------------------------------------------------------------
def smoke() -> dict:
    model, tok, graph, _ = build_live()
    store = MatrixStore(model)
    attn = next(m for m in graph["matrices"] if m["role"] == "attn")
    return {"source": "live", "n_matrices": len(graph["matrices"]),
            "n_connections": len(graph["connections"]),
            "sample_matrix": attn["name"],
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
        print(smoke())
        return
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
