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
from viz.runtime import build_tiny, stream_generate
from viz.datasets import (list_datasets, resolve_spec, load_pretrain,
                          PretrainBatches, SFTBatches)
from viz.train_engine import SimpleEngine, HifiTrainer, HifiSFTTrainer
from viz.stats import ActivationRecorder, WeightTracker, snapshot_matrices
from viz.weights import MatrixStore, load_checkpoint

from src.config import load_config
from src.model import GPT
from src.tokenizer import QwenTokenizer

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


def clamp_sft_len(configured: int, model) -> int:
    """把配置的 SFT 最大长度收窄到模型 ctx_len，避免 RoPE 越界。

    build_tiny 固定 ctx_len=64，而 --sft-max-len 默认 256；若不收窄，
    补长后的 batch 会让 apply_rope 的 cos[offset:offset+t] 取空而抛错。
    """
    ctx = getattr(getattr(model, "cfg", None), "ctx_len", None)
    if ctx is None:
        return int(configured)
    return min(int(configured), int(ctx))


def validate_train_spec(dataset: str, mode: str) -> dict:
    """训练启动前的轻量校验。

    未知 id 抛 KeyError、方式/类型不符抛 ValueError、文件缺失抛
    FileNotFoundError、数据集为空抛 ValueError；由 API 层转成 400，
    避免拖到训练线程里变成 500。
    """
    if not dataset:
        raise ValueError("数据集不能为空")
    spec = resolve_spec(dataset, mode)
    if not spec.get("builtin") and not Path(spec["path"]).exists():
        raise FileNotFoundError(f"数据集文件不存在：{spec['path']}")
    return spec


# ----------------------------------------------------------------------------
# 数据源构建：实时小模型 / 真实 checkpoint
# ----------------------------------------------------------------------------
class CkptUnavailable(RuntimeError):
    pass


_QWEN = None


def _qwen_tokenizer():
    global _QWEN
    if _QWEN is None:
        _QWEN = QwenTokenizer.load("data/tokenizer/qwen2.5-0.5b",
                                   add_image_token=False)
    return _QWEN


def build_live():
    """tiny 结构 + Qwen 词表；数据集只影响训练数据，不影响模型结构。"""
    from viz.datasets import DATASETS
    from viz.runtime import CORPUS
    tok = _qwen_tokenizer()
    model = build_tiny(tok.vocab_size, "cpu")
    graph = build_graph(model, source="live")
    return model, tok, graph, (DATASETS["corpus_qwen"],)


class State:
    def __init__(self, args) -> None:
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hub = Hub()
        self.tracker = WeightTracker()
        self.recorder = ActivationRecorder()
        self.has_ckpt = bool(args.ckpt)
        self.mode = "weights" if self.has_ckpt else "live"
        self._live = None   # (model, tok, graph)
        self._ckpt = None   # (model, tok, graph)
        self._scratch = None  # (model, tok, graph)
        self._stores: dict[str, MatrixStore] = {}
        self.trainer = None
        self.train_engine = None
        self._train_meta: dict = {}
        self._train_ctx = None  # (model, graph)
        self.train_target = "live"
        self._train_finished = False
        self._last_saved: str | None = None

    def ensure_live(self):
        if self._live is None:
            model, tok, graph, _ = build_live()
            model = model.to(self.device)
            self._live = (model, tok, graph)
            self.recorder.attach(model)
        return self._live

    def ensure_ckpt(self):
        if not self.has_ckpt:
            raise CkptUnavailable(
                "服务端未加载 checkpoint，请用 --ckpt <path> 启动后重试。")
        if self._ckpt is None:
            model, cfg = load_checkpoint(self.args.ckpt, self.args.config)
            model.to(self.device)
            graph = build_graph(model, source="ckpt")
            tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
            self._ckpt = (model, tok, graph)
            self.recorder.attach(model)
        return self._ckpt

    def ensure_scratch(self):
        if self._scratch is None:
            cfg = load_config(self.args.config)
            model = GPT(cfg.model).to(self.device)
            graph = build_graph(model, source="scratch")
            tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
            self._scratch = (model, tok, graph)
            self.recorder.attach(model)
        return self._scratch

    def source(self, which: str):
        if which == "scratch":
            return self.ensure_scratch()
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

    def start_train(self, target="live", mode="pretrain", dataset="corpus_qwen",
                    engine="simple", params=None):
        params = params or {}
        if target not in ("live", "ckpt", "scratch"):
            target = "live"
        if target == "ckpt" and not self.has_ckpt:
            raise CkptUnavailable("服务端未加载 checkpoint，请用 --ckpt <path> 启动后重试。")
        spec = validate_train_spec(dataset, mode)
        if engine not in ("simple", "hifi"):
            engine = "simple"
        if self.trainer:
            self.trainer.stop()
        # 教学注释：在加载新数据/构造新引擎之前就清空旧 trainer 与上下文，
        # 否则中途失败会留下上一轮已停止的 trainer，之后 stop 会把它按新
        # 文件名保存，写错的内容。
        self.trainer = None
        self._train_ctx = None
        self._train_meta = {}
        self._train_finished = False
        model, tok, graph = self.source(target)

        max_steps = max(1, int(params.get("max_steps", 200)))
        lr = float(params.get("lr", getattr(self.args, "lr", 3e-4)))
        batch_size = max(1, int(params.get("batch_size", 8)))
        grad_accum = max(1, int(params.get("grad_accum", 1)))
        block = int(self.args.block)
        sft_len = clamp_sft_len(self.args.sft_max_len, model)

        if mode == "pretrain":
            data = load_pretrain(spec, tok)
            make_batches = lambda: PretrainBatches(data, batch_size, block, self.device)
            hifi_data = data
        else:
            path = spec["path"]
            pad_id = tok.special_id("pad")
            make_batches = lambda: SFTBatches(path, batch_size, sft_len,
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
                                              max_len=sft_len,
                                              device=self.device, on_step=self._on_step,
                                              on_done=self._on_done)
            else:
                self.trainer = HifiTrainer(model, cfg, hifi_data, tokenizer=tok,
                                           device=self.device, ctx_len=block,
                                           on_step=self._on_step, on_done=self._on_done)
        else:
            batches = make_batches()
            self.trainer = SimpleEngine(model, batches.next, self.device, self._on_step,
                                        lr=lr, max_steps=max_steps, grad_accum=grad_accum,
                                        weight_decay=0.05, warmup_steps=0,
                                        on_done=self._on_done)
        self.train_target = target
        self.train_engine = engine
        self.trainer.start()

    def _finish(self, trainer):
        """保存本次 trainer 并广播一次 training:false，保证只发生一次。"""
        path = self._save_path()
        try:
            trainer.save(path)
        except Exception as e:  # noqa: BLE001
            print(f"[viz] 保存失败：{e}")
            path = None
        if self.trainer is trainer:
            self.trainer = None
        self._train_finished = True
        self._last_saved = path
        self.hub.publish({"type": "status", "training": False, "saved": path})
        return path

    def _on_done(self):
        """训练线程自然跑完时的回调：保存并通知前端。"""
        trainer = self.trainer
        if trainer is None:
            return
        self._finish(trainer)

    def stop_train(self):
        trainer = self.trainer
        if not trainer:
            # 没有在跑的训练：仅在从未结束时补一条停止状态，避免与自然
            # 结束的 on_done 重复广播。
            if not self._train_finished:
                self.hub.publish({"type": "status", "training": False, "saved": None})
            return self._last_saved
        trainer.stop()
        # stop() 会 join 训练线程；若它其实已自然结束并由 on_done 保存/清空，
        # 则直接返回，不再二次保存或广播。
        if self.trainer is not trainer:
            return self._last_saved
        return self._finish(trainer)

    def _save_path(self):
        import datetime
        meta = self._train_meta
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = ROOT / "out" / "viz"
        out.mkdir(parents=True, exist_ok=True)
        return str(out / f"{meta.get('target','live')}-{meta.get('mode','pretrain')}"
                          f"-{meta.get('dataset','corpus_qwen')}-{ts}.pt")

    def _on_step(self, step, loss, lr):
        model, graph = self._train_ctx
        vals = snapshot_matrices(model, graph["matrices"], tracker=self.tracker)
        msg = {"type": "tick", "step": step,
               "loss": loss if math.isfinite(loss) else 0.0, "lr": lr,
               "target": self.train_target, "values": vals}
        msg.update(self._train_meta)
        self.hub.publish(msg)
        self.tracker.capture(model)

    def infer(self, which, prompt, max_new_tokens=40, temperature=0.9, top_k=20,
              top_p=0.95, top_n=8, seed=None):
        model, tok, graph = self.source(which)
        self.recorder.clear()

        def on_token(text, token_id, attn_pack, topn, kv):
            vals = {m["name"]: {"act": self.recorder.for_matrix(m["name"])}
                    for m in graph["matrices"]}
            self.hub.publish({"type": "token", "token": (text[-1] if text else ""),
                              "id": token_id, "text": text, "values": vals,
                              "attn": attn_pack, "topn": topn, "kv": kv})

        stream_generate(model, tok, prompt, on_token, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_k=top_k, top_p=top_p,
                        top_n=top_n, attn_layer=0, seed=seed)


state: State | None = None


@asynccontextmanager
async def lifespan(_app):
    state.hub.bind(asyncio.get_running_loop())
    yield


app = FastAPI(title="mini-llm-lab 3D 可视化", lifespan=lifespan)


@app.exception_handler(CkptUnavailable)
async def _ckpt_unavailable(_req, exc):
    return JSONResponse({"error": str(exc)}, status_code=400)


# ----------------------------------------------------------------------------
# HTTP 端点
# ----------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/model")
async def api_model(source: str = "live"):
    graph = await asyncio.to_thread(state.graph, source)
    return JSONResponse(graph)


@app.get("/api/datasets")
async def api_datasets():
    return JSONResponse(list_datasets())


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
async def api_train_start(payload: dict | None = None):
    body = payload or {}
    target = str(body.get("target", "live"))
    mode = str(body.get("mode", "pretrain"))
    dataset = str(body.get("dataset", "corpus_qwen"))
    engine = str(body.get("engine", "simple"))
    params = body.get("params") or {}
    try:
        validate_train_spec(dataset, mode)
    except KeyError:
        return JSONResponse({"error": f"未知数据集：{dataset}"}, status_code=400)
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: state.start_train(target, mode, dataset, engine, params))
    total_steps = int(params.get("max_steps", 200))
    state.hub.publish({"type": "status", "training": True, "target": target,
                       "mode": mode, "dataset": dataset, "engine": engine,
                       "total_steps": total_steps})
    return {"ok": True, "target": target, "mode": mode, "dataset": dataset,
            "engine": engine, "total_steps": total_steps}


@app.post("/api/train/stop")
async def api_train_stop():
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, state.stop_train)
    return {"ok": True, "saved": path}


@app.post("/api/infer")
async def api_infer(payload: dict):
    which = str(payload.get("source", "live"))
    seed = payload.get("seed")
    seed = int(seed) if seed not in (None, "") else None
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, state.infer, which,
        str(payload.get("prompt", "人工智能")),
        int(payload.get("max_new_tokens", 40)),
        float(payload.get("temperature", 0.9)),
        int(payload.get("top_k", 20)),
        float(payload.get("top_p", 0.95)),
        int(payload.get("top_n", 8)),
        seed,
    )
    return {"ok": True}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    q = state.hub.register()
    try:
        await websocket.send_json(
            {"type": "init", "has_ckpt": state.has_ckpt,
             "source": "ckpt" if state.mode == "weights" else "live"})
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
    model = build_tiny(256, "cpu")
    store = MatrixStore(model)
    graph = build_graph(model, source="live")
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
    ap.add_argument("--config", default="configs/gpt_fineweb.yaml")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--sft-max-len", type=int, default=256)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    state = State(args)
    if args.check:
        print(smoke())
        return
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
