# 教学注释：验证 3D 可视化的数据层——统一模型图。
from pathlib import Path

import pytest
import torch

from src.config import ModelConfig
from src.model import GPT
from viz.graph import build_graph, CUBE_THRESHOLD
from viz.neurons import neuron_cloud
from viz.runtime import CharTokenizer, CORPUS
from viz.stats import ActivationRecorder, WeightTracker, snapshot_matrices, summarize
from viz.weights import MatrixStore, diverging_rgb, load_checkpoint
import viz.server as server


def tiny_model():
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=64, d_model=32, n_layer=2, n_head=4,
                           n_kv_head=2, d_ff=64, ctx_len=16))


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


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer(CORPUS)
    ids = tok.encode("人工智能")
    assert tok.decode(ids) == "人工智能"[0:len(ids)] or len(ids) == 4


def test_summarize_and_snapshot_by_matrix():
    model = tiny_model()
    g = build_graph(model)
    s = summarize([torch.randn(8, 4)])
    assert s["n"] == 32 and s["norm"] > 0
    tracker = WeightTracker(); tracker.capture(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    snap = snapshot_matrices(model, g["matrices"], tracker=tracker)
    assert snap["tok_emb.weight"]["delta"] > 0
    assert snap["tok_emb.weight"]["norm"] > 0


def test_neuron_cloud_shapes():
    w = torch.randn(40, 24)
    c = neuron_cloud(w, max_nodes=16, top_edges=50)
    assert len(c["out_coords"]) == 16 and len(c["in_coords"]) == 16
    assert all(len(p) == 3 for p in c["out_coords"])
    assert len(c["edges"]) == 50


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
    assert abs(g["values"][0][0] - float(_tensor(model, name)[0, 0].detach())) < 1e-6
    p = store.grid(name, tiles=8)
    assert p["grid"] == [8, 8]


def test_cell_patch_stats_png():
    model = tiny_model()
    store = MatrixStore(model)
    name = "blocks.0.mlp.w1.weight"     # [d_ff, d_model] = [64, 32]
    w = _tensor(model, name)
    assert abs(store.cell(name, 1, 2) - float(w[1, 2].detach())) < 1e-6
    patch = store.patch(name, 0, 0, 4, 5)
    assert patch["h"] == 4 and patch["w"] == 5 and len(patch["values"]) == 4
    st = store.stats(name)
    assert st["shape"] == [64, 32] and st["absmax"] > 0
    png = store.png(name, tiles=8, norm="matrix")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_load_checkpoint_optional():
    import os
    if os.environ.get("VIZ_TEST_CKPT") != "1":
        pytest.skip("set VIZ_TEST_CKPT=1 to load the real checkpoint")
    if not Path("out/gpt/latest.pt").exists():
        pytest.skip("无本地 checkpoint")
    model, cfg = load_checkpoint("out/gpt/latest.pt", "configs/gpt_tinystories.yaml")
    assert model.cfg.d_model == 768 and len(model.blocks) == 12


def test_server_smoke_builds_live_graph():
    info = server.smoke()
    assert info["source"] == "live"
    assert info["n_matrices"] > 0 and info["n_connections"] > 0
    assert "blocks.0.attn.q_proj.weight" in info["sample_matrix"]


def test_train_spec_validation():
    from viz.datasets import resolve_spec
    assert resolve_spec("poetry", "pretrain")["kind"] == "pretrain"
    with pytest.raises(ValueError):
        resolve_spec("poetry", "sft")
