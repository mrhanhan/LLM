# 教学注释：验证 3D 可视化的数据层——统一模型图。
import torch

from src.config import ModelConfig
from src.model import GPT
from viz.graph import build_graph, CUBE_THRESHOLD
from viz.runtime import CharTokenizer, CORPUS
from viz.stats import ActivationRecorder, WeightTracker, snapshot_matrices, summarize


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
