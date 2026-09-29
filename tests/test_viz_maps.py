import torch

from src.config import ModelConfig
from src.model import GPT
from viz.maps import SEQUENTIAL_HI, SEQUENTIAL_LO, matrix_map, pool_map, sequential_rgb
from viz.maps import GradCache
from viz.stats import snapshot_gradients
from viz.graph import build_graph


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
