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
