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


def test_simple_engine_on_done_fires_once_on_completion():
    import time
    model = _model()
    done = []

    def batch_fn():
        x = torch.randint(0, 64, (2, 8))
        return x, x.clone()

    eng = SimpleEngine(model, batch_fn, "cpu", lambda *a: None, lr=1e-3,
                       max_steps=3, grad_accum=1, log_interval=1,
                       on_done=lambda: done.append(1))
    eng.start()
    deadline = time.time() + 10
    while time.time() < deadline and not done:
        time.sleep(0.01)
    eng.stop()  # 自然跑完后再 stop，也不应再次触发 on_done
    assert done == [1]
    assert eng.step == 3


def test_simple_engine_stop_suppresses_on_done():
    import time
    model = _model()
    done = []
    eng = SimpleEngine(model, lambda: (torch.zeros(2, 8, dtype=torch.long),) * 2,
                       "cpu", lambda *a: None, max_steps=100000, log_interval=100000,
                       on_done=lambda: done.append(1))
    eng.start()
    time.sleep(0.05)
    eng.stop()
    assert done == []
    assert eng.step < 100000
