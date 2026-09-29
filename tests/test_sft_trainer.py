import numpy as np
from src.config import ModelConfig, TrainConfig
from src.model import GPT
from src.tokenizer import CharTokenizer
from src.sft_data import SFTDataset, save_sft_npz
from src.trainer import SFTTrainer


def _ds(tmp_path):
    tok = CharTokenizer.train(["你好世界答问abc"])
    examples = []
    for _ in range(8):
        examples.append({"messages": [{"role": "user", "content": "你好"},
                                      {"role": "assistant", "content": "你好答"}]})
    p = tmp_path / "t.npz"
    save_sft_npz(tok, examples, str(p), max_len=64)
    return tok, SFTDataset(str(p))


def test_sft_trainer_overfits_tiny_set(tmp_path):
    tok, ds = _ds(tmp_path)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layer=2, n_head=4,
                      n_kv_head=2, d_ff=128, ctx_len=64)
    tcfg = TrainConfig(batch_size=4, grad_accum=1, lr=3e-3, warmup_steps=1,
                       max_steps=150, eval_interval=10_000, save_interval=10_000,
                       log_interval=10_000, out_dir=str(tmp_path / "out"))
    model = GPT(cfg)
    tr = SFTTrainer(model, tcfg, ds, tok, max_len=64, val_ds=ds, device="cpu")
    tr.train()
    assert tr._last_loss < 0.3, f"SFT 过拟合失败：loss={tr._last_loss:.3f}"
