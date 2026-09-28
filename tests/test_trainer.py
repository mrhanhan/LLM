# 教学注释：用极小的假数据验证——训练会造成参数变化、checkpoint 能续训、
# metrics.jsonl 正常写入。
import os
import numpy as np
import torch
from src.config import ModelConfig, TrainConfig
from src.model import GPT
from src.trainer import Trainer


def make_data(tmp_path, n=5000):
    p = tmp_path / "d.bin"
    rng = np.random.randint(0, 100, size=n).astype(np.uint32)
    rng.tofile(str(p))
    return np.memmap(str(p), dtype=np.uint32, mode="r")


def test_trainer_runs_and_resumes(tmp_path):
    cfg = ModelConfig(vocab_size=100, d_model=64, n_layer=2, n_head=4,
                      n_kv_head=2, d_ff=128, ctx_len=32)
    model = GPT(cfg)
    tcfg = TrainConfig(batch_size=2, grad_accum=1, lr=1e-2, warmup_steps=2,
                       max_steps=6, eval_interval=3, eval_iters=2,
                       save_interval=6, log_interval=1, out_dir=str(tmp_path / "out"),
                       dtype="fp32", compile=False)
    data = make_data(tmp_path)
    trainer = Trainer(model, tcfg, data, data)
    before = model.tok_emb.weight.detach().clone()
    trainer.train()
    after = model.tok_emb.weight.detach()
    assert not torch.allclose(before, after), "参数应被更新"
    assert os.path.exists(os.path.join(tcfg.out_dir, "latest.pt"))
    assert os.path.exists(os.path.join(tcfg.out_dir, "metrics.jsonl"))

    # 续训：从 checkpoint 恢复，步数应继续
    tcfg2 = TrainConfig(**{**tcfg.__dict__, "max_steps": 9})
    model2 = GPT(cfg)
    trainer2 = Trainer(model2, tcfg2, data, data)
    start_step = trainer2.load(os.path.join(tcfg.out_dir, "latest.pt"))
    assert start_step == 6
