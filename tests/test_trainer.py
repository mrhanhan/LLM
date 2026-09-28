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


def tiny_cfg():
    return ModelConfig(vocab_size=100, d_model=64, n_layer=2, n_head=4,
                       n_kv_head=2, d_ff=128, ctx_len=32)


def tiny_tcfg(out_dir, **over):
    base = dict(batch_size=2, grad_accum=1, lr=1e-2, warmup_steps=2,
                max_steps=6, eval_interval=3, eval_iters=2, save_interval=6,
                log_interval=1, out_dir=str(out_dir), dtype="fp32", compile=False)
    base.update(over)
    return TrainConfig(**base)


def test_resume_finished_checkpoint_preserves_step(tmp_path):
    # 回归：续训一个"已训完"的 checkpoint（start_step >= max_steps）时，
    # train() 的循环不会执行，末尾 save() 绝不能把步数写回 0。
    cfg = tiny_cfg()
    data = make_data(tmp_path)
    out = tmp_path / "out"
    Trainer(GPT(cfg), tiny_tcfg(out), data, data).train()

    trainer = Trainer(GPT(cfg), tiny_tcfg(out, max_steps=6), data, data)
    assert trainer.load(str(out / "latest.pt")) == 6
    trainer.train()  # max_steps == start_step，循环为空

    resumed = torch.load(str(out / "latest.pt"), map_location="cpu", weights_only=False)
    assert resumed["step"] == 6, "续训完成态 ckpt 不应把步数清零"


def test_compile_true_falls_back_to_eager(tmp_path, monkeypatch):
    # 回归：torch.compile 惰性编译，首次前向才真正编译；warmup 前向必须把
    # 延迟触发的编译错误捕获并回退 eager，而不是让训练崩溃。
    cfg = tiny_cfg()
    data = make_data(tmp_path)

    def fake_compile(model):
        class _Broken:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("triton not available")
        return _Broken()

    monkeypatch.setattr(torch, "compile", fake_compile)
    tcfg = tiny_tcfg(tmp_path / "out", compile=True, max_steps=2,
                     save_interval=2, eval_interval=2)
    trainer = Trainer(GPT(cfg), tcfg, data, data)
    assert trainer.model is trainer.raw_model, "编译失败应回退到 eager 原始模型"
    trainer.train()
    assert os.path.exists(str(tmp_path / "out" / "latest.pt"))
