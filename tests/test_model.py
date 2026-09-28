# 教学注释：验证前向形状、权重共享、参数量统计与反向可训练。
import torch
from src.config import ModelConfig, TrainConfig
from src.model import GPT, RMSNorm, SwiGLU


def tiny_cfg():
    return ModelConfig(vocab_size=300, d_model=64, n_layer=2, n_head=4,
                       n_kv_head=2, d_ff=128, ctx_len=32)


def test_rmsnorm_and_swiglu_shapes():
    n = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    assert n(x).shape == x.shape
    m = SwiGLU(16, 32)
    assert m(x).shape == x.shape


def test_gpt_forward_and_loss():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    logits, loss, _ = model(x, targets=y)
    assert logits.shape == (2, cfg.ctx_len, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_weight_tying():
    cfg = tiny_cfg()
    model = GPT(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_num_params_reasonable():
    cfg = tiny_cfg()
    model = GPT(cfg)
    assert model.num_params() > model.num_params(non_embedding=True)
    assert model.num_params(non_embedding=True) > 0


def test_backward_works():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    _, loss, _ = model(x, targets=y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_configure_optimizers():
    model = GPT(tiny_cfg())
    opt = model.configure_optimizers(TrainConfig(weight_decay=0.1))
    assert len(opt.param_groups) == 2  # 衰减组 + 不衰减组
