# 教学注释：如果模型连一批固定数据都记不住，说明实现有 bug。
# 这是最快的"端到端可用性"检查，比看几小时 loss 曲线快得多。
import torch
from src.config import ModelConfig
from src.model import GPT


def test_overfit_single_batch():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=260, d_model=64, n_layer=2, n_head=4,
                      n_kv_head=2, d_ff=128, ctx_len=32, dropout=0.0)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.ctx_len))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for step in range(200):
        _, loss, _ = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.2, f"过拟合失败：{first:.3f} -> {loss.item():.3f}"
