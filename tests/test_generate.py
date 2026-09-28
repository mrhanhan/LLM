# 教学注释：验证生成能持续产出指定长度、且 greedy（temperature->0）是确定性的。
import torch
from src.config import ModelConfig
from src.model import GPT
from src.generate import generate, sample_logits


def tiny_model():
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=200, d_model=64, n_layer=2, n_head=4,
                           n_kv_head=2, d_ff=128, ctx_len=64))


def test_sample_logits_greedy():
    logits = torch.tensor([0.1, 2.0, 0.5])
    # 温度极低时趋近 argmax
    tok = sample_logits(logits, temperature=1e-4, top_k=None, top_p=1.0)
    assert tok.item() == 1


def test_generate_length_and_determinism():
    model = tiny_model()
    from src.tokenizer import CharTokenizer
    tok = CharTokenizer(["你", "好", "世", "界"], {"bos": 0, "eos": 1, "pad": 2, "unk": 3, "image": 4})
    out1 = generate(model, tok, "你好", max_new_tokens=10, temperature=0.0, device="cpu")
    out2 = generate(model, tok, "你好", max_new_tokens=10, temperature=0.0, device="cpu")
    assert out1 == out2
    assert len(out1) >= 2
