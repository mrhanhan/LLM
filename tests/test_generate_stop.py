import torch
from src.generate import generate
from src.tokenizer import CharTokenizer


class _FakeModel(torch.nn.Module):
    """永远输出固定 token 的假模型，用来验证 stop_ids 行为。"""

    def __init__(self, vocab: int, token_id: int):
        super().__init__()
        self.vocab, self.token_id = vocab, token_id

    def forward(self, idx, past_kvs=None, use_cache=False):
        b, t = idx.shape
        logits = torch.full((b, t, self.vocab), -1e9)
        logits[..., self.token_id] = 0.0
        return logits, None, None


def test_generate_stops_on_stop_id():
    tok = CharTokenizer.train(["你好abc"])
    stop = tok.special_id("eos")
    model = _FakeModel(tok.vocab_size, stop)
    out = generate(model, tok, "你好", max_new_tokens=20, temperature=0.0,
                   stop_ids=[stop], device="cpu")
    assert out == ""                                # 第一个 token 就是 stop，立即停止


def test_generate_runs_to_max_without_matching_stop():
    tok = CharTokenizer.train(["你好abc"])
    model = _FakeModel(tok.vocab_size, tok.special_id("unk"))
    out = generate(model, tok, "你好", max_new_tokens=3, temperature=0.0,
                   stop_ids=[tok.special_id("eos")], device="cpu")
    assert len(tok.encode(out)) >= 1
