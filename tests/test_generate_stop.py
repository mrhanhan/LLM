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


class _RecordingModel(_FakeModel):
    """额外记录每次前向首个 token 的模型，用于验证 add_bos 与停止步数。"""

    def __init__(self, vocab: int, token_id: int):
        super().__init__(vocab, token_id)
        self.first_tokens: list[int] = []

    def forward(self, idx, past_kvs=None, use_cache=False):
        self.first_tokens.append(int(idx[0, 0]))
        return super().forward(idx, past_kvs=past_kvs, use_cache=use_cache)


def test_generate_stops_on_custom_stop_id_not_eos():
    # stop 用普通字符 token（非 eos）：若实现忽略 stop_ids 而退回默认 eos，就停不下来。
    tok = CharTokenizer.train(["你好abc"])
    x = tok.encode("a")[0]
    assert x != tok.special_id("eos")
    model = _FakeModel(tok.vocab_size, x)
    out = generate(model, tok, "你好", max_new_tokens=20, temperature=0.0,
                   stop_ids=[x], device="cpu")
    assert out == ""                                # 第一个 token 就是 stop，立即停止


def test_generate_default_stop_is_eos():
    tok = CharTokenizer.train(["你好abc"])
    model = _FakeModel(tok.vocab_size, tok.special_id("eos"))
    out = generate(model, tok, "你好", max_new_tokens=20, temperature=0.0, device="cpu")
    assert out == ""                                # 不传 stop_ids 时默认遇 eos 停止


def test_generate_empty_stop_ids_means_no_early_stop():
    # 空列表表示"没有停止 token"，不是"回退默认 eos"：模型每步都吐 eos 也应跑满。
    tok = CharTokenizer.train(["你好abc"])
    eos = tok.special_id("eos")
    model = _RecordingModel(tok.vocab_size, eos)
    out = generate(model, tok, "你好", max_new_tokens=3, temperature=0.0,
                   stop_ids=[], device="cpu")
    assert len(model.first_tokens) == 3             # 未被 eos 提前打断
    assert len(tok.encode(out)) >= 1


def test_generate_runs_to_max_without_matching_stop():
    tok = CharTokenizer.train(["你好abc"])
    model = _FakeModel(tok.vocab_size, tok.special_id("unk"))
    out = generate(model, tok, "你好", max_new_tokens=3, temperature=0.0,
                   stop_ids=[tok.special_id("eos")], device="cpu")
    assert len(tok.encode(out)) >= 1


def test_generate_add_bos_flag():
    # CharTokenizer 的 bos / eos / 首个字符 id 互不相同，可清楚区分三种前缀。
    tok = CharTokenizer.train(["你好abc"])
    prompt = "你好"
    assert tok.special_id("bos") != tok.special_id("eos")
    assert tok.encode(prompt)[0] != tok.special_id("bos")

    stop = tok.special_id("eos")
    no_bos = _RecordingModel(tok.vocab_size, stop)
    generate(no_bos, tok, prompt, max_new_tokens=1, temperature=0.0,
             stop_ids=[stop], device="cpu", add_bos=False)
    assert no_bos.first_tokens[0] == tok.encode(prompt)[0]   # 首token是提示词首字符

    with_bos = _RecordingModel(tok.vocab_size, stop)
    generate(with_bos, tok, prompt, max_new_tokens=1, temperature=0.0,
             stop_ids=[stop], device="cpu", add_bos=True)
    assert with_bos.first_tokens[0] == tok.special_id("bos")


def test_generate_return_count_zero_when_stopped_immediately():
    tok = CharTokenizer.train(["你好abc"])
    x = tok.encode("a")[0]
    model = _FakeModel(tok.vocab_size, x)
    out = generate(model, tok, "你好", max_new_tokens=20, temperature=0.0,
                   stop_ids=[x], device="cpu", return_count=True)
    assert isinstance(out, tuple)
    text, n = out
    assert text == "" and n == 0                    # 首 token 即 stop，追加数为 0


def test_generate_return_count_equals_max_without_stop():
    tok = CharTokenizer.train(["你好abc"])
    eos = tok.special_id("eos")
    model = _FakeModel(tok.vocab_size, eos)
    text, n = generate(model, tok, "你好", max_new_tokens=3, temperature=0.0,
                       stop_ids=[], device="cpu", return_count=True)
    assert n == 3                                   # 无停止 token 时跑满上限


def test_generate_default_returns_str():
    tok = CharTokenizer.train(["你好abc"])
    model = _FakeModel(tok.vocab_size, tok.special_id("eos"))
    out = generate(model, tok, "你好", max_new_tokens=3, temperature=0.0, device="cpu")
    assert isinstance(out, str)                     # 默认保持原有 str 返回
