# 教学注释：验证生成能持续产出、采样器行为，以及 KV 缓存与朴素重算结果一致。
import torch
from src.config import ModelConfig
from src.model import GPT
from src.generate import generate, sample_logits, _apply_repetition_penalty
from src.tokenizer import CharTokenizer


def tiny_model(vocab_size: int = 9):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab_size, d_model=64, n_layer=2, n_head=4,
                           n_kv_head=2, d_ff=128, ctx_len=64))


def char_tokenizer():
    # 词表 5 个特殊符号 + 4 个字符 = 9，与 tiny_model 的 vocab_size 对齐，decode 不会越界。
    return CharTokenizer(["你", "好", "世", "界"],
                         {"bos": 0, "eos": 1, "pad": 2, "unk": 3, "image": 4})


class _IdTokenizer(CharTokenizer):
    """把 decode 重写成逗号分隔的 id，从而能把 generate 的输出当成 id 序列比较。"""

    def decode(self, ids):
        return ",".join(str(int(i)) for i in ids)


class _SpyGPT(GPT):
    """记录每次前向喂入的序列长度与缓存长度，用来证明 generate 真的增量使用了缓存。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.call_input_lens = []
        self.call_past_lens = []

    def forward(self, idx=None, inputs_embeds=None, targets=None, past_kvs=None, use_cache=False):
        self.call_input_lens.append(0 if idx is None else idx.shape[1])
        self.call_past_lens.append(0 if past_kvs is None else past_kvs[0][0].shape[-2])
        return super().forward(idx=idx, inputs_embeds=inputs_embeds, targets=targets,
                               past_kvs=past_kvs, use_cache=use_cache)


def spy_model(vocab_size: int = 9):
    torch.manual_seed(0)
    return _SpyGPT(ModelConfig(vocab_size=vocab_size, d_model=64, n_layer=2, n_head=4,
                               n_kv_head=2, d_ff=128, ctx_len=64))


def test_sample_logits_greedy():
    logits = torch.tensor([0.1, 2.0, 0.5])
    # temperature<=0 走显式 argmax 分支
    tok0 = sample_logits(logits, temperature=0.0, top_k=None, top_p=None)
    assert tok0.dim() == 1
    assert tok0.item() == int(torch.argmax(logits))

    # 温度极低时 softmax 饱和，同样趋近 argmax
    tok = sample_logits(logits, temperature=1e-4, top_k=None, top_p=1.0)
    assert tok.item() == 1


def test_kv_cache_logits_match_full_forward():
    """KV 缓存逐步前向的 logits 必须与一次性整段前向完全一致。

    这能抓住 offset 错误、重复喂入整段、或 use_cache 误用等问题。
    """
    model = tiny_model(vocab_size=9)
    model.eval()
    idx = torch.randint(0, 9, (1, 8))

    full, _, _ = model(idx, past_kvs=None, use_cache=False)

    past = None
    steps = []
    for t in range(idx.shape[1]):
        logits_t, _, past = model(idx[:, t:t + 1], past_kvs=past, use_cache=True)
        steps.append(logits_t)
    incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-4)


def _naive_greedy_ids(model, tokenizer, prompt, max_new_tokens):
    """朴素参考实现：每一步都从头喂入完整序列，不使用 KV 缓存。"""
    model.eval()
    eos = tokenizer.special_id("eos")
    seq = list(tokenizer.encode(prompt, add_bos=True))
    out = []
    for _ in range(max_new_tokens):
        idx = torch.tensor([seq], dtype=torch.long)
        logits, _, _ = model(idx, past_kvs=None, use_cache=False)
        tid = int(sample_logits(logits[0, -1].float(), temperature=0.0).item())
        if tid == eos:
            break
        out.append(tid)
        seq.append(tid)
    return out


def test_generate_matches_naive_recompute():
    """generate 的缓存增量结果必须与逐步整段重算的贪心结果逐 token 相同。"""
    model = tiny_model(vocab_size=9)
    tok = _IdTokenizer(["你", "好", "世", "界"],
                       {"bos": 0, "eos": 1, "pad": 2, "unk": 3, "image": 4})

    cached = generate(model, tok, "你好", max_new_tokens=12, temperature=0.0, device="cpu")
    cached_ids = [int(x) for x in cached.split(",")] if cached else []

    reference = _naive_greedy_ids(model, tok, "你好", max_new_tokens=12)

    assert cached_ids == reference


def test_generate_actually_uses_incremental_kv_cache():
    """证明 generate 首步喂整段、之后每次只喂 1 个 token 且缓存递增。

    若实现改成每步重喂整段、或 use_cache=False，则这里的前向长度/缓存长度断言会失败。
    """
    model = spy_model(vocab_size=9)
    tok = _IdTokenizer(["你", "好", "世", "界"],
                       {"bos": 0, "eos": 1, "pad": 2, "unk": 3, "image": 4})

    out = generate(model, tok, "你好", max_new_tokens=12, temperature=0.0, device="cpu")
    n_gen = len(out.split(",")) if out else 0

    assert model.call_input_lens[0] == 3            # [bos] + 你好
    assert all(n == 1 for n in model.call_input_lens[1:])
    assert model.call_past_lens[0] == 0
    assert model.call_past_lens == sorted(model.call_past_lens)
    assert len(model.call_input_lens) in (n_gen, n_gen + 1)  # 命中 EOS 时多一次前向


def test_sample_logits_top_k_restricts_choices():
    logits = torch.tensor([0.0, 5.0, 1.0, 4.0])
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        tok = sample_logits(logits, temperature=1.0, top_k=2, top_p=None, generator=gen)
        assert tok.item() in (1, 3)   # top-2 只可能是最大的两个


def test_sample_logits_top_p_one_is_noop():
    logits = torch.tensor([0.3, 1.2, -0.7, 2.5])
    with_p1 = sample_logits(logits, temperature=1.0, top_k=None, top_p=1.0,
                            generator=torch.Generator().manual_seed(7))
    no_p = sample_logits(logits, temperature=1.0, top_k=None, top_p=None,
                         generator=torch.Generator().manual_seed(7))
    assert with_p1.item() == no_p.item()   # top_p=1.0 不截断，与不传 top_p 等价


def test_repetition_penalty_reshapes_logits():
    logits = torch.tensor([2.0, -4.0, 1.0])
    _apply_repetition_penalty(logits, [0, 1], penalty=2.0)
    assert logits[0].item() == 1.0    # 正 logits 除以 penalty
    assert logits[1].item() == -8.0   # 负 logits 乘以 penalty（更负）
    assert logits[2].item() == 1.0    # 未见过的 token 不变


def test_generate_length_and_determinism():
    model = tiny_model(vocab_size=9)
    tok = char_tokenizer()
    out1 = generate(model, tok, "你好", max_new_tokens=10, temperature=0.0, device="cpu")
    out2 = generate(model, tok, "你好", max_new_tokens=10, temperature=0.0, device="cpu")
    assert isinstance(out1, str)
    assert out1 == out2
