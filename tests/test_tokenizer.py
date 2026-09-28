# 教学注释：分词器必须满足 decode(encode(x)) == x（字节级可逆），
# 特殊符号不参与 BPE 合并，且能稳定保存/加载。
from src.tokenizer import BPETokenizer, CharTokenizer, SPECIALS


CORPUS = [
    "从前有座山，山里有座庙。",
    "小猫在草地上跑来跑去，非常开心。",
    "The quick brown fox jumps over the lazy dog.",
    "太阳 sun 月亮 moon 星星 star。",
] * 40


def test_bpe_roundtrip():
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=600, min_frequency=1, verbose=False)
    for s in CORPUS[:5]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_specials_present():
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=600, min_frequency=1, verbose=False)
    for name in SPECIALS:
        assert 0 <= tok.special_id(name) < tok.vocab_size


def test_bpe_save_load(tmp_path):
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=600, min_frequency=1, verbose=False)
    tok.save(str(tmp_path))
    tok2 = BPETokenizer.load(str(tmp_path))
    s = "小猫在草地上"
    assert tok2.decode(tok2.encode(s)) == s
    assert tok2.vocab_size == tok.vocab_size


def test_bpe_bos_eos():
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=600, min_frequency=1, verbose=False)
    ids = tok.encode("你好", add_bos=True, add_eos=True)
    assert ids[0] == tok.special_id("bos")
    assert ids[-1] == tok.special_id("eos")


def test_char_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer.train(iter(CORPUS))
    for s in CORPUS[:5]:
        assert tok.decode(tok.encode(s)) == s
    tok.save(str(tmp_path))
    tok2 = CharTokenizer.load(str(tmp_path))
    assert tok2.decode(tok2.encode("小猫")) == "小猫"
