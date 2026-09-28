# 教学注释：验证二进制存储可回读、批张量形状与"下一 token"标签对齐。
import numpy as np
import torch
from src.tokenizer import BPETokenizer, SPECIALS
from src.data import build_text_bin, load_bin, get_batch

CORPUS = ["甲乙丙丁戊己庚辛壬癸", "子丑寅卯辰巳午未申酉"] * 200


def test_build_and_load(tmp_path):
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=400, min_frequency=1, verbose=False)
    tbin = tmp_path / "train.bin"
    vbin = tmp_path / "val.bin"
    n_train, n_val = build_text_bin(
        tok, iter(CORPUS), str(tbin), val_bin_path=str(vbin), val_ratio=0.1
    )
    assert n_train > 0 and n_val > 0
    data = load_bin(str(tbin))
    assert len(data) == n_train


def test_get_batch_shapes(tmp_path):
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=400, min_frequency=1, verbose=False)
    tbin = tmp_path / "train.bin"
    build_text_bin(tok, iter(CORPUS), str(tbin), val_bin_path=str(tmp_path / "v.bin"))
    data = load_bin(str(tbin))
    x, y = get_batch(data, batch_size=4, ctx_len=16, device="cpu")
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])  # 标签是输入右移一位
    assert x.dtype == torch.long
