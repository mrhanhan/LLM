"""VLM 标签右移（next-token）回归测试。

教学注释：这是 VLM 阶段最关键缺陷的回归测试——因果语言模型的 labels 必须是
input_ids **右移一位**（labels[t] 监督 input_ids[t+1]），而不是与 input_ids 对齐。
旧实现把 labels 直接对齐到 input_ids（图像位填 -100），本文件的断言会立即失败，
所以它锁住的是真实不变式，而不是同义反复。
"""
import random

from src.data import (
    SyntheticImageDataset,
    build_caption_example,
    build_vqa_example,
    collate_vlm,
)
from src.tokenizer import CharTokenizer

# 固定特殊符号 id；普通字符从 len(SPECIALS)=5 开始编号。
SPECIALS = {"bos": 0, "eos": 1, "pad": 2, "unk": 3, "image": 4}


def _tok(*texts: str) -> CharTokenizer:
    chars = sorted({c for t in texts for c in t})
    return CharTokenizer(chars, dict(SPECIALS))


def _first_supervised(labels) -> int:
    return next(t for t, v in enumerate(labels) if v != -100)


def _assert_shift_invariant(input_ids, labels):
    """labels[t] != -100 处必须满足 labels[t] == input_ids[t+1]（下一 token）。"""
    assert len(labels) == len(input_ids)
    for t in range(len(labels)):
        if labels[t] != -100:
            assert t + 1 < len(input_ids), "监督位不能落在最后一个位置（否则不是右移）"
            assert labels[t] == input_ids[t + 1], f"位置 {t} 的标签不是下一 token"


def test_caption_labels_are_next_token():
    caption = "图中有两个图形：一个红色的圆形在左边"
    N = 4
    tok = _tok(caption)
    ids, labels = build_caption_example(tok, caption, N)

    _assert_shift_invariant(ids, labels)

    first = _first_supervised(labels)
    # 第一监督位紧随 <image>×N：最后一张图预测第一个文本 token。
    assert first == N
    assert all(v == -100 for v in labels[:N])          # 图像块位置全部屏蔽
    assert ids[first] == tok.special_id("image")
    assert labels[first] == tok.encode(caption)[0]
    assert labels[-1] == -100                          # 末尾留给"最后 token 预测 eos"


def test_vqa_labels_are_next_token():
    q, a = "图中有几个图形？", "两个"
    N = 3
    tok = _tok(q, a, "答：")   # VQA 提示词由 question + "答：" 拼成，需覆盖这两个字
    ids, labels = build_vqa_example(tok, q, a, N)

    _assert_shift_invariant(ids, labels)

    prompt_ids = [tok.special_id("bos")] + [tok.special_id("image")] * N + tok.encode(q + "答：")
    first = _first_supervised(labels)
    # 第一监督位预测答案首 token，其输入位是 "答：" 的最后一个 token。
    assert first == len(prompt_ids) - 1
    assert all(v == -100 for v in labels[:first])
    assert ids[first] == tok.encode(q + "答：")[-1]
    assert labels[first] == tok.encode(a)[0]
    assert tok.decode(ids[: first + 1]).endswith("答：")


def test_collate_vqa_shift_after_padding():
    random.seed(1234)  # 固定随机选题，保证可复现
    ds = SyntheticImageDataset(num_samples=6, img_size=32, seed=0)
    batch = [ds[i] for i in range(3)]
    for b in batch:
        b["use_qa"] = True

    # 覆盖合成问答里常见的字符；未覆盖的字符会成为 unk，不影响结构性不变式。
    tok = _tok("图中有几个图形？答：两没有左边右边中间圆形方形三角形红色绿色蓝色黄色一个")
    out = collate_vlm(batch, tok, num_image_tokens=4)

    ids = out["input_ids"].tolist()
    labels = out["labels"].tolist()
    pad_id = tok.special_id("pad")
    assert len(ids) == len(labels) == len(batch)

    for row_ids, row_labels in zip(ids, labels):
        _assert_shift_invariant(row_ids, row_labels)   # padding 后依然成立
        for t, token in enumerate(row_ids):
            if token == pad_id:
                assert row_labels[t] == -100            # pad 位置必须屏蔽
