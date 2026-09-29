import json
import random
from collections import Counter

from src.sft_data import (iter_sft_examples, normalize_alpaca, normalize_firefly,
                          synthesize_dialogue)


def _write_alpaca(tmp_path, n):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps([{"instruction": f"u{k}", "input": "", "output": f"a{k}"}
                    for k in range(n)]),
        encoding="utf-8")
    return str(path)


def test_normalize_alpaca_with_input():
    msgs = normalize_alpaca({"instruction": "翻译", "input": "hello", "output": "你好"})["messages"]
    assert msgs == [{"role": "user", "content": "翻译\nhello"},
                    {"role": "assistant", "content": "你好"}]


def test_normalize_alpaca_without_input():
    msgs = normalize_alpaca({"instruction": "你好", "input": "", "output": "嗨"})["messages"]
    assert msgs[0]["content"] == "你好"


def test_normalize_firefly():
    msgs = normalize_firefly({"kind": "NLI", "input": "前提", "target": "中立"})["messages"]
    assert msgs == [{"role": "user", "content": "前提"},
                    {"role": "assistant", "content": "中立"}]


def test_synthesize_dialogue_alternates_roles():
    pairs = [("问1", "答1"), ("问2", "答2"), ("问3", "答3")]
    conv = synthesize_dialogue(pairs, random.Random(0), max_turns=3)
    roles = [m["role"] for m in conv["messages"]]
    assert roles == ["user", "assistant"] * (len(roles) // 2)
    assert 2 <= len(conv["messages"]) <= 6


def test_iter_sft_examples_two_formats(tmp_path):
    alp = tmp_path / "alpaca.json"
    alp.write_text(json.dumps([{"instruction": "i1", "input": "", "output": "o1"}]),
                   encoding="utf-8")
    ff = tmp_path / "firefly.jsonl"
    ff.write_text(json.dumps({"kind": "k", "input": "i2", "target": "o2"}) + "\n",
                  encoding="utf-8")
    got = list(iter_sft_examples([str(alp), str(ff)], synth_prob=0.0))
    assert len(got) == 2
    assert all(len(g["messages"]) == 2 for g in got)


def test_iter_sft_examples_deterministic(tmp_path):
    path = _write_alpaca(tmp_path, 6)
    first = list(iter_sft_examples([path], synth_prob=1.0, seed=7))
    second = list(iter_sft_examples([path], synth_prob=1.0, seed=7))
    assert first == second


def test_iter_sft_examples_no_pair_loss(tmp_path):
    path = _write_alpaca(tmp_path, 6)
    convs = list(iter_sft_examples([path], synth_prob=1.0, seed=0))
    assert any(len(c["messages"]) > 2 for c in convs)
    got = Counter(m["content"] for c in convs for m in c["messages"]
                  if m["role"] == "user")
    assert got == Counter(f"u{k}" for k in range(6))


def test_iter_sft_examples_alternates_roles(tmp_path):
    path = _write_alpaca(tmp_path, 6)
    for conv in iter_sft_examples([path], synth_prob=1.0, seed=3):
        roles = [m["role"] for m in conv["messages"]]
        assert roles == ["user", "assistant"] * (len(roles) // 2)
        assert len(roles) >= 2


import numpy as np
import torch
from src.sft_data import (SFTDataset, collate_sft, save_sft_npz, tokenize_examples)
from src.tokenizer import CharTokenizer

SAMPLE = [{"messages": [{"role": "user", "content": "问"} ,
                        {"role": "assistant", "content": "答"}]}]


def _tok():
    return CharTokenizer.train(["问答复你好"], max_chars=100)


def test_tokenize_and_npz_roundtrip(tmp_path):
    tok = _tok()
    p = tmp_path / "t.npz"
    n = save_sft_npz(tok, SAMPLE, str(p), max_len=64)
    assert n == 1
    ds = SFTDataset(str(p))
    assert len(ds) == 1
    ids, labels = ds[0]
    assert len(ids) == len(labels)
    assert any(l != -100 for l in labels)


def test_tokenize_skips_examples_without_supervision():
    tok = _tok()
    bad = [{"messages": [{"role": "user", "content": "只有用户"}]}]
    arrays, n = tokenize_examples(tok, bad, max_len=64)
    assert n == 0 and len(arrays["offsets"]) == 1


def test_collate_pads_and_masks():
    tok = _tok()
    batch = [([1, 2, 3, 4], [-100, -100, 5, 6]), ([1, 2], [-100, 7])]
    out = collate_sft(batch, pad_id=0)
    assert out["input_ids"].shape == (2, 4)
    assert out["labels"].tolist()[1][2:] == [-100, -100]
    assert isinstance(out["input_ids"], torch.Tensor)
