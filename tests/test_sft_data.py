import json
import random
from src.sft_data import (iter_sft_examples, normalize_alpaca, normalize_firefly,
                          synthesize_dialogue)


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
