"""SFT 数据：归一化 alpaca/firefly、合成多轮对话、分词落盘、Dataset 与 collate。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator


def normalize_alpaca(obj: dict) -> dict:
    """alpaca 格式：instruction/input/output -> 单轮 user/assistant。"""
    user = obj["instruction"]
    if obj.get("input"):
        user = f"{user}\n{obj['input']}"
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": obj["output"]}]}


def normalize_firefly(obj: dict) -> dict:
    """firefly 格式：input/target -> 单轮 user/assistant（kind 丢弃）。"""
    return {"messages": [{"role": "user", "content": obj["input"]},
                         {"role": "assistant", "content": obj["target"]}]}


def synthesize_dialogue(pairs: list[tuple[str, str]], rng: random.Random,
                        max_turns: int = 3) -> dict:
    """把若干单轮 (user, assistant) 拼成一段多轮对话。

    教学注释：alpaca/firefly 都是单轮数据，靠拼接让模型见过"多轮"的序列形态。
    这是有意的数据增广，不改变单条样本的语义。
    """
    n = min(rng.randint(2, max_turns), len(pairs)) if pairs else 0
    chosen = rng.sample(pairs, n) if n else []
    msgs = []
    for u, a in chosen:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return {"messages": msgs}


def _read_messages(path: str, max_items: int | None) -> Iterator[list[dict]]:
    """按扩展名解析：.jsonl = firefly，.json = alpaca。"""
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_items is not None and i >= max_items:
                    break
                line = line.strip()
                if line:
                    yield normalize_firefly(json.loads(line))["messages"]
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        for i, obj in enumerate(data):
            if max_items is not None and i >= max_items:
                break
            yield normalize_alpaca(obj)["messages"]


def iter_sft_examples(paths: list[str], max_items: int | None = None,
                      synth_prob: float = 0.5, seed: int = 0) -> Iterator[dict]:
    """读取并归一化所有 SFT 文件；按 synth_prob 把相邻单轮拼成多轮对话。"""
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for path in paths:
        for msgs in _read_messages(path, max_items):
            pairs.append((msgs[0]["content"], msgs[1]["content"]))
    i = 0
    while i < len(pairs):
        if synth_prob > 0 and rng.random() < synth_prob and i + 1 < len(pairs):
            k = min(rng.randint(2, 3), len(pairs) - i)
            yield synthesize_dialogue(pairs[i:i + k], rng, max_turns=3)
            i += k
        else:
            u, a = pairs[i]
            yield {"messages": [{"role": "user", "content": u},
                                {"role": "assistant", "content": a}]}
            i += 1
