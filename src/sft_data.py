"""SFT 数据：归一化 alpaca/firefly、合成多轮对话、分词落盘、Dataset 与 collate。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator


def _turn(user: str, assistant: str) -> list[dict]:
    """构造一轮 user/assistant 消息（全局唯一的一处构造点）。"""
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": assistant}]


def normalize_alpaca(obj: dict) -> dict:
    """alpaca 格式：instruction/input/output -> 单轮 user/assistant。"""
    user = obj["instruction"]
    if obj.get("input"):
        user = f"{user}\n{obj['input']}"
    return {"messages": _turn(user, obj["output"])}


def normalize_firefly(obj: dict) -> dict:
    """firefly 格式：input/target -> 单轮 user/assistant（kind 丢弃）。"""
    return {"messages": _turn(obj["input"], obj["target"])}


def synthesize_dialogue(pairs: list[tuple[str, str]], rng: random.Random,
                        max_turns: int = 3) -> dict:
    """把给定窗口内的单轮 (user, assistant) 全部拼成一段多轮对话。

    教学注释：alpaca/firefly 都是单轮数据，靠拼接让模型见过"多轮"的序列形态。
    这是有意的数据增广，不改变单条样本的语义。

    约定：消费窗口内的**所有** pair（n = min(len(pairs), max_turns)），
    调用方（iter_sft_examples）据此按窗口长度推进，保证不丢样本。
    """
    n = min(len(pairs), max_turns)
    chosen = rng.sample(pairs, n)
    msgs = [m for pair in chosen for m in _turn(*pair)]
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
            yield {"messages": _turn(*pairs[i])}
            i += 1


def tokenize_examples(tok, examples, max_len: int):
    """把对话样本编码成扁平 int32 数组 + offsets；无监督信号的样本丢弃。

    返回 (arrays, n_kept)，arrays 含 ids/labels/offsets 三个键。
    """
    import numpy as np

    from src.chat_format import build_sft_example, has_supervision

    ids_all: list[int] = []
    lab_all: list[int] = []
    offsets: list[int] = [0]
    n_kept = 0
    for ex in examples:
        ids, labels = build_sft_example(tok, ex["messages"], max_len=max_len)
        if not ids or not has_supervision(labels):
            continue
        ids_all.extend(int(i) for i in ids)
        lab_all.extend(int(l) for l in labels)
        offsets.append(len(ids_all))
        n_kept += 1
    arrays = {
        "ids": np.array(ids_all, dtype=np.int32),
        "labels": np.array(lab_all, dtype=np.int32),
        "offsets": np.array(offsets, dtype=np.int64),
    }
    return arrays, n_kept


def save_sft_npz(tok, examples, path: str, max_len: int) -> int:
    """编码并写 npz，返回保留的样本数。"""
    import numpy as np

    arrays, n = tokenize_examples(tok, examples, max_len)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return n


class SFTDataset:
    """读取 npz 的 SFT 数据集：__getitem__ 返回 (ids, labels)（变长）。"""

    def __init__(self, npz_path: str):
        import numpy as np

        z = np.load(npz_path)
        self.ids = z["ids"]
        self.labels = z["labels"]
        self.offsets = z["offsets"]

    def __len__(self) -> int:
        return int(len(self.offsets) - 1)

    def __getitem__(self, i: int):
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.ids[s:e].astype("int64"), self.labels[s:e].astype("int64")


def collate_sft(batch, pad_id: int, max_len: int | None = None) -> dict:
    """按 batch 内最大长度 padding；ids 补 pad_id，labels 补 -100。"""
    import torch

    maxlen = max(len(ids) for ids, _ in batch)
    if max_len is not None:
        maxlen = min(maxlen, max_len)
    inputs, labels = [], []
    for ids, lab in batch:
        ids = list(ids)[:maxlen]
        lab = list(lab)[:maxlen]
        pad = maxlen - len(ids)
        inputs.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
