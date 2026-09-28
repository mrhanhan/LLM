"""分词器：手写字节级 BPE（教学核心）+ 字符级 + Qwen 适配（见 Task 3）。

字节级 BPE 思想（GPT-2 同款）：
1) 把文本先按正则切成"词"，每个词再拆成 utf-8 字节序列；
2) 反复统计相邻字节/符号对的频率，把最高频的一对合并成新符号；
3) 重复 vocab_size - 初始符号数 次，得到 merges 表；
4) 编码时按 merges 的出现顺序（rank 小者优先）对字节序列做合并。
好处：任何字符都不会 OOV（未登录），中文天然可处理。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import regex as re

# 特殊符号：pad 填充 / bos 起始 / eos 结束 / unk / image 视觉占位
SPECIALS: dict[str, str] = {
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
    "unk": "<unk>",
    "image": "<image>",
}

# 预切分：先单独切出英文缩写、单个汉字、英文单词、数字、其它符号、空白。
# 注意 \w 在 unicode 下包含汉字，所以汉字要显式列出。
_PRE_TOKEN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[\u4e00-\u9fff\u3400-\u4dbf]| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]|\s+"
)


def _get_pairs(symbols: tuple[int, ...]) -> Counter:
    """统计相邻符号对的频次。"""
    pairs: Counter = Counter()
    prev = symbols[0]
    for cur in symbols[1:]:
        pairs[(prev, cur)] += 1
        prev = cur
    return pairs


def _merge(symbols: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """把序列中的所有 pair 替换成 new_id。"""
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class _BaseTokenizer:
    """统一接口。子类实现 encode/decode 与 vocab_size。"""

    specials = SPECIALS

    def _special_map(self) -> dict[str, int]:
        raise NotImplementedError

    def special_id(self, name: str) -> int:
        return self._special_map()[name]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError


class BPETokenizer(_BaseTokenizer):
    def __init__(self, merges: list[tuple[int, int]], vocab: dict[int, bytes], special_to_id: dict[str, int]):
        self.merges = merges
        self.vocab = vocab                                   # id -> 字节串
        self._special_to_id = special_to_id
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        # 布局：[特殊符号][0..255 字节][merge 产生的新符号]
        self._byte_offset = len(special_to_id)
        self._base_size = len(special_to_id) + 256

    @property
    def vocab_size(self) -> int:
        # 布局 = 特殊符号 + 256 字节 + merges，_base_size 已含特殊符号，勿重复相加
        return self._base_size + len(self.merges)

    def _special_map(self) -> dict[str, int]:
        return self._special_to_id

    def _ids_for_text(self, text: str) -> list[int]:
        """把普通文本转成 id 序列（不含特殊符号）。"""
        ids: list[int] = []
        for chunk in _PRE_TOKEN.findall(text):
            # 初始符号用字节的"全局 id"（byte_offset + b），与 train 的词表布局一致
            symbols = tuple(self._byte_offset + b for b in chunk.encode("utf-8"))
            while len(symbols) >= 2:
                pairs = _get_pairs(symbols)
                # 在所有相邻对里，选择 merge rank 最小（最早学到）的一对合并
                pair = min(
                    (p for p in pairs if p in self.merge_ranks),
                    key=lambda p: self.merge_ranks[p],
                    default=None,
                )
                if pair is None:
                    break
                symbols = _merge(symbols, pair, self.merge_ranks[pair] + self._base_size)
            ids.extend(symbols)
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._ids_for_text(text)
        if add_bos:
            ids = [self.special_id("bos")] + ids
        if add_eos:
            ids = ids + [self.special_id("eos")]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        id_to_special = {v: k for k, v in self._special_to_id.items()}
        out, buf = [], bytearray()
        for i in ids:
            i = int(i)
            if i in id_to_special:
                if buf:
                    out.append(bytes(buf).decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(self.specials[id_to_special[i]])
            else:
                buf.extend(self.vocab[i])
        if buf:
            out.append(bytes(buf).decode("utf-8", errors="replace"))
        return "".join(out)

    def save(self, save_dir: str) -> None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "merges.json").write_text(
            json.dumps([[a, b] for a, b in self.merges]), encoding="utf-8"
        )
        (d / "vocab.json").write_text(
            json.dumps({str(k): list(v) for k, v in self.vocab.items()}), encoding="utf-8"
        )
        (d / "specials.json").write_text(
            json.dumps(self._special_to_id, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, save_dir: str) -> "BPETokenizer":
        d = Path(save_dir)
        merges = [tuple(p) for p in json.loads((d / "merges.json").read_text(encoding="utf-8"))]
        vocab = {int(k): bytes(v) for k, v in json.loads((d / "vocab.json").read_text(encoding="utf-8")).items()}
        special_to_id = json.loads((d / "specials.json").read_text(encoding="utf-8"))
        return cls(merges, vocab, special_to_id)

    @classmethod
    def train(
        cls,
        texts: Iterator[str],
        vocab_size: int = 16384,
        min_frequency: int = 2,
        specials: dict[str, str] = SPECIALS,
        max_bytes: int = 20_000_000,
        verbose: bool = True,
    ) -> "BPETokenizer":
        special_to_id = {name: i for i, name in enumerate(specials)}
        base_size = len(specials) + 256
        byte_offset = len(specials)
        vocab = {byte_offset + b: bytes([b]) for b in range(256)}

        # 1) 统计词频（同一 word 只处理一次，加速）
        #    注意：初始符号统一用全局 id（byte_offset + 字节值），与 encode 保持一致
        word_freq: Counter = Counter()
        consumed = 0
        for text in texts:
            for chunk in _PRE_TOKEN.findall(text):
                word_freq[tuple(byte_offset + b for b in chunk.encode("utf-8"))] += 1
            consumed += len(text.encode("utf-8"))
            if consumed >= max_bytes:
                break

        words = list(word_freq.keys())
        freqs = [word_freq[w] for w in words]

        merges: list[tuple[int, int]] = []
        target_merges = max(0, vocab_size - base_size)
        for step in range(target_merges):
            pair_freq: Counter = Counter()
            for w, f in zip(words, freqs):
                if len(w) < 2:
                    continue
                for p, c in _get_pairs(w).items():
                    pair_freq[p] += c * f
            if not pair_freq:
                break
            best_pair, best_count = pair_freq.most_common(1)[0]
            if best_count < min_frequency:
                break
            new_id = byte_offset + 256 + len(merges)
            merges.append(best_pair)
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
            words = [_merge(w, best_pair, new_id) for w in words]
            if verbose and (step + 1) % 500 == 0:
                print(f"[BPE] merge {step + 1}/{target_merges} 当前词表 {byte_offset + 256 + len(merges)}")

        tok = cls(merges, vocab, special_to_id)
        return tok


class CharTokenizer(_BaseTokenizer):
    """字符级分词器：每个字符一个 id，最简单，便于对照理解。"""

    def __init__(self, chars: list[str], special_to_id: dict[str, int]):
        self.chars = chars
        self._special_to_id = special_to_id
        self._char_to_id = {c: i + len(special_to_id) for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self._special_to_id) + len(self.chars)

    def _special_map(self) -> dict[str, int]:
        return self._special_to_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        unk = self.special_id("unk")
        ids = [self._char_to_id.get(c, unk) for c in text]
        if add_bos:
            ids = [self.special_id("bos")] + ids
        if add_eos:
            ids = ids + [self.special_id("eos")]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        id_to_special = {v: k for k, v in self._special_to_id.items()}
        out = []
        for i in ids:
            i = int(i)
            if i in id_to_special:
                out.append(self.specials[id_to_special[i]])
            else:
                out.append(self.chars[i - len(self._special_to_id)])
        return "".join(out)

    def save(self, save_dir: str) -> None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "chars.json").write_text(json.dumps(self.chars, ensure_ascii=False), encoding="utf-8")
        (d / "specials.json").write_text(json.dumps(self._special_to_id, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, save_dir: str) -> "CharTokenizer":
        d = Path(save_dir)
        chars = json.loads((d / "chars.json").read_text(encoding="utf-8"))
        special_to_id = json.loads((d / "specials.json").read_text(encoding="utf-8"))
        return cls(chars, special_to_id)

    @classmethod
    def train(cls, texts: Iterator[str], specials: dict[str, str] = SPECIALS, max_chars: int = 2_000_000) -> "CharTokenizer":
        seen: dict[str, None] = {}
        consumed = 0
        for t in texts:
            for c in t:
                seen[c] = None
            consumed += len(t)
            if consumed >= max_chars:
                break
        return cls(list(seen.keys()), {name: i for i, name in enumerate(specials)})
