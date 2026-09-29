"""viz 训练用数据集注册表与批次提供者（写死，不扫描目录）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATASETS: dict[str, dict] = {
    "corpus_qwen": {
        "id": "corpus_qwen", "label": "内置中文语料（最快）",
        "kind": "pretrain", "tokenizer": "qwen", "builtin": True,
        "note": "viz.runtime.CORPUS 用 Qwen 分词",
    },
    "tiny_stories": {
        "id": "tiny_stories", "label": "TinyStories",
        "kind": "pretrain", "tokenizer": "qwen",
        "path": "data/processed/text/train.bin",
    },
    "poetry": {
        "id": "poetry", "label": "中文诗词",
        "kind": "pretrain", "tokenizer": "qwen",
        "path": "data/processed/text/poetry.train.bin",
    },
    "advertise": {
        "id": "advertise", "label": "AdvertiseGen",
        "kind": "pretrain", "tokenizer": "qwen",
        "path": "data/processed/text/advertise.train.bin",
    },
    "fineweb_cmn": {
        "id": "fineweb_cmn", "label": "FineWeb 中文（大文件）",
        "kind": "pretrain", "tokenizer": "qwen",
        "path": "data/processed/text/fineweb_cmn.train.bin",
    },
    "sft_chat": {
        "id": "sft_chat", "label": "Alpaca+Firefly 多轮",
        "kind": "sft", "tokenizer": "qwen",
        "path": "data/processed/sft/train.npz",
    },
    "sft_poetry": {
        "id": "sft_poetry", "label": "诗词续写",
        "kind": "sft", "tokenizer": "qwen",
        "path": "data/processed/sft/poetry.train.npz",
    },
    "sft_advertise": {
        "id": "sft_advertise", "label": "广告文案",
        "kind": "sft", "tokenizer": "qwen",
        "path": "data/processed/sft/advertise.train.npz",
    },
}


def _available(spec: dict) -> bool:
    if spec.get("builtin"):
        return True
    return Path(spec["path"]).exists()


def list_datasets() -> list[dict]:
    out = []
    for spec in DATASETS.values():
        out.append({
            "id": spec["id"], "label": spec["label"], "kind": spec["kind"],
            "tokenizer": spec["tokenizer"], "available": _available(spec),
            "note": spec.get("note", ""),
        })
    return out


def resolve_spec(dataset_id: str, mode: str) -> dict:
    spec = DATASETS[dataset_id]  # 未知 id -> KeyError
    if spec["kind"] != mode:
        raise ValueError(f"数据集 {dataset_id} 属于 {spec['kind']}，不能用于 {mode}")
    return spec


def load_pretrain(spec: dict, tok) -> np.ndarray:
    if spec.get("builtin"):
        from viz.runtime import CORPUS
        return np.asarray(tok.encode(CORPUS), dtype=np.int64)
    from src.data import load_bin
    return np.asarray(load_bin(spec["path"]), dtype=np.int64)


class PretrainBatches:
    """随机取样 (x, y=next-token) 的预训练批次。"""

    def __init__(self, data: np.ndarray, batch_size: int, block: int, device: str):
        self.data = data
        self.batch_size = int(batch_size)
        self.block = int(block)
        self.device = device

    def next(self):
        import torch
        n = len(self.data)
        hi = max(n - self.block - 1, 1)
        ix = np.random.randint(0, hi, size=self.batch_size)
        x = np.stack([self.data[i:i + self.block] for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.block] for i in ix])
        return (torch.from_numpy(x.astype("int64")).to(self.device),
                torch.from_numpy(y.astype("int64")).to(self.device))


class SFTBatches:
    """从 npz 随机取样本，按批内最大长度 padding。"""

    def __init__(self, npz_path: str, batch_size: int, max_len: int,
                 pad_id: int, device: str):
        from src.sft_data import SFTDataset
        self.ds = SFTDataset(npz_path)
        self.batch_size = int(batch_size)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.device = device

    def next(self):
        import torch
        from src.sft_data import collate_sft
        n = len(self.ds)
        idx = np.random.randint(0, max(n, 1), size=self.batch_size)
        batch = [self.ds[int(k)] for k in idx]
        out = collate_sft(batch, self.pad_id, self.max_len)
        return (out["input_ids"].to(self.device), out["labels"].to(self.device))
