"""数据下载与预处理：全部落到 data/ 分层目录，走 hf-mirror。"""
from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Iterator

from torch.utils.data import Dataset

# 教学注释：强制把下载端点指向国内镜像，避免超时。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "added_tokens.json", "special_tokens_map.json", "config.json", "generation_config.json",
]


def download_qwen_tokenizer(dest_dir: str) -> str:
    """把 Qwen2.5 分词器文件下载到项目内 dest_dir，不依赖 HF 缓存。"""
    from huggingface_hub import snapshot_download
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B",
        local_dir=dest_dir,
        allow_patterns=_TOKENIZER_FILES,
    )
    return dest_dir


def download_tinystories(dest_dir: str) -> list[str]:
    """下载中文 TinyStories 压缩包并解压，返回所有 jsonl 路径。"""
    import tarfile as _tar
    from huggingface_hub import hf_hub_download

    root = Path(dest_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(
        repo_id="adam89/TinyStoriesChinese",
        filename="TinyStories_all_data_zh.tar.gz",
        repo_type="dataset",  # 教学注释：该仓库是 dataset，不指定会在 model 命名空间 404
        local_dir=dest_dir,
    )
    jsonl_dir = root / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    with _tar.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".jsonl")]
        for m in members:
            m.name = Path(m.name).name  # 去掉目录前缀，避免路径穿越
            tar.extract(m, jsonl_dir, filter="data")
    return sorted(str(p) for p in jsonl_dir.glob("*.jsonl"))


def iter_texts(paths: list[str]) -> Iterator[str]:
    """逐行读取 jsonl，兼容 story_zh/story/text/content 四种字段名。

    教学注释：adam89/TinyStoriesChinese 每行同时有英文 story 和中文 story_zh，
    这里优先取中文 story_zh，避免把英文原文喂给中文语料训练。
    """
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("story_zh") or obj.get("story") or obj.get("text") or obj.get("content")
                if text:
                    yield text
