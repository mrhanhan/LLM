# 教学注释：索引只需 image 相对路径 + caption，Dataset 读取时再解析图片。
import json
from pathlib import Path
from PIL import Image
from src.data import build_caption_index, CaptionDataset
import pytest


def test_build_index_and_dataset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32), (i * 50, 100, 150)).save(raw / f"{i}.png")
        (raw / f"{i}.txt").write_text(f"这是第{i}张图", encoding="utf-8")
    out = tmp_path / "index.jsonl"
    n = build_caption_index(str(raw), str(out))
    assert n == 3
    ds = CaptionDataset(str(out), img_size=64, raw_dir=str(raw))
    assert len(ds) == 3
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["caption"].startswith("这是第")


def test_build_index_skips_missing_caption(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (16, 16)).save(raw / "a.png")   # 没有对应 txt
    out = tmp_path / "index.jsonl"
    n = build_caption_index(str(raw), str(out))
    assert n == 0
