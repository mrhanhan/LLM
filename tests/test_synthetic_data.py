# 教学注释：合成数据必须"自洽"——图与描述同源生成，
# 所以模型能学会时是真正的视觉理解，而非背语料。
import torch
from src.data import SyntheticImageDataset, COLORS, SHAPES


def test_shapes_and_range():
    ds = SyntheticImageDataset(num_samples=5, img_size=64, seed=1)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["image"].min() >= -1.01 and item["image"].max() <= 1.01
    assert isinstance(item["caption"], str) and len(item["caption"]) > 0
    assert len(item["qa"]) >= 1


def test_determinism():
    a = SyntheticImageDataset(num_samples=3, img_size=64, seed=7)
    b = SyntheticImageDataset(num_samples=3, img_size=64, seed=7)
    assert torch.equal(a[1]["image"], b[1]["image"])
    assert a[1]["caption"] == b[1]["caption"]


def test_caption_mentions_generated_attributes():
    ds = SyntheticImageDataset(num_samples=40, img_size=64, seed=3)
    caps = [ds[i]["caption"] for i in range(40)]
    assert any(any(c in cap for c in COLORS) for cap in caps)
    assert any(any(s in cap for s in SHAPES) for cap in caps)


def test_collate_shapes():
    from src.tokenizer import CharTokenizer
    ds = SyntheticImageDataset(num_samples=4, img_size=64, seed=0)
    batch = [ds[i] for i in range(2)]
    tok = CharTokenizer(["图", "中", "有", "个", "形", "一", "两", "三"], {"bos": 0, "eos": 1, "pad": 2, "unk": 3, "image": 4})
    out = __import__("src.data", fromlist=["collate_vlm"]).collate_vlm(batch, tok, num_image_tokens=64)
    assert out["input_ids"].shape[0] == 2
    assert out["pixel_values"].shape == (2, 3, 64, 64)
    assert out["input_ids"].shape == out["labels"].shape
