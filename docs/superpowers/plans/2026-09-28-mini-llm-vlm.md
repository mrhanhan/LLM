# mini-llm-lab — 视觉 VLM 实现计划（Plan 2/2）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已训练的中文 GPT 上「装眼睛」，从零手写 ViT + 投影层，实现图片中文描述与简单 VQA。

**Architecture:** 手写 ViT 编码图片 → MLP 投影到 LLM 维度 → 用视觉特征替换 `<image>` 占位符的 embedding（LLaVA 式拼接），损失只在文本 token 上计算。先用合成几何图文验证链路，再接真实儿童图文数据集。LLM 权重从 Plan 1 的 checkpoint 迁移。

**Tech Stack:** 同 Plan 1，另用 `pillow` 生成/读取图片。

**Spec:** `docs/superpowers/specs/2026-09-28-mini-llm-lab-design.md`

**前置依赖:** Plan 1 已完成（`src/model.py`、`src/trainer.py`、`src/generate.py`、`data/tokenizer/qwen2.5-0.5b`、`out/gpt/latest.pt`）。VLM 复用同款 Qwen 分词器（已含 `<image>` 符号，词表一致，无需 resize）。

## Global Constraints

- 运行环境与解释器同 Plan 1（`.venv\Scripts\python.exe`）；数据一律在 `data/` 下分类存放。
- 图像统一 resize 到 `img_size=128`，`patch_size=16` → **64 个 patch** → 每个样本固定 64 个 `<image>` token。
- 视觉编码器默认：`d_vision=384, depth=6, n_head=6`。
- 损失**只在文本 token** 上计算，图像占位位置标签为 `-100`。
- 代码带**详细中文注释**；`torch.compile` 默认关闭。
- 真实图文数据落盘 `data/raw/image/children_caption/`，索引落盘 `data/processed/image/children/index.jsonl`。

---

### Task V1: 合成几何图文数据集（快速验证 VLM 链路）

**Files:**
- Modify: `src/data.py`（追加图像工具与 `SyntheticImageDataset`）
- Create: `scripts/make_synthetic_samples.py`
- Create: `tests/test_synthetic_data.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `src.data.COLORS, SHAPES, ZONES`（中文字符串常量）
  - `src.data.image_to_tensor(img) -> Tensor`（`(3, H, W)` float32，归一化到 [-1,1]）
  - `src.data.SyntheticImageDataset(num_samples: int, img_size: int = 128, seed: int = 0)`
    - `__len__ -> int`
    - `__getitem__(i) -> dict{"image": Tensor, "caption": str, "qa": list[tuple[str, str]]}`
  - `src.data.collate_vlm(batch: list[dict], tokenizer, num_image_tokens: int) -> dict`（返回 `input_ids/labels/pixel_values` 张量）

- [ ] **Step 1: 写失败测试** `tests/test_synthetic_data.py`

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_synthetic_data.py -v`
Expected: FAIL（`cannot import name 'SyntheticImageDataset'`）

- [ ] **Step 3: 追加实现到 `src/data.py`**

在文件末尾追加：

```python
# ---------------------------------------------------------------------------
# 视觉部分：合成图文数据 + 图像工具
# ---------------------------------------------------------------------------
COLORS = {"红色": (220, 40, 40), "绿色": (40, 200, 80), "蓝色": (50, 90, 230), "黄色": (240, 210, 50)}
SHAPES = ["圆形", "方形", "三角形"]
ZONES = {"左": 0.22, "中": 0.5, "右": 0.78}
_N_CN = {1: "一", 2: "两", 3: "三"}

_MEAN = 0.5
_STD = 0.5


def image_to_tensor(img):
    """PIL 图 -> (3,H,W) float32，归一化到 [-1,1]。用 numpy 实现，避免额外依赖。"""
    import numpy as np
    import torch

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _draw_sample(rng, size: int):
    """随机画 1~3 个彩色图形，并同时生成中文描述与问答（图与文本同源）。"""
    import random
    from PIL import Image, ImageDraw

    n = rng.randint(1, 3)
    # 三个槽位（左/中/右）随机选 n 个，保证位置不重叠
    zones = rng.sample(list(ZONES.keys()), n)
    zones.sort(key=lambda z: ZONES[z])
    objects = []
    for z in zones:
        shape = rng.choice(SHAPES)
        color = rng.choice(list(COLORS.keys()))
        objects.append((shape, color, z))

    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    r = size // 9
    for shape, color, z in objects:
        cx = int(ZONES[z] * size)
        cy = size // 2 + rng.randint(-size // 10, size // 10)
        rgb = COLORS[color]
        if shape == "圆形":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
        elif shape == "方形":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
        else:
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb)

    desc = "，".join(f"一个{color}的{shape}在{z}边" for shape, color, z in objects)
    caption = f"图中有{_N_CN[n]}个图形：{desc}。"

    qa = [("图中有几个图形？", f"{_N_CN[n]}个")]
    for shape, color, z in objects:
        qa.append((f"{z}边是什么图形？", f"{color}的{shape}"))
        qa.append((f"图中有{color}的{shape}吗？", "有"))
    # 加一个否定样本
    absent = rng.choice(SHAPES)
    if absent not in [o[0] for o in objects]:
        qa.append((f"图中有{absent}吗？", "没有"))
    return img, caption, qa


class SyntheticImageDataset(Dataset):
    """程序生成的几何图文数据：image + 中文描述 + 简单 VQA。"""

    def __init__(self, num_samples: int, img_size: int = 128, seed: int = 0):
        self.num_samples = num_samples
        self.img_size = img_size
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i: int):
        import random
        rng = random.Random(self.seed * 1000003 + i)
        img, caption, qa = _draw_sample(rng, self.img_size)
        return {"image": image_to_tensor(img), "caption": caption, "qa": qa}


def build_caption_example(tokenizer, caption: str, num_image_tokens: int):
    """构造"图片 -> 描述"样本：<bos> <image>*N <描述> <eos>，仅文本部分计入损失。"""
    bos = tokenizer.special_id("bos")
    eos = tokenizer.special_id("eos")
    img = tokenizer.special_id("image")
    text_ids = tokenizer.encode(caption)
    input_ids = [bos] + [img] * num_image_tokens + text_ids + [eos]
    labels = [-100] * (1 + num_image_tokens) + text_ids + [eos]
    return input_ids, labels


def build_vqa_example(tokenizer, question: str, answer: str, num_image_tokens: int):
    """构造 VQA 样本：<bos> <image>*N 问题 答：<答案> <eos>。"""
    bos = tokenizer.special_id("bos")
    eos = tokenizer.special_id("eos")
    img = tokenizer.special_id("image")
    prompt_ids = [bos] + [img] * num_image_tokens + tokenizer.encode(question + "答：")
    answer_ids = tokenizer.encode(answer) + [eos]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return input_ids, labels


def collate_vlm(batch: list[dict], tokenizer, num_image_tokens: int):
    """把一批样本整理成训练张量：图像 token 用 <image> 占位符，按 batch 内最大长度补齐。

    教学注释：batch 内各样本文本长度不同，需 padding 到同长；label 的 pad 位为 -100，
    由于是因果注意力，末尾的 pad 不会影响前面的预测。
    """
    import torch

    all_ids, all_labels, pix = [], [], []
    for item in batch:
        if "qa" in item and item.get("use_qa"):
            q, a = item["qa"][0]
            ids, labels = build_vqa_example(tokenizer, q, a, num_image_tokens)
        else:
            ids, labels = build_caption_example(tokenizer, item["caption"], num_image_tokens)
        all_ids.append(ids)
        all_labels.append(labels)
        pix.append(item["image"])

    pad_id = tokenizer.special_id("pad")
    max_len = max(len(x) for x in all_ids)
    input_ids, labels = [], []
    for ids, lab in zip(all_ids, all_labels):
        n_pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * n_pad)
        labels.append(lab + [-100] * n_pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "pixel_values": torch.stack(pix, dim=0),
    }
```

- [ ] **Step 4: 创建抽样脚本** `scripts/make_synthetic_samples.py`

```python
"""生成若干合成图文样本并保存为 png + 文本，肉眼确认图与描述一致。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import SyntheticImageDataset

OUT = Path("data/processed/image/synthetic_preview")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ds = SyntheticImageDataset(num_samples=6, img_size=128, seed=42)
    for i in range(len(ds)):
        item = ds[i]
        img = (item["image"] * 0.5 + 0.5).clamp(0, 1) * 255
        from PIL import Image
        import numpy as np
        Image.fromarray(img.permute(1, 2, 0).numpy().astype("uint8")).save(OUT / f"{i}.png")
        (OUT / f"{i}.txt").write_text(item["caption"], encoding="utf-8")
    print(f"已保存 {len(ds)} 组样本到 {OUT}")
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_synthetic_data.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/data.py scripts/make_synthetic_samples.py tests/test_synthetic_data.py
git commit -m "feat: synthetic geometric image-caption data and VLM collate"
```

---

### Task V2: 真实中文儿童图文数据下载与索引

**Files:**
- Modify: `src/data.py`（追加 `download_children_captions`, `build_caption_index`, `CaptionDataset`）
- Create: `scripts/prepare_image_data.py`
- Create: `tests/test_caption_dataset.py`

**Interfaces:**
- Consumes: `image_to_tensor`
- Produces:
  - `src.data.download_children_captions(dest_dir: str) -> str`
  - `src.data.build_caption_index(raw_dir: str, out_jsonl: str, max_items: int | None = None) -> int`
  - `src.data.CaptionDataset(index_jsonl: str, img_size: int = 128)`
    - `__len__ -> int`，`__getitem__(i) -> dict{"image": Tensor, "caption": str}`

- [ ] **Step 1: 写失败测试** `tests/test_caption_dataset.py`

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_caption_dataset.py -v`
Expected: FAIL（`cannot import name 'build_caption_index'`）

- [ ] **Step 3: 追加实现到 `src/data.py`**

```python
def download_children_captions(dest_dir: str) -> str:
    """下载中文儿童图文数据集（png + 同名 txt 描述）到项目内目录。"""
    from huggingface_hub import snapshot_download
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="svjack/Chinese_Children_Image_Captioning_Dataset_Split0",
        local_dir=dest_dir,
        allow_patterns=["*.png", "*.txt"],
    )
    return dest_dir


def build_caption_index(raw_dir: str, out_jsonl: str, max_items: int | None = None) -> int:
    """扫描 png，配对同名 txt，写出 {image, caption} 的 jsonl 索引。"""
    root = Path(raw_dir)
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for png in sorted(root.rglob("*.png")):
            txt = png.with_suffix(".txt")
            if not txt.exists():
                continue
            caption = txt.read_text(encoding="utf-8", errors="ignore").strip()
            if not caption:
                continue
            f.write(json.dumps({"image": str(png.relative_to(root)).replace("\\", "/"),
                                "caption": caption}, ensure_ascii=False) + "\n")
            count += 1
            if max_items is not None and count >= max_items:
                break
    return count


class CaptionDataset(Dataset):
    """真实图文数据集：按索引读取图片与中文描述。"""

    def __init__(self, index_jsonl: str, img_size: int = 128, raw_dir: str | None = None):
        self.items = [json.loads(l) for l in Path(index_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.img_size = img_size
        # index 位于 data/processed/image/children/index.jsonl，
        # parents[3] 即 data/ 目录；找不到时回退到约定路径
        if raw_dir:
            self.raw_dir = Path(raw_dir)
        else:
            self.raw_dir = Path(index_jsonl).resolve().parents[3] / "raw" / "image" / "children_caption"
        self.items = [it for it in self.items if (self.raw_dir / it["image"]).exists()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image
        it = self.items[i]
        img = Image.open(self.raw_dir / it["image"]).convert("RGB").resize((self.img_size, self.img_size))
        return {"image": image_to_tensor(img), "caption": it["caption"]}
```

- [ ] **Step 4: 创建下载与索引脚本** `scripts/prepare_image_data.py`

```python
"""下载真实中文儿童图文数据并建立索引。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import download_children_captions, build_caption_index


def main(max_items=None):
    raw = download_children_captions("data/raw/image/children_caption")
    n = build_caption_index(raw, "data/processed/image/children/index.jsonl", max_items=max_items)
    print(f"下载到 {raw}，建立索引 {n} 条")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_caption_dataset.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/data.py scripts/prepare_image_data.py tests/test_caption_dataset.py
git commit -m "feat: real Chinese image-caption download, index and dataset"
```

---

### Task V3: 手写 ViT 视觉编码器

**Files:**
- Create: `src/vision.py`
- Create: `tests/test_vision.py`

**Interfaces:**
- Consumes: `src.attention.MultiHeadAttention`, `build_rope_cache`
- Produces:
  - `src.vision.VisionEncoder(d_vision=384, depth=6, n_head=6, img_size=128, patch_size=16, in_chans=3, dropout=0.0)`
    - `forward(pixel_values: Tensor) -> Tensor`（`(B, N, d_vision)`）
    - `num_patches -> int`
    - `num_params() -> int`

- [ ] **Step 1: 写失败测试** `tests/test_vision.py`

```python
# 教学注释：只用**双向**注意力编码 patch 序列（无因果 mask），
# 输出每个 patch 的特征，供后续投影到语言模型维度。
import torch
from src.vision import VisionEncoder


def test_vit_output_shape_and_patch_count():
    vit = VisionEncoder(d_vision=64, depth=2, n_head=4, img_size=32, patch_size=16)
    x = torch.randn(2, 3, 32, 32)
    out = vit(x)
    assert vit.num_patches == 4          # (32/16)^2
    assert out.shape == (2, 4, 64)


def test_vit_is_bidirectional():
    torch.manual_seed(0)
    vit = VisionEncoder(d_vision=32, depth=1, n_head=4, img_size=32, patch_size=16)
    vit.eval()
    x = torch.randn(1, 3, 32, 32)
    o1 = vit(x)
    x2 = x.clone()
    x2[:, :, :16, :16] = torch.randn(3, 16, 16)  # 只改左上 patch
    o2 = vit(x2)
    # 双向注意力：改一个 patch 会影响**所有** patch 的输出
    assert not torch.allclose(o1[0, 0], o2[0, 0], atol=1e-5)
    assert not torch.allclose(o1[0, 3], o2[0, 3], atol=1e-5)


def test_vit_params_positive():
    vit = VisionEncoder(d_vision=64, depth=2, n_head=4, img_size=32, patch_size=16)
    assert vit.num_params() > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_vision.py -v`
Expected: FAIL（`No module named 'src.vision'`）

- [ ] **Step 3: 实现 `src/vision.py`**

```python
"""手写视觉编码器 ViT：patch embedding + 位置编码 + 双向 Transformer 编码。"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention, build_rope_cache


class PatchEmbed(nn.Module):
    """用 stride=patch_size 的卷积把图片切成 patch 并映射到 d_vision 维。"""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, d_vision: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size 必须能被 patch_size 整除"
        self.grid = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, d_vision, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                       # (B, d_vision, grid, grid)
        x = x.flatten(2).transpose(1, 2)       # (B, N, d_vision)
        return x


class ViTBlock(nn.Module):
    """双向 Transformer 块（复用 MultiHeadAttention，causal=False）。"""

    def __init__(self, d_vision: int, n_head: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_vision)
        self.attn = MultiHeadAttention(d_vision, n_head, n_head, dropout=dropout, causal=False)
        self.n2 = nn.LayerNorm(d_vision)
        hidden = int(d_vision * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_vision, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_vision), nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        h, _ = self.attn(self.n1(x), cos, sin)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, d_vision: int = 384, depth: int = 6, n_head: int = 6,
                 img_size: int = 128, patch_size: int = 16, in_chans: int = 3, dropout: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, d_vision)
        self.num_patches = self.patch_embed.grid ** 2
        # 可学习的位置编码（ViT 原论文做法）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_vision))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(d_vision, n_head, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_vision)
        head_dim = d_vision // n_head
        cos, sin = build_rope_cache(head_dim, self.num_patches, 10000.0, "cpu", torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, pixel_values):
        x = self.patch_embed(pixel_values) + self.pos_embed
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        return self.norm(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_vision.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/vision.py tests/test_vision.py
git commit -m "feat: hand-written ViT vision encoder"
```

---

### Task V4: 多模态模型（投影层 + 视觉特征替换 + 权重迁移）

**Files:**
- Create: `src/vlm.py`
- Modify: `src/trainer.py`（追加 `VLMTrainer`）
- Create: `tests/test_vlm.py`

**Interfaces:**
- Consumes: `src.model.GPT`, `src.vision.VisionEncoder`, `src.data.collate_vlm`, `Trainer`
- Produces:
  - `src.vlm.VisionProjector(d_vision, d_model)`
  - `src.vlm.MiniVLM(gpt: GPT, vision: VisionEncoder, projector: VisionProjector, image_token_id: int)`
    - `forward(input_ids, pixel_values, labels=None) -> (logits, loss, None)`
    - `replace_image_embeds(input_ids, pixel_values) -> Tensor`
    - `num_params() -> int`
  - `src.vlm.build_vlm(model_cfg, vlm_cfg, gpt_ckpt: str | None = None) -> MiniVLM`
  - `src.trainer.VLMTrainer(model, train_cfg, dataset, tokenizer, num_image_tokens, device=None)`

- [ ] **Step 1: 写失败测试** `tests/test_vlm.py`

```python
# 教学注释：验证三点——输出形状、图像位置标签被屏蔽、以及 patch 特征确实
# 被投影后替换进了 <image> 的位置（改变图片会改变 logits）。
import torch
from src.config import ModelConfig
from src.model import GPT
from src.vision import VisionEncoder
from src.vlm import VisionProjector, MiniVLM


def build_tiny(image_token_id=5, num_patches=4):
    gpt = GPT(ModelConfig(vocab_size=20, d_model=32, n_layer=1, n_head=4,
                          n_kv_head=2, d_ff=64, ctx_len=64))
    vit = VisionEncoder(d_vision=32, depth=1, n_head=4, img_size=32, patch_size=16)
    proj = VisionProjector(32, 32)
    return MiniVLM(gpt, vit, proj, image_token_id), num_patches


def test_vlm_forward_and_loss():
    model, N = build_tiny()
    B = 2
    # 文本 token 一律用 6..20，避免与 image id（5）混淆；第 0 位是 bos。
    input_ids = torch.randint(6, 20, (B, 1 + N + 6), dtype=torch.long)
    input_ids[:, 0] = 0
    input_ids[:, 1:1 + N] = 5
    labels = input_ids.clone()
    labels[:, :1 + N] = -100
    pix = torch.randn(B, 3, 32, 32)
    logits, loss, _ = model(input_ids, pix, labels)
    assert logits.shape[:2] == input_ids.shape
    assert loss.item() > 0


def test_vlm_label_masking_excludes_image_positions():
    model, N = build_tiny()
    input_ids = torch.randint(6, 20, (1, 1 + N + 3), dtype=torch.long)
    input_ids[0, 0] = 0
    input_ids[0, 1:1 + N] = 5
    labels = input_ids.clone()
    labels[:, :1 + N] = -100
    assert (labels[:, :1 + N] == -100).all()


def test_image_features_really_used():
    torch.manual_seed(0)
    model, N = build_tiny()
    model.eval()
    input_ids = torch.randint(6, 20, (1, 1 + N + 3), dtype=torch.long)
    input_ids[0, 0] = 0
    input_ids[0, 1:1 + N] = 5
    a = model(input_ids, torch.randn(1, 3, 32, 32))[0]
    b = model(input_ids, torch.randn(1, 3, 32, 32))[0]
    assert not torch.allclose(a, b, atol=1e-5), "换图片后输出应变化"


def test_build_vlm_loads_gpt_weights(tmp_path):
    from src.config import ModelConfig
    import torch as t
    gpt = GPT(ModelConfig(vocab_size=20, d_model=32, n_layer=1, n_head=4, n_kv_head=2, d_ff=64, ctx_len=64))
    ckpt = {"model": gpt.state_dict()}
    p = tmp_path / "g.pt"
    t.save(ckpt, str(p))
    from src.vlm import build_vlm
    model = build_vlm(gpt.cfg, {"d_vision": 32, "depth": 1, "n_head": 4, "img_size": 32, "patch_size": 16},
                      gpt_ckpt=str(p), image_token_id=5)
    assert t.allclose(model.gpt.tok_emb.weight, gpt.tok_emb.weight)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_vlm.py -v`
Expected: FAIL（`No module named 'src.vlm'`）

- [ ] **Step 3: 实现 `src/vlm.py`**

```python
"""多模态模型：ViT 视觉特征 -> MLP 投影 -> 替换 <image> 占位符的 embedding（LLaVA 式）。"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model import GPT
from src.vision import VisionEncoder


class VisionProjector(nn.Module):
    """把视觉特征投影到语言模型维度（两层 MLP + GELU）。"""

    def __init__(self, d_vision: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_vision, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MiniVLM(nn.Module):
    def __init__(self, gpt: GPT, vision: VisionEncoder, projector: VisionProjector, image_token_id: int):
        super().__init__()
        self.gpt = gpt
        self.vision = vision
        self.projector = projector
        self.image_token_id = image_token_id

    def replace_image_embeds(self, input_ids, pixel_values):
        """先把 token 转成 embedding，再用视觉特征填充 <image> 所在位置。"""
        embeds = self.gpt.embed(input_ids)                       # (B, T, d_model)
        vis = self.projector(self.vision(pixel_values))          # (B, N, d_model)
        mask = input_ids == self.image_token_id                  # (B, T)
        # 每行恰有 N 个 True，按行优先展开后与 vis.reshape(-1, d) 一一对应
        embeds = embeds.clone()
        embeds[mask] = vis.reshape(-1, vis.size(-1)).to(embeds.dtype)
        return embeds

    def forward(self, input_ids, pixel_values, labels=None):
        embeds = self.replace_image_embeds(input_ids, pixel_values)
        return self.gpt(inputs_embeds=embeds, targets=labels)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, train_cfg):
        """把**全部**参数（LLM + ViT + 投影层）分成衰减/不衰减两组。"""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        try:
            return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                     betas=(train_cfg.beta1, train_cfg.beta2),
                                     fused=torch.cuda.is_available())
        except TypeError:
            return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                     betas=(train_cfg.beta1, train_cfg.beta2))


def build_vlm(model_cfg, vlm_cfg: dict, gpt_ckpt: str | None = None, image_token_id: int = 0,
              device: str = "cpu") -> MiniVLM:
    """组装 MiniVLM；若给了 gpt_ckpt，则把 Plan 1 训练好的 LLM 权重迁移进来。"""
    gpt = GPT(model_cfg)
    if gpt_ckpt:
        ckpt = torch.load(gpt_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = gpt.load_state_dict(state, strict=False)
        print(f"LLM 权重迁移：missing={len(missing)} unexpected={len(unexpected)}")
    vision = VisionEncoder(
        d_vision=vlm_cfg.get("d_vision", 384),
        depth=vlm_cfg.get("depth", 6),
        n_head=vlm_cfg.get("n_head", 6),
        img_size=vlm_cfg.get("img_size", 128),
        patch_size=vlm_cfg.get("patch_size", 16),
    )
    projector = VisionProjector(vision.patch_embed.proj.out_channels, model_cfg.d_model)
    model = MiniVLM(gpt, vision, projector, image_token_id)
    return model.to(device)
```

- [ ] **Step 4: 追加 `VLMTrainer` 到 `src/trainer.py`**

```python
class VLMTrainer(Trainer):
    """多模态训练器：批数据由 collate_vlm 生成，loss 在图像位置上被屏蔽。"""

    def __init__(self, model, train_cfg, dataset, tokenizer, num_image_tokens,
                 device: str | None = None, use_qa: bool = False):
        from torch.utils.data import DataLoader

        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.use_qa = use_qa
        self.loader = DataLoader(
            dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True,
            collate_fn=self._collate, num_workers=0,
        )
        self._iter = None
        super().__init__(model, train_cfg, None, None, tokenizer=tokenizer, device=device)

    def _collate(self, batch):
        from src.data import collate_vlm
        for b in batch:
            b["use_qa"] = self.use_qa
        return collate_vlm(batch, self.tokenizer, self.num_image_tokens)

    @torch.no_grad()
    def _estimate_val(self) -> float:
        # 简化：用训练分布上的若干 batch 估计，避免额外验证集管线
        self.model.eval()
        losses = []
        for _ in range(max(1, min(5, self.cfg.eval_iters))):
            b = self._next_batch()
            with _autocast_ctx(self.cfg, self.device):
                _, loss, _ = self.model(b["input_ids"].to(self.device),
                                        b["pixel_values"].to(self.device),
                                        b["labels"].to(self.device))
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def _next_batch(self):
        """从 DataLoader 取一批（自动循环 epoch）。"""
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            b = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            b = next(self._iter)
        return b

    def _forward_loss(self, b, _unused=None):
        """基类以 (x, y) 调用；这里第一个参数即为 collate 后的 batch 字典。"""
        _, loss, _ = self.model(b["input_ids"].to(self.device),
                                b["pixel_values"].to(self.device),
                                b["labels"].to(self.device))
        return loss
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_vlm.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/vlm.py src/trainer.py tests/test_vlm.py
git commit -m "feat: MiniVLM with vision projector, feature splicing and weight transfer"
```

---

### Task V5: VLM 训练脚本、图文推理与 VQA

**Files:**
- Create: `scripts/train_vlm.py`
- Create: `src/vlm_generate.py`
- Create: `configs/vlm_children.yaml`
- Create: `tests/test_vlm_generate.py`

**Interfaces:**
- Consumes: `build_vlm`, `VLMTrainer`, `SyntheticImageDataset`, `CaptionDataset`, `build_caption_example`, `build_vqa_example`
- Produces:
  - `src.vlm_generate.generate_caption(model, tokenizer, image_tensor, max_new_tokens=64, temperature=0.0, device="cuda") -> str`
  - `src.vlm_generate.answer_question(model, tokenizer, image_tensor, question, max_new_tokens=32, temperature=0.0, device="cuda") -> str`

- [ ] **Step 1: 写失败测试** `tests/test_vlm_generate.py`

```python
# 教学注释：推理时复用训练同款 prompt 模板，才与训练分布一致。
import torch
from src.config import ModelConfig
from src.model import GPT
from src.vision import VisionEncoder
from src.vlm import VisionProjector, MiniVLM
from src.tokenizer import CharTokenizer
from src.vlm_generate import generate_caption, answer_question


def build(image_token_id=0):
    gpt = GPT(ModelConfig(vocab_size=50, d_model=32, n_layer=1, n_head=4,
                          n_kv_head=2, d_ff=64, ctx_len=128))
    vit = VisionEncoder(d_vision=32, depth=1, n_head=4, img_size=32, patch_size=16)
    return MiniVLM(gpt, vit, VisionProjector(32, 32), image_token_id=image_token_id)


def test_generate_caption_and_qa_return_strings():
    # 关键：image 特殊符号排在前面（id=0），字符 id 从 len(specials)=5 开始，避免冲突。
    specials = {"image": 0, "bos": 1, "eos": 2, "pad": 3, "unk": 4}
    tok = CharTokenizer(["你", "好", "红", "色", "圆", "形", "有", "没", "个"], specials)
    model = build(image_token_id=0)
    img = torch.randn(3, 32, 32)
    cap = generate_caption(model, tok, img, max_new_tokens=8, device="cpu")
    ans = answer_question(model, tok, img, "有几个图形？", max_new_tokens=4, device="cpu")
    assert isinstance(cap, str) and isinstance(ans, str)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_generate.py -v`
Expected: FAIL（`No module named 'src.vlm_generate'`）

- [ ] **Step 3: 实现 `src/vlm_generate.py`**

```python
"""VLM 推理：图片描述与 VQA（复用训练 prompt 模板 + KV 缓存）。"""
from __future__ import annotations

import torch

from src.data import build_caption_example, build_vqa_example
from src.generate import sample_logits


@torch.no_grad()
def _decode(model, tokenizer, prompt_ids, first_embeds, max_new_tokens,
            temperature, top_k, top_p, device):
    """从给定的首步 embedding 起，自回归解码。"""
    eos_id = tokenizer.special_id("eos")
    past = None
    # 第一步：整段 prompt（含视觉特征）作为输入
    logits, _, past = model.gpt(inputs_embeds=first_embeds.to(device), use_cache=True)
    out = []
    next_logits = logits[0, -1].float()
    for _ in range(max_new_tokens):
        tid = int(sample_logits(next_logits, temperature, top_k, top_p).item())
        if tid == eos_id:
            break
        out.append(tid)
        tok_t = torch.tensor([[tid]], device=device, dtype=torch.long)
        logits, _, past = model.gpt(inputs_embeds=model.gpt.embed(tok_t), past_kvs=past, use_cache=True)
        next_logits = logits[0, -1].float()
    return tokenizer.decode(out)


@torch.no_grad()
def generate_caption(model, tokenizer, image_tensor, max_new_tokens: int = 64,
                     temperature: float = 0.0, top_k=None, top_p: float | None = 0.9,
                     device: str = "cuda") -> str:
    """图片 -> 中文描述。"""
    model.eval().to(device)
    n_img = model.vision.num_patches
    prompt_ids = [tokenizer.special_id("bos")] + [model.image_token_id] * n_img
    input_ids = torch.tensor([prompt_ids], device=device)
    pix = image_tensor.unsqueeze(0).to(device)
    embeds = model.replace_image_embeds(input_ids, pix)
    return _decode(model, tokenizer, prompt_ids, embeds, max_new_tokens,
                   temperature, top_k, top_p, device)


@torch.no_grad()
def answer_question(model, tokenizer, image_tensor, question: str, max_new_tokens: int = 32,
                    temperature: float = 0.0, top_k=None, top_p: float | None = 0.9,
                    device: str = "cuda") -> str:
    """VQA：图片 + 问题 -> 答案。"""
    model.eval().to(device)
    n_img = model.vision.num_patches
    prompt_ids = ([tokenizer.special_id("bos")] + [model.image_token_id] * n_img
                  + tokenizer.encode(question + "答："))
    input_ids = torch.tensor([prompt_ids], device=device)
    pix = image_tensor.unsqueeze(0).to(device)
    embeds = model.replace_image_embeds(input_ids, pix)
    return _decode(model, tokenizer, prompt_ids, embeds, max_new_tokens,
                   temperature, top_k, top_p, device)
```

- [ ] **Step 4: 创建训练脚本** `scripts/train_vlm.py`

```python
"""训练 MiniVLM：合成数据快速验证 -> 真实儿童图文 + VQA。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml

from src.config import ModelConfig, TrainConfig, load_config, apply_overrides
from src.data import SyntheticImageDataset, CaptionDataset
from src.tokenizer import QwenTokenizer
from src.trainer import VLMTrainer
from src.utils import set_seed, human_params
from src.vlm import build_vlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm_children.yaml")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = apply_overrides(load_config(args.config), args.set)
    set_seed(cfg.train.seed)

    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
    cfg.model.vocab_size = tok.vocab_size

    vlm_cfg = raw.get("vlm", {})
    model = build_vlm(cfg.model, vlm_cfg, gpt_ckpt=raw.get("gpt_ckpt"), image_token_id=tok.special_id("image"))
    n_img = model.vision.num_patches
    print(f"MiniVLM 参数量 {human_params(model.num_params())}，图像 patch 数 {n_img}")

    data_kind = raw.get("data_kind", "synthetic")
    use_qa = bool(raw.get("use_qa", False))
    if data_kind == "synthetic":
        dataset = SyntheticImageDataset(num_samples=raw.get("num_samples", 20000),
                                        img_size=vlm_cfg.get("img_size", 128), seed=cfg.train.seed)
    else:
        dataset = CaptionDataset("data/processed/image/children/index.jsonl",
                                 img_size=vlm_cfg.get("img_size", 128))

    trainer = VLMTrainer(model, cfg.train, dataset, tok, num_image_tokens=n_img, use_qa=use_qa)
    if args.resume:
        print(f"从 {args.resume} 恢复，已完成 {trainer.load(args.resume)} 步")
    trainer.train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 创建配置** `configs/vlm_children.yaml`

```yaml
gpt_ckpt: out/gpt/latest.pt
data_kind: synthetic          # synthetic | children
use_qa: false
num_samples: 20000
vlm:
  d_vision: 384
  depth: 6
  n_head: 6
  img_size: 128
  patch_size: 16
model:
  vocab_size: 151937          # 会被 tokenizer 实际词表覆盖
  d_model: 768
  n_layer: 12
  n_head: 12
  n_kv_head: 4
  d_ff: 2048
  ctx_len: 1024
data:
  tokenizer_kind: qwen
  tokenizer_dir: data/tokenizer/qwen2.5-0.5b
train:
  batch_size: 8
  grad_accum: 2
  lr: 1.0e-4
  min_lr: 1.0e-5
  warmup_steps: 100
  max_steps: 4000
  eval_interval: 200
  save_interval: 500
  log_interval: 10
  out_dir: out/vlm
  dtype: bf16
  compile: false
  seed: 42
```

- [ ] **Step 6: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_vlm_generate.py -v`
Expected: PASS

- [ ] **Step 7: 冒烟训练（合成数据，少量步数链路验证）**

Run: `.venv\Scripts\python scripts/train_vlm.py --set train.max_steps=100 train.save_interval=50 train.batch_size=4`
Expected: 打印 loss 并生成 `out/vlm/latest.pt`、`out/vlm/loss.png`

- [ ] **Step 8: 过拟合小样本验证（合成数据应当学到颜色/形状/位置）**

Run: `.venv\Scripts\python scripts/train_vlm.py --set train.max_steps=800 train.lr=3e-4`
Expected: loss 明显下降（<1.0 量级）；用 `generate_caption` 检查描述接近真实

- [ ] **Step 9: 提交**

```bash
git add scripts/train_vlm.py src/vlm_generate.py configs/vlm_children.yaml tests/test_vlm_generate.py
git commit -m "feat: VLM training script, captioning and VQA inference"
```

---

### Task V6: Gradio 网页（文本 + 图片）与 VLM 原理文档

**Files:**
- Create: `scripts/webapp.py`
- Create: `docs/04-vlm.md`
- Modify: `README.md`（追加 VLM 使用说明）

**Interfaces:**
- Consumes: `generate`, `generate_caption`, `answer_question`, `GPT`, `MiniVLM`, tokenizers
- Produces: 可启动的网页应用

- [ ] **Step 1: 实现 `scripts/webapp.py`**

```python
"""Gradio 网页：文本聊天 + 图片描述/VQA。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
import torch
import gradio as gr
from PIL import Image

from src.config import ModelConfig, load_config
from src.model import GPT
from src.vlm import build_vlm
from src.tokenizer import QwenTokenizer
from src.generate import generate
from src.vlm_generate import generate_caption, answer_question
from src.data import image_to_tensor

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(args):
    tok = QwenTokenizer.load(args.tokenizer_dir)

    text_cfg = load_config(args.text_config)
    text_cfg.model.vocab_size = tok.vocab_size
    text_model = GPT(text_cfg.model)
    text_model.load_state_dict(torch.load(args.text_ckpt, map_location="cpu", weights_only=False)["model"])

    vlm = None
    if Path(args.vlm_ckpt).exists():
        raw = yaml.safe_load(Path(args.vlm_config).read_text(encoding="utf-8"))
        vlm_cfg = dict(raw["model"])
        vlm_cfg["vocab_size"] = tok.vocab_size
        vlm = build_vlm(ModelConfig(**vlm_cfg), raw.get("vlm", {}),
                        gpt_ckpt=None, image_token_id=tok.special_id("image"), device=DEV)
        vlm.load_state_dict(torch.load(args.vlm_ckpt, map_location="cpu", weights_only=False)["model"])
    return tok, text_model, vlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", default="data/tokenizer/qwen2.5-0.5b")
    ap.add_argument("--text_config", default="configs/gpt_tinystories.yaml")
    ap.add_argument("--text_ckpt", default="out/gpt/latest.pt")
    ap.add_argument("--vlm_config", default="configs/vlm_children.yaml")
    ap.add_argument("--vlm_ckpt", default="out/vlm/latest.pt")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    tok, text_model, vlm = load_models(args)

    def chat_fn(prompt, max_tokens, temperature, top_p):
        return generate(text_model, tok, prompt, max_new_tokens=int(max_tokens),
                        temperature=float(temperature), top_p=float(top_p), device=DEV)

    def caption_fn(image, question):
        if vlm is None:
            return "未找到 VLM checkpoint，请先训练：python scripts/train_vlm.py"
        if image is None:
            return "请上传图片"
        img = image.convert("RGB").resize((vlm.vision.patch_embed.grid * vlm.vision.patch_embed.proj.kernel_size[0],) * 2)
        tensor = image_to_tensor(img)
        if question and question.strip():
            return answer_question(vlm, tok, tensor, question.strip(), device=DEV)
        return generate_caption(vlm, tok, tensor, device=DEV)

    with gr.Blocks(title="mini-llm-lab") as demo:
        gr.Markdown("# mini-llm-lab（从零手写 LLM + VLM）")
        with gr.Tab("文本聊天"):
            prompt = gr.Textbox(label="提示词")
            with gr.Row():
                max_tokens = gr.Slider(16, 256, value=128, step=16, label="生成长度")
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
            out = gr.Textbox(label="生成结果")
            gr.Button("生成").click(chat_fn, [prompt, max_tokens, temperature, top_p], out)
        with gr.Tab("图片描述 / VQA"):
            image = gr.Image(type="pil", label="上传图片")
            question = gr.Textbox(label="问题（留空 = 生成描述）", placeholder="例如：图中有几个图形？")
            out2 = gr.Textbox(label="回答")
            gr.Button("运行").click(caption_fn, [image, question], out2)

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 `docs/04-vlm.md`**

讲解内容：为什么语言模型"看不见"；ViT 如何把图片变成 patch 序列；
投影层的作用（对齐两种特征空间）；为什么用 `<image>` 占位符 + 特征替换（LLaVA 式）；
为什么损失只算文本；如何用合成数据自检；从语言模型迁移权重（"先教说话再装眼睛"）；
caption 与 VQA prompt 模板的构造；常见失败与排查（描述变成通用套话 = 视觉特征没被用上）。

- [ ] **Step 3: 更新 `README.md`**

追加：`scripts/prepare_image_data.py`、`scripts/train_vlm.py`（合成 → 真实两阶段）、
`scripts/webapp.py` 的用法，以及"VLM 效果自检方法"。

- [ ] **Step 4: 启动冒烟测试（人工确认可打开）**

Run: `.venv\Scripts\python scripts/webapp.py`
Expected: 终端出现 `Running on local URL: http://127.0.0.1:7860`；浏览器能打开两个 Tab，文本与图片链路均可用（Ctrl+C 退出）

- [ ] **Step 5: 提交**

```bash
git add scripts/webapp.py docs/04-vlm.md README.md
git commit -m "feat: Gradio web app for text chat and image caption/VQA, plus VLM guide"
```

---

## 验收（Plan 2）

- `pytest tests/ -v` 全部通过。
- `scripts/prepare_image_data.py` 能下载真实图文并生成索引。
- `scripts/train_vlm.py` 在合成数据上 loss 明显下降，`generate_caption` 能正确描述颜色/形状/位置。
- `answer_question` 能回答合成数据的计数/颜色/有无问题。
- 切换到 `data_kind: children` 后能在真实儿童图文上产出通顺中文描述。
- `scripts/webapp.py` 可启动，文本与图片两个 Tab 均可用。
