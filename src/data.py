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


def build_text_bin(
    tokenizer,
    texts: Iterator[str],
    out_path: str,
    val_bin_path: str | None = None,
    max_docs: int | None = None,
    val_ratio: float = 0.01,
) -> tuple[int, int]:
    """把所有文本编码成一个 uint32 数组并按 val_ratio 切分。

    教学注释：Qwen 词表 >65535，所以用 uint32；先全量编码再切分，
    避免训练时反复调用分词器（分词是 CPU 瓶颈）。
    """
    import numpy as np

    eos = tokenizer.special_id("eos")
    ids: list[int] = []
    for i, t in enumerate(texts):
        if max_docs is not None and i >= max_docs:
            break
        ids.extend(tokenizer.encode(t))
        ids.append(eos)

    n_val = int(len(ids) * val_ratio)
    train_ids = ids[: len(ids) - n_val]
    val_ids = ids[len(ids) - n_val:]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.array(train_ids, dtype=np.uint32).tofile(out_path)
    if val_bin_path:
        Path(val_bin_path).parent.mkdir(parents=True, exist_ok=True)
        np.array(val_ids, dtype=np.uint32).tofile(val_bin_path)
    return len(train_ids), len(val_ids)


def load_bin(path: str):
    """以内存映射方式打开 .bin，避免一次性载入内存。"""
    import numpy as np
    return np.memmap(path, dtype=np.uint32, mode="r")


def get_batch(data, batch_size: int, ctx_len: int, device: str, generator=None):
    """随机采样 batch_size 个长度为 ctx_len 的片段，返回 (x, y=右移一位)。"""
    import numpy as np
    import torch

    max_start = len(data) - ctx_len - 1
    if max_start <= 0:
        raise ValueError(f"数据太短：len={len(data)}, ctx_len={ctx_len}")
    ix = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([np.asarray(data[i: i + ctx_len], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(data[i + 1: i + 1 + ctx_len], dtype=np.int64) for i in ix])
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y


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
    """构造"图片 -> 描述"样本：<bos> <image>*N <描述> <eos>，仅文本部分计入损失。

    教学注释：因果语言模型预测的是"下一个 token"，而 GPT.forward 用
    `logits[t]` 对齐 `labels[t]`，所以 labels 必须是 input_ids **右移一位**，
    并在不需要预测的位置填 -100（图像占位与序列末尾）。
    最后一张图的向量负责预测第一个文本 token，最后一个文本 token 负责预测 eos。
    """
    bos = tokenizer.special_id("bos")
    eos = tokenizer.special_id("eos")
    img = tokenizer.special_id("image")
    text_ids = tokenizer.encode(caption)
    input_ids = [bos] + [img] * num_image_tokens + text_ids + [eos]
    labels = [-100] * num_image_tokens + text_ids + [eos] + [-100]
    return input_ids, labels


def build_vqa_example(tokenizer, question: str, answer: str, num_image_tokens: int):
    """构造 VQA 样本：<bos> <image>*N 问题 答：<答案> <eos>。

    同样地，labels 右移一位：从"答："之后开始监督答案，最后一个 token 预测 eos。
    """
    bos = tokenizer.special_id("bos")
    eos = tokenizer.special_id("eos")
    img = tokenizer.special_id("image")
    prompt_ids = [bos] + [img] * num_image_tokens + tokenizer.encode(question + "答：")
    answer_ids = tokenizer.encode(answer) + [eos]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * (len(prompt_ids) - 1) + answer_ids + [-100]
    return input_ids, labels


def collate_vlm(batch: list[dict], tokenizer, num_image_tokens: int):
    """把一批样本整理成训练张量：图像 token 用 <image> 占位符，按 batch 内最大长度补齐。

    教学注释：batch 内各样本文本长度不同，需 padding 到同长；label 的 pad 位为 -100，
    由于是因果注意力，末尾的 pad 不会影响前面的预测。
    """
    import torch

    import random

    all_ids, all_labels, pix = [], [], []
    for item in batch:
        if "qa" in item and item.get("use_qa"):
            # 教学注释：qa[0] 恒为"数图形"问题，若只取它则形状/颜色/位置/
            # 否定等题型永远学不到；随机抽一题使各种问题类型都被覆盖。
            q, a = item["qa"][random.randrange(len(item["qa"]))]
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


# ---------------------------------------------------------------------------
# 真实中文儿童图文数据：下载、建索引、读取
# ---------------------------------------------------------------------------
def download_children_captions(dest_dir: str) -> str:
    """下载中文儿童图文数据集（png + 同名 txt 描述）到项目内目录。

    教学注释：repo 是 dataset 类型，必须显式指定 repo_type，
    否则会在 model 命名空间 404；只拉 png/txt，避免下载无关文件。
    """
    from huggingface_hub import snapshot_download
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="svjack/Chinese_Children_Image_Captioning_Dataset_Split0",
        repo_type="dataset",
        local_dir=dest_dir,
        allow_patterns=["*.png", "*.txt"],
    )
    return dest_dir


def build_caption_index(raw_dir: str, out_jsonl: str, max_items: int | None = None) -> int:
    """扫描 png，配对同名 txt，写出 {image, caption} 的 jsonl 索引。

    教学注释：索引只存相对路径（正斜杠，跨平台一致），
    图片文件可能在磁盘上移动，读取时再由 Dataset 拼接绝对路径。
    max_items 用于大仓库时截断，保证建索引过程可控。
    """
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
    """真实图文数据集：按索引读取图片与中文描述。

    教学注释：__getitem__ 里按需打开并缩放图片，避免一次性把所有图读进内存。
    """

    def __init__(self, index_jsonl: str, img_size: int = 128, raw_dir: str | None = None):
        self.items = [json.loads(l) for l in Path(index_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.img_size = img_size
        # index 位于 data/processed/image/children/index.jsonl，
        # parents[3] 即 data/ 目录；找不到时回退到约定路径
        if raw_dir:
            self.raw_dir = Path(raw_dir)
        else:
            self.raw_dir = Path(index_jsonl).resolve().parents[3] / "raw" / "image" / "children_caption"
        # 教学注释：过滤掉图片已被删除的索引项，保证 len 与实际可读取数量一致。
        self.items = [it for it in self.items if (self.raw_dir / it["image"]).exists()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image
        it = self.items[i]
        img = Image.open(self.raw_dir / it["image"]).convert("RGB").resize((self.img_size, self.img_size))
        return {"image": image_to_tensor(img), "caption": it["caption"]}
