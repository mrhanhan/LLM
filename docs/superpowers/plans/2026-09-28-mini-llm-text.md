# mini-llm-lab — 文本 LLM 实现计划（Plan 1/2）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零手写一个中文 GPT（默认 Qwen 分词器，主干 ~80M），完成「分词 → 数据 → 模型 → 训练 → 生成/聊天」闭环。

**Architecture:** 纯 PyTorch 手写 Transformer（RoPE + RMSNorm + SwiGLU + GQA + 权重共享），配置驱动，bf16 + 梯度累积，支持断点续训与 loss 曲线。全部数据落在 `data/` 下分层目录。

**Tech Stack:** Python 3.12（`.venv`）、PyTorch 2.14、numpy、PyYAML、huggingface_hub、transformers（仅取 Qwen 分词器）、matplotlib、pytest。

**Spec:** `docs/superpowers/specs/2026-09-28-mini-llm-lab-design.md`

## Global Constraints

- 运行环境：Windows + PowerShell 7；解释器一律用 `P:\Demo\LLM\.venv\Scripts\python.exe`。
- **所有**数据/分词器/产物放在 `data/` 下：`data/tokenizer/`、`data/raw/`、`data/processed/`；不得依赖 `~/.cache/huggingface`。
- 下载统一设 `HF_ENDPOINT=https://hf-mirror.com`。
- 默认模型结构：`d_model=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2048, ctx_len=1024`；默认分词器 = Qwen（本地目录 `data/tokenizer/qwen2.5-0.5b`）。
- `torch.compile`：配置开关，默认 `false`；开启失败自动回退 eager 并打印原因。
- 代码需带**详细中文注释**（本项目为教学用途，覆盖默认"不写注释"规则）。
- checkpoint/日志/曲线统一输出到 `out/`。
- 训练精度 bf16（autocast），主权重 fp32。

---

### Task 1: 项目骨架、配置系统与工具函数

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/utils.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `tests/test_utils.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `src.config.ModelConfig / TrainConfig / DataConfig / Config`
  - `src.config.load_config(path) -> Config`
  - `src.config.apply_overrides(cfg: Config, overrides: list[str]) -> Config`（格式 `"train.lr=3e-4"`）
  - `src.config.save_config(cfg: Config, path) -> None`
  - `src.utils.set_seed(seed: int) -> None`
  - `src.utils.get_lr(step: int, cfg: TrainConfig) -> float`
  - `src.utils.human_params(n: int) -> str`
  - `src.utils.plot_loss(jsonl_path: str, out_png: str) -> None`

- [ ] **Step 1: 创建骨架文件**

`.gitignore`:
```gitignore
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
out/
*.png
!docs/**/*.png
```

`requirements.txt`:
```text
torch>=2.4
numpy
PyYAML
pillow
huggingface_hub
transformers
matplotlib
pytest
gradio
```

`src/__init__.py` 与 `tests/__init__.py` 为空文件。

- [ ] **Step 2: 写失败测试** `tests/test_config.py`

```python
# 教学注释：配置系统测试——加载 YAML、命令行覆盖、保存回读一致性。
import textwrap
from pathlib import Path
from src.config import load_config, apply_overrides, save_config


def test_load_config_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  n_layer: 4\ntrain:\n  lr: 0.001\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.model.n_layer == 4
    assert cfg.train.lr == 0.001
    # 未提供的字段用默认值
    assert cfg.model.d_model == 768
    assert cfg.data.tokenizer_kind == "qwen"


def test_apply_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lr: 0.001\n", encoding="utf-8")
    cfg = load_config(str(p))
    cfg = apply_overrides(cfg, ["train.lr=3e-4", "model.n_layer=2", "train.compile=true"])
    assert cfg.train.lr == 3e-4
    assert cfg.model.n_layer == 2
    assert cfg.train.compile is True


def test_save_and_reload(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  d_model: 256\n", encoding="utf-8")
    cfg = load_config(str(p))
    out = tmp_path / "saved.yaml"
    save_config(cfg, str(out))
    cfg2 = load_config(str(out))
    assert cfg2.model.d_model == 256
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.config'`）

- [ ] **Step 4: 实现 `src/config.py`**

```python
"""配置系统：YAML <-> dataclass，支持命令行覆盖。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """模型结构超参（教学注释：这些数字决定参数量与显存占用）。"""
    vocab_size: int = 151936      # Qwen2.5 词表大小；用自定义 BPE 时改为 16384
    d_model: int = 768            # 隐藏维度
    n_layer: int = 12             # Transformer 层数
    n_head: int = 12              # 查询头数
    n_kv_head: int = 4            # KV 头数（GQA：少于 n_head 可省显存）
    d_ff: int = 2048              # SwiGLU 中间维度
    ctx_len: int = 1024           # 最大上下文长度
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True   # 输入 embedding 与输出投影权重共享


@dataclass
class TrainConfig:
    batch_size: int = 8           # 单次前向的 micro-batch
    grad_accum: int = 8           # 梯度累积步数（等效 batch = batch_size*grad_accum）
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    max_steps: int = 20000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 100
    save_interval: int = 1000
    log_interval: int = 20
    out_dir: str = "out/gpt"
    dtype: str = "bf16"           # bf16 | fp32
    compile: bool = False         # Windows 上默认关闭
    seed: int = 42


@dataclass
class DataConfig:
    tokenizer_kind: str = "qwen"  # qwen | bpe | char
    tokenizer_dir: str = "data/tokenizer/qwen2.5-0.5b"
    train_bin: str = "data/processed/text/train.bin"
    val_bin: str = "data/processed/text/val.bin"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        data=DataConfig(**raw.get("data", {})),
    )


def _coerce(value: str, target_type):
    """把字符串命令行的值转成目标字段类型。"""
    if target_type is bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """支持形如 'train.lr=3e-4' 的命令行覆盖。"""
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=")[0]:
            raise ValueError(f"非法覆盖参数: {item!r}，应形如 train.lr=3e-4")
        key, value = item.split("=", 1)
        section, attr = key.split(".", 1)
        obj = getattr(cfg, section)
        if not hasattr(obj, attr):
            raise AttributeError(f"未知配置项: {section}.{attr}")
        target_type = type(getattr(obj, attr))
        setattr(obj, attr, _coerce(value, target_type))
    return cfg


def save_config(cfg: Config, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
```

- [ ] **Step 5: 写测试** `tests/test_utils.py`

```python
# 教学注释：验证学习率调度（warmup+余弦）与参数量格式化。
import json
from src.config import TrainConfig
from src.utils import set_seed, get_lr, human_params


def test_get_lr_warmup_and_decay():
    cfg = TrainConfig(warmup_steps=10, max_steps=110, lr=1.0, min_lr=0.0)
    # warmup 阶段线性上升
    assert get_lr(0, cfg) == 0.1
    assert abs(get_lr(9, cfg) - 1.0) < 1e-9
    # 训练结束后固定在 min_lr
    assert get_lr(110, cfg) == 0.0
    # 中点在 warmup 与 min 之间
    mid = get_lr(60, cfg)
    assert 0.0 <= mid <= 1.0


def test_human_params():
    assert human_params(1_234) == "1.2K"
    assert human_params(2_500_000) == "2.5M"
    assert human_params(3_100_000_000) == "3.1B"


def test_set_seed_reproducible():
    import torch
    set_seed(0)
    a = torch.randn(3)
    set_seed(0)
    b = torch.randn(3)
    assert torch.equal(a, b)
```

- [ ] **Step 6: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_utils.py -v`
Expected: FAIL（`No module named 'src.utils'`）

- [ ] **Step 7: 实现 `src/utils.py`**

```python
"""通用工具：随机种子、学习率调度、参数量格式化、日志、loss 曲线。"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch

from src.config import TrainConfig


def set_seed(seed: int) -> None:
    """固定 python / numpy / torch 随机种子，保证可复现。"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, cfg: TrainConfig) -> float:
    """warmup 线性上升 + 余弦退火到 min_lr。"""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * ratio))


def human_params(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


def gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    used = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{used:.2f}/{total:.1f}GB"


class JsonlLogger:
    """把每步指标追加写入 metrics.jsonl，便于之后画曲线。"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def plot_loss(jsonl_path: str, out_png: str) -> None:
    """读取 metrics.jsonl，画出 train/val loss 曲线（教学：观察是否收敛/过拟合）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses, val_steps, val_losses = [], [], [], []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if "loss" in rec and rec.get("split") == "train":
            steps.append(rec["step"])
            losses.append(rec["loss"])
        if "val_loss" in rec:
            val_steps.append(rec["step"])
            val_losses.append(rec["val_loss"])

    plt.figure(figsize=(8, 5))
    if steps:
        plt.plot(steps, losses, label="train loss")
    if val_steps:
        plt.plot(val_steps, val_losses, label="val loss", marker="o", ms=3)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
```

- [ ] **Step 8: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_config.py tests/test_utils.py -v`
Expected: PASS（全部）

- [ ] **Step 9: 初始化仓库并提交**

```bash
git init
git add .gitignore requirements.txt src/__init__.py src/config.py src/utils.py tests/__init__.py tests/test_config.py tests/test_utils.py
git commit -m "chore: project skeleton, config system and utils"
```

---

### Task 2: 手写字节级 BPE 分词器 + 字符级分词器

**Files:**
- Create: `src/tokenizer.py`
- Create: `tests/test_tokenizer.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `src.tokenizer.SPECIALS: dict[str, str]`（`pad/bos/eos/unk/image`）
  - `src.tokenizer.CharTokenizer.train(texts, save_dir, specials=SPECIALS, max_chars=2_000_000) -> CharTokenizer`
  - `src.tokenizer.CharTokenizer.load(save_dir) -> CharTokenizer`
  - `src.tokenizer.BPETokenizer.train(texts, vocab_size=16384, min_frequency=2, specials=SPECIALS, max_bytes=20_000_000, verbose=True) -> BPETokenizer`
  - `src.tokenizer.BPETokenizer.load(save_dir) -> BPETokenizer`
  - 两者均提供：`encode(text, add_bos=False, add_eos=False) -> list[int]`、`decode(ids) -> str`、`vocab_size`、`special_id(name) -> int`
  - `save(save_dir) -> None`

- [ ] **Step 1: 写失败测试** `tests/test_tokenizer.py`

```python
# 教学注释：分词器必须满足 decode(encode(x)) == x（字节级可逆），
# 特殊符号不参与 BPE 合并，且能稳定保存/加载。
from src.tokenizer import BPETokenizer, CharTokenizer, SPECIALS


CORPUS = [
    "从前有座山，山里有座庙。",
    "小猫在草地上跑来跑去，非常开心。",
    "The quick brown fox jumps over the lazy dog.",
    "太阳 sun 月亮 moon 星星 star。",
] * 40


def test_bpe_roundtrip():
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=600, min_frequency=1, verbose=False)
    for s in CORPUS[:5]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_specials_present():
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=600, min_frequency=1, verbose=False)
    for name in SPECIALS:
        assert 0 <= tok.special_id(name) < tok.vocab_size


def test_bpe_save_load(tmp_path):
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=600, min_frequency=1, verbose=False)
    tok.save(str(tmp_path))
    tok2 = BPETokenizer.load(str(tmp_path))
    s = "小猫在草地上"
    assert tok2.decode(tok2.encode(s)) == s
    assert tok2.vocab_size == tok.vocab_size


def test_bpe_bos_eos():
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=600, min_frequency=1, verbose=False)
    ids = tok.encode("你好", add_bos=True, add_eos=True)
    assert ids[0] == tok.special_id("bos")
    assert ids[-1] == tok.special_id("eos")


def test_char_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer.train(iter(CORPUS))
    for s in CORPUS[:5]:
        assert tok.decode(tok.encode(s)) == s
    tok.save(str(tmp_path))
    tok2 = CharTokenizer.load(str(tmp_path))
    assert tok2.decode(tok2.encode("小猫")) == "小猫"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_tokenizer.py -v`
Expected: FAIL（`No module named 'src.tokenizer'`）

- [ ] **Step 3: 实现 `src/tokenizer.py`**

```python
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
        return self._base_size + len(self.merges) + len(self._special_to_id)

    def _special_map(self) -> dict[str, int]:
        return self._special_to_id

    def _ids_for_text(self, text: str) -> list[int]:
        """把普通文本转成 id 序列（不含特殊符号）。"""
        ids: list[int] = []
        for chunk in _PRE_TOKEN.findall(text):
            symbols = tuple(chunk.encode("utf-8"))
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
                symbols = _merge(symbols, pair, self.merge_ranks[pair] + self._byte_offset)
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
        word_freq: Counter = Counter()
        consumed = 0
        for text in texts:
            for chunk in _PRE_TOKEN.findall(text):
                word_freq[tuple(chunk.encode("utf-8"))] += 1
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_tokenizer.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add src/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: hand-written byte-level BPE and char tokenizers"
```

---

### Task 3: Qwen 分词器落地到项目目录 + 适配

**Files:**
- Modify: `src/tokenizer.py`（追加 `QwenTokenizer`）
- Create: `src/data.py`（本任务只加下载与文本工具，Dataset 在 Task 4）
- Create: `scripts/prepare_data_part1.py`
- Create: `tests/test_qwen_tokenizer.py`

**Interfaces:**
- Consumes: `src.tokenizer.SPECIALS`
- Produces:
  - `src.data.download_qwen_tokenizer(dest_dir: str) -> str`（返回本地目录）
  - `src.data.download_tinystories(dest_dir: str) -> list[str]`（返回 jsonl 路径列表）
  - `src.data.iter_texts(paths: list[str]) -> Iterator[str]`
  - `src.tokenizer.QwenTokenizer.load(local_dir: str, add_image_token: bool = True) -> QwenTokenizer`
  - `QwenTokenizer.encode/decode/vocab_size/special_id/save`

- [ ] **Step 1: 实现下载工具 `src/data.py`**

```python
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
        local_dir=dest_dir,
    )
    jsonl_dir = root / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    with _tar.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".jsonl")]
        for m in members:
            m.name = Path(m.name).name  # 去掉目录前缀，避免路径穿越
            tar.extract(m, jsonl_dir)
    return sorted(str(p) for p in jsonl_dir.glob("*.jsonl"))


def iter_texts(paths: list[str]) -> Iterator[str]:
    """逐行读取 jsonl，兼容 story/text/content 三种字段名。"""
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("story") or obj.get("text") or obj.get("content")
                if text:
                    yield text
```

- [ ] **Step 2: 写失败测试** `tests/test_qwen_tokenizer.py`

```python
# 教学注释：Qwen 分词器从项目内目录加载；<image> 特殊符号需可用。
import shutil
from pathlib import Path
import pytest
from src.tokenizer import QwenTokenizer

QWEN_DIR = "data/tokenizer/qwen2.5-0.5b"


@pytest.mark.skipif(not Path(QWEN_DIR).exists(), reason="需先运行 scripts/prepare_data_part1.py 下载分词器")
def test_qwen_roundtrip_and_image_token():
    tok = QwenTokenizer.load(QWEN_DIR)
    s = "你好，世界！Hello 123。"
    assert tok.decode(tok.encode(s)) == s
    assert 0 <= tok.special_id("image") < tok.vocab_size
    ids = tok.encode("测试", add_bos=True, add_eos=True)
    assert len(ids) == len(tok.encode("测试")) + 2
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_qwen_tokenizer.py -v`
Expected: FAIL（`QwenTokenizer` 未定义）或 SKIP（若未下载）——先看报错信息

- [ ] **Step 4: 追加 `QwenTokenizer` 到 `src/tokenizer.py`**

在文件末尾追加：

```python
class QwenTokenizer(_BaseTokenizer):
    """适配 Qwen2.5 分词器（从项目内本地目录加载）。

    教学注释：Qwen 用 byte-level BPE，词表约 15 万，中文/代码表现好；
    我们额外注册一个 <image> 特殊符号供后续 VLM 使用，并把改动持久化到本地目录。
    """

    def __init__(self, hf, image_token_id: int):
        self.hf = hf
        self._image_token_id = image_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.hf)

    def _special_map(self) -> dict[str, int]:
        hf = self.hf
        pad = hf.pad_token_id if hf.pad_token_id is not None else hf.eos_token_id
        bos = hf.bos_token_id if hf.bos_token_id is not None else hf.eos_token_id
        eos = hf.eos_token_id
        unk = hf.unk_token_id if hf.unk_token_id is not None else hf.eos_token_id
        return {"pad": pad, "bos": bos, "eos": eos, "unk": unk, "image": self._image_token_id}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.hf.encode(text, add_special_tokens=False)
        if add_bos:
            ids = [self.special_id("bos")] + ids
        if add_eos:
            ids = ids + [self.special_id("eos")]
        return ids

    def decode(self, ids) -> str:
        return self.hf.decode([int(i) for i in ids], skip_special_tokens=False)

    def save(self, save_dir: str) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.hf.save_pretrained(save_dir)

    @classmethod
    def load(cls, local_dir: str, add_image_token: bool = True) -> "QwenTokenizer":
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(local_dir)
        if add_image_token and "<image>" not in hf.get_vocab():
            hf.add_special_tokens({"additional_special_tokens": ["<image>"]})
            hf.save_pretrained(local_dir)  # 持久化，保证重启后 <image> 仍在
        image_token_id = hf.convert_tokens_to_ids("<image>")
        return cls(hf, image_token_id)
```

> 注意：`BPETokenizer.load` 与 `train` 都通过 `__init__` 设定了 `_byte_offset`/`_base_size`，两者行为一致，无需额外处理。

- [ ] **Step 5: 创建下载脚本** `scripts/prepare_data_part1.py`

```python
"""下载 Qwen 分词器与中文 TinyStories 到 data/（走 hf-mirror）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import download_qwen_tokenizer, download_tinystories

if __name__ == "__main__":
    d1 = download_qwen_tokenizer("data/tokenizer/qwen2.5-0.5b")
    print("tokenizer ->", d1)
    files = download_tinystories("data/raw/text/tinystories_zh")
    print(f"tinystories jsonl: {len(files)} 个")
```

- [ ] **Step 6: 运行下载脚本**

Run: `.venv\Scripts\python scripts/prepare_data_part1.py`
Expected: 打印 tokenizer 目录与 jsonl 文件数量（>0）

- [ ] **Step 7: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_qwen_tokenizer.py -v`
Expected: PASS

- [ ] **Step 8: 提交**

```bash
git add src/tokenizer.py src/data.py scripts/prepare_data_part1.py tests/test_qwen_tokenizer.py
git commit -m "feat: local Qwen tokenizer adapter with image token and hf-mirror downloads"
```

---

### Task 4: 文本二进制预处理与随机批采样

**Files:**
- Modify: `src/data.py`（追加 `build_text_bin`, `load_bin`, `get_batch`）
- Create: `scripts/prepare_data_part2.py`
- Create: `tests/test_text_data.py`

**Interfaces:**
- Consumes: `tokenizer.encode`, `iter_texts`
- Produces:
  - `src.data.build_text_bin(tokenizer, texts: Iterator[str], out_path: str, max_docs: int | None = None, val_ratio: float = 0.01) -> tuple[int, int]`（返回 (train_tokens, val_tokens)）
  - `src.data.load_bin(path: str) -> "np.memmap"`
  - `src.data.get_batch(data, batch_size: int, ctx_len: int, device: str, generator=None) -> tuple[torch.Tensor, torch.Tensor]`

- [ ] **Step 1: 写失败测试** `tests/test_text_data.py`

```python
# 教学注释：验证二进制存储可回读、批张量形状与"下一 token"标签对齐。
import numpy as np
import torch
from src.tokenizer import BPETokenizer, SPECIALS
from src.data import build_text_bin, load_bin, get_batch

CORPUS = ["甲乙丙丁戊己庚辛壬癸", "子丑寅卯辰巳午未申酉"] * 200


def test_build_and_load(tmp_path):
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=400, min_frequency=1, verbose=False)
    tbin = tmp_path / "train.bin"
    vbin = tmp_path / "val.bin"
    n_train, n_val = build_text_bin(
        tok, iter(CORPUS), str(tbin), val_bin_path=str(vbin), val_ratio=0.1
    )
    assert n_train > 0 and n_val > 0
    data = load_bin(str(tbin))
    assert len(data) == n_train


def test_get_batch_shapes(tmp_path):
    tok = BPETokenizer.train(iter(CORPUS), vocab_size=400, min_frequency=1, verbose=False)
    tbin = tmp_path / "train.bin"
    build_text_bin(tok, iter(CORPUS), str(tbin), val_bin_path=str(tmp_path / "v.bin"))
    data = load_bin(str(tbin))
    x, y = get_batch(data, batch_size=4, ctx_len=16, device="cpu")
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])  # 标签是输入右移一位
    assert x.dtype == torch.long
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_text_data.py -v`
Expected: FAIL（`cannot import name 'build_text_bin'`）

- [ ] **Step 3: 追加实现到 `src/data.py`**

```python
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
```

- [ ] **Step 4: 创建预处理脚本** `scripts/prepare_data_part2.py`

```python
"""把 TinyStories 中文语料编码成 data/processed/text/{train,val}.bin。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import iter_texts, build_text_bin
from src.tokenizer import QwenTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", default="data/tokenizer/qwen2.5-0.5b")
    ap.add_argument("--raw_dir", default="data/raw/text/tinystories_zh/jsonl")
    ap.add_argument("--out", default="data/processed/text/train.bin")
    ap.add_argument("--val", default="data/processed/text/val.bin")
    ap.add_argument("--max_docs", type=int, default=None)
    args = ap.parse_args()

    tok = QwenTokenizer.load(args.tokenizer_dir)
    files = sorted(str(p) for p in Path(args.raw_dir).glob("*.jsonl"))
    print(f"语料文件 {len(files)} 个，开始编码…")
    n_train, n_val = build_text_bin(
        tok, iter_texts(files), args.out, val_bin_path=args.val, max_docs=args.max_docs
    )
    print(f"完成：train={n_train} tokens, val={n_val} tokens")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_text_data.py -v`
Expected: PASS

- [ ] **Step 6: 跑真实预处理（可先只取少量文档验证链路）**

Run: `.venv\Scripts\python scripts/prepare_data_part2.py --max_docs 2000`
Expected: 生成 `data/processed/text/train.bin` 与 `val.bin`，打印 token 数

- [ ] **Step 7: 提交**

```bash
git add src/data.py scripts/prepare_data_part2.py tests/test_text_data.py
git commit -m "feat: text bin preprocessing and random batch sampling"
```

---

### Task 5: RoPE 与因果多头注意力（含 GQA）

**Files:**
- Create: `src/attention.py`
- Create: `tests/test_attention.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `src.attention.build_rope_cache(head_dim: int, max_seq: int, theta: float, device, dtype) -> tuple[Tensor, Tensor]`
  - `src.attention.apply_rope(x: Tensor, cos: Tensor, sin: Tensor, offset: int = 0) -> Tensor`
  - `src.attention.MultiHeadAttention(d_model, n_head, n_kv_head, dropout=0.0, causal=True, bias=False)`
    - `forward(x, cos, sin, past_kv=None, offset=0) -> (out: Tensor, new_kv: tuple[Tensor, Tensor])`

- [ ] **Step 1: 写失败测试** `tests/test_attention.py`

```python
# 教学注释：核心正确性——因果性（未来 token 不能影响过去位置的输出）。
import torch
from src.attention import build_rope_cache, apply_rope, MultiHeadAttention


def test_rope_preserves_shape_and_first_position():
    cos, sin = build_rope_cache(8, 16, 10000.0, "cpu", torch.float32)
    x = torch.randn(2, 3, 5, 8)
    out = apply_rope(x, cos, sin, offset=0)
    assert out.shape == x.shape
    # 位置 0 的旋转角为 0，数值应保持不变
    assert torch.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6)


def test_causal_masking():
    torch.manual_seed(0)
    attn = MultiHeadAttention(32, 4, 4, dropout=0.0, causal=True)
    cos, sin = build_rope_cache(8, 32, 10000.0, "cpu", torch.float32)
    x = torch.randn(1, 10, 32)
    out1, _ = attn(x, cos, sin)
    x2 = x.clone()
    x2[:, 9] = torch.randn(32)  # 篡改最后一个位置
    out2, _ = attn(x2, cos, sin)
    # 前 9 个位置输出必须完全一致
    assert torch.allclose(out1[:, :9], out2[:, :9], atol=1e-6)


def test_gqa_shapes_and_cache():
    attn = MultiHeadAttention(32, 8, 2, dropout=0.0, causal=True)
    cos, sin = build_rope_cache(32 // 8, 32, 10000.0, "cpu", torch.float32)
    x = torch.randn(2, 6, 32)
    out, (k, v) = attn(x, cos, sin)
    assert out.shape == x.shape
    assert k.shape[1] == 2  # KV 头数为 2
    # 用缓存增量解码一个 token，输出形状正确
    x1 = torch.randn(2, 1, 32)
    out1, (k1, _) = attn(x1, cos, sin, past_kv=(k, v), offset=6)
    assert out1.shape == (2, 1, 32)
    assert k1.shape[2] == 7
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_attention.py -v`
Expected: FAIL（`No module named 'src.attention'`）

- [ ] **Step 3: 实现 `src/attention.py`**

```python
"""手写注意力：旋转位置编码 RoPE + 因果多头注意力（支持 GQA 与 KV 缓存）。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_rope_cache(head_dim: int, max_seq: int, theta: float, device, dtype):
    """预计算 RoPE 的 cos/sin 表。

    RoPE 把位置编码成"旋转角度"：第 i 对维度用频率 1/theta^(2i/d)。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (max_seq, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """对 (B, H, T, D) 施加旋转位置编码。offset 用于增量解码时的位置偏移。"""
    t = x.shape[-2]
    c = cos[offset: offset + t].view(1, 1, t, -1)
    s = sin[offset: offset + t].view(1, 1, t, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class MultiHeadAttention(nn.Module):
    """多头自注意力：QKV 投影 -> RoPE -> （可选）拼接缓存 -> 缩放点积 -> 因果 mask -> 输出投影。

    GQA（分组查询注意力）：KV 头数 n_kv_head 少于 Q 头数 n_head，多个 Q 头共享一组 KV，
    可显著减小 KV 缓存的显存占用（推理时关键）。
    """

    def __init__(self, d_model: int, n_head: int, n_kv_head: int, dropout: float = 0.0,
                 causal: bool = True, bias: bool = False):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        assert n_head % n_kv_head == 0, "n_head 必须能被 n_kv_head 整除"
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.causal = causal
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_head * self.head_dim, d_model, bias=bias)

    def forward(self, x, cos, sin, past_kv=None, offset=0):
        B, T, _ = x.shape
        hd = self.head_dim
        q = self.q_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, hd).transpose(1, 2)

        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)

        # GQA：把 KV 头复制到与 Q 头数相同
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        if self.causal:
            tq, tk = q.shape[-2], k.shape[-2]
            mask = torch.triu(
                torch.ones(tq, tk, device=x.device, dtype=torch.bool),
                diagonal=tk - tq + 1,
            )
            att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = torch.dropout(att, self.dropout_p, train=self.training)
        out = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), new_kv
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_attention.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/attention.py tests/test_attention.py
git commit -m "feat: RoPE and causal multi-head attention with GQA and KV cache"
```

---

### Task 6: RMSNorm / SwiGLU / Block / GPT 模型

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `src.config.ModelConfig`, `src.attention.*`
- Produces:
  - `src.model.RMSNorm(d_model, eps=1e-6)`
  - `src.model.SwiGLU(d_model, d_ff, dropout=0.0)`
  - `src.model.Block(cfg: ModelConfig)`
  - `src.model.GPT(cfg: ModelConfig)`
    - `embed(idx) -> Tensor`
    - `forward(idx=None, inputs_embeds=None, targets=None, past_kvs=None, use_cache=False) -> (logits, loss, new_kvs)`
    - `num_params(non_embedding: bool = False) -> int`
    - `configure_optimizers(train_cfg) -> torch.optim.Optimizer`

- [ ] **Step 1: 写失败测试** `tests/test_model.py`

```python
# 教学注释：验证前向形状、权重共享、参数量统计与反向可训练。
import torch
from src.config import ModelConfig, TrainConfig
from src.model import GPT, RMSNorm, SwiGLU


def tiny_cfg():
    return ModelConfig(vocab_size=300, d_model=64, n_layer=2, n_head=4,
                       n_kv_head=2, d_ff=128, ctx_len=32)


def test_rmsnorm_and_swiglu_shapes():
    n = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    assert n(x).shape == x.shape
    m = SwiGLU(16, 32)
    assert m(x).shape == x.shape


def test_gpt_forward_and_loss():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    logits, loss, _ = model(x, targets=y)
    assert logits.shape == (2, cfg.ctx_len, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_weight_tying():
    cfg = tiny_cfg()
    model = GPT(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_num_params_reasonable():
    cfg = tiny_cfg()
    model = GPT(cfg)
    assert model.num_params() > model.num_params(non_embedding=True)
    assert model.num_params(non_embedding=True) > 0


def test_backward_works():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    _, loss, _ = model(x, targets=y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_configure_optimizers():
    model = GPT(tiny_cfg())
    opt = model.configure_optimizers(TrainConfig(weight_decay=0.1))
    assert len(opt.param_groups) == 2  # 衰减组 + 不衰减组
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_model.py -v`
Expected: FAIL（`No module named 'src.model'`）

- [ ] **Step 3: 实现 `src/model.py`**

```python
"""GPT 模型：RMSNorm + SwiGLU + 残差 Block，权重共享的 decoder-only 语言模型。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import MultiHeadAttention, build_rope_cache
from src.config import ModelConfig, TrainConfig


class RMSNorm(nn.Module):
    """均方根归一化：比 LayerNorm 少一个均值项，更快且效果好。"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    """门控前馈：w3(silu(w1(x)) * w2(x))，比普通 MLP 更强。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    """Pre-Norm 残差块：x + Attn(Norm(x)) 再 x + MLP(Norm(x))。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_head, cfg.n_kv_head,
                                       dropout=cfg.dropout, causal=True)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)

    def forward(self, x, cos, sin, past_kv=None, offset=0):
        h, new_kv = self.attn(self.n1(x), cos, sin, past_kv=past_kv, offset=offset)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        head_dim = cfg.d_model // cfg.n_head
        cos, sin = build_rope_cache(head_dim, cfg.ctx_len, cfg.rope_theta, "cpu", torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed(self, idx):
        return self.drop(self.tok_emb(idx))

    def forward(self, idx=None, inputs_embeds=None, targets=None, past_kvs=None, use_cache=False):
        x = inputs_embeds if inputs_embeds is not None else self.embed(idx)
        offset = past_kvs[0][0].shape[-2] if past_kvs is not None else 0
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pk = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, self.cos, self.sin, past_kv=pk, offset=offset)
            if use_cache:
                new_kvs.append(new_kv)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss, new_kvs

    @torch.no_grad()
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def configure_optimizers(self, train_cfg: TrainConfig) -> torch.optim.Optimizer:
        """把参数分成"要 weight decay"和"不要"两组（Norm 与 bias 不衰减）。"""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = torch.cuda.is_available()
        try:
            return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                     betas=(train_cfg.beta1, train_cfg.beta2), fused=fused)
        except TypeError:
            return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                     betas=(train_cfg.beta1, train_cfg.beta2))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_model.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: GPT model with RMSNorm, SwiGLU, weight tying"
```

---

### Task 7: 过拟合小批测试（模型确实能学的证据）

**Files:**
- Create: `tests/test_overfit.py`

**Interfaces:**
- Consumes: `src.model.GPT`, `src.config.ModelConfig`, `src.utils.get_lr`
- Produces: 无（仅测试）

- [ ] **Step 1: 写测试**

```python
# 教学注释：如果模型连一批固定数据都记不住，说明实现有 bug。
# 这是最快的"端到端可用性"检查，比看几小时 loss 曲线快得多。
import torch
from src.config import ModelConfig
from src.model import GPT


def test_overfit_single_batch():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=260, d_model=64, n_layer=2, n_head=4,
                      n_kv_head=2, d_ff=128, ctx_len=32, dropout=0.0)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.ctx_len))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for step in range(200):
        _, loss, _ = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.2, f"过拟合失败：{first:.3f} -> {loss.item():.3f}"
```

- [ ] **Step 2: 运行测试**

Run: `.venv\Scripts\python -m pytest tests/test_overfit.py -v`
Expected: PASS（loss 降到 0.2 以下）

- [ ] **Step 3: 提交**

```bash
git add tests/test_overfit.py
git commit -m "test: overfit a tiny batch to prove the model can learn"
```

---

### Task 8: 训练器（AMP / 梯度累积 / 断点续训 / loss 曲线）+ 训练脚本

**Files:**
- Create: `src/trainer.py`
- Create: `scripts/train_gpt.py`
- Create: `configs/gpt_tinystories.yaml`
- Create: `configs/gpt_bpe.yaml`
- Create: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `GPT`, `get_batch`, `load_bin`, `QwenTokenizer`, `get_lr`, `JsonlLogger`, `plot_loss`, `save_config`, `set_seed`
- Produces:
  - `src.trainer.Trainer(model, train_cfg: TrainConfig, train_data, val_data, tokenizer=None)`
    - `train() -> None`
    - `save(path: str) -> None`
    - `load(path: str, resume: bool = True) -> int`（返回已训练步数）

- [ ] **Step 1: 写失败测试** `tests/test_trainer.py`

```python
# 教学注释：用极小的假数据验证——训练会造成参数变化、checkpoint 能续训、
# metrics.jsonl 正常写入。
import os
import numpy as np
import torch
from src.config import ModelConfig, TrainConfig
from src.model import GPT
from src.trainer import Trainer


def make_data(tmp_path, n=5000):
    p = tmp_path / "d.bin"
    rng = np.random.randint(0, 100, size=n).astype(np.uint32)
    rng.tofile(str(p))
    return np.memmap(str(p), dtype=np.uint32, mode="r")


def test_trainer_runs_and_resumes(tmp_path):
    cfg = ModelConfig(vocab_size=100, d_model=64, n_layer=2, n_head=4,
                      n_kv_head=2, d_ff=128, ctx_len=32)
    model = GPT(cfg)
    tcfg = TrainConfig(batch_size=2, grad_accum=1, lr=1e-2, warmup_steps=2,
                       max_steps=6, eval_interval=3, eval_iters=2,
                       save_interval=6, log_interval=1, out_dir=str(tmp_path / "out"),
                       dtype="fp32", compile=False)
    data = make_data(tmp_path)
    trainer = Trainer(model, tcfg, data, data)
    before = model.tok_emb.weight.detach().clone()
    trainer.train()
    after = model.tok_emb.weight.detach()
    assert not torch.allclose(before, after), "参数应被更新"
    assert os.path.exists(os.path.join(tcfg.out_dir, "latest.pt"))
    assert os.path.exists(os.path.join(tcfg.out_dir, "metrics.jsonl"))

    # 续训：从 checkpoint 恢复，步数应继续
    tcfg2 = TrainConfig(**{**tcfg.__dict__, "max_steps": 9})
    model2 = GPT(cfg)
    trainer2 = Trainer(model2, tcfg2, data, data)
    start_step = trainer2.load(os.path.join(tcfg.out_dir, "latest.pt"))
    assert start_step == 6
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_trainer.py -v`
Expected: FAIL（`No module named 'src.trainer'`）

- [ ] **Step 3: 实现 `src/trainer.py`**

```python
"""训练器：bf16 混合精度 + 梯度累积 + 裁剪 + 学习率调度 + 断点续训 + 日志/曲线。"""
from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.config import TrainConfig
from src.utils import JsonlLogger, get_lr, gpu_mem_str, human_params, plot_loss


def _autocast_ctx(train_cfg: TrainConfig):
    """bf16 训练不需要 GradScaler，直接 autocast 即可。"""
    if train_cfg.dtype == "bf16" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class Trainer:
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, train_data, val_data,
                 tokenizer=None, device: str | None = None):
        self.cfg = train_cfg
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.opt = model.configure_optimizers(train_cfg)
        self.raw_model = model  # 保存未 compile 的原始模型引用

        if train_cfg.compile:
            # 教学注释：torch.compile 在 Windows 上需要 triton，可能失败；失败自动回退。
            try:
                self.model = torch.compile(model)
                print("[compile] torch.compile 已启用")
            except Exception as e:  # noqa: BLE001
                print(f"[compile] 失败，回退 eager：{e}")
                self.model = model

        self.out_dir = Path(train_cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = JsonlLogger(str(self.out_dir / "metrics.jsonl"))
        self.start_step = 0

    def _forward_loss(self, x, y):
        with _autocast_ctx(self.cfg):
            _, loss, _ = self.model(x, targets=y)
        return loss

    @torch.no_grad()
    def _estimate_val(self) -> float:
        self.model.eval()
        losses = []
        for _ in range(self.cfg.eval_iters):
            x, y = self._get_batch(self.val_data)
            losses.append(self._forward_loss(x, y).item())
        self.model.train()
        return sum(losses) / len(losses)

    def _get_batch(self, data):
        from src.data import get_batch
        return get_batch(data, self.cfg.batch_size, self.model.cfg.ctx_len, self.device)

    def _next_batch(self):
        """取一个训练 batch（子类可覆写以支持多模态等不同数据形态）。"""
        return self._get_batch(self.train_data)

    def train(self):
        cfg = self.cfg
        self.model.train()
        t0 = time.time()
        for step in range(self.start_step, cfg.max_steps):
            self._current_step = step + 1
            lr = get_lr(step, cfg)
            for g in self.opt.param_groups:
                g["lr"] = lr

            # 梯度累积：多个 micro-batch 的梯度平均后再更新一次
            self.opt.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum):
                x, y = self._next_batch()
                loss = self._forward_loss(x, y) / cfg.grad_accum
                loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()

            if step % cfg.log_interval == 0:
                dt = time.time() - t0
                print(f"step {step:>6}/{cfg.max_steps} | loss {loss.item() * cfg.grad_accum:.4f} "
                      f"| lr {lr:.2e} | {dt:.1f}s | mem {gpu_mem_str()}")
                self.logger.log({"step": step, "split": "train",
                                 "loss": loss.item() * cfg.grad_accum, "lr": lr})
            if step > 0 and step % cfg.eval_interval == 0:
                v = self._estimate_val()
                print(f"  [eval] step {step} val_loss {v:.4f}")
                self.logger.log({"step": step, "val_loss": v})
            if step > 0 and step % cfg.save_interval == 0:
                self.save(str(self.out_dir / "latest.pt"))
                plot_loss(str(self.out_dir / "metrics.jsonl"), str(self.out_dir / "loss.png"))

        self.save(str(self.out_dir / "latest.pt"))
        plot_loss(str(self.out_dir / "metrics.jsonl"), str(self.out_dir / "loss.png"))
        print(f"训练完成，总耗时 {time.time() - t0:.1f}s，参数量 {human_params(self.raw_model.num_params())}")

    def save(self, path: str):
        from src.config import save_config
        ckpt = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "step": getattr(self, "_current_step", 0),
            "cfg": self.raw_model.cfg,
            "train_cfg": self.cfg,
        }
        torch.save(ckpt, path)

    def load(self, path: str, resume: bool = True) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model"])
        if resume:
            try:
                self.opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:  # noqa: BLE001
                print(f"优化器状态恢复失败（可忽略）：{e}")
            self.start_step = int(ckpt.get("step", 0))
        return self.start_step
```

- [ ] **Step 4: 创建训练脚本** `scripts/train_gpt.py`

```python
"""训练文本 GPT：读取 yaml 配置，支持命令行覆盖与 --resume。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, apply_overrides
from src.data import load_bin
from src.model import GPT
from src.tokenizer import QwenTokenizer, BPETokenizer, CharTokenizer
from src.trainer import Trainer
from src.utils import human_params, set_seed


def build_tokenizer(cfg):
    kind = cfg.data.tokenizer_kind
    if kind == "qwen":
        return QwenTokenizer.load(cfg.data.tokenizer_dir)
    if kind == "bpe":
        return BPETokenizer.load(cfg.data.tokenizer_dir)
    return CharTokenizer.load(cfg.data.tokenizer_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpt_tinystories.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint 路径")
    ap.add_argument("--set", nargs="*", default=[], help="覆盖项，如 train.lr=3e-4")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    set_seed(cfg.train.seed)

    tok = build_tokenizer(cfg)
    cfg.model.vocab_size = tok.vocab_size  # 词表大小必须与分词器一致
    print(f"分词器 {cfg.data.tokenizer_kind}，词表 {tok.vocab_size}")

    train_data = load_bin(cfg.data.train_bin)
    val_data = load_bin(cfg.data.val_bin)
    model = GPT(cfg.model)
    print(f"模型参数量 {human_params(model.num_params())}（不含词表 "
          f"{human_params(model.num_params(non_embedding=True))}）")

    trainer = Trainer(model, cfg.train, train_data, val_data, tokenizer=tok)
    if args.resume:
        step = trainer.load(args.resume)
        print(f"从 {args.resume} 恢复，已完成 {step} 步")
    trainer.train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 创建配置文件** `configs/gpt_tinystories.yaml`

```yaml
model:
  vocab_size: 151936
  d_model: 768
  n_layer: 12
  n_head: 12
  n_kv_head: 4
  d_ff: 2048
  ctx_len: 1024
  dropout: 0.0
  tie_embeddings: true
data:
  tokenizer_kind: qwen
  tokenizer_dir: data/tokenizer/qwen2.5-0.5b
  train_bin: data/processed/text/train.bin
  val_bin: data/processed/text/val.bin
train:
  batch_size: 8
  grad_accum: 8
  lr: 6.0e-4
  min_lr: 6.0e-5
  warmup_steps: 200
  max_steps: 20000
  eval_interval: 250
  save_interval: 1000
  log_interval: 20
  out_dir: out/gpt
  dtype: bf16
  compile: false
  seed: 42
```

`configs/gpt_bpe.yaml`：复制上面，改 `model.vocab_size: 16384`、`data.tokenizer_kind: bpe`、`data.tokenizer_dir: data/tokenizer/bpe_16k`、`train.out_dir: out/gpt_bpe`。

- [ ] **Step 6: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_trainer.py -v`
Expected: PASS

- [ ] **Step 7: 冒烟训练（小步数验证整条链路）**

Run: `.venv\Scripts\python scripts/train_gpt.py --set train.max_steps=40 train.save_interval=20 train.grad_accum=2 train.batch_size=4`
Expected: 打印 loss、生成 `out/gpt/latest.pt` 与 `out/gpt/loss.png`

- [ ] **Step 8: 提交**

```bash
git add src/trainer.py scripts/train_gpt.py configs/ tests/test_trainer.py
git commit -m "feat: trainer with AMP, grad accumulation, resume and loss curves"
```

---

### Task 9: 采样生成（KV 缓存）与命令行聊天

**Files:**
- Create: `src/generate.py`
- Create: `scripts/chat.py`
- Create: `tests/test_generate.py`

**Interfaces:**
- Consumes: `GPT`, tokenizer, `src.config`
- Produces:
  - `src.generate.sample_logits(logits: Tensor, temperature, top_k, top_p, generator=None) -> Tensor`（返回单个 token id）
  - `src.generate.generate(model, tokenizer, prompt: str, max_new_tokens=128, temperature=0.8, top_k=None, top_p=0.9, repetition_penalty=1.0, device="cuda") -> str`

- [ ] **Step 1: 写失败测试** `tests/test_generate.py`

```python
# 教学注释：验证生成能持续产出指定长度、且 greedy（temperature->0）是确定性的。
import torch
from src.config import ModelConfig
from src.model import GPT
from src.generate import generate, sample_logits


def tiny_model():
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=200, d_model=64, n_layer=2, n_head=4,
                           n_kv_head=2, d_ff=128, ctx_len=64))


def test_sample_logits_greedy():
    logits = torch.tensor([0.1, 2.0, 0.5])
    # 温度极低时趋近 argmax
    tok = sample_logits(logits, temperature=1e-4, top_k=None, top_p=1.0)
    assert tok.item() == 1


def test_generate_length_and_determinism():
    model = tiny_model()
    from src.tokenizer import CharTokenizer
    tok = CharTokenizer(["你", "好", "世", "界"], {"bos": 0, "eos": 1, "pad": 2, "unk": 3, "image": 4})
    out1 = generate(model, tok, "你好", max_new_tokens=10, temperature=0.0, device="cpu")
    out2 = generate(model, tok, "你好", max_new_tokens=10, temperature=0.0, device="cpu")
    assert out1 == out2
    assert len(out1) >= 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python -m pytest tests/test_generate.py -v`
Expected: FAIL（`No module named 'src.generate'`）

- [ ] **Step 3: 实现 `src/generate.py`**

```python
"""自回归采样生成（带 KV 缓存）：temperature / top-k / top-p / 重复惩罚。"""
from __future__ import annotations

import torch


def sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_k=None,
                  top_p: float | None = None, generator=None) -> torch.Tensor:
    """从一维 logits 中采样一个 token。temperature<=0 时退化为贪心。"""
    if temperature <= 0:
        return torch.argmax(logits).unsqueeze(0)

    logits = logits / max(temperature, 1e-8)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < vals[-1], float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # 保留累积概率达到 top_p 的最小集合
        cutoff = cum > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _apply_repetition_penalty(logits, prev_ids, penalty):
    if penalty and penalty != 1.0 and prev_ids:
        for tid in set(int(i) for i in prev_ids):
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_k=None, top_p: float | None = 0.9,
             repetition_penalty: float = 1.0, device: str = "cuda",
             generator=None) -> str:
    """给定提示词，自回归生成文本。使用 KV 缓存避免重复计算。"""
    model.eval()
    model.to(device)
    ids = tokenizer.encode(prompt, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.special_id("eos")
    generated = []

    past = None
    for _ in range(max_new_tokens):
        cur = idx if past is None else idx[:, -1:]
        logits, _, past = model(cur, past_kvs=past, use_cache=True)
        step_logits = logits[0, -1].float().clone()
        _apply_repetition_penalty(step_logits, ids + generated, repetition_penalty)
        nxt = sample_logits(step_logits, temperature, top_k, top_p, generator)
        tid = int(nxt.item())
        if tid == eos_id:
            break
        generated.append(tid)
        idx = torch.cat([idx, nxt.view(1, 1).to(device)], dim=1)

    return tokenizer.decode(generated)
```

- [ ] **Step 4: 创建聊天脚本** `scripts/chat.py`

```python
"""命令行聊天：加载训练好的 checkpoint 续写。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.config import load_config
from src.model import GPT
from src.tokenizer import QwenTokenizer
from src.generate import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpt_tinystories.yaml")
    ap.add_argument("--ckpt", default="out/gpt/latest.pt")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
    cfg.model.vocab_size = tok.vocab_size
    model = GPT(cfg.model)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print("模型已加载。输入文字开始续写，输入 /quit 退出。")
    while True:
        prompt = input("你> ").strip()
        if prompt in ("/quit", "/exit", ""):
            break
        print("AI> " + generate(model, tok, prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, top_p=args.top_p))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv\Scripts\python -m pytest tests/test_generate.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/generate.py scripts/chat.py tests/test_generate.py
git commit -m "feat: KV-cache sampling generation and CLI chat"
```

---

### Task 10: README 与原理文档

**Files:**
- Create: `README.md`
- Create: `docs/01-tokenizer.md`
- Create: `docs/02-transformer.md`
- Create: `docs/03-training.md`
- Create: `src/__init__.py` 已存在

**Interfaces:**
- Consumes: 全部已完成模块
- Produces: 文档

- [ ] **Step 1: 写 `README.md`**

内容需包含：项目简介、环境安装（`.venv` 已有依赖，补 `pip install -r requirements.txt`）、
数据准备命令、训练命令、续训命令、聊天命令、目录结构、学习路线（对应里程碑）、常见问题
（显存不足怎么办：减小 `batch_size` 增大 `grad_accum`；`torch.compile` 失败属正常）。

- [ ] **Step 2: 写 `docs/01-tokenizer.md`**

讲解：为什么需要分词、字符级 vs 字节级 BPE、merges 表的合并过程、特殊符号、
`decode(encode(x))==x` 的意义、Qwen 分词器与本项目手写 BPE 的对比实验方法。

- [ ] **Step 3: 写 `docs/02-transformer.md`**

讲解：Embedding → N×Block（RMSNorm/Attention/SwiGLU）→ 输出投影；
RoPE 的旋转直觉；因果 mask 为什么必要；GQA 如何省显存；权重共享；参数量逐项计算。

- [ ] **Step 4: 写 `docs/03-training.md`**

讲解：交叉熵与"下一 token 预测"；warmup+余弦；梯度累积=用时间换显存；
梯度裁剪；bf16 混合精度；如何读 `loss.png` 判断欠拟合/过拟合；checkpoint 内容与续训原理。

- [ ] **Step 5: 提交**

```bash
git add README.md docs/01-tokenizer.md docs/02-transformer.md docs/03-training.md
git commit -m "docs: README and text-LLM principle guides"
```

---

## 验收（Plan 1）

- `pytest tests/ -v` 全部通过。
- `scripts/prepare_data_part1.py` + `part2.py` 能产出 `data/processed/text/{train,val}.bin`。
- `scripts/train_gpt.py` 能跑通，中断后 `--resume out/gpt/latest.pt` 继续，`out/gpt/loss.png` 有下降曲线。
- `scripts/chat.py` 能生成可读中文。
