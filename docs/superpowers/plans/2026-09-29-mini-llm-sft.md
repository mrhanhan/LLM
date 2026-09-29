# SFT（预训练语料扩充 + 指令微调 + 多轮对话）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用开源中文预训练语料训出一个 2048 上下文的基座模型，再做指令微调（只在助手回答上计损失），使模型能完成简单多轮中文对话，并提供 CLI/网页入口与自动评测。

**Architecture:** 在既有文本闭环上新增两条数据链：预训练用 `HfFileSystem` 远程按 row-group 采样 `fineweb-2/cmn_Hani` 文本 → 编码成 `uint32` 二进制；SFT 把 `alpaca-zh` + `firefly` 归一化成统一 `messages` 格式、合成多轮对话 → 预分词成 npz。模型主干不变、`ctx_len=2048`；对话模板复用 Qwen 现有 `<|im_start|>/<|im_end|>` 特殊 token，SFT 只监督助手段（含轮末 `<|im_end|>`），`SFTTrainer` 复用 `Trainer` 的两个接缝。

**Tech Stack:** Python 3.12、PyTorch（bf16）、`huggingface_hub`（`HfFileSystem`）、`pyarrow`、`fsspec`、Qwen2.5-0.5B 分词器、Gradio、pytest。

**Spec:** `docs/superpowers/specs/2026-09-29-mini-llm-sft-design.md`

## Global Constraints

- 解释器一律用 `P:\Demo\LLM\.venv\Scripts\python.exe`；命令在项目根 `P:\Demo\LLM` 下执行。
- 打印中文的脚本必须在文件头做 `sys.stdout.reconfigure(encoding="utf-8")`；测试与命令可加 `$env:PYTHONUTF8=1`。
- 数据/产物只落 `data/`、`out/`（已被 `.gitignore` 忽略），不入库。
- HF 访问默认走本机代理 `http://127.0.0.1:7890` 直连 `huggingface.co`；`hf-mirror.com` 仅作回退。
- 不使用 gated 数据集（`BAAI/Infinity-Instruct` 需要授权，本阶段不用）。
- 对话模板：复用 Qwen 现有特殊 token（`<|im_start|>`=151644、`<|im_end|>`=151645，用 `convert_tokens_to_ids` 动态取，禁止写死）；**不扩词表、不改 embedding**。
- 模型结构固定：`d_model=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2048`，`ctx_len=2048`，`vocab_size=151666`。
- SFT 损失只在 assistant 段的 content token 与其后的 `<|im_end|>` 上计算，其余 label 一律 `-100`。
- `src/model.py` 除本计划明确要求外不得改动；`nn.Dropout(cfg.dropout)` 必须保留。
- 每个 Task 结束必须跑测试并提交。

---

## File Structure

| 文件 | 职责 | 动作 |
|---|---|---|
| `src/hfenv.py` | 统一设置 HF 代理/端点环境变量 | 新增 |
| `src/config.py` | `DataConfig` 增 HF/SFT 字段 | 修改 |
| `src/data.py` | 新增 fineweb-2 远程读取；移除写死镜像 | 修改 |
| `src/chat_format.py` | 对话模板、SFT 样本构造、停止/剥离/history 转换 | 新增 |
| `src/sft_data.py` | SFT 归一化、多轮合成、分词落盘、Dataset/collate | 新增 |
| `src/trainer.py` | 新增 `SFTTrainer` | 修改 |
| `src/generate.py` | `generate` 增 `stop_ids` | 修改 |
| `src/chat_eval.py` | 评测纯逻辑 + 固定题库 | 新增 |
| `scripts/prepare_pretrain_data.py` | 预训练数据准备 | 新增 |
| `scripts/prepare_sft_data.py` | SFT 数据准备 | 新增 |
| `scripts/train_sft.py` | SFT 训练入口 | 新增 |
| `scripts/chat.py` | 多轮对话（保留 `base`） | 修改 |
| `scripts/eval_chat.py` | 自动评测 | 新增 |
| `scripts/webapp.py` | 文本页改多轮 | 修改 |
| `configs/gpt_fineweb.yaml`、`configs/sft_zh.yaml` | 新配置 | 新增 |
| 7 个测试文件（hfenv / fineweb_reader / chat_format / sft_data / sft_trainer / generate_stop / chat_eval） | 单元测试 | 新增 |
| `docs/05-sft.md`、`README.md` | 文档 | 新增/修改 |

---

## Task 1: HF 访问配置

**Files:**
- Create: `src/hfenv.py`
- Modify: `src/config.py`（`DataConfig`）、`src/data.py`（删写死镜像）、`scripts/prepare_data_part1.py`
- Test: `tests/test_hfenv.py`

**Interfaces:**
- Produces: `src.hfenv.setup_hf(endpoint: str | None, proxy: str | None) -> None`
- Produces: `DataConfig` 新字段 `hf_endpoint: str = ""`、`proxy: str = "http://127.0.0.1:7890"`、`sft_train: str = "data/processed/sft/train.npz"`、`sft_val: str = "data/processed/sft/val.npz"`、`max_len: int = 2048`

- [ ] **Step 1: 写失败测试** `tests/test_hfenv.py`

```python
import os
from src.hfenv import setup_hf


def test_setup_hf_proxy_direct(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "HF_ENDPOINT", "NO_PROXY"):
        monkeypatch.delenv(k, raising=False)
    setup_hf("", "http://127.0.0.1:7890")
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert "HF_ENDPOINT" not in os.environ          # 直连：不能残留镜像端点
    assert "127.0.0.1" in os.environ["NO_PROXY"]


def test_setup_hf_endpoint_no_proxy(monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    setup_hf("https://hf-mirror.com", None)
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert "HTTP_PROXY" not in os.environ
```

- [ ] **Step 2: 运行确认失败** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_hfenv.py -q`；Expected: FAIL（`No module named 'src.hfenv'`）。

- [ ] **Step 3: 实现** `src/hfenv.py`

```python
"""统一配置 HuggingFace 访问：代理直连或镜像端点。

教学注释：本机通过 HTTP 代理（默认 127.0.0.1:7890）直连 huggingface.co；
若无代理则可用 hf-mirror.com 回退。集中在这里设置，避免各脚本各写一份。
"""
from __future__ import annotations

import os

_LOCAL = "127.0.0.1,localhost,::1"


def setup_hf(endpoint: str | None = None, proxy: str | None = None) -> None:
    """设置 HF_ENDPOINT 与 HTTP(S)_PROXY。

    - proxy 非空：走代理；endpoint 为空则直连官方 HF（清除镜像端点）。
    - endpoint 非空：使用该镜像端点。
    """
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        existing = os.environ.get("NO_PROXY", "")
        parts = [p for p in existing.split(",") if p]
        for host in _LOCAL.split(","):
            if host not in parts:
                parts.append(host)
        os.environ["NO_PROXY"] = ",".join(parts)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        # 直连（或仅走代理）：移除镜像端点，否则会打到 hf-mirror
        os.environ.pop("HF_ENDPOINT", None)
```

- [ ] **Step 4: 运行确认通过** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_hfenv.py -q`；Expected: PASS（2 passed）。

- [ ] **Step 5: 给 `DataConfig` 加字段**（`src/config.py`，整体替换 `DataConfig`）

```python
@dataclass
class DataConfig:
    tokenizer_kind: str = "qwen"  # qwen | bpe | char
    tokenizer_dir: str = "data/tokenizer/qwen2.5-0.5b"
    train_bin: str = "data/processed/text/train.bin"
    val_bin: str = "data/processed/text/val.bin"
    # 教学注释：HF 访问方式与 SFT 数据路径，配合本阶段新增脚本使用。
    hf_endpoint: str = ""                              # 空 = 直连 HF
    proxy: str = "http://127.0.0.1:7890"               # 本机代理
    sft_train: str = "data/processed/sft/train.npz"
    sft_val: str = "data/processed/sft/val.npz"
    max_len: int = 2048                                # SFT 序列最大长度
```

- [ ] **Step 6: 移除 `src/data.py` 写死镜像**（删掉 `src/data.py` 第 12-13 行）

```python
# 教学注释：强制把下载端点指向国内镜像，避免超时。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

- [ ] **Step 7: `scripts/prepare_data_part1.py` 接入 `setup_hf`**

在 argparse 增加 `--endpoint`（默认 `https://hf-mirror.com`，兼容旧行为）与 `--proxy`（默认 `None`），并在调用下载函数前：

```python
from src.hfenv import setup_hf
setup_hf(args.endpoint or None, args.proxy or None)
```

- [ ] **Step 8: 跑全量测试并提交**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/ -q`；Expected: 全绿。

```bash
git add src/hfenv.py src/config.py src/data.py scripts/prepare_data_part1.py tests/test_hfenv.py
git commit -m "feat: configurable HF access (proxy direct / mirror) for SFT data pipeline (Task 1)"
```

---

## Task 2: fineweb-2 远程文本读取

**Files:**
- Modify: `src/data.py`（追加函数）
- Test: `tests/test_fineweb_reader.py`

**Interfaces:**
- Produces: `FINEWEB_REPO = "HuggingFaceFW/fineweb-2"`
- Produces: `iter_parquet_text(pf, text_col="text", max_bytes=None) -> Iterator[str]`
- Produces: `open_remote_parquet(repo, path_in_repo, block_size=4*1024*1024) -> tuple`
- Produces: `read_fineweb_cached(repo, lang, split, shard, max_bytes, cache_path) -> Iterator[str]`

- [ ] **Step 1: 写失败测试** `tests/test_fineweb_reader.py`

```python
import gzip
import pyarrow as pa
import pyarrow.parquet as pq
from src.data import iter_parquet_text, read_fineweb_cached


def _make_parquet(path, texts):
    pq.write_table(pa.table({"text": texts}), path)


def test_iter_parquet_text_respects_byte_budget(tmp_path):
    p = tmp_path / "a.parquet"
    _make_parquet(p, ["你好世界", "再见了", "第三句"])
    got = list(iter_parquet_text(pq.ParquetFile(p), max_bytes=6))   # "你好世界"=12 字节
    assert got == ["你好世界"]


def test_iter_parquet_text_no_budget_reads_all(tmp_path):
    p = tmp_path / "b.parquet"
    _make_parquet(p, ["a", "b"])
    assert list(iter_parquet_text(pq.ParquetFile(p))) == ["a", "b"]


def test_read_fineweb_cached_uses_cache_without_network(tmp_path):
    cache = tmp_path / "train.jsonl.gz"
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        f.write("第一行\n第二行\n")
    out = list(read_fineweb_cached("n/a", "cmn_Hani", "train", "x.parquet",
                                   max_bytes=None, cache_path=str(cache)))
    assert out == ["第一行", "第二行"]
```

- [ ] **Step 2: 运行确认失败** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_fineweb_reader.py -q`；Expected: FAIL（`cannot import name 'iter_parquet_text'`）。

- [ ] **Step 3: 实现**（追加到 `src/data.py` 末尾）

```python
FINEWEB_REPO = "HuggingFaceFW/fineweb-2"


def iter_parquet_text(pf, text_col: str = "text", max_bytes: int | None = None):
    """遍历 ParquetFile 的 row-group，逐条产出 text，累计字节数达到 max_bytes 即停。

    教学注释：fineweb-2 单片约 4.8GB，全下太浪费；按 row-group 读取可以让
    HfFileSystem 只请求需要的分块（每个分块约十几 MB）。
    """
    total = 0
    for gi in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(gi, columns=[text_col]).column(text_col)
        for t in col.to_pylist():
            if not t:
                continue
            yield t
            total += len(t.encode("utf-8"))
            if max_bytes is not None and total >= max_bytes:
                return


def open_remote_parquet(repo: str, path_in_repo: str, block_size: int = 4 * 1024 * 1024):
    """用 HfFileSystem 远程打开 parquet。返回 (ParquetFile, file_obj)，用完需关闭 file_obj。"""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    f = fs.open(f"datasets/{repo}/{path_in_repo}", "rb", block_size=block_size)
    return pq.ParquetFile(f), f


def read_fineweb_cached(repo: str, lang: str, split: str, shard: str,
                        max_bytes: int | None, cache_path: str):
    """读取 fineweb 文本：cache_path 存在则直接读 gz 缓存，否则远程读并写缓存。

    教学注释：抽取出的文本落盘缓存，既省代理流量，也保证断点可复现。
    """
    import gzip
    from pathlib import Path

    cp = Path(cache_path)
    if cp.exists():
        with gzip.open(cp, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield line
        return

    path_in_repo = f"data/{lang}/{split}/{shard}"
    pf, fobj = open_remote_parquet(repo, path_in_repo)
    cp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(cp, "wt", encoding="utf-8") as out:
            for t in iter_parquet_text(pf, max_bytes=max_bytes):
                out.write(t.replace("\n", " ") + "\n")
                yield t
    finally:
        fobj.close()
```

- [ ] **Step 4: 运行确认通过** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_fineweb_reader.py -q`；Expected: PASS（3 passed）。

- [ ] **Step 5: 提交**

```bash
git add src/data.py tests/test_fineweb_reader.py
git commit -m "feat: stream fineweb-2 parquet by row-group with local text cache (Task 2)"
```

---

## Task 3: 预训练数据准备脚本 + 预训练配置

**Files:**
- Create: `scripts/prepare_pretrain_data.py`、`configs/gpt_fineweb.yaml`

**Interfaces:**
- Consumes: `read_fineweb_cached`、`build_text_bin`、`src.hfenv.setup_hf`
- Produces: `data/processed/text/fineweb_cmn.{train,val}.bin`、`configs/gpt_fineweb.yaml`

- [ ] **Step 1: 写脚本** `scripts/prepare_pretrain_data.py`

```python
"""准备预训练语料：采样 fineweb-2 中文文本并编码成 uint32 二进制。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import load_config
from src.data import FINEWEB_REPO, build_text_bin, read_fineweb_cached
from src.hfenv import setup_hf
from src.tokenizer import QwenTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpt_fineweb.yaml")
    ap.add_argument("--endpoint", default="", help="空=直连 HF；可传 https://hf-mirror.com")
    ap.add_argument("--proxy", default=None, help="HTTP 代理，如 http://127.0.0.1:7890")
    ap.add_argument("--max_bytes", type=int, default=1_000_000_000, help="预训练文本字节上限")
    ap.add_argument("--out", default="data/processed/text/fineweb_cmn.train.bin")
    ap.add_argument("--val", default="data/processed/text/fineweb_cmn.val.bin")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_hf(args.endpoint, args.proxy or cfg.data.proxy)
    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)

    train_txt = read_fineweb_cached(FINEWEB_REPO, "cmn_Hani", "train",
                                    "000_00000.parquet", args.max_bytes,
                                    "data/raw/text/fineweb_cmn/train.jsonl.gz")
    val_txt = read_fineweb_cached(FINEWEB_REPO, "cmn_Hani", "test",
                                  "000_00000.parquet", max(2_000_000, args.max_bytes // 200),
                                  "data/raw/text/fineweb_cmn/val.jsonl.gz")
    n_train, n_val = build_text_bin(tok, train_txt, args.out, val_bin_path=args.val)
    print(f"完成：train={n_train} tokens, val={n_val} tokens")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写配置** `configs/gpt_fineweb.yaml`

```yaml
model:
  vocab_size: 151666
  d_model: 768
  n_layer: 12
  n_head: 12
  n_kv_head: 4
  d_ff: 2048
  ctx_len: 2048
  dropout: 0.0
  tie_embeddings: true
data:
  tokenizer_kind: qwen
  tokenizer_dir: data/tokenizer/qwen2.5-0.5b
  train_bin: data/processed/text/fineweb_cmn.train.bin
  val_bin: data/processed/text/fineweb_cmn.val.bin
  hf_endpoint: ""
  proxy: "http://127.0.0.1:7890"
train:
  batch_size: 8
  grad_accum: 8
  lr: 6.0e-4
  min_lr: 6.0e-5
  warmup_steps: 100
  max_steps: 2000
  eval_interval: 250
  save_interval: 500
  log_interval: 20
  out_dir: out/gpt_pretrain
  dtype: bf16
  compile: false
  seed: 42
```

- [ ] **Step 3: 冒烟运行**（小数据验证链路）

Run:
```powershell
$env:PYTHONUTF8=1
& ".venv\Scripts\python.exe" scripts/prepare_pretrain_data.py --max_bytes 2000000
```
Expected: 打印 `完成：train=... tokens, val=... tokens`；`data/raw/text/fineweb_cmn/train.jsonl.gz` 与两份 `.bin` 存在。

- [ ] **Step 4: 提交**

```bash
git add scripts/prepare_pretrain_data.py configs/gpt_fineweb.yaml
git commit -m "feat: pretrain data prep script + fineweb 2048 config (Task 3)"
```

---

## Task 4: 预训练跑通（ctx 2048）

**Files:**
- 复用 `scripts/train_gpt.py`、`configs/gpt_fineweb.yaml`
- 产物：`out/gpt_pretrain/{latest.pt,metrics.jsonl,loss.png}`

**Interfaces:**
- Produces: `out/gpt_pretrain/latest.pt`（供 Task 8 初始化）

- [ ] **Step 1: 全量数据准备**

Run:
```powershell
$env:PYTHONUTF8=1
& ".venv\Scripts\python.exe" scripts/prepare_pretrain_data.py
```
Expected: 抽取约 1GB 文本并编码；`fineweb_cmn.train.bin` 约 1.2GB。

- [ ] **Step 2: 冒烟训练（20 步验证 2048 上下文）**

Run:
```powershell
& ".venv\Scripts\python.exe" scripts/train_gpt.py --config configs/gpt_fineweb.yaml --set train.max_steps=20 train.eval_interval=10
```
Expected: 正常打印 step/loss/lr/mem，显存 < 16GB，生成 `out/gpt_pretrain/latest.pt`。

- [ ] **Step 3: 正式训练**

Run:
```powershell
& ".venv\Scripts\python.exe" scripts/train_gpt.py --config configs/gpt_fineweb.yaml
```
Expected: 2000 步跑完；`loss.png` 的 val loss 相对首步下降 > 40%。

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "chore: pretrain run at ctx=2048 (Task 4)" --allow-empty
```

---

## Task 5: 对话模板 `src/chat_format.py`

**Files:**
- Create: `src/chat_format.py`
- Test: `tests/test_chat_format.py`

**Interfaces:**
- Produces: `SYSTEM_PROMPT: str`
- Produces: `im_start_id(tok) -> int`、`im_end_id(tok) -> int`
- Produces: `render_messages(tok, messages, add_generation_prompt=False) -> str`
- Produces: `build_sft_example(tok, messages, max_len=2048) -> tuple[list[int], list[int]]`
- Produces: `has_supervision(labels) -> bool`
- Produces: `stop_ids(tok) -> list[int]`
- Produces: `strip_special_text(tok, text: str) -> str`（去掉回答里残留的特殊 token 字符串）
- Produces: `history_to_messages(history) -> list[dict]`

- [ ] **Step 1: 写失败测试** `tests/test_chat_format.py`

```python
from src.chat_format import (SYSTEM_PROMPT, build_sft_example, has_supervision,
                             history_to_messages, im_end_id, im_start_id,
                             render_messages, stop_ids, strip_special_text)
from src.tokenizer import QwenTokenizer

TOK = QwenTokenizer.load("data/tokenizer/qwen2.5-0.5b")


def test_im_tokens_single_id():
    assert TOK.encode("<|im_start|>") == [im_start_id(TOK)]
    assert TOK.encode("<|im_end|>") == [im_end_id(TOK)]


def test_render_messages_contains_roles_and_generation_prompt():
    s = render_messages(TOK, [{"role": "user", "content": "你好"}],
                        add_generation_prompt=True)
    assert s.startswith(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    assert "<|im_start|>user\n你好<|im_end|>\n" in s
    assert s.endswith("<|im_start|>assistant\n")


def test_sft_example_only_labels_assistant():
    msgs = [{"role": "user", "content": "你好"},
            {"role": "assistant", "content": "很高兴见到你"}]
    ids, labels = build_sft_example(TOK, msgs, max_len=256)
    assert len(ids) == len(labels)
    sup_ids = [ids[i] for i, l in enumerate(labels) if l != -100]
    assert has_supervision(labels)
    assert TOK.encode("很高兴见到你")[0] in sup_ids
    assert im_end_id(TOK) in sup_ids
    # 用户段所有位置都不监督（用前缀长度精确定位用户内容起点）
    user_start = len(TOK.encode(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"))
    for k in range(len(TOK.encode("你好"))):
        assert labels[user_start + k] == -100
    # 序列最后是 "\n"(-100)，其前是助手轮末 <|im_end|>(被监督)
    assert labels[-1] == -100 and labels[-2] != -100


def test_sft_example_truncates_keeping_last_turns():
    msgs = [{"role": "user", "content": "第一轮问题" * 30},
            {"role": "assistant", "content": "第一轮回答" * 30},
            {"role": "user", "content": "最后一问"},
            {"role": "assistant", "content": "最后答案"}]
    ids, labels = build_sft_example(TOK, msgs, max_len=64)
    assert len(ids) <= 64 and len(ids) == len(labels)
    assert has_supervision(labels)
    assert TOK.encode("最后答案")[0] in [ids[i] for i, l in enumerate(labels) if l != -100]


def test_stop_ids_and_strip_special():
    assert im_end_id(TOK) in stop_ids(TOK)
    assert strip_special_text(TOK, "你好<|im_end|>") == "你好"
    assert strip_special_text(TOK, "<|endoftext|>再见") == "再见"


def test_history_to_messages_both_formats():
    pairs = [["你好", "嗨"]]                       # gradio 元组形式
    assert history_to_messages(pairs) == [{"role": "user", "content": "你好"},
                                          {"role": "assistant", "content": "嗨"}]
    dicts = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert history_to_messages(dicts) == dicts
```

- [ ] **Step 2: 运行确认失败** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_chat_format.py -q`；Expected: FAIL（`No module named 'src.chat_format'`）。

- [ ] **Step 3: 实现** `src/chat_format.py`

```python
"""对话模板：复用 Qwen 的 <|im_start|>/<|im_end|>，构造 SFT 样本（只监督助手）。

教学注释：Qwen2.5 分词器词表里已有 <|im_start|>(151644) 与 <|im_end|>(151645)，
所以既不扩表也不改 embedding；id 一律用 convert_tokens_to_ids 动态取，避免写死。
"""
from __future__ import annotations

SYSTEM_PROMPT = "你是一个乐于助人的中文助手。"


def _ids(tok, text: str) -> list[int]:
    return tok.encode(text)


def im_start_id(tok) -> int:
    if hasattr(tok, "hf"):
        return tok.hf.convert_tokens_to_ids("<|im_start|>")
    return _ids(tok, "<|im_start|>")[0]


def im_end_id(tok) -> int:
    if hasattr(tok, "hf"):
        return tok.hf.convert_tokens_to_ids("<|im_end|>")
    return _ids(tok, "<|im_end|>")[0]


def render_messages(tok, messages: list[dict], add_generation_prompt: bool = False) -> str:
    """把消息列表渲染成 Qwen 对话格式字符串。"""
    out = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


def _segment(tok, role: str, content: str):
    """返回 (ids, labels)：仅 assistant 段的 content 与其后 <|im_end|> 计入损失。"""
    head = _ids(tok, f"<|im_start|>{role}\n")
    body = _ids(tok, content)
    end = [im_end_id(tok)]
    nl = _ids(tok, "\n")
    ids = head + body + end + nl
    if role == "assistant":
        labels = [-100] * len(head) + body + end + [-100] * len(nl)
    else:
        labels = [-100] * len(ids)
    return ids, labels


def build_sft_example(tok, messages: list[dict], max_len: int = 2048):
    """构造 SFT 的 (input_ids, labels)；超长则从最早的消息开始丢弃，保留最近若干轮。

    教学注释：与 render_messages 一样自动补 system 段，保证训练与推理的序列前缀一致。
    """
    msgs = list(messages)
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    segs = [_segment(tok, m["role"], m["content"]) for m in msgs]
    out_ids: list[int] = []
    out_labels: list[int] = []
    for ids, labels in reversed(segs):
        if out_ids and len(ids) + len(out_ids) > max_len:
            break
        out_ids = ids + out_ids
        out_labels = labels + out_labels
    if len(out_ids) > max_len:                      # 单条消息就超长：保守截断
        out_ids = out_ids[:max_len]
        out_labels = out_labels[:max_len]
    return out_ids, out_labels


def has_supervision(labels: list[int]) -> bool:
    return any(l != -100 for l in labels)


def stop_ids(tok) -> list[int]:
    """生成时的停止 token：正常情况下模型应输出 <|im_end|>，eos 兜底。"""
    return [im_end_id(tok), tok.special_id("eos")]


def strip_special_text(tok, text: str) -> str:
    """去掉回答里可能残留的特殊 token 字符串（显示用）。"""
    for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>",
                  "<image>", "<pad>", "<bos>", "<eos>", "<unk>"):
        text = text.replace(token, "")
    return text.strip()


def history_to_messages(history) -> list[dict]:
    """把 Gradio 的 history（元组或 dict 形式）转成 messages 列表。"""
    msgs: list[dict] = []
    for item in history or []:
        if isinstance(item, dict):
            msgs.append({"role": item["role"], "content": item["content"]})
        else:
            user, bot = item[0], item[1]
            if user:
                msgs.append({"role": "user", "content": user})
            if bot:
                msgs.append({"role": "assistant", "content": bot})
    return msgs
```

- [ ] **Step 4: 运行确认通过** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_chat_format.py -q`；Expected: PASS（6 passed）。

- [ ] **Step 5: 提交**

```bash
git add src/chat_format.py tests/test_chat_format.py
git commit -m "feat: Qwen chat template + assistant-only SFT labels (Task 5)"
```

---

## Task 6: SFT 数据归一化与多轮合成

**Files:**
- Create: `src/sft_data.py`
- Test: `tests/test_sft_data.py`

**Interfaces:**
- Produces: `normalize_alpaca(obj) -> dict`、`normalize_firefly(obj) -> dict`
- Produces: `synthesize_dialogue(pairs, rng, max_turns=3) -> dict`
- Produces: `iter_sft_examples(paths, max_items=None, synth_prob=0.5, seed=0) -> Iterator[dict]`

- [ ] **Step 1: 写失败测试** `tests/test_sft_data.py`

```python
import json
import random
from src.sft_data import (iter_sft_examples, normalize_alpaca, normalize_firefly,
                          synthesize_dialogue)


def test_normalize_alpaca_with_input():
    msgs = normalize_alpaca({"instruction": "翻译", "input": "hello", "output": "你好"})["messages"]
    assert msgs == [{"role": "user", "content": "翻译\nhello"},
                    {"role": "assistant", "content": "你好"}]


def test_normalize_alpaca_without_input():
    msgs = normalize_alpaca({"instruction": "你好", "input": "", "output": "嗨"})["messages"]
    assert msgs[0]["content"] == "你好"


def test_normalize_firefly():
    msgs = normalize_firefly({"kind": "NLI", "input": "前提", "target": "中立"})["messages"]
    assert msgs == [{"role": "user", "content": "前提"},
                    {"role": "assistant", "content": "中立"}]


def test_synthesize_dialogue_alternates_roles():
    pairs = [("问1", "答1"), ("问2", "答2"), ("问3", "答3")]
    conv = synthesize_dialogue(pairs, random.Random(0), max_turns=3)
    roles = [m["role"] for m in conv["messages"]]
    assert roles == ["user", "assistant"] * (len(roles) // 2)
    assert 2 <= len(conv["messages"]) <= 6


def test_iter_sft_examples_two_formats(tmp_path):
    alp = tmp_path / "alpaca.json"
    alp.write_text(json.dumps([{"instruction": "i1", "input": "", "output": "o1"}]),
                   encoding="utf-8")
    ff = tmp_path / "firefly.jsonl"
    ff.write_text(json.dumps({"kind": "k", "input": "i2", "target": "o2"}) + "\n",
                  encoding="utf-8")
    got = list(iter_sft_examples([str(alp), str(ff)], synth_prob=0.0))
    assert len(got) == 2
    assert all(len(g["messages"]) == 2 for g in got)
```

- [ ] **Step 2: 运行确认失败** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_sft_data.py -q`；Expected: FAIL（`No module named 'src.sft_data'`）。

- [ ] **Step 3: 实现（第一部分）** `src/sft_data.py`

```python
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
```

- [ ] **Step 4: 运行确认通过** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_sft_data.py -q`；Expected: PASS（5 passed）。

- [ ] **Step 5: 提交**

```bash
git add src/sft_data.py tests/test_sft_data.py
git commit -m "feat: SFT data normalization + multi-turn synthesis (Task 6)"
```

---

## Task 7: SFT 分词落盘 + Dataset + collate + 数据脚本

**Files:**
- Modify: `src/sft_data.py`（追加）
- Create: `scripts/prepare_sft_data.py`、`configs/sft_zh.yaml`
- Test: `tests/test_sft_data.py`（追加）

**Interfaces:**
- Produces: `tokenize_examples(tok, examples, max_len) -> tuple[dict, int]`
- Produces: `save_sft_npz(tok, examples, path, max_len) -> int`
- Produces: `class SFTDataset(npz_path)`（`__getitem__ -> (list[int], list[int])`）
- Produces: `collate_sft(batch, pad_id, max_len=None) -> dict`（`{"input_ids","labels"}`）

- [ ] **Step 1: 追加失败测试**（`tests/test_sft_data.py` 末尾）

```python
import numpy as np
import torch
from src.sft_data import (SFTDataset, collate_sft, save_sft_npz, tokenize_examples)
from src.tokenizer import CharTokenizer

SAMPLE = [{"messages": [{"role": "user", "content": "问"} ,
                        {"role": "assistant", "content": "答"}]}]


def _tok():
    return CharTokenizer.train(["问答复你好"], max_chars=100)


def test_tokenize_and_npz_roundtrip(tmp_path):
    tok = _tok()
    p = tmp_path / "t.npz"
    n = save_sft_npz(tok, SAMPLE, str(p), max_len=64)
    assert n == 1
    ds = SFTDataset(str(p))
    assert len(ds) == 1
    ids, labels = ds[0]
    assert len(ids) == len(labels)
    assert any(l != -100 for l in labels)


def test_tokenize_skips_examples_without_supervision():
    tok = _tok()
    bad = [{"messages": [{"role": "user", "content": "只有用户"}]}]
    arrays, n = tokenize_examples(tok, bad, max_len=64)
    assert n == 0 and len(arrays["offsets"]) == 1


def test_collate_pads_and_masks():
    tok = _tok()
    batch = [([1, 2, 3, 4], [-100, -100, 5, 6]), ([1, 2], [-100, 7])]
    out = collate_sft(batch, pad_id=0)
    assert out["input_ids"].shape == (2, 4)
    assert out["labels"].tolist()[1][2:] == [-100, -100]
    assert isinstance(out["input_ids"], torch.Tensor)
```

- [ ] **Step 2: 运行确认失败** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_sft_data.py -q`；Expected: FAIL（`cannot import name 'SFTDataset'`）。

- [ ] **Step 3: 实现（第二部分，追加到 `src/sft_data.py`）**

```python
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
```

- [ ] **Step 4: 运行确认通过** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_sft_data.py -q`；Expected: PASS（8 passed）。

- [ ] **Step 5: 写配置与数据脚本**（`configs/sft_zh.yaml`、`scripts/prepare_sft_data.py`）

```python
"""准备 SFT 数据：下载 alpaca-zh + firefly 子集，归一化/合成多轮，编码成 npz。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import load_config
from src.hfenv import setup_hf
from src.sft_data import iter_sft_examples, save_sft_npz
from src.tokenizer import QwenTokenizer


def download_sft(dest_dir: str, firefly_items: int, proxy: str | None):
    """下载 alpaca-zh（整文件）与 firefly 前 N 条（流式，不整下 1.17GB）。"""
    from huggingface_hub import HfFileSystem

    root = Path(dest_dir)
    root.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    alp = root / "alpaca_gpt4_data_zh.json"
    if not alp.exists():
        with fs.open("datasets/shibing624/alpaca-zh/alpaca_gpt4_data_zh.json", "rb") as f:
            alp.write_bytes(f.read())
    ff = root / "firefly_subset.jsonl"
    if not ff.exists():
        with fs.open("datasets/YeungNLP/firefly-train-1.1M/firefly-train-1.1M.jsonl", "rb") as f:
            import io
            with ff.open("w", encoding="utf-8") as out:
                for i, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
                    if i >= firefly_items:
                        break
                    out.write(line)
    return [str(alp), str(ff)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_zh.yaml")
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--firefly_items", type=int, default=40000)
    ap.add_argument("--out", default="data/processed/sft/train.npz")
    ap.add_argument("--val", default="data/processed/sft/val.npz")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_hf(args.endpoint, args.proxy or cfg.data.proxy)
    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
    paths = download_sft("data/raw/sft", args.firefly_items, args.proxy or cfg.data.proxy)

    # 先全部读成列表，切分 train/val（固定顺序，保证可复现）
    examples = list(iter_sft_examples(paths, synth_prob=0.5, seed=0))
    n_val = max(1, int(len(examples) * args.val_ratio))
    train_ex, val_ex = examples[:-n_val], examples[-n_val:]
    n_tr = save_sft_npz(tok, train_ex, args.out, cfg.data.max_len)
    n_va = save_sft_npz(tok, val_ex, args.val, cfg.data.max_len)
    print(f"完成：train={n_tr}, val={n_va}（共 {len(examples)} 段对话）")


if __name__ == "__main__":
    main()
```

同一步内写入 `configs/sft_zh.yaml`（SFT 训练与数据准备共用）：

```yaml
model:
  vocab_size: 151666
  d_model: 768
  n_layer: 12
  n_head: 12
  n_kv_head: 4
  d_ff: 2048
  ctx_len: 2048
  dropout: 0.0
  tie_embeddings: true
data:
  tokenizer_kind: qwen
  tokenizer_dir: data/tokenizer/qwen2.5-0.5b
  train_bin: data/processed/text/fineweb_cmn.train.bin
  val_bin: data/processed/text/fineweb_cmn.val.bin
  hf_endpoint: ""
  proxy: "http://127.0.0.1:7890"
  sft_train: data/processed/sft/train.npz
  sft_val: data/processed/sft/val.npz
  max_len: 2048
train:
  batch_size: 8
  grad_accum: 4
  lr: 1.0e-4
  min_lr: 1.0e-5
  warmup_steps: 50
  max_steps: 1500
  eval_interval: 100
  save_interval: 300
  log_interval: 20
  out_dir: out/sft
  dtype: bf16
  compile: false
  seed: 42
```

- [ ] **Step 6: 冒烟运行**

Run:
```powershell
$env:PYTHONUTF8=1
& ".venv\Scripts\python.exe" scripts/prepare_sft_data.py --firefly_items 200
```
Expected: 下载两个文件并打印 `完成：train=...`；`data/processed/sft/{train,val}.npz` 生成。

- [ ] **Step 7: 提交**

```bash
git add src/sft_data.py scripts/prepare_sft_data.py configs/sft_zh.yaml tests/test_sft_data.py
git commit -m "feat: SFT tokenize-to-npz, Dataset/collate, data prep script, and SFT config (Task 7)"
```

---

## Task 8: SFTTrainer + 训练脚本 + 配置

**Files:**
- Modify: `src/trainer.py`（追加 `SFTTrainer`）
- Create: `scripts/train_sft.py`（配置 `configs/sft_zh.yaml` 由 Task 7 创建）
- Test: `tests/test_sft_trainer.py`

**Interfaces:**
- Consumes: `SFTDataset`、`collate_sft`、`Trainer`、`_autocast_ctx`
- Produces: `class SFTTrainer(Trainer)`，签名 `(model, train_cfg, train_ds, tokenizer, max_len=2048, val_ds=None, device=None)`
- Produces: `out/sft/latest.pt`

- [ ] **Step 1: 确认配置文件存在**（`configs/sft_zh.yaml` 已由 Task 7 创建）

Run: `Test-Path configs/sft_zh.yaml`；Expected: `True`。

- [ ] **Step 2: 写失败测试** `tests/test_sft_trainer.py`

```python
import numpy as np
from src.config import ModelConfig, TrainConfig
from src.model import GPT
from src.tokenizer import CharTokenizer
from src.sft_data import SFTDataset, save_sft_npz
from src.trainer import SFTTrainer


def _ds(tmp_path):
    tok = CharTokenizer.train(["你好世界答问abc"])
    examples = []
    for _ in range(8):
        examples.append({"messages": [{"role": "user", "content": "你好"},
                                      {"role": "assistant", "content": "你好答"}]})
    p = tmp_path / "t.npz"
    save_sft_npz(tok, examples, str(p), max_len=64)
    return tok, SFTDataset(str(p))


def test_sft_trainer_overfits_tiny_set(tmp_path):
    tok, ds = _ds(tmp_path)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layer=2, n_head=4,
                      n_kv_head=2, d_ff=128, ctx_len=64)
    tcfg = TrainConfig(batch_size=4, grad_accum=1, lr=3e-3, warmup_steps=1,
                       max_steps=150, eval_interval=10_000, save_interval=10_000,
                       log_interval=10_000, out_dir=str(tmp_path / "out"))
    model = GPT(cfg)
    tr = SFTTrainer(model, tcfg, ds, tok, max_len=64, val_ds=ds, device="cpu")
    tr.train()
    assert tr._last_loss < 0.3, f"SFT 过拟合失败：loss={tr._last_loss:.3f}"
```

- [ ] **Step 3: 运行确认失败** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_sft_trainer.py -q`；Expected: FAIL（`cannot import name 'SFTTrainer'`）。

- [ ] **Step 4: 实现**（追加到 `src/trainer.py` 末尾）

```python
class SFTTrainer(Trainer):
    """指令微调训练器：文本序列 + label 掩码（只监督助手），复用 Trainer 的优化/日志/续训。

    教学注释：与 VLMTrainer 一样，只覆写"怎么取一批数据"和"怎么算 loss"两个接缝，
    训练循环完全复用基类——这就是把数据形态与训练循环解耦的价值。
    """

    def __init__(self, model, train_cfg, train_ds, tokenizer, max_len: int = 2048,
                 val_ds=None, device: str | None = None):
        from torch.utils.data import DataLoader

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.special_id("pad")
        self.train_loader = DataLoader(
            train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True,
            collate_fn=self._collate, num_workers=0,
        )
        self.val_loader = None
        if val_ds is not None and len(val_ds) > 0:
            self.val_loader = DataLoader(
                val_ds, batch_size=train_cfg.batch_size, shuffle=False, drop_last=False,
                collate_fn=self._collate, num_workers=0,
            )
        self._iter = None
        super().__init__(model, train_cfg, None, None, tokenizer=tokenizer, device=device)

    def _collate(self, batch):
        from src.sft_data import collate_sft
        return collate_sft(batch, self.pad_id, self.max_len)

    def _next_batch(self):
        if self._iter is None:
            self._iter = iter(self.train_loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.train_loader)
            return next(self._iter)

    def _forward_loss(self, batch):
        with _autocast_ctx(self.cfg, self.device):
            _, loss, _ = self.model(batch["input_ids"].to(self.device),
                                    targets=batch["labels"].to(self.device))
        self._last_loss = loss.item()
        return loss

    @torch.no_grad()
    def _estimate_val(self) -> float:
        if self.val_loader is None:
            return self._forward_loss(self._next_batch()).item()
        self.model.eval()
        losses = []
        for b in self.val_loader:
            with _autocast_ctx(self.cfg, self.device):
                _, loss, _ = self.model(b["input_ids"].to(self.device),
                                        targets=b["labels"].to(self.device))
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(1, len(losses))
```

- [ ] **Step 5: 运行确认通过** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_sft_trainer.py -q`；Expected: PASS（1 passed，约 10–40s）。

- [ ] **Step 6: 写训练脚本** `scripts/train_sft.py`

```python
"""SFT 训练：从预训练 checkpoint 初始化，用对话数据做指令微调。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.config import apply_overrides, load_config
from src.model import GPT
from src.sft_data import SFTDataset
from src.tokenizer import QwenTokenizer
from src.trainer import SFTTrainer
from src.utils import human_params, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_zh.yaml")
    ap.add_argument("--init", default="out/gpt_pretrain/latest.pt", help="预训练权重")
    ap.add_argument("--resume", default=None, help="从 SFT checkpoint 续训")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    set_seed(cfg.train.seed)
    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
    cfg.model.vocab_size = tok.vocab_size

    model = GPT(cfg.model)
    if args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = args.init
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"])
        print(f"已从 {ckpt_path} 加载权重")
    else:
        print(f"[warn] 未找到 {ckpt_path}，将从头训练")
    print(f"模型参数量 {human_params(model.num_params())}")

    train_ds = SFTDataset(cfg.data.sft_train)
    val_ds = SFTDataset(cfg.data.sft_val) if Path(cfg.data.sft_val).exists() else None
    print(f"SFT 样本：train={len(train_ds)}, val={len(val_ds) if val_ds else 0}")
    trainer = SFTTrainer(model, cfg.train, train_ds, tok, max_len=cfg.data.max_len, val_ds=val_ds)
    if args.resume:
        trainer.load(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 全量测试并提交**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/ -q`；Expected: 全绿。

```bash
git add src/trainer.py scripts/train_sft.py tests/test_sft_trainer.py
git commit -m "feat: SFTTrainer (assistant-masked loss) + train_sft entry (Task 8)"
```

---

## Task 9: `generate` 支持停止 token + `chat.py` 多轮

**Files:**
- Modify: `src/generate.py`、`scripts/chat.py`
- Test: `tests/test_generate_stop.py`

**Interfaces:**
- Produces: `generate(..., stop_ids: list[int] | None = None)`（默认 `[eos]`）
- Produces: `scripts/chat.py --mode {chat,base}`

- [ ] **Step 1: 写失败测试** `tests/test_generate_stop.py`

```python
import torch
from src.generate import generate
from src.tokenizer import CharTokenizer


class _FakeModel(torch.nn.Module):
    """永远输出固定 token 的假模型，用来验证 stop_ids 行为。"""

    def __init__(self, vocab: int, token_id: int):
        super().__init__()
        self.vocab, self.token_id = vocab, token_id

    def forward(self, idx, past_kvs=None, use_cache=False):
        b, t = idx.shape
        logits = torch.full((b, t, self.vocab), -1e9)
        logits[..., self.token_id] = 0.0
        return logits, None, None


def test_generate_stops_on_stop_id():
    tok = CharTokenizer.train(["你好abc"])
    stop = tok.special_id("eos")
    model = _FakeModel(tok.vocab_size, stop)
    out = generate(model, tok, "你好", max_new_tokens=20, temperature=0.0,
                   stop_ids=[stop], device="cpu")
    assert out == ""                                # 第一个 token 就是 stop，立即停止


def test_generate_runs_to_max_without_matching_stop():
    tok = CharTokenizer.train(["你好abc"])
    model = _FakeModel(tok.vocab_size, tok.special_id("unk"))
    out = generate(model, tok, "你好", max_new_tokens=3, temperature=0.0,
                   stop_ids=[tok.special_id("eos")], device="cpu")
    assert len(tok.encode(out)) >= 1
```

- [ ] **Step 2: 运行确认失败** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_generate_stop.py -q`；Expected: FAIL（`generate() got an unexpected keyword argument 'stop_ids'`）。

- [ ] **Step 3: 修改 `src/generate.py`**

把 `generate` 签名与停止逻辑改为：

```python
@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_k=None, top_p: float | None = 0.9,
             repetition_penalty: float = 1.0, device: str = "cuda",
             generator=None, stop_ids: list[int] | None = None) -> str:
    """给定提示词，自回归生成文本。使用 KV 缓存避免重复计算。

    教学注释：stop_ids 支持"遇到任一指定 token 就停"，聊天场景用 <|im_end|> 停止，
    基座续写沿用 eos。stop token 本身不进入输出。
    """
    model.eval()
    model.to(device)
    ids = tokenizer.encode(prompt, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stop = set(stop_ids) if stop_ids is not None else {tokenizer.special_id("eos")}
    generated = []

    past = None
    for _ in range(max_new_tokens):
        cur = idx if past is None else idx[:, -1:]
        logits, _, past = model(cur, past_kvs=past, use_cache=True)
        step_logits = logits[0, -1].float().clone()
        _apply_repetition_penalty(step_logits, ids + generated, repetition_penalty)
        nxt = sample_logits(step_logits, temperature, top_k, top_p, generator)
        tid = int(nxt.item())
        if tid in stop:
            break
        generated.append(tid)
        idx = torch.cat([idx, nxt.view(1, 1).to(device)], dim=1)

    return tokenizer.decode(generated)
```

- [ ] **Step 4: 运行确认通过** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_generate_stop.py -q`；Expected: PASS（2 passed）。

- [ ] **Step 5: 改造 `scripts/chat.py`**（整体替换）

```python
"""命令行聊天：--mode chat 走对话模板多轮，--mode base 走原来的续写。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.config import load_config
from src.generate import generate
from src.model import GPT
from src.tokenizer import QwenTokenizer


def load_model(cfg_path, ckpt, tok):
    cfg = load_config(cfg_path)
    cfg.model.vocab_size = tok.vocab_size
    model = GPT(cfg.model)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chat", "base"], default="chat")
    ap.add_argument("--config", default="configs/sft_zh.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    if args.ckpt is None:
        args.ckpt = "out/sft/latest.pt" if args.mode == "chat" else "out/gpt/latest.pt"
    tok = QwenTokenizer.load("data/tokenizer/qwen2.5-0.5b")
    model = load_model(args.config, args.ckpt, tok)

    if args.mode == "base":
        print("模型已加载（续写模式）。输入 /quit 退出。")
        while True:
            prompt = input("你> ").strip()
            if prompt in ("/quit", "/exit", ""):
                break
            print("AI> " + generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_p=args.top_p))
        return

    from src.chat_format import (render_messages, stop_ids, strip_special_text)
    print("模型已加载（对话模式）。输入 /reset 清空历史，/quit 退出。")
    messages = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    while True:
        user = input("你> ").strip()
        if user in ("/quit", "/exit", ""):
            break
        if user == "/reset":
            messages = []
            print("[已清空对话历史]")
            continue
        messages.append({"role": "user", "content": user})
        prompt = render_messages(tok, messages, add_generation_prompt=True)
        reply = generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_p=args.top_p,
                         device=device, stop_ids=stop_ids(tok))
        reply = strip_special_text(tok, reply)
        messages.append({"role": "assistant", "content": reply})
        print("AI> " + reply)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 冒烟测试对话**（需 Task 8 已产出 `out/sft/latest.pt`；若没有，用 `--mode base` 验证脚本可跑）

Run:
```powershell
$env:PYTHONUTF8=1
& ".venv\Scripts\python.exe" scripts/chat.py --mode chat --max_new_tokens 32
```
Expected: 进入对话，输入「你好」后能打印一段中文回答（或至少在 `out/sft/latest.pt` 缺失时打印 warn 并仍能加载随机权重跑通）。

- [ ] **Step 7: 全量测试并提交**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/ -q`；Expected: 全绿。

```bash
git add src/generate.py scripts/chat.py tests/test_generate_stop.py
git commit -m "feat: stop_ids in generate + multi-turn chat CLI with --mode (Task 9)"
```

---

## Task 10: 自动评测（`src/chat_eval.py` + `scripts/eval_chat.py`）

**Files:**
- Create: `src/chat_eval.py`（纯函数 + 固定题库，可单测）、`scripts/eval_chat.py`（运行器）
- Test: `tests/test_chat_eval.py`

**Interfaces:**
- Produces: `src.chat_eval.check_output(text: str, stopped: bool) -> dict`（键 `non_empty/no_repeat/stopped/ok`）
- Produces: `src.chat_eval.recall_ok(text: str, keyword: str) -> bool`
- Produces: `src.chat_eval.PROMPTS: list[str]`、`src.chat_eval.MULTI_TURN: list[tuple[str, str]]`
- Produces: `out/sft/eval.md`

- [ ] **Step 1: 写失败测试** `tests/test_chat_eval.py`

```python
from src.chat_eval import check_output, recall_ok


def test_check_output_flags_empty_and_repetition():
    assert check_output("", stopped=True)["ok"] is False
    rep = "哈哈哈哈" * 40
    assert check_output(rep, stopped=True)["no_repeat"] is False
    assert check_output(rep, stopped=True)["ok"] is False


def test_check_output_requires_stop():
    assert check_output("这是一个正常的回答。", stopped=False)["ok"] is False
    assert check_output("这是一个正常的回答。", stopped=True)["ok"] is True


def test_recall_ok():
    assert recall_ok("北京的简称是京。", "北京") is True
    assert recall_ok("不知道。", "北京") is False
```

- [ ] **Step 2: 运行确认失败** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_chat_eval.py -q`；Expected: FAIL（`No module named 'src.chat_eval'`）。

- [ ] **Step 3: 实现纯逻辑** `src/chat_eval.py`

```python
"""对话评测的纯逻辑与固定题库（不依赖模型/网络，便于单测）。"""
from __future__ import annotations

from collections import Counter

PROMPTS = [
    "你好，请介绍一下你自己。", "什么是人工智能？", "给我讲一个一句话的笑话。",
    "把“今天天气很好”翻译成英文。", "列出三种水果。", "1 加 1 等于几？",
    "用一句话解释什么是太阳。", "推荐一本适合初学者读的书。", "什么是水循环？",
    "写一句鼓励人的话。", "苹果和香蕉哪个通常更甜？", "中国首都是哪里？",
    "用三个词形容大海。", "为什么要多喝水？", "把“谢谢你”改写得更正式一些。",
    "简单说说地球为什么有四季。", "什么是计算机？", "给“小猫”写一个比喻句。",
]
MULTI_TURN = [("我叫小明，请记住。", "小明"), ("我刚才说我叫什么？", "小明")]


def check_output(text: str, stopped: bool) -> dict:
    """检查单条回答：非空、无明显重复、能正常停止。

    教学注释：小模型最容易出的两种毛病是"空/极短"和"复读循环"，
    用最高频二元组占比来近似检测复读。
    """
    t = (text or "").strip()
    non_empty = len(t) >= 2
    grams = [t[i:i + 2] for i in range(len(t) - 1)]
    rep = (max(Counter(grams).values()) / len(grams)) if grams else 1.0
    no_repeat = rep < 0.5
    return {"non_empty": non_empty, "no_repeat": no_repeat, "stopped": stopped,
            "ok": non_empty and no_repeat and stopped}


def recall_ok(text: str, keyword: str) -> bool:
    """多轮召回：回答里是否出现前文的关键词。"""
    return bool(keyword) and keyword in (text or "")
```

- [ ] **Step 4: 运行确认通过** — Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_chat_eval.py -q`；Expected: PASS（3 passed）。

- [ ] **Step 5: 写运行器** `scripts/eval_chat.py`

```python
"""自动评测：固定留出问题 → 生成 → 检查非空/不重复/正常停止/多轮引用。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_zh.yaml")
    ap.add_argument("--ckpt", default="out/sft/latest.pt")
    ap.add_argument("--out", default="out/sft/eval.md")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    import torch
    from src.chat_eval import MULTI_TURN, PROMPTS, check_output, recall_ok
    from src.chat_format import render_messages, stop_ids, strip_special_text
    from src.config import load_config
    from src.generate import generate
    from src.model import GPT
    from src.tokenizer import QwenTokenizer

    tok = QwenTokenizer.load("data/tokenizer/qwen2.5-0.5b")
    cfg = load_config(args.config)
    cfg.model.vocab_size = tok.vocab_size
    model = GPT(cfg.model)
    if Path(args.ckpt).exists():
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
    else:
        print(f"[warn] 未找到 {args.ckpt}，使用随机权重")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stop = stop_ids(tok)

    def ask(messages):
        prompt = render_messages(tok, messages, add_generation_prompt=True)
        raw = generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                       temperature=0.7, top_p=0.9, device=device, stop_ids=stop)
        n_new = len(tok.encode(raw))
        return strip_special_text(tok, raw), n_new < args.max_new_tokens

    lines, n_ok = ["# SFT 对话评测", ""], 0
    for q in PROMPTS:
        out, stopped = ask([{"role": "user", "content": q}])
        res = check_output(out, stopped)
        n_ok += int(res["ok"])
        lines.append(f"- **Q**: {q}\n  - A: {out}\n  - {res}")

    messages = []
    for q, kw in MULTI_TURN:
        messages.append({"role": "user", "content": q})
        out, _ = ask(messages)
        messages.append({"role": "assistant", "content": out})
        ok = recall_ok(out, kw)
        n_ok += int(ok)
        lines.append(f"- **多轮 Q**: {q}\n  - A: {out}\n  - recall({kw})={ok}")

    total = len(PROMPTS) + len(MULTI_TURN)
    lines.insert(1, f"\n通过 {n_ok}/{total}。\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"评测完成：{n_ok}/{total} 通过，结果写入 {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 提交**

```bash
git add src/chat_eval.py scripts/eval_chat.py tests/test_chat_eval.py
git commit -m "feat: automated chat eval (non-empty / no-repeat / stop / multi-turn recall) (Task 10)"
```

---

## Task 11: webapp 多轮文本聊天

**Files:**
- Modify: `scripts/webapp.py`

**Interfaces:**
- Consumes: `render_messages`、`stop_ids`、`history_to_messages`
- Produces: `build_demo` 的文本页为多轮 `gr.ChatInterface`

- [ ] **Step 1: 新增文本模型加载（改 `load_models`）**

在 `load_models` 中，除了现有 `text_model`（基座续写），再加载一个 SFT 对话模型：

```python
def load_chat_model(args, tok):
    """加载 SFT 对话模型（存在才用）。"""
    if not Path(args.sft_ckpt).exists():
        return None
    cfg = load_config(args.sft_config)
    cfg.model.vocab_size = tok.vocab_size
    m = GPT(cfg.model)
    m.load_state_dict(torch.load(args.sft_ckpt, map_location="cpu", weights_only=False)["model"])
    print(f"已加载对话模型：{args.sft_ckpt}")
    return m
```

并在 `main()` 的 argparse 增加 `--sft_config`（默认 `configs/sft_zh.yaml`）、`--sft_ckpt`（默认 `out/sft/latest.pt`）。

- [ ] **Step 2: 把「文本聊天」页换成多轮 `ChatInterface`**

在 `build_demo` 中，用下面替换原来的单轮 `gr.Textbox` 文本页（保留图片页）：

```python
    def chat_reply(message, history, max_tokens, temperature, top_p):
        from src.chat_format import (history_to_messages, render_messages,
                                     stop_ids, strip_special_text)
        if chat_model is None:
            return "未找到对话模型，请先训练：python scripts/train_sft.py"
        messages = history_to_messages(history)
        messages.append({"role": "user", "content": message})
        prompt = render_messages(tok, messages, add_generation_prompt=True)
        out = generate(chat_model, tok, prompt, max_new_tokens=int(max_tokens),
                       temperature=float(temperature), top_p=float(top_p), device=DEV,
                       stop_ids=stop_ids(tok))
        return strip_special_text(tok, out)
```

```python
        with gr.Tab("文本对话"):
            gr.ChatInterface(
                chat_reply,
                additional_inputs=[
                    gr.Slider(16, 512, value=256, step=16, label="生成长度"),
                    gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="temperature"),
                    gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p"),
                ],
            )
```

`build_demo` 需要能拿到 `chat_model`：把签名改为 `build_demo(tok, chat_model, vlm)`，
`main()` 里 `build_demo(tok, load_chat_model(args, tok), vlm)`。

- [ ] **Step 3: 冒烟检查**

Run:
```powershell
& ".venv\Scripts\python.exe" scripts/webapp.py --check
```
Expected: 打印 `[check] 界面构建成功；VLM 可用：True`（且不报 ChatInterface 参数错误）。

- [ ] **Step 4: 提交**

```bash
git add scripts/webapp.py
git commit -m "feat: multi-turn Gradio ChatInterface for SFT chat (Task 11)"
```

---

## Task 12: 文档与最终验收

**Files:**
- Create: `docs/05-sft.md`
- Modify: `README.md`
- 产物：`out/sft/eval.md`

- [ ] **Step 1: 写 `docs/05-sft.md`**

内容必须覆盖：
1. 为什么需要 SFT（基座只会续写）与整体流程（预训练 → SFT → 对话）。
2. 数据：fineweb-2 `cmn_Hani`、alpaca-zh、firefly；**多轮合成**的做法与理由（单轮来源）。
3. 对话模板：`<|im_start|>/<|im_end|>` 序列、**只监督 assistant**、`-100` 掩码、截断策略。
4. 训练命令（预训练 / SFT / 续训）与预期 loss 曲线。
5. 对话用法：`chat.py --mode chat`、`webapp.py`、`eval_chat.py`。
6. 能力边界与常见问题（胡言乱语、不停止、context 2048、代理下载）。

- [ ] **Step 2: 更新 `README.md`**

- 里程碑表新增 M9–M12（✅）。
- 新增「12. 对话（SFT）」小节：环境（代理）、数据准备、预训练、SFT、对话、评测命令。
- 顶部项目简介补一句「先教说话 → 装眼睛 → 学会对话」。

- [ ] **Step 3: 端到端验收**

Run:
```powershell
$env:PYTHONUTF8=1
& ".venv\Scripts\python.exe" -m pytest tests/ -q
& ".venv\Scripts\python.exe" scripts/eval_chat.py
```
Expected: 测试全绿；评测打印通过数与 `out/sft/eval.md`，其中非空/不重复/停止三项应基本全通过（多轮召回作为观察项）。

- [ ] **Step 4: 提交**

```bash
git add docs/05-sft.md README.md
git commit -m "docs: SFT guide + README milestones and chat usage (Task 12)"
```

---

## Self-Review（对照 spec 检查）

- **Spec §4.1 预训练源** → Task 2/3 覆盖（`iter_parquet_text`/`read_fineweb_cached` + `prepare_pretrain_data.py`）。
- **Spec §4.2 SFT 源与多轮合成** → Task 6/7 覆盖。
- **Spec §5.1 hfenv** → Task 1 覆盖。
- **Spec §5.2 chat_format（只监督助手/停止/剥离/history）** → Task 5 覆盖。
- **Spec §5.3 SFTDataset/collate/SFTTrainer** → Task 7/8 覆盖。
- **Spec §5.4 ctx=2048 预训练** → Task 3/4 覆盖。
- **Spec §5.5 stop_ids/chat.py/webapp/eval** → Task 9/10/11 覆盖。
- **Spec §8 测试与验收** → 5 个新测试文件 + Task 12 端到端验收覆盖。
- **Spec §6 依赖** → Task 1 前需 `pyarrow`（已装）；无新增未列依赖。
- **类型一致性**：`build_sft_example`/`has_supervision`/`collate_sft`/`SFTDataset`/`SFTTrainer`/`stop_ids`/`history_to_messages` 在各 Task 中名称与签名一致。
- **已消除的易错点**：`generate` 增加 `stop_ids` 而非改返回值；`im_*_id` 对无 `.hf` 的分词器有回退（便于用 `CharTokenizer` 做快速过拟合测试）；`_segment` 把轮末 `\n` 与 `<|im_end|>` 拆开，避免误监督换行；`build_sft_example` 自动补 system 段，与 `render_messages` 前缀一致（训练/推理不漂移）；Task 10 的纯逻辑放 `src/chat_eval.py`，避免测试去 import `scripts/`。
