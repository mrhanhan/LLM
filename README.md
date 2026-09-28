# mini-llm-lab

从零手写一个**中文 GPT** 的教学项目：不依赖任何 HuggingFace 高层模型封装，
只用 PyTorch 原生算子实现「分词 → 数据 → 模型 → 训练 → 生成/聊天」的完整闭环。

> 主线叙事：**先教它说话**（文本 LLM）→ **再给它装眼睛**（手写 ViT）→ **让它开口描述世界**（多模态）。
> 当前仓库已完成第一阶段（文本闭环），第二阶段的实现计划见
> `docs/superpowers/plans/2026-09-28-mini-llm-vlm.md`。

## 1. 项目简介

| 组件 | 实现 |
|---|---|
| 分词器 | 「手写字节级 BPE（教学核心）」+「字符级」+「Qwen2.5 分词器适配」三种，配置切换 |
| 数据管道 | 从 `hf-mirror` 下载中文 TinyStories → 解析 jsonl → 分词 → 拼成 `uint32` 二进制 |
| 模型 | 纯 PyTorch 手写 decoder-only Transformer：RoPE + RMSNorm + SwiGLU + GQA + 权重共享 |
| 训练 | bf16 混合精度 + 梯度累积 + 梯度裁剪 + warmup/余弦调度 + 断点续训 + loss 曲线 |
| 生成 | 带 KV 缓存的采样（temperature / top-k / top-p / 重复惩罚）+ 命令行聊天 |

默认结构：`d_model=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2048, ctx_len=1024`。
**主干（不含词表）约 75.5M 参数**；Qwen 词表 151666，与输出投影共享，**总体约 192M**。
16GB 显存即可训练（显存不够时用梯度累积降 micro-batch，见第 8 节）。

原理讲解见 `docs/`：

- [docs/01-tokenizer.md](docs/01-tokenizer.md) —— 分词器原理与对比实验
- [docs/02-transformer.md](docs/02-transformer.md) —— 模型结构与参数量逐项计算
- [docs/03-training.md](docs/03-training.md) —— 训练原理、读曲线与续训

## 2. 环境安装

环境：Windows + PowerShell 7，Python 3.12。项目自带 `.venv`，解释器统一用
`.venv\Scripts\python.exe`。补齐依赖（缺失时）：

```powershell
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
```

`requirements.txt` 包含：`torch>=2.4`、`numpy`、`PyYAML`、`huggingface_hub`、
`transformers`（只用来加载 Qwen 分词器）、`matplotlib`（画 loss 曲线）、`pytest`。

## 3. 数据准备

所有数据、分词器与产物都落在 `data/` 下，**不依赖 HF 缓存目录**。
下载端点强制走国内镜像 `HF_ENDPOINT=https://hf-mirror.com`（在 `src/data.py` 中设置）。

**第一步：下载 Qwen 分词器 + 中文 TinyStories**

```powershell
& ".venv\Scripts\python.exe" scripts/prepare_data_part1.py
```

- Qwen2.5 分词器文件 → `data/tokenizer/qwen2.5-0.5b/`（本地加载，并注册 `<image>` 特殊符号）
- `adam89/TinyStoriesChinese` 压缩包 → `data/raw/text/tinystories_zh/`（解压出 jsonl）

**第二步：编码成训练用的二进制**

```powershell
& ".venv\Scripts\python.exe" scripts/prepare_data_part2.py --max_docs 2000
```

- 逐行读取 jsonl，**优先取中文字段 `story_zh`**，用 Qwen 分词器编码
- 每篇末尾补 `<eos>`，全部拼成一个 `uint32` 数组（Qwen 词表 > 65535，必须用 4 字节）
- 按 1% 切分出验证集，写出 `data/processed/text/train.bin` 与 `val.bin`
- `--max_docs 2000` 只取前 2000 篇，便于快速跑通；去掉则用全部语料

## 4. 训练

```powershell
& ".venv\Scripts\python.exe" scripts/train_gpt.py
```

- 默认读取 `configs/gpt_tinystories.yaml`（Qwen 分词器 + TinyStories 中文）
- 任何字段都能在命令行覆盖，例如改学习率、改输出目录：

```powershell
& ".venv\Scripts\python.exe" scripts/train_gpt.py --set train.lr=3e-4 train.batch_size=4
```

- 启动时会打印分词器词表大小、模型总参数量与主干参数量
- 训练产物统一输出到 `out/gpt/`：
  - `latest.pt` —— 最新 checkpoint（模型 + 优化器 + 步数 + 配置）
  - `metrics.jsonl` —— 逐步记录的 train/val loss、lr
  - `loss.png` —— 由 `metrics.jsonl` 画出的 loss 曲线

**开启 `torch.compile`**（可选，默认关闭；Windows 上失败会自动回退 eager）：

```powershell
& ".venv\Scripts\python.exe" scripts/train_gpt.py --set train.compile=true
```

## 5. 续训

训练中断后，用 `--resume` 从 checkpoint 继续（会恢复已训练步数与优化器状态）：

```powershell
& ".venv\Scripts\python.exe" scripts/train_gpt.py --resume out/gpt/latest.pt
```

> 注意：`max_steps` 是**总步数**。若 checkpoint 步数已 ≥ `max_steps`，训练会立即结束；
> 想继续多训，需要同时把 `max_steps` 调大，例如 `--set train.max_steps=40000`。

## 6. 聊天 / 续写

```powershell
& ".venv\Scripts\python.exe" scripts/chat.py
```

加载 `out/gpt/latest.pt` 后进入交互式续写，输入 `/quit` 或 `/exit` 退出。可调参数：

```powershell
& ".venv\Scripts\python.exe" scripts/chat.py --temperature 0.8 --top_p 0.9 --max_new_tokens 128
```

> 基座模型是「续写」而非「问答」，更擅长接龙续写故事，而不是回答问题。
> 若要图文推理，请用后续 VLM 计划里的 `vlm_generate.py` / `webapp.py`。

## 7. 目录结构

```
P:\Demo\LLM\
├─ README.md                  # 本文件
├─ requirements.txt
├─ configs/
│  ├─ gpt_tinystories.yaml    # 默认：Qwen 分词器 + TinyStories 中文
│  └─ gpt_bpe.yaml            # 对比：自定义 16k BPE 分词器
├─ docs/
│  ├─ 01-tokenizer.md         # 分词原理
│  ├─ 02-transformer.md       # 模型结构
│  ├─ 03-training.md          # 训练原理
│  └─ superpowers/            # 设计与实现计划（spec / plan）
├─ src/
│  ├─ config.py               # YAML ↔ dataclass 配置，支持命令行覆盖
│  ├─ tokenizer.py            # 手写 BPE + 字符级 + Qwen 适配
│  ├─ data.py                 # 下载、解析、预处理、随机批采样
│  ├─ attention.py            # RoPE + 因果多头注意力 + GQA + KV 缓存
│  ├─ model.py                # RMSNorm / SwiGLU / Block / GPT
│  ├─ trainer.py              # 训练循环（AMP / 累积 / 裁剪 / 续训 / 曲线）
│  ├─ generate.py             # KV-cache 自回归采样
│  └─ utils.py                # 种子、LR 调度、日志、loss 曲线
├─ scripts/
│  ├─ prepare_data_part1.py   # 下载分词器与语料
│  ├─ prepare_data_part2.py   # 编码成 train.bin / val.bin（--tokenizer_kind qwen|bpe|char）
│  ├─ train_tokenizer.py      # 训练并保存手写 BPE 词表
│  ├─ train_gpt.py            # 训练入口
│  └─ chat.py                 # 命令行聊天
├─ tests/                     # pytest 单元测试
├─ data/                      # 所有数据（.gitignore 忽略）
│  ├─ tokenizer/qwen2.5-0.5b/ # Qwen 分词器（项目内本地）
│  ├─ raw/text/tinystories_zh/# 原始 jsonl
│  └─ processed/text/         # train.bin / val.bin
└─ out/                       # checkpoint / loss.png / metrics.jsonl（忽略）
```

## 8. 学习路线（对应里程碑）

| 里程碑 | 内容 | 产出 | 状态 |
|---|---|---|---|
| M1 分词器 | 手写字节级 BPE + 字符级 + Qwen 适配 | `src/tokenizer.py` | ✅ |
| M2 数据管道 | hf-mirror 下载 + 预处理成 memmap | `scripts/prepare_data_*.py`、`data/*.bin` | ✅ |
| M3 模型 | RoPE / Attention / Block / GPT | `src/attention.py`、`src/model.py` | ✅ |
| M4 训练 | Trainer、配置、续训、loss 曲线 | `scripts/train_gpt.py`、`out/gpt/*` | ✅ |
| M5 生成 | KV-cache 采样、CLI 聊天 | `src/generate.py`、`scripts/chat.py` | ✅ |
| M6 视觉 | 手写 ViT + 投影层 | `src/vision.py` | ⏳ 计划 |
| M7 多模态训练 | 合成图文 → 真实图文 → 简单 VQA | `scripts/train_vlm.py` | ⏳ 计划 |
| M8 交互与文档 | Gradio 网页 + 原理文档 | `scripts/webapp.py`、`docs/04-vlm.md` | ⏳ 计划 |

## 9. 运行测试

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ -v
```

测试覆盖：配置加载/覆盖、分词往返（`decode(encode(x)) == x`）、因果 mask 正确性、
GQA 与 KV 缓存形状、模型前向与权重共享、小批过拟合（证明模型能学）、训练器续训、
编译失败回退 eager。

## 10. 常见问题（FAQ）

**Q1：训练显存不足（CUDA out of memory）怎么办？**
减小 `batch_size`、增大 `grad_accum`，两者乘积（等效 batch）尽量不变，等于「用时间换显存」：

```powershell
& ".venv\Scripts\python.exe" scripts/train_gpt.py --set train.batch_size=4 train.grad_accum=16
```

也可减小 `ctx_len`，或把 `train.dtype` 设为 `fp32`（更慢更占显存，一般不必）。

**Q2：`torch.compile` 报错 / 失败，正常吗？**
正常。`torch.compile` 是可选加速项，Windows 上通常需要 triton，安装不全就会失败。
代码会用一个小 dummy batch 提前触发编译，失败时**自动回退 eager 并打印原因**，不影响训练。
默认配置 `compile: false`。

**Q3：为什么预处理用 `uint32` 而不是 `uint16`？**
Qwen 词表有 151666 个 token，超过 `uint16` 上限 65535，必须用 4 字节。手写 16k BPE 可用
`uint16`，但为了统一，本项目一律 `uint32`。

**Q4：控制台打印中文报 `UnicodeEncodeError`？**
`scripts/train_gpt.py`、`scripts/train_tokenizer.py`、`scripts/prepare_data_part1.py`
与 `scripts/prepare_data_part2.py` 都已强制 `sys.stdout.reconfigure(encoding="utf-8")`。
`scripts/chat.py` 未加此保护，若报编码错误，先执行 `$env:PYTHONUTF8=1`（或设置同名系统环境变量）再运行。

**Q5：下载超时 / 连接失败？**
项目统一走 `HF_ENDPOINT=https://hf-mirror.com`。如果仍失败，检查网络或手动确认该镜像可用。

**Q6：`data/` 和 `out/` 会进 git 吗？**
不会，已在 `.gitignore` 忽略。分词器、语料、checkpoint、曲线都不入库。

**Q7：怎么换用手写 BPE 分词器做对比？**
`scripts/prepare_data_part2.py` 支持 `--tokenizer_kind {qwen,bpe,char}`（默认 `qwen`）。
改用手写 BPE 只需三步：

```powershell
# 1) 用同一批 jsonl 语料训练并保存一份 BPE 词表
& ".venv\Scripts\python.exe" scripts/train_tokenizer.py --max_bytes 3000000

# 2) 用该 BPE 词表重新编码一份 train.bin / val.bin（注意输出名带 bpe_，避免覆盖 Qwen 数据）
& ".venv\Scripts\python.exe" scripts/prepare_data_part2.py --tokenizer_kind bpe `
  --tokenizer_dir data/tokenizer/bpe_16k `
  --out data/processed/text/bpe_train.bin --val data/processed/text/bpe_val.bin --max_docs 2000

# 3) 用 BPE 配置训练（gpt_bpe.yaml 已指向上面两份 bpe_*.bin）
& ".venv\Scripts\python.exe" scripts/train_gpt.py --config configs/gpt_bpe.yaml
```

> 说明：`train_tokenizer.py` 默认目标词表 16384；若语料子集较小，`min_frequency=2`
> 会在高频对被合并完时提前停止，实际词表可能小于目标（`train_gpt.py` 会用分词器
> 真实 `vocab_size` 覆盖配置）。细节见 [docs/01-tokenizer.md](docs/01-tokenizer.md)。
