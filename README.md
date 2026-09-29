# mini-llm-lab

从零手写一个**中文 GPT** 的教学项目：不依赖任何 HuggingFace 高层模型封装，
只用 PyTorch 原生算子实现「分词 → 数据 → 模型 → 训练 → 生成/聊天」的完整闭环。

> 主线叙事：**先教说话**（预训练文本 LLM）→ **装眼睛**（手写 ViT）→ **学会对话**（SFT 指令微调 + 聊天）。
> 文本闭环（分词/预训练/生成）与多模态（VLM）均已跑通；第二阶段的实现计划见
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
- [docs/04-vlm.md](docs/04-vlm.md) —— 手写 ViT、多模态融合与图文/VQA
- [docs/05-sft.md](docs/05-sft.md) —— 指令微调（SFT）、对话模板与聊天/评测
- [docs/06-architecture-compare.md](docs/06-architecture-compare.md) —— 与 Qwen3 / DeepSeek V3→V4.1 的架构对比
- [docs/07-visualization.md](docs/07-visualization.md) —— 3D 权重可视化（实时训练/推理）
- [docs/models/README.md](docs/models/README.md) —— 17 个真实模型的逐层架构文档（GLM / Qwen / MiMo / Kimi / DeepSeek）+ [整体对比](docs/models/compare.html)

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
HF 访问统一由 `src/hfenv.py` 设置：本机默认走代理 `http://127.0.0.1:7890`
直连官方 `huggingface.co`，无代理时可用国内镜像 `--endpoint https://hf-mirror.com`
（旧的 TinyStories 准备脚本默认镜像端点）。

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

`scripts/chat.py` 有两种模式：

```powershell
# 对话模式（默认）：加载 out/sft/latest.pt，多轮问答，/reset 清空，/quit 退出
& ".venv\Scripts\python.exe" scripts/chat.py --mode chat

# 基座续写模式：加载 out/gpt/latest.pt，输入 /quit 或 /exit 退出
& ".venv\Scripts\python.exe" scripts/chat.py --mode base
```

可调参数（两模式通用）：

```powershell
& ".venv\Scripts\python.exe" scripts/chat.py --mode chat --temperature 0.7 --top_p 0.9 --max_new_tokens 256
```

> **基座模型是「续写」而非「问答」**，更擅长接龙续写故事，而不是回答问题——这一
> 提醒仍然适用于 `--mode base`。要问答请用 `--mode chat`（需先完成第 12 节的 SFT）。
> 若要图文推理，请用 `vlm_generate.py` / `webapp.py`。

## 7. 目录结构

```
P:\Demo\LLM\
├─ README.md                  # 本文件
├─ requirements.txt
├─ configs/
│  ├─ gpt_tinystories.yaml    # 默认：Qwen 分词器 + TinyStories 中文
│  ├─ gpt_bpe.yaml            # 对比：自定义 16k BPE 分词器
│  ├─ gpt_fineweb.yaml        # 预训练：fineweb-2 中文，ctx 2048
│  └─ sft_zh.yaml             # SFT：对话数据 + 对话模板
├─ docs/
│  ├─ 01-tokenizer.md         # 分词原理
│  ├─ 02-transformer.md       # 模型结构
│  ├─ 03-training.md          # 训练原理
│  ├─ 04-vlm.md               # 视觉语言模型原理
│  ├─ 05-sft.md               # 指令微调 / 对话原理
│  ├─ 06-architecture-compare.md  # 与 Qwen3 / DeepSeek V3→V4.1 对比
│  ├─ 07-visualization.md     # 3D 权重可视化说明
│  ├─ superpowers/            # 设计与实现计划（spec / plan）
│  └─ models/                 # 12 个真实模型的逐层架构文档（由架构 JSON 生成）
├─ src/
│  ├─ config.py               # YAML ↔ dataclass 配置，支持命令行覆盖
│  ├─ hfenv.py                # 统一 HF 代理 / 镜像端点设置
│  ├─ tokenizer.py            # 手写 BPE + 字符级 + Qwen 适配
│  ├─ data.py                 # 下载、解析、预处理、随机批采样
│  ├─ attention.py            # RoPE + 因果多头注意力 + GQA + KV 缓存
│  ├─ model.py                # RMSNorm / SwiGLU / Block / GPT
│  ├─ trainer.py              # Trainer / VLMTrainer / SFTTrainer（AMP / 累积 / 续训 / 曲线）
│  ├─ generate.py             # KV-cache 自回归采样（stop_ids / add_bos）
│  ├─ chat_format.py          # 对话模板、assistant-only 掩码、停止符
│  ├─ sft_data.py             # SFT 归一化 / 多轮合成 / npz Dataset
│  ├─ chat_eval.py            # 评测题库与通过判定（纯逻辑）
│  ├─ vision.py / vlm.py      # 手写 ViT + 多模态融合
│  ├─ vlm_generate.py         # 图片描述 / VQA 推理
│  └─ utils.py                # 种子、LR 调度、日志、loss 曲线
├─ scripts/
│  ├─ prepare_data_part1.py   # 下载分词器与语料
│  ├─ prepare_data_part2.py   # 编码成 train.bin / val.bin（--tokenizer_kind qwen|bpe|char）
│  ├─ prepare_pretrain_data.py # fineweb-2 中文 → fineweb_cmn.*.bin
│  ├─ prepare_sft_data.py     # alpaca-zh + firefly → sft/*.npz
│  ├─ train_tokenizer.py      # 训练并保存手写 BPE 词表
│  ├─ train_gpt.py            # 预训练入口
│  ├─ train_sft.py            # SFT 训练入口
│  ├─ train_vlm.py            # VLM 训练入口
│  ├─ chat.py                 # 命令行聊天（--mode chat|base）
│  ├─ eval_chat.py            # 自动对话评测 → out/sft/eval.md
│  ├─ gen_model_docs.py       # 由 viz/architectures/*.json 生成 docs/models/**
│  └─ webapp.py               # Gradio 网页（对话 + 图文）
├─ viz/                       # 3D 权重可视化 + 架构浏览器（Python 后端 + Three.js 前端）
│  ├─ server.py               # FastAPI + WebSocket 服务
│  ├─ topology.py             # 拓扑 / 权重统计 / 神经元点云 / 激活 hook
│  ├─ runtime.py              # 小字符级 GPT、实时训练、逐 token 推理
│  ├─ arch.py                 # 架构 JSON 加载 / 逐层展开
│  ├─ architectures/*.json    # 12 个真实模型的架构规格（文档与网页同源）
│  └─ static/                 # 前端页面 + 本地 vendored Three.js / KaTeX
├─ reference/                 # 仅本地：Qwen3 / DeepSeek 建模代码（.gitignore 忽略）
├─ tests/                     # pytest 单元测试
├─ data/                      # 所有数据（.gitignore 忽略）
│  ├─ tokenizer/qwen2.5-0.5b/ # Qwen 分词器（项目内本地）
│  ├─ raw/text/tinystories_zh/# 原始 jsonl
│  ├─ raw/text/fineweb_cmn/   # fineweb-2 中文缓存
│  ├─ raw/sft/                # alpaca-zh / firefly 原始文件
│  ├─ processed/text/         # *.bin 预训练二进制
│  └─ processed/sft/          # train.npz / val.npz
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
| M6 视觉 | 手写 ViT + 投影层 | `src/vision.py`、`src/vlm.py` | ✅ |
| M7 多模态训练 | 合成图文 → 真实图文 → 简单 VQA | `scripts/train_vlm.py` | ✅ |
| M8 交互与文档 | Gradio 网页 + 原理文档 | `scripts/webapp.py`、`docs/04-vlm.md` | ✅ |
| M9 预训练数据 | fineweb-2 `cmn_Hani` 流式采样 + HF 代理 | `scripts/prepare_pretrain_data.py`、`src/hfenv.py` | ✅ |
| M10 预训练（ctx 2048） | 大词表中文基座，显存调参 | `configs/gpt_fineweb.yaml`、`out/gpt_pretrain/*` | ✅ |
| M11 SFT | 对话模板 + assistant-only 掩码 + 多轮合成 | `src/chat_format.py`、`src/sft_data.py`、`scripts/train_sft.py` | ✅ |
| M12 对话与评测 | 多轮聊天、自动评测、网页 | `scripts/chat.py`、`scripts/eval_chat.py`、`docs/05-sft.md` | ✅ |
| M13 架构对比 + 3D 可视化 | 克隆 Qwen3/DeepSeek 源码对比；实时看权重变化 | `reference/`、`docs/06-architecture-compare.md`、`viz/`、`docs/07-visualization.md` | ✅ |
| M14 多模型逐层架构库 | GLM/Qwen/MiMo/Kimi/DeepSeek 17 个版本规格 + 文档 + 架构浏览器 + 整体对比 | `viz/architectures/*.json`、`scripts/gen_model_docs.py`、`scripts/gen_compare_html.py`、`docs/models/**` | ✅ |

## 9. 运行测试

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ -v
```

测试覆盖：配置加载/覆盖、分词往返（`decode(encode(x)) == x`）、因果 mask 正确性、
GQA 与 KV 缓存形状、模型前向与权重共享、小批过拟合（证明模型能学）、训练器续训、
编译失败回退 eager、合成图文数据与 collate、ViT 形状与双向注意力、VLM 特征替换与权重迁移、
图文/问答推理；以及 SFT 相关：HF 代理/镜像设置（`src/hfenv.py`）、对话模板与
assistant-only 掩码/截断（`src/chat_format.py`）、SFT 数据归一化与多轮合成、
SFTTrainer 只监督助手、fineweb-2 流式读取、`generate` 的 `stop_ids` 停止、
对话评测通过判定（`src/chat_eval.py`）；以及可视化：拓扑构建、权重统计、
更新幅度（delta）、神经元点云 PCA、字符分词器、激活 hook 与逐 token 推理（`tests/test_viz.py`）、
架构 JSON 的良构性与逐层展开（`tests/test_architectures.py`）。

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
`scripts/` 下的训练/准备/聊天脚本都已强制
`sys.stdout.reconfigure(encoding="utf-8")`。若仍报错，先执行 `$env:PYTHONUTF8=1`
（或设置同名系统环境变量）再运行。

**Q5：下载超时 / 连接失败？**
默认走本机代理 `http://127.0.0.1:7890` 直连官方 HF；无代理时改用镜像
`--endpoint https://hf-mirror.com`。两条路径都由 `src/hfenv.py` 统一设置。

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

## 11. 视觉版（VLM）使用

原理见 [docs/04-vlm.md](docs/04-vlm.md)：手写 ViT 把图片编码成 64 个 patch 向量，
经投影层对齐后用视觉特征替换 `<image>` 占位符的 embedding（LLaVA 式），
LLM 权重从文本阶段的 checkpoint 迁移（"先教说话，再装眼睛"）。

```powershell
# 1) 合成几何图文上快速训练（分钟级，先验证链路；会自动复用 out/gpt/latest.pt 的 LLM 权重）
& ".venv\Scripts\python.exe" scripts/train_vlm.py --set train.max_steps=600 train.batch_size=4

# 2) 训练 VQA 版本（问题 -> 答案）
& ".venv\Scripts\python.exe" scripts/train_vlm.py --use_qa `
  --set train.out_dir=out/vlm_vqa train.max_steps=600 train.batch_size=4

# 3) 真实中文儿童图文（需先成功下载数据）
& ".venv\Scripts\python.exe" scripts/prepare_image_data.py --max_items 3000
& ".venv\Scripts\python.exe" scripts/train_vlm.py --data_kind children

# 4) 网页交互：文本聊天 + 图片描述/VQA
& ".venv\Scripts\python.exe" scripts/webapp.py
```

> VLM 默认配置 `configs/vlm_children.yaml`：`img_size=128`、`patch_size=16` → 每个样本
> 64 个 `<image>` token；`d_vision=384, depth=6`；`data_kind: synthetic`。
> 生成效果差时，优先检查标签是否相对输入**右移一位**（见 docs/04-vlm.md 第 5 节）。

## 12. 对话（SFT）使用

原理见 [docs/05-sft.md](docs/05-sft.md)：预训练基座只会"续写"，SFT 用
"用户问 / 助手答"的对话数据教它**只对助手的话算损失**（其余 token 标签为 `-100`），
复用 Qwen 的 `<|im_start|>/<|im_end|>` 做对话分隔（不扩词表）。

**环境（HF 访问）**：默认直连官方并走本机代理 `http://127.0.0.1:7890`
（`configs/*.yaml` 的 `data.proxy`，命令行 `--proxy`）；无代理可用镜像
`--endpoint https://hf-mirror.com`。

```powershell
# 1) 预训练数据：fineweb-2 中文（默认约 1GB 文本）
& ".venv\Scripts\python.exe" scripts/prepare_pretrain_data.py

# 2) 预训练基座（ctx 2048，约 192M 参数）
#    本机 16GB 显存务必用配置默认的 batch_size=1, grad_accum=64（见 docs/05-sft.md 第 4 节）
& ".venv\Scripts\python.exe" scripts/train_gpt.py --config configs/gpt_fineweb.yaml
#    续训：
& ".venv\Scripts\python.exe" scripts/train_gpt.py --config configs/gpt_fineweb.yaml `
  --resume out/gpt_pretrain/latest.pt

# 3) SFT 数据：alpaca-zh + firefly（默认 40000 条），合成多轮对话
& ".venv\Scripts\python.exe" scripts/prepare_sft_data.py

# 4) SFT 训练（默认从 out/gpt_pretrain/latest.pt 初始化；--init 可换权重）
& ".venv\Scripts\python.exe" scripts/train_sft.py
#    续训：& ".venv\Scripts\python.exe" scripts/train_sft.py --resume out/sft/latest.pt

# 5) 命令行对话（/reset 清空历史，/quit 退出）
& ".venv\Scripts\python.exe" scripts/chat.py --mode chat

# 6) 自动评测（18 条单轮 + 2 轮多轮召回，写入 out/sft/eval.md）
& ".venv\Scripts\python.exe" scripts/eval_chat.py

# 7) 网页（文本对话 + 图片描述/VQA；--check 只做冒烟构建）
& ".venv\Scripts\python.exe" scripts/webapp.py
& ".venv\Scripts\python.exe" scripts/webapp.py --check
```

> **能力边界**：这个 SFT 模型总计约 192M 参数，只在一个有界的中文对话样本上微调，
> 是"**会简单聊天**"的教学模型，**不是可靠的事实助手**。`eval_chat.py` 只检查
> 非空 / 不重复 / 能停止并记录通过数，**不设阈值、不会失败**；多轮召回是观察项。

## 13. 架构对比与 3D 可视化

```powershell
# 3D 可视化（实时训练 + 实时推理），打开 http://127.0.0.1:7861
& ".venv\Scripts\python.exe" viz/server.py
# 冒烟构建（不启动服务）
& ".venv\Scripts\python.exe" viz/server.py --check
```

- 页面有 **分层网络图**（球=层，颜色/大小=权重，连线粗细=权重强度，点击展开层内子模块）、
  **神经元点云**（点=采样神经元，连线粗细=|权重|，暖色=正/冷色=负）和
  **架构浏览器**（12 个真实模型的逐层 3D 展示，点层看公式与源码）三个视图。
- 点「开始训练」实时看权重更新；点「推理」逐 token 看激活与注意力变化。
- 与 Qwen3 / DeepSeek V3→V4.1 的逐项架构对比见
  [docs/06-architecture-compare.md](docs/06-architecture-compare.md)；用法与数据协议见
  [docs/07-visualization.md](docs/07-visualization.md)。
- **17 个真实模型的逐层架构文档**（GLM-4/4.5/5/5.3/5.3-Flash、Qwen3/3.5/3.8/3-Next、
  MiMo/MiMo-VL、Kimi K2/K2.5/K3、DeepSeek V3/V3.2/V4.1）见
  [docs/models/README.md](docs/models/README.md)；**横向对比**见
  [docs/models/compare.html](docs/models/compare.html)（或在可视化服务打开 `/compare`）。文档与网页都由
  `viz/architectures/*.json` 生成，新增模型只需加一份 JSON：

```powershell
& ".venv\Scripts\python.exe" scripts/gen_model_docs.py            # 生成全部文档 + 索引
& ".venv\Scripts\python.exe" scripts/gen_model_docs.py kimi-k3     # 只生成某个
& ".venv\Scripts\python.exe" scripts/gen_compare_html.py           # 生成整体对比 HTML
```
