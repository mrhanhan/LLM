# mini-llm-lab 设计文档

- 日期：2026-09-28
- 状态：待评审
- 类型：架构级（全新项目）

## 1. 背景与目标

用户希望**从零手写**理解大语言模型，并在同一套代码上扩展出一个**支持视觉的版本**。
环境已就绪：RTX 5080 16GB、CUDA 13、项目内 `.venv`（Python 3.12.10），已安装
`torch 2.14.0+cu130`、`transformers 5.17.0`、`accelerate`、`pillow`、`tokenizers 0.23.2`。
训练数据从 **hf-mirror**（`https://hf-mirror.com`）下载。

### 目标
1. **教学优先**：每个核心模块（分词、注意力、Transformer、训练循环、视觉编码、多模态融合）
   都手写实现，不依赖 HF 的高层模型封装。
2. **可运行**：文本版能训出通顺中文；视觉版能对图片输出中文描述，并回答简单 VQA。
3. **可复现、可迭代**：配置文件驱动、支持断点续训、loss 曲线可视化。
4. **成体系的学习产出**：`docs/` 下有原理讲解，代码带详细中文注释。

### 非目标（YAGNI）
- 不追求 SOTA 效果、不训练大模型、不做分布式/多卡。
- 不做 RLHF、不做 RAG、不做长上下文（默认 1024）。
- 不实现 FlashAttention 等底层算子优化（用 PyTorch 原生）。
- 视觉版不做复杂对话/多图/视频。

## 2. 学习路线（里程碑叙事线）

> **先教它说话 → 再给它装眼睛 → 让它开口描述世界**

| 里程碑 | 内容 | 产出 |
|---|---|---|
| M1 分词器 | 手写 BPE + Qwen tokenizer 适配 | `train_tokenizer.py`、词表文件 |
| M2 数据管道 | hf-mirror 下载 + 预处理成 memmap | `prepare_data.py`、`data/*.bin` |
| M3 模型 | Attention / Block / GPT | `model.py` + 单测 |
| M4 训练 | Trainer、配置、续训、曲线 | `train_gpt.py`、`loss.png`、`ckpt` |
| M5 生成 | KV-cache 采样、CLI 聊天 | `generate.py`、`chat.py` |
| M6 视觉 | 手写 ViT + 投影层 + 多模态模型 | `vision.py`、`vlm.py` |
| M7 多模态训练 | 合成图文 → 真实中文图文 → 简单 VQA | `train_vlm.py` |
| M8 交互与文档 | Gradio 网页 + 原理文档 | `webapp.py`、`docs/*.md` |

每个里程碑结束都应有**可独立运行的脚本**与**可观测的结果**（loss、样例输出、图片描述）。

## 3. 目录结构

```
P:\Demo\LLM\
├─ README.md                  # 总览 + 学习路线 + 怎么跑每一步
├─ requirements.txt
├─ configs/
│  ├─ gpt_tinystories.yaml    # 默认：Qwen 分词器 + TinyStories 中文
│  ├─ gpt_bpe.yaml            # 对比：自定义 BPE 分词器
│  └─ vlm_children.yaml
├─ docs/
│  ├─ 01-tokenizer.md
│  ├─ 02-transformer.md
│  ├─ 03-training.md
│  ├─ 04-vlm.md
│  └─ superpowers/specs/      # 本设计文档
├─ src/
│  ├─ config.py               # YAML + dataclass 配置
│  ├─ tokenizer.py            # 手写 BPE + 字符级 + Qwen 适配
│  ├─ data.py                 # 下载、预处理、Dataset
│  ├─ attention.py            # MHA + RoPE + 因果 mask + GQA
│  ├─ model.py                # RMSNorm / SwiGLU / Block / GPT
│  ├─ vision.py               # 手写 ViT
│  ├─ vlm.py                  # 投影层 + 多模态融合
│  ├─ trainer.py              # 训练循环
│  ├─ generate.py             # 采样生成
│  ├─ vlm_generate.py         # 图文推理
│  └─ utils.py                # 日志/曲线/checkpoint/种子/显存
├─ tests/
│  ├─ test_tokenizer.py
│  ├─ test_attention.py
│  ├─ test_model.py
│  └─ test_overfit.py         # 过拟合小批（模型能学信号的证据）
├─ scripts/
│  ├─ train_tokenizer.py
│  ├─ prepare_data.py
│  ├─ train_gpt.py
│  ├─ train_vlm.py
│  ├─ chat.py                 # CLI
│  └─ webapp.py               # Gradio
├─ data/                      # 所有数据/分词器，按目录分类（见 4.3）
│  ├─ tokenizer/
│  │  ├─ qwen2.5-0.5b/        # Qwen 分词器文件（落盘到项目内）
│  │  └─ bpe_16k/             # 手写 BPE 产物
│  ├─ raw/
│  │  ├─ text/                # 原始文本数据
│  │  └─ image/               # 原始图文数据
│  └─ processed/
│     ├─ text/                # train.bin / val.bin
│     └─ image/               # 图文索引
└─ out/                       # checkpoint / loss.png / metrics.jsonl（.gitignore）
```

### 3.1 数据目录规范（强制）
- **所有**下载数据与分词器产物一律放在 `data/` 下，按 `tokenizer / raw / processed`
  与 `text / image` 分层，不在 HF 缓存目录里"隐身"使用。
- Qwen 分词器文件下载到 `data/tokenizer/qwen2.5-0.5b/`，加载时**从该本地路径**读取，
  不依赖 `~/.cache/huggingface`。
- 每个子目录附 `README.md` 说明来源 URL、文件清单、生成方式。

## 4. 组件设计

### 4.1 配置系统 `src/config.py`
- YAML 文件 + dataclass（`ModelConfig` / `TrainConfig` / `DataConfig`）；VLM 配置段直接以原始 dict 传入。
- 支持命令行覆盖（如 `--train.lr 3e-4 --model.n_layer 10`）。
- 配置随 checkpoint 一起保存，续训时校验结构一致性。

### 4.2 分词器 `src/tokenizer.py`
统一接口：`encode(str)->list[int]`、`decode(list[int])->str`、`vocab_size`、`special_tokens`。

1. **`BPETokenizer`（手写，教学重点）**
   - 字节级 BPE：先正则预切分，再把词拆成字节/字符，迭代统计相邻对 → 合并最高频对。
   - 训练在语料**子集**上进行（默认取前 N 万篇/约几十 MB），显示进度，可控耗时。
   - 默认词表 16384；产物存 `data/tokenizer/bpe_16k/`（`vocab.json` + `merges.txt`）；
     支持增量 `encode` 时按 merges 排序合并。
   - 特殊符号：`<pad> <bos> <eos> <unk>` 与视觉占位 `<image>`。
2. **`CharTokenizer`**：字符级，最简单，便于对照理解。
3. **`QwenTokenizer`**：把 `Qwen/Qwen2.5-0.5B` 的 tokenizer 文件（`tokenizer.json`、
   `tokenizer_config.json`、`vocab.json`、`merges.txt` 等）**下载到项目内**
   `data/tokenizer/qwen2.5-0.5b/`，并用 `AutoTokenizer.from_pretrained(该本地路径)` 加载，
   不读 HF 缓存。
   - **默认使用**（结论：Qwen 分词对中文/代码效果最好）。
   - 注意代价：Qwen 词表 ~151,936 → token embedding 约 **116M** 参数（权重共享，无独立 lm_head）。
     因此启用 Qwen 时模型**总体参数约 200M**，Transformer 主干仍约 80M。16GB 显存完全可训练。

**决策记录**：用户要求「既用 Qwen 分词器，也手写一个 BPE」。故两者都实现，靠配置切换；
默认配置用 Qwen，`gpt_bpe.yaml` 用作教学对比。

### 4.3 数据管道 `src/data.py`
- 下载逻辑：设置 `HF_ENDPOINT=https://hf-mirror.com`，用 `huggingface_hub` 拉取原始文件，
  **全部落到 `data/raw/` 对应子目录**。不使用 `datasets` 库（避免黑盒与额外依赖），自己解析 jsonl/tar。
- **文本**：`adam89/TinyStoriesChinese`（`TinyStories_all_data_zh.tar.gz` → 解压 jsonl）→
  存 `data/raw/text/tinystories_zh/`。流程：下载 → 解析 → 分词 → 拼接为 `uint32`（Qwen 词表 >65535）
  → 存 `data/processed/text/train.bin`、`val.bin`（memmap，训练时按固定长度随机切块）。
  可选补充中文维基子集到 `data/raw/text/wikipedia_zh/`。
- **图文**：
  - 合成：程序实时生成几何图形（圆/方/三角 + 颜色 + 位置）+ 对应中文描述与 VQA 问答，**无需下载**；
    可选缓存在 `data/processed/image/synthetic/`。
  - 真实：`svjack/Chinese_Children_Image_Captioning_Dataset_Split0`（png + 同名 txt 中文描述），
    下载到 `data/raw/image/children_caption/`，构建 `(image_path, caption)` 索引
    （`data/processed/image/`），图像统一 resize 到配置尺寸（默认 128×128）。

### 4.4 模型 `src/attention.py` + `src/model.py`
- `RoPE`：手写旋转位置编码。
- `CausalSelfAttention`：手写 Q/K/V/O 投影、多头、上三角 mask；支持 GQA（`n_kv_head`）。
- `RMSNorm`、`SwiGLU` MLP、`Block`（Pre-Norm + 残差）。
- `GPT`：Token Embedding + N×Block + 最终 RMSNorm + 输出投影（与 embedding **权重共享**）。
- **默认结构**：`d_model=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2048, ctx=1024`。
  主干 ≈ 80M 参数（不含词表 embedding）。所有字段可配置。
- 提供 `num_params()` 与结构打印，便于观察每个模块参数量。

### 4.5 视觉编码 `src/vision.py`
- 手写 ViT：`Conv2d(patch_size, stride=patch_size)` 做 patch embedding → 展平 → 可学习位置编码
  → N×ViTBlock（LayerNorm + GELU，双向注意力，无因果 mask；内部复用 4.4 的 `MultiHeadAttention`）→ LayerNorm。
- 默认 `img_size=128, patch_size=16 → 64 patches, d_vision=384, depth=6, heads=6`，约 19M。

### 4.6 多模态融合 `src/vlm.py`
- 结构：`VisionEncoder → MLP Projector(d_vision→d_model) → 拼接进 LLM 序列`。
- 序列模板：`<bos> <image>×N <中文文本> <eos>`；`<image>` 占位符对应的 embedding 用投影后的
  视觉特征**替换**（LLaVA 式），损失**只在文本 token 上计算**（图像位置 label 设为 -100）。
- **初始化策略**：LLM 权重从已训练的 GPT checkpoint 加载（迁移），ViT/投影层随机初始化。
- `<image>` 特殊 token 需加入 tokenizer，并相应扩展 embedding（Qwen 路径下 `resize_token_embeddings`）。

### 4.7 训练器 `src/trainer.py`
- bf16 autocast（主权重 fp32），AdamW，梯度累积 + 梯度裁剪(1.0)，warmup + 余弦退火，
  weight decay 0.1。
- 断点续训：checkpoint 保存 `model/optimizer/scheduler/step/rng/config`，`--resume` 恢复。
- 日志：控制台 + `out/metrics.jsonl`；`utils.plot_loss` 输出 `out/loss.png`（matplotlib）。
- 验证：周期性算 val loss + 采样生成，观察"从乱码到通顺"的过程。
- `torch.compile`：**可开关，默认关闭**；开启失败自动回退 eager 并打印原因（Windows 无 triton 时）。

### 4.8 生成与交互
- `generate.py`：KV-cache 自回归采样，支持 `temperature / top_k / top_p / repetition_penalty`。
- `vlm_generate.py`：图片 → 中文描述；支持 VQA 指令模板（如"图里有几个红色圆形？"）。
- `chat.py`：命令行多轮（基座模型为续写，若后续做指令微调再升级）。
- `webapp.py`：Gradio —— 文本聊天页 + 图片上传页。

## 5. 数据来源（hf-mirror）
| 用途 | 数据集/模型 | 拉取文件 | 落盘位置 |
|---|---|---|---|
| 中文文本 | `adam89/TinyStoriesChinese` | `TinyStories_all_data_zh.tar.gz` | `data/raw/text/tinystories_zh/` |
| 分词器 | `Qwen/Qwen2.5-0.5B` | tokenizer 相关文件 | `data/tokenizer/qwen2.5-0.5b/` |
| 真实图文 | `svjack/Chinese_Children_Image_Captioning_Dataset_Split0` | `*.png` + `*.txt` | `data/raw/image/children_caption/` |
| 图文（备选） | 程序合成 | 运行时生成 | `data/processed/image/synthetic/` |

## 6. 参数量与显存预算
| 组件 | 参数 |
|---|---|
| GPT 主干（d768×12） | ≈ 80M |
| + Qwen 词表 embedding（共享） | +117M |
| **文本模型合计** | **≈ 200M** |
| ViT（d384×6） | ≈ 19M |
| 投影层 | ≈ 0.9M |

训练 200M 模型：fp32 主权重 ~0.8GB + AdamW 状态 ~1.6GB + 激活（bs=8, ctx=1024）约数 GB，
16GB 显存内可跑；显存不足时用**梯度累积**降低 micro-batch。代码内置显存峰值打印。

## 7. 风险与对策
| 风险 | 对策 |
|---|---|
| Windows 上 `torch.compile` 失败 | 开关 + 自动回退 eager；不影响训练 |
| 手写 BPE 纯 Python 慢 | 仅在语料子集训练；加进度条；给出词表大小/耗时说明 |
| Qwen 词表使模型超 80M 目标 | 文档明确说明；提供 16k BPE 配置用于对比 |
| TinyStories 数据规模/清洗 | 先抽样跑通再全量；预留中文维基备选 |
| VLM 小数据易过拟合 | 合成数据先验证链路；真实数据加大增广/早停，只训必要层可作可选项 |
| 真实图文下载体积大 | snapshot 支持按需/断点；儿童图文数据集较小，优先使用 |

## 8. 测试与验收
**单元测试（tests/）**
- 分词器：`decode(encode(x)) == x`（字节级往返）。
- 注意力：因果 mask 正确性（篡改未来 token 不影响过去位置输出）。
- 模型：前向输出形状、`num_params()` 合理、embedding 权重共享生效。
- 过拟合：在极小批上训练若干步，loss 应显著下降到接近 0（证明模型能学）。

**端到端验收**
1. `train_gpt.py` 能跑通并产出下降的 `loss.png`；中断后 `--resume` 能继续。
2. 采样生成中文**可读、无明显乱码**（TinyStories 风格）。
3. `train_vlm.py` 在合成数据上能生成正确描述（颜色/形状/位置）；在真实儿童图文上能出通顺中文描述。
4. VQA 能回答合成数据的计数/颜色问题。
5. `webapp.py` 可启动，文本与图片两条链路可用。

## 9. 交付里程碑（实现顺序）
M1→M2→M3→M4→M5（文本闭环）→M6→M7（视觉闭环）→M8（交互+文档）。
每个里程碑完成后运行对应测试/脚本并展示证据，再进行下一个。
