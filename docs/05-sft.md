# 05 · 指令微调（SFT）与对话

> 对应代码：`src/hfenv.py`、`src/chat_format.py`、`src/sft_data.py`、`src/chat_eval.py`、
> `src/trainer.py: SFTTrainer`、`scripts/prepare_pretrain_data.py`、
> `scripts/prepare_sft_data.py`、`scripts/train_sft.py`、`scripts/chat.py`、
> `scripts/eval_chat.py`、`scripts/webapp.py`
> 配置：`configs/gpt_fineweb.yaml`（预训练）、`configs/sft_zh.yaml`（SFT）

这一章讲清楚：为什么光有预训练还不能对话、怎么用真实中文语料把模型喂到 "会接话"、
对话模板长什么样、损失只算在谁头上、以及怎么去聊天、评测和排错。

---

## 1. 为什么需要 SFT：基座只会"续写"

预训练的目标只有一个：**给定前文，预测下一个 token**。它训练出来的模型叫**基座
（base）**，学到的是"语言接着往下写"的统计规律。你输入"中国的首都是"，
它会续出"北京。"；但你输入"请介绍一下你自己。"这种**指令**，它不会"回答你"，
而是把你这句话当成一段待续写的文本，接着往下编——它没学过"被提问后该回答"这件事。

SFT（Supervised Fine-Tuning，有监督微调）就是在预训练之后，用 **"用户问 / 助手答"
成对的对话数据**再训练一轮，让模型学会三件事：

1. **识别对话结构**：知道 `<|im_start|>user` 后面是问题、`<|im_start|>assistant`
   后面该轮到自己回答；
2. **扮演助手角色**：按"乐于助人的中文助手"的口吻回答，而不是把指令当续写素材；
3. **知道何时停止**：回答结束输出 `<|im_end|>`，而不是无限复读。

所以整体流程是：

```text
预训练（fineweb-2 中文语料，学会说话）
        │  out/gpt_pretrain/latest.pt
        ▼
SFT（alpaca-zh + firefly 对话数据，学会对话）
        │  out/sft/latest.pt
        ▼
对话（chat.py --mode chat / webapp.py）
```

预训练负责"语言能力"，SFT 负责"对话格式"，两者是叠加关系，而不是替换关系。

---

## 2. 环境与网络

环境：Windows + PowerShell 7，解释器统一用 `.venv\Scripts\python.exe`。

HuggingFace 访问集中在 `src/hfenv.py` 里配置：

- **默认直连官方 `huggingface.co`，走本机代理 `http://127.0.0.1:7890`**
  （在 `configs/*.yaml` 的 `data.proxy` 里，命令行可 `--proxy` 覆盖）；
- 没有代理时，可用国内镜像回退：`--endpoint https://hf-mirror.com`。

> 脚本里 `setup_hf()` 的规则：传了 `proxy` 就走代理并在 `NO_PROXY` 里排除本地地址；
> `endpoint` 为空则**清除**镜像端点直连官方，非空则改打指定镜像。

---

## 3. 预训练数据：fineweb-2 中文

数据源是 `HuggingFaceFW/fineweb-2` 的 `cmn_Hani` 配置（简体/汉字中文）。它体量极大，
不能整份下载，项目**按 parquet 的 row-group 流式读取，用字节预算控制样本量**，
再缓存成 `jsonl.gz`：

```powershell
# 默认 --max_bytes 1000000000（约 1GB 文本）
& ".venv\Scripts\python.exe" scripts/prepare_pretrain_data.py
```

- 缓存目录：`data/raw/text/fineweb_cmn/`（`train.jsonl.gz` / `val.jsonl.gz`）；
- 输出（uint32 二进制）：`data/processed/text/fineweb_cmn.train.bin` /
  `fineweb_cmn.val.bin`；
- 验证集取自 fineweb-2 的 **`test` split**，不是从训练集里切出来的。

本项目实际产出的规模：

| 集合 | tokens | 大小 |
|---|---|---|
| train | 261,604,487 | 997.9 MiB |
| val | 1,296,386 | 4.9 MiB |

---

## 4. 预训练配置与那个"显存坑"

`configs/gpt_fineweb.yaml` 的关键参数：

- `ctx_len=2048`、`vocab_size=151666`（Qwen2.5 词表）；
- 主干：`d_model=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2048`；
- 参数量：**非词表部分约 75.5M，含词表约 192M**。

> **显存注意（本机实测）**：在 16GB 的 RTX 5080 上，计划最初写的
> `batch_size=8, grad_accum=8` **会 OOM**——单是 fp32 的 loss logits
> 就要约 **9.3GiB**。真正能跑的是 **`batch_size: 1, grad_accum: 64`**
> （峰值约 **11.4GB**，约 **8.7 s/step**）。把 `batch_size` 提到 ≥2 会把优化器/
> 中间张量挤到共享内存，速度反而慢约 3 倍。所以**别改大 micro-batch**，
> 要等效大 batch 就加 `grad_accum`。

训练与续训：

```powershell
# 从头训练
& ".venv\Scripts\python.exe" scripts/train_gpt.py --config configs/gpt_fineweb.yaml

# 断点续训（恢复步数与优化器状态）
& ".venv\Scripts\python.exe" scripts/train_gpt.py --config configs/gpt_fineweb.yaml `
  --resume out/gpt_pretrain/latest.pt
```

产物在 `out/gpt_pretrain/`：`latest.pt`（checkpoint）、`metrics.jsonl`（逐步 loss）、
`loss.png`（曲线），与 `docs/03-training.md` 里讲的一致。

---

## 5. 对话模板：复用 Qwen 的 `<|im_start|>` / `<|im_end|>`

Qwen2.5 分词器词表里本来就有 `<|im_start|>`（id **151644**）与 `<|im_end|>`
（id **151645**），所以 SFT **既不加词表也不改 embedding**，直接用它们当对话分隔符。
模板格式（`src/chat_format.py`）：

```text
<|im_start|>system
你是一个乐于助人的中文助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮你的吗？<|im_end|>
```

- 固定 system 提示词是 `你是一个乐于助人的中文助手。`，训练与推理**共用同一个**，
  避免分布漂移；
- `render_messages(..., add_generation_prompt=True)` 会在末尾补
  `<|im_start|>assistant\n`，让模型从这句话之后开始答。

### 只监督 assistant：`-100` 掩码

`build_sft_example` 给每个 token 生成 `labels`，但**只有 assistant 的内容和它这一轮
结尾的 `<|im_end|>` 参与 loss，其余位置全部设成 `-100`**（交叉熵会忽略 `-100`）：

```text
ids    : <|im_start|> system … <|im_start|> user … <|im_start|> assistant 回答<|im_end|> \n
labels : -100 -100 -100 … -100 -100 … -100 -100 -100 label label label …
                                             └── 只有这段被监督 ──┘
```

**为什么**：如果让模型也去"预测"用户的问题和 system 段，等于让它学会说用户的话，
既浪费容量又会让它抢话。只监督回答，才是"学会当助手"。

### 截断策略

单条样本超过 `max_len=2048` 时，`build_sft_example` 的规则是：

1. **始终保留 system 前缀**（训练/推理一致）；
2. 从**最早的轮次**开始丢弃，优先保留最近的对话；
3. 若单条消息本身还是超长，则保留它的**尾部**，确保结尾的助手回答及其
   `<|im_end|>` 被监督到。

### 生成端的两点约束（`scripts/chat.py`）

- 生成**不加 bos**（`generate(..., add_bos=False)`）：Qwen 的 bos 会映射成
  `<|endoftext|>`，而训练样本并不以它开头；
- 停止符是 `<|im_end|>`（`stop_ids()` 兜底带上 eos），命中即停，且停止符本身不进输出。

---

## 6. SFT 数据来源与"多轮合成"

SFT 数据来自两个公开中文数据集（都是**单轮**问答）：

| 数据源 | 说明 |
|---|---|
| `shibing624/alpaca-zh` | 48,818 条 `instruction/input/output` 单轮样本 |
| `YeungNLP/firefly-train-1.1M` | 取前 **40,000** 行（`input/target`），流式截取，不整下 1.17GB |

```powershell
# 默认 --firefly_items 40000
& ".venv\Scripts\python.exe" scripts/prepare_sft_data.py
```

**多轮合成（有意的数据增广）**：这两个来源都是单轮的，模型光看单轮学不会"记住上文"
的序列形态。于是 `prepare_sft_data.py` 以 `synth_prob=0.5` 的概率，把相邻的 1–3 条
单轮问答**拼成一段多轮对话**（见 `src/sft_data.py: synthesize_dialogue`）。它只是把
独立的问题连起来，**不改变任何单条样本的语义**，目的是让模型见过"user/assistant 交替"
的序列。

本项目实际产出的规模（`max_len=2048`）：**49,837 条 train / 1,017 条 val 对话**，
落盘为 `data/processed/sft/train.npz`（约 115MB）与 `val.npz`（约 2.7MB）。
npz 里是三个扁平数组：`ids`、`labels`（含 `-100` 掩码）、`offsets`（每条对话的边界）。

---

## 7. SFT 训练与 loss 曲线

`configs/sft_zh.yaml` 关键参数：`lr=1e-4`、`batch_size=8, grad_accum=4`、
`max_steps=1500`、`out_dir=out/sft`。

```powershell
# 从预训练 checkpoint 初始化（默认 out/gpt_pretrain/latest.pt）
& ".venv\Scripts\python.exe" scripts/train_sft.py

# 换个预训练权重
& ".venv\Scripts\python.exe" scripts/train_sft.py --init out/gpt_pretrain/latest.pt

# 从 SFT checkpoint 续训
& ".venv\Scripts\python.exe" scripts/train_sft.py --resume out/sft/latest.pt
```

**怎么读 loss 曲线**（产物 `out/sft/metrics.jsonl` / `loss.png`）：

- SFT 的 loss **只统计 assistant 段**，起点通常比预训练低（因为只在回答上学），
  随训练稳步下降；
- 训练 loss 持续下降而 val loss 早早走平甚至回升，说明开始**过拟合**这个小样本，
  可以少训几步，或把 `max_steps` 调小；
- 注意 loss 低 ≠ 回答对。SFT 学的是"对话的格式与口吻"，事实性不在此保证
  （见第 9 节）。

---

## 8. 聊天 / 评测 / 网页

### 命令行

```powershell
# 对话模式（默认）：加载 out/sft/latest.pt
& ".venv\Scripts\python.exe" scripts/chat.py --mode chat
```

- 输入 `/reset` 清空会话历史（清空后再问"我叫什么"，它自然答不上来）；
- 输入 `/quit`（或 `/exit`、空行）退出；
- 可调 `--temperature 0.7 --top_p 0.9 --max_new_tokens 256`。

```powershell
# 基座续写模式：加载 out/gpt/latest.pt，仍是"续写"而非"问答"
& ".venv\Scripts\python.exe" scripts/chat.py --mode base
```

### 自动评测

```powershell
& ".venv\Scripts\python.exe" scripts/eval_chat.py
```

- 用 **18 条固定单轮 prompt + 2 轮多轮召回检查** 跑一遍生成；
- 单轮检查三项：**非空 / 无明显重复 / 能正常停止**（`src/chat_eval.py`）；
- 多轮检查回答里是否复现了上文关键词（如"小明"）；
- 采样使用**固定随机种子**，保证可复现；
- 结果写入 `out/sft/eval.md`。

> 注意：`eval_chat.py` 目前只**打印并记录通过数**，**不设阈值、不会因此失败**。
> 其中多轮召回只是观察项（小模型常常记不住），不要把它当硬指标。

### 网页

```powershell
& ".venv\Scripts\python.exe" scripts/webapp.py            # 启动 Gradio
& ".venv\Scripts\python.exe" scripts/webapp.py --check    # 只构建模型与界面并退出（冒烟）
```

界面有两个标签页：**文本对话**（多轮 `ChatInterface`，有生成长度 / temperature /
top_p 滑块）与**图片描述 / VQA**（复用 VLM 那条链路）。对话页若没找到
`out/sft/latest.pt` 会提示先去训练。

---

## 9. 能力边界与常见问题（FAQ）

**先明确边界**：这个 SFT 模型**总共只有约 192M 参数**，且只在一个**有界的中文对话
样本**上微调过。它是一个"**会简单聊天**"的教学模型，**不是可靠的事实助手**——
它可能一本正经地答错事实题。请把它当作"对话格式跑通了"的证据，而不是生产级助手。

**Q1：回答胡言乱语 / 答非所问？**
几个常见原因，按概率排查：① 预训练本身步数太少、语言能力不足（先确认
`out/gpt_pretrain` 的 loss 已明显下降）；② SFT 数据是有限样本，问到了分布外的话题；
③ 生成温度过高，试 `--temperature 0.6`；④ 训练/推理的模板或 system 前缀不一致
（本项目用同一处 `chat_format`，正常不会漂移）。

**Q2：回答重复、复读停不下来？**
先看它有没有正常输出 `<|im_end|>`。若一直不停：确认用的是 `--mode chat`（会自动
停止并剥掉特殊 token），而不是 `--mode base`。若在 chat 模式下仍复读，多半是 SFT
训得过少、还没学会收尾，或温度/采样设置不当；减小 `--max_new_tokens` 也能限制影响。

**Q3：为什么上下文只有 2048？**
`ctx_len=2048` 是预训练就定下的位置编码长度。多轮对话里 `build_sft_example` 会在
超长时**从最早的轮次开始丢弃**，所以聊得太久，前面的内容会被"忘掉"。这是按设计的行为。

**Q4：下载卡住 / 连接失败？**
确认代理 `http://127.0.0.1:7890` 在运行；没有代理就换镜像
`--endpoint https://hf-mirror.com`。两条路径都由 `src/hfenv.py` 统一设置。

**Q5：预训练 `batch_size=8` 就 OOM？**
正常，别开那么大。用配置默认的 `batch_size: 1, grad_accum: 64`（见第 4 节）。
需要更大的等效 batch 时，只加 `grad_accum`，不要加 `batch_size`。

**Q6：`eval_chat.py` 通过数不高，是不是坏了？**
不一定。它只做**非空 / 不重复 / 能停止**的基础检查，多轮召回是观察项；模型小、
样本有限，能稳定"说人话、会停"就算这条教学链路成功。

---

## 10. 与代码的对应关系

| 概念 | 代码 |
|---|---|
| HF 代理 / 镜像配置 | `src/hfenv.py: setup_hf` |
| 对话模板渲染 | `src/chat_format.py: render_messages` |
| SFT 样本与 `-100` 掩码 / 截断 | `src/chat_format.py: build_sft_example` |
| 停止符 / 历史转换 | `src/chat_format.py: stop_ids / history_to_messages` |
| 数据归一化与多轮合成 | `src/sft_data.py: normalize_* / synthesize_dialogue` |
| npz Dataset 与 padding | `src/sft_data.py: SFTDataset / collate_sft` |
| 只监督助手的训练循环 | `src/trainer.py: SFTTrainer` |
| 预训练语料准备 | `scripts/prepare_pretrain_data.py` |
| SFT 数据准备 | `scripts/prepare_sft_data.py` |
| SFT 训练入口 | `scripts/train_sft.py` |
| 命令行对话 | `scripts/chat.py`（`--mode chat|base`） |
| 自动评测 | `scripts/eval_chat.py`、`src/chat_eval.py` |
| 网页对话 / 图文 | `scripts/webapp.py` |
