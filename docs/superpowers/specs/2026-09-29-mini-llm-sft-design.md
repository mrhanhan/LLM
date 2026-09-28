# mini-llm-lab SFT 设计文档（预训练语料扩充 + 指令微调 + 多轮对话）

- 日期：2026-09-29
- 状态：待评审
- 类型：架构级（在既有文本/VLM 闭环上新增「预训练数据 + SFT + 对话」子系统）
- 前置：`docs/superpowers/specs/2026-09-28-mini-llm-lab-design.md`（M1–M8 已完成）

## 1. 背景与目标

现状：M1–M8 的闭环代码已跑通，但对话质量不可用——基座模型只在
`adam89/TinyStoriesChinese` 的前 2000 篇（约 32 万 token）上训练了 40 步，
`out/gpt/latest.pt` 实质接近随机权重，无法回答问题。

目标（本阶段）：
1. 拉取**开源中文预训练语料**并真正预训练一个基座模型（`ctx_len=2048`）。
2. 拉取**开源中文指令（SFT）数据**，用**对话模板**做监督微调（只在助手回答上计损失）。
3. 让模型能完成**简单多轮对话**（自我介绍、常识问答、改写、追问），并可复现、可评测。
4. 保持教学属性：数据来源、模板、掩码、评测逻辑都手写、带中文注释、有文档。

### 非目标（YAGNI）
- 不追求 SOTA；不训练 >0.25B 模型；不做多卡/分布式。
- 不做 RLHF / DPO / RAG / 工具调用。
- 不做长上下文（本阶段 2048 封顶）；不改分词器词表（复用 Qwen 现有特殊 token）。
- 不覆盖 VLM 链路（VLM 配置独立，不受本阶段影响）。
- 不做 gated 数据集的授权流程（见 4.2）。

## 2. 环境与网络约束（已实测）

- 训练机：RTX 5080 16GB；P 盘剩余约 495GB；解释器 `.venv\Scripts\python.exe`。
- 网络：**本机 HTTP 代理 `http://127.0.0.1:7890`**（Clash，注册表 `ProxyServer` 记录，
  端口在监听）。经代理可**直连 `huggingface.co`**。
- 镜像 `hf-mirror.com` 的**递归列表/`/parquet` 端点对大仓库不可用**（超时/403），
  故本阶段默认走**代理直连 HF**；镜像作为可用回退（小仓库）。

### 2.1 实测结论（决定选型）

| 数据源 | 结论 | 证据 |
|---|---|---|
| `HuggingFaceFW/fineweb-2` | ✅ 可用（直连） | 列表 9597 文件/6.8s；中文配置 `cmn_Hani`（371 训练分片）；`data/cmn_Hani/train/000_00000.parquet` HEAD 200，**单片约 4.84GB**；`text` 列 |
| `BAAI/Infinity-Instruct` | ⛔ 不可用 | `gated=auto`，无 token → resolve 401 |
| `shibing624/alpaca-zh` | ✅ 可用 | `alpaca_gpt4_data_zh.json` 35MB，n=48818，字段 `instruction/input/output` |
| `YeungNLP/firefly-train-1.1M` | ✅ 可用 | `firefly-train-1.1M.jsonl` 1.17GB，字段 `kind/input/target`，**单轮任务集（非多轮）** |
| `opencsg/Chinese-FineWeb-Edu-V2` | ✅ 可用（备选） | 625×`data/NNNNN.parquet`，每片约 973MB，列 `text/score/__index__/source` |

其他已核实事实：
- Qwen 分词器特殊 token 已在词表内：`<|im_start|>`=151644、`<|im_end|>`=151645、
  `<|endoftext|>`(=eos)=151643；`encode` 时即使 `add_special_tokens=False` 也能切成单个 id。
  → **无需扩表、不改 embedding**。
- `src/model.py` 的 loss 用 `F.cross_entropy(..., ignore_index=-100)`（默认对非 -100 位置求均值），
  天然支持「只在助手 token 上计损失」。
- `HfFileSystem` 经代理可**按 row-group 远程读取** parquet（只下所需分块），
  无需整片下载：已成功远程读出 fineweb-2 test 片 schema（`cols=[text,id,dump,url,date,file_path,language,language_score,language_script,minhash_cluster_size,top_langs]`，33296 行/34 组）。

## 3. 目标与验收

**验收（可自动检查）**：`scripts/eval_chat.py` 跑固定 ~20 条留出问题（含 2 条多轮追问），
全部满足：① 非空且长度合理；② 无短串重复循环；③ 在 `<|im_end|>` 正常停止（非撞 `max_new_tokens`）；
④ 多轮追问能引用上文关键词。人工抽查输出记录进 `out/sft/eval.md`。

**性能目标**：预训练 ~2–2.5 亿 token（约 1h 内，可续训）；SFT 约 5–8 万段对话（10–20min）。

## 4. 数据设计

### 4.1 预训练语料（fineweb-2 `cmn_Hani`）

- 新增 `src/data.py: iter_fineweb_text(...)`：
  用 `HfFileSystem().open("datasets/HuggingFaceFW/fineweb-2/data/cmn_Hani/train/000_00000.parquet", "rb", block_size=4MB)`
  配合 `pyarrow.parquet.ParquetFile`，**逐 row-group** 读取 `text` 列，累计到 `max_bytes` 停止。
  - train：从 `train/000_00000.parquet` 起顺序读取；val：读 `test/000_00000.parquet`，天然与 train 不重叠。
  - 默认 `max_bytes ≈ 1.0GB`（UTF-8 中文约 3 字节/字 → 约 3.5 亿字 → Qwen 下约 2–2.5 亿 token）；
    「更大」只需调大该参数。
- **缓存**：抽取的文本先写 `data/raw/text/fineweb_cmn/{train,val}.jsonl.gz`；已存在则跳过网络读取。
- 编码复用现有 `build_text_bin` → `data/processed/text/fineweb_cmn.{train,val}.bin`（uint32）。
- 新增 `scripts/prepare_pretrain_data.py`（参数：`--max_bytes`、`--proxy`、`--endpoint`、`--out/--val`）。

### 4.2 SFT 数据（alpaca-zh + firefly）

- 下载（经代理 / `HfFileSystem`）：
  - `shibing624/alpaca-zh` → `data/raw/sft/alpaca_gpt4_data_zh.json`（35MB，整文件）。
  - `YeungNLP/firefly-train-1.1M` → **流式取前 N 条**（默认 40000）写入
    `data/raw/sft/firefly_subset.jsonl`，不下载整份 1.17GB。
- **归一化**为统一中间格式 `{"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]}`：
  - alpaca-zh：`user = instruction (+ "\n" + input 若非空)`，`assistant = output`。
  - firefly：`user = input`，`assistant = target`（丢弃 `kind`，或作为可选前缀）。
- **多轮合成**（关键）：真实来源均单轮，故按概率把 1–3 条单轮样本**拼接**成一段多轮对话
  （`user/assistant` 交替，首轮可选注入固定 system），使模型见过多轮格式。
  - 该合成是**有意的数据增广**，在文档与代码注释中明确标注。
- 预分词产物：`data/processed/sft/{train,val}.npz`，键 `ids`(int32 flat)、`labels`(int32 flat)、
  `offsets`(int64)，val 占总样本约 2%。
- 新增 `scripts/prepare_sft_data.py`。

### 4.3 数据目录

```
data/
├─ raw/
│  ├─ text/fineweb_cmn/{train,val}.jsonl.gz     # 预训练原始文本缓存
│  └─ sft/{alpaca_gpt4_data_zh.json, firefly_subset.jsonl}
└─ processed/
   ├─ text/fineweb_cmn.{train,val}.bin
   └─ sft/{train,val}.npz
```

## 5. 组件设计

### 5.1 HF 访问配置 `src/hfenv.py`（新增）

- `setup_hf(endpoint: str | None, proxy: str | None)`：
  - `proxy` 非空 → 设 `HTTP_PROXY`/`HTTPS_PROXY`（并保证 `NO_PROXY` 含本机）。
  - `endpoint` 非空 → 设 `HF_ENDPOINT`；若两者都空 → **直连 HF**（不写死镜像）。
- `DataConfig` 增字段：`hf_endpoint: str = ""`、`proxy: str = "http://127.0.0.1:7890"`。
- 移除 `src/data.py` 顶部写死镜像的 `os.environ.setdefault("HF_ENDPOINT", ...)`，改为在脚本入口调用
  `setup_hf`（旧脚本提供 `--endpoint https://hf-mirror.com` 即可回退镜像，行为向后兼容）。

### 5.2 对话模板 `src/chat_format.py`（新增）

- 从 tokenizer 动态取 id：`im_start = convert_tokens_to_ids("<|im_start|>")`、`im_end=.../<|im_end|>`。
- `SYSTEM_PROMPT`：固定一句简短中文（可为空串），**SFT 与推理必须一致**。
- `render_messages(messages, add_generation_prompt=False) -> str`：
  ```
  <|im_start|>system\n{S}<|im_end|>\n
  <|im_start|>user\n{U}<|im_end|>\n
  <|im_start|>assistant\n{A}<|im_end|>\n
  ...
  ```
- `build_sft_example(tokenizer, messages, max_len=2048) -> (input_ids, labels)`：
  - 逐段编码（`add_special_tokens=False` 以保留特殊 token 单 id）；
  - `labels` 仅对 **assistant 的内容 token + 该轮结尾 `<|im_end|>`** 置真值，其余 `-100`；
  - 超长时**从左侧截断**（保留最近若干轮），并保证至少保留一段 assistant（否则丢弃该样本）。
- `STOP_IDS = [im_end, eos]`（正常情况下模型应生成 `<|im_end|>`，eos 作为兜底）。
- `format_history_for_generate`：推理时把历史渲染到 `assistant\n` 结束。

### 5.3 SFT 数据与训练

- `src/sft_data.py`（新增）：
  - `SFTDataset(npz_path)`：`np.load(..., mmap_mode="r")`，`__getitem__` 返回 `(ids, labels)`。
  - `collate_sft(batch, pad_id, max_len)`：按 batch 内最大长度 pad，ids 补 `pad_id`、labels 补 `-100`；
    超出 `max_len` 截断。
  - `iter_sft_examples(paths, max_items)`：解析 & 归一化（见 4.2）。
- `src/trainer.py`：新增 `SFTTrainer(Trainer)`，仿 `VLMTrainer` 只覆写两个接缝：
  - `_next_batch()` → 返回 collate 后的 dict `{"input_ids","labels"}`；
  - `_forward_loss(batch)` → `model(input_ids, targets=labels)`（外包 `_autocast_ctx`）；
  - `_estimate_val()` → 用 val npz 的前若干 batch。
- `scripts/train_sft.py` + `configs/sft_zh.yaml`：从 `out/gpt_pretrain/latest.pt` 初始化 LLM 权重，
  产物 `out/sft/`（`latest.pt`/`metrics.jsonl`/`loss.png`）。

### 5.4 预训练

- 新增 `configs/gpt_fineweb.yaml`：`ctx_len=2048`、`vocab_size=151666`，其余沿用
  `d_model=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2048`；`out_dir=out/gpt_pretrain`。
- 直接复用 `scripts/train_gpt.py`（已支持 `--set` / `--resume`）。
- RoPE 缓存按 `cfg.ctx_len` 预计算（`model.py:68`），`ctx_len=2048` 自动生效。

### 5.5 推理与交互

- `src/generate.py`：`generate(...)` 增 `stop_ids: list[int] | None`（默认 `[eos]`），命中即停。
- `scripts/chat.py` 改造：`--mode {chat,base}`（默认 `chat`）：
  - `chat`：多轮历史（`messages`）、system、`/reset`、`/quit`；用 `chat_format` 渲染，
    **不额外加 bos**（模板自带起始特殊 token，与训练一致），经 `generate` 在 `<|im_end|>` 停止；
    显示前剥离特殊 token。
  - `base`：保留原「续写」行为（向后兼容 README 既有用法）。
- `scripts/webapp.py`：文本页改为 Gradio **多轮 `ChatInterface`**（维护 messages 历史，应用同一模板）；
  图片描述 / VQA 两个页保留。
- 新增 `scripts/eval_chat.py`：内置 ~20 条留出问题 → 生成 → 自动检查（见 3）→ 打印表格并写
  `out/sft/eval.md`。

## 6. 配置与依赖

- `requirements.txt` 增：`pyarrow`、`aiohttp`（本机已装）；`fsspec` 随依赖已有。
- 新增配置：`configs/gpt_fineweb.yaml`、`configs/sft_zh.yaml`。
- 新增/改动源码：`src/hfenv.py`、`src/chat_format.py`、`src/sft_data.py`（新增）；
  `src/config.py`、`src/data.py`、`src/trainer.py`、`src/generate.py`（改动）；
  `scripts/chat.py`、`scripts/webapp.py`（改动）。
- 新增脚本：`prepare_pretrain_data.py`、`prepare_sft_data.py`、`train_sft.py`、`eval_chat.py`。

## 7. 里程碑

| 里程碑 | 内容 | 产出 |
|---|---|---|
| M9 预训练数据 | HF 直连/代理 + fineweb-2 采样 + 编码 | `prepare_pretrain_data.py`、`fineweb_cmn.*.bin` |
| M10 预训练 | ctx=2048 基座训练 | `out/gpt_pretrain/*`、loss 曲线 |
| M11 SFT | 数据归一化 + 模板掩码 + 训练 | `prepare_sft_data.py`、`train_sft.py`、`out/sft/*` |
| M12 对话交付 | 多轮 chat.py/webapp + 评测 + 文档 | `eval_chat.py`、`out/sft/eval.md`、`docs/05-sft.md` |

## 8. 测试与验收（TDD）

新增 `tests/`：
- `test_chat_format.py`：特殊 token 往返；**只有 assistant 段有 label**（其余 -100）；
  超长截断保留末轮且不截空 assistant；`STOP_IDS` 正确。
- `test_sft_data.py`：alpaca/firefly 归一化正确；多轮合成结构与角色交替正确；
  train/val 不重叠；npz 读写往返；`collate_sft` padding/mask 正确。
- `test_fineweb_reader.py`：用本地极小 parquet 夹具验证「按 row-group + 字节预算」读取、
  train/val 不重叠（不访问网络）。
- `test_sft_trainer.py`：小样本过拟合 loss→接近 0，且能生成训练目标（证明能学）。
- `test_chat_eval.py`：检查函数对「空输出/重复循环/未停止」判负，对正常输出判正。

**端到端验收**：`eval_chat.py` 全项通过 + 人工抽查；`pytest tests/` 全绿；
README/`docs/05-sft.md` 记录数据、训练命令与前后对比样例。

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| 依赖本机代理，代理不可用则下载失败 | 抽取文本落 `data/raw/**` 缓存，断点续跑；`--endpoint` 可切镜像回退 |
| fineweb-2 单片 4.84GB，直接下满浪费 | `HfFileSystem` 按 row-group 远程读，只下所需分块；字节预算封顶 |
| 真实 SFT 数据均单轮，多轮能力不足 | 合成 1–3 轮对话（有意增广）；评测含多轮追问项 |
| 长样本截断把 assistant 截空 | 从左侧截断 + 保证至少一段 assistant，否则丢弃；测试覆盖 |
| 75M 主体 + 简单 SFT 仍只是「简单对话」 | 文档如实说明能力边界，不夸大；不承诺事实正确率 |
| gated 数据集需要授权 | 本阶段不使用 gated 源；如需 Infinity，后续提供 token 再评估 |
| `<|im_end|>` 未纳入监督导致不会停止 | 明确把该 token 计入 assistant label；评测检查停止行为 |

## 10. 预算估算

- 模型：主干 ≈75.5M（不含词表），含 Qwen 词表总计 ≈192M。
- 预训练：`tokens/step = batch(8) × grad_accum(8) × ctx(2048) = 131072`；2–2.5 亿 token ≈ 1500–2000 步，
  16GB 显存内可跑，预计约 1h（可续训）。
- SFT：约 5–8 万段对话、平均 ~600 token；`tokens/step = 8×8×截断长度`，预计 10–20min。
- 磁盘：预训练原始文本 ≤ ~1GB，二进制 ~1.2GB，SFT npz 十至数百 MB，均远低于 495GB 余量。

## 11. 决策记录

| 决策 | 结论 | 理由 |
|---|---|---|
| 训练预算 | 轻量快速（数小时） | 用户选择 |
| 预训练源 | `fineweb-2/cmn_Hani`（原选） | 代理直连后可用；`opencsg/Chinese-FineWeb-Edu-V2` 作备选 |
| SFT 源 | `alpaca-zh` + `firefly` 子集 | Infinity-Instruct gated 不可用，改用非 gated 替代 |
| 模板/特殊 token | 复用 Qwen `<|im_start|>/<|im_end|>` | 已在词表内，零扩表/零改 embedding |
| 模型 | 主干 768/12 不变，`ctx_len=2048`，从零重训 | 用户选择；旧 40 步 checkpoint 视为无效 |
| 损失 | 只在 assistant token（含轮末 `<|im_end|>`） | 让模型学会「回答」与「停止」 |
| 入口 | 改造 `chat.py` + `webapp` 为多轮 | 用户选择；旧续写保留为 `--mode base` |
| 验收 | 留出问题集 + 自动检查 | 用户选择 |
