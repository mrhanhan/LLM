# viz 训练配置与实时刷新 设计（Spec）

**日期：** 2026-09-29
**状态：** 已与用户对齐（两轮 AskUserQuestion）

## 目标

让 `viz` 左栏成为一个可配置的训练控制台：选择模型/训练方式/数据集/超参后开始训练，
并在训练过程中让 3D 视图与右侧抽屉**实时反映权重变化**。

## 决策（来自用户访谈）

1. 适用范围：`live` / `ckpt` / `scratch` **三种目标都支持**选择训练方式与数据集。
2. 分词器：**tiny live 也用 Qwen 词表**（不再有字符级内置分词器作为训练默认）。
   - 因此 live 模型仍是很小的结构（d_model=64 / 4 层），但 `vocab_size = QwenTokenizer.vocab_size`。
3. 训练循环：**两套引擎，可切换**
   - `simple`：轻量循环（可随时停、逐 tick 推权重），支持 pretrain/SFT、max_steps、lr、grad_accum、grad_clip。
   - `hifi`：复用 `src.trainer.Trainer` / `SFTTrainer`（bf16、warmup/cosine 调度、梯度累积），
     用子类注入 `on_step` 回调与中断标志，仍逐 step 推权重。
4. 训练配置位置：**左栏四段式**（模型 / 训练方式 / 数据集 / 超参 / 高保真勾选 / 开始停止）。
5. 暴露的超参：`max_steps`、`lr`、`batch_size`、`grad_accum`。
6. 实时刷新：**分级节流**
   - 3D 薄板与立方块：每个 `tick` 更新（沿用现有 `shelf.updateValues`）。
   - 右侧抽屉的热力图/统计：打开「矩阵」页且训练中，约 **500ms 节流**重绘。
   - 展开层的平面/立方纹理：每 **N 个 tick**（默认 20）重建一次。
7. 数据集来源：**写死注册表** `viz/datasets.py`（不扫描目录）。
8. SFT 前置：**不限制**（允许从随机权重 SFT，不警告）。
9. 结束保存：停止/完成后保存 checkpoint 到 `out/viz/<target>-<mode>-<dataset>-<ts>.pt`。

## 数据集注册表（写死）

| id | 名称 | kind | 数据 |
|---|---|---|---|
| `corpus_qwen` | 内置中文语料 | pretrain | `viz.runtime.CORPUS`，Qwen 分词（默认，最快） |
| `tiny_stories` | TinyStories | pretrain | `data/processed/text/train.bin` |
| `poetry` | 中文诗词 | pretrain | `data/processed/text/poetry.train.bin` |
| `advertise` | AdvertiseGen | pretrain | `data/processed/text/advertise.train.bin` |
| `fineweb_cmn` | FineWeb 中文（大文件） | pretrain | `data/processed/text/fineweb_cmn.train.bin` |
| `sft_chat` | Alpaca+Firefly 多轮 | sft | `data/processed/sft/train.npz` |
| `sft_poetry` | 诗词续写 | sft | `data/processed/sft/poetry.train.npz` |
| `sft_advertise` | 广告文案 | sft | `data/processed/sft/advertise.train.npz` |

所有条目 `tokenizer = "qwen"`（`data/tokenizer/qwen2.5-0.5b`）。

## 后端接口（新增/变更）

- `GET /api/datasets` → `[{id,label,kind,tokenizer,available,note}]`
- `POST /api/train/start` body：
  ```json
  {"target":"live|ckpt|scratch","mode":"pretrain|sft","dataset":"poetry",
   "engine":"simple|hifi","params":{"max_steps":200,"lr":3e-4,"batch_size":8,"grad_accum":1}}
  ```
  返回 `{ok,target,mode,dataset,engine,total_steps}`
- `tick` 消息新增 `total_steps`、`mode`、`dataset`、`engine`。
- `POST /api/train/stop` → 停止并保存。

## 非目标

- 不做数据集的在线下载（用 `scripts/prepare_modelscope_data.py` 离线准备）。
- 不做多进程/多卡；单进程后台线程。
- 不改动 `src/` 的训练语义（hifi 通过子类扩展，不修改 `Trainer` 本体）。
