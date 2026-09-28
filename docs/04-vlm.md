# 04 · 视觉语言模型（VLM）原理

> 目标：在已经会"说话"的中文 GPT 基础上"装上眼睛"，使其能看图片并输出中文描述、
> 回答简单问题。核心思路一句话：**把图片变成一串向量，塞进语言模型的输入序列里。**

## 1. 语言模型为什么"看不见"

文本 GPT 的输入是一串 token id，经过 `nn.Embedding` 变成 `(B, T, d_model)` 的向量。
它只认识词表里的 token，没有任何接收像素的通道。要让它能看图，我们要解决两件事：

1. 把图片编码成和文本同维度的向量序列；
2. 让这些向量出现在正确的位置，并且让语言模型知道"这里是一张图"。

## 2. ViT：把图片变成"视觉词"

人看一张图，可以把它切成 128×128、每块 16×16 的小方块（patch）。默认配置下
`grid = 128/16 = 8`，所以一张图有 `8×8 = 64` 个 patch。每个 patch 拉平成 16×16×3
的向量，用一个卷积（`PatchEmbed`，等价于"按块切分 + 线性投影"）映射到 `d_vision` 维，
就得到 64 个"视觉词"。

之后交给和语言模型同款的 Transformer 编码器（`src/vision.py`），区别只有一点：

- 文本有先后顺序，注意力是**因果的**（只能看左边，`causal=True`）；
- 图片内部没有顺序，注意力是**双向的**（`causal=False`）。

```text
图片 (3,128,128) → PatchEmbed → 64 个 patch 向量
                → 加可学习位置编码 → 6 层双向 Transformer → (64, d_vision)
```

## 3. 投影层：对齐两个"特征空间"

ViT 输出的维度是 `d_vision=384`，而语言模型需要 `d_model=768`，两者语义空间也不同。
中间加一个两层 MLP（`VisionProjector`）做"翻译"：`384 → 768 → 768`。

这一步很关键：它把视觉特征搬到语言模型"能读懂"的空间里。

## 4. 融合：用 `<image>` 占位符 + 特征替换（LLaVA 式）

在词表里加一个特殊 token `<image>`。构造输入时，用 64 个 `<image>` 占住图像的 64 个
位置：

```text
<bos> <image> <image> ... <image> 图中有一个红色圆形。 <eos>
         └──────── 64 个 ────────┘
```

模型一开始把 `<image>` 当成普通 token 得到 embedding；随后我们把这 64 个位置的向量
**替换**成 ViT 投影后的 64 个视觉向量（`MiniVLM.replace_image_embeds`）。对语言模型
来说，这就是一条普通序列——它不需要知道哪些是图、哪些是字。

## 5. 损失只算在文本上

图像位置不应该被"预测"（我们不想让模型去重建像素），所以这些位置的标签设为 `-100`
（`PyTorch` 交叉熵会忽略它）。模型只需学会：**看到这 64 个视觉向量，写出对应的中文。**

> 因果语言模型预测的是"下一个 token"，因此 `labels` 必须相对 `input_ids` **右移一位**
> （见 `src/data.py: build_caption_example`）。若标签与输入对齐，模型会学到"复制输入"
> 的捷径，loss 迅速变 0 却完全不会生成——这是最常见、也最隐蔽的坑。

## 6. 训练策略：先教说话，再装眼睛

MiniVLM 的 LLM 部分直接从**文本阶段训练好的 checkpoint** 加载（`gpt_ckpt`），
只有 ViT 和投影层是随机初始化。这比从零同时训两半快得多，也更稳定。

指令/问答（VQA）在同一个框架下实现：把问题接在图像占位符后面，只监督答案部分：

```text
<bos> <image>*64 图中有几个图形？答： 两个 <eos>
                                    └── 只在这里算损失 ──┘
```

## 7. 数据

1. **合成数据**（`SyntheticImageDataset`）：程序随机画 1~3 个彩色几何图形，同时生成
   "一个红色圆形在左边"这样的中文描述与问答。图与文本同源，能快速验证整条链路是否
   真的"看图说话"，而不是背语料。
2. **真实数据**：中文儿童图文数据集（png + 同名 txt 描述），下载与索引见
   `scripts/prepare_image_data.py`。

## 8. 怎么用

```powershell
# 合成数据上快速训练（先跑通链路）
& ".venv\Scripts\python.exe" scripts/train_vlm.py --set train.max_steps=600 train.batch_size=4

# 训练 VQA 版本
& ".venv\Scripts\python.exe" scripts/train_vlm.py --use_qa --set train.out_dir=out/vlm_vqa train.max_steps=600

# 真实图文（需先成功下载数据）
& ".venv\Scripts\python.exe" scripts/prepare_image_data.py
& ".venv\Scripts\python.exe" scripts/train_vlm.py --data_kind children

# 网页交互（文本 + 图片）
& ".venv\Scripts\python.exe" scripts/webapp.py
```

## 9. 效果自检与常见失败

- **合成数据上 loss 应明显下降**：本项目 100 步内 loss 从 ~9 降到 ~0.2。
- **描述变成"图图图图…""红色红色…"这类重复**：几乎总是训练/推理的输入格式或标签
  错位（见第 5 节），或标签没右移。
- **描述正确但颜色/形状随机**：说明文本模板学会了，但视觉特征没被真正用上——检查
  投影层是否参与训练、`<image>` 位置是否真的被替换（可对比"换一张图，logits 是否变化"，
  见 `tests/test_vlm.py: test_image_features_really_used`）。
- **VQA 只学会套话**：检查 `use_qa` 样本构造与 `<image>` 数量是否与训练一致。

## 10. 与代码的对应关系

| 概念 | 代码 |
|---|---|
| Patch 切分 + 投影 | `src/vision.py: PatchEmbed` |
| 双向视觉编码器 | `src/vision.py: VisionEncoder` |
| 模态对齐投影 | `src/vlm.py: VisionProjector` |
| 特征替换融合 | `src/vlm.py: MiniVLM.replace_image_embeds` |
| 图文/问答样本构造 | `src/data.py: build_caption_example / build_vqa_example` |
| 批量整理与补齐 | `src/data.py: collate_vlm` |
| 多模态训练循环 | `src/trainer.py: VLMTrainer` |
| 图片描述 / VQA 推理 | `src/vlm_generate.py` |
| 合成 / 真实数据 | `src/data.py: SyntheticImageDataset / CaptionDataset` |
