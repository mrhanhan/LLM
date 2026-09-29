# 06 · 架构对比：Qwen3 / DeepSeek V3→V4.1 vs 我们的 mini-llm-lab

本文把本项目「从零手写的中文 GPT」放到真实工业模型的坐标系里对照，回答一个问题：

> **我们抄到的骨架（RoPE + RMSNorm + SwiGLU + GQA + 权重共享）和一线大模型差在哪？**

结论先行：**主干骨架几乎一致，差距全部在"规模化 + 省算力/省显存"的工程改造上**。
Qwen3 基本就是"放大版 + 加了 QK-Norm 的我们"；DeepSeek 则是在同一个骨架上做了
MLA / MoE / 稀疏注意力 / 多 token 预测等结构性替换。

---

## 0. 参考源码（本地已克隆）

为便于对照，第三方代码放在 `reference/`（已在 `.gitignore` 忽略，不进本仓库），
全部用 **blobless 浅克隆 / 单文件下载**，只取建模相关代码：

| 目录 | 来源 | 取到的内容 |
|---|---|---|
| `reference/Qwen3/` | `github.com/QwenLM/Qwen3`（官方仓库）+ HF transformers | 官方 README/文档；`hf_transformers/modeling_qwen3.py`、`configuration_qwen3.py` |
| `reference/DeepSeek-V3/` | `github.com/deepseek-ai/DeepSeek-V3` | `inference/model.py`（MLA+MoE 参考实现）、configs；HF `modeling_deepseek_v3.py` |
| `reference/DeepSeek-V3.2-Exp/` | `github.com/deepseek-ai/DeepSeek-V3.2-Exp` | `inference/model.py`（DSA 稀疏注意力）、HF `modeling_deepseek_v32.py` |
| `reference/DeepSeek-V4.1/` | HF `deepseek-ai/DeepSeek-V4.1-Flash` + transformers | `hf_config.json`、`hf_README.md`、`hf_transformers/modeling_deepseek_v4.py` |

> 说明：Qwen3 官方仓库本身只有文档/部署脚本，**建模代码在 HuggingFace transformers**，
> 所以从 transformers 抓了 `modeling_qwen3.py`。DeepSeek V4/V4.1 **没有放出官方 GitHub
> 建模仓库**，只有 HF 权重 + config + 技术报告，因此以 HF config/README + transformers
> `modeling_deepseek_v4.py`（通用 V4 实现）为据，下文会对"V4.1 独有、代码里没有"的部分
> 单独标注。

复现命令（PowerShell，网络走本机代理 `http://127.0.0.1:7890`）：

```powershell
# 1) 稀疏浅克隆（只下代码，不下权重）
git clone --depth 1 --filter=blob:none https://github.com/deepseek-ai/DeepSeek-V3.git reference/DeepSeek-V3
git clone --depth 1 --filter=blob:none https://github.com/deepseek-ai/DeepSeek-V3.2-Exp.git reference/DeepSeek-V3.2-Exp
git clone --depth 1 --filter=blob:none --sparse https://github.com/QwenLM/Qwen3.git reference/Qwen3
git -C reference/Qwen3 sparse-checkout set docs/source

# 2) Qwen3 / DeepSeek 的 HF modeling 代码（官方仓库没有，从 transformers 取）
#    base=https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models
#    qwen3/modeling_qwen3.py、deepseek_v3/modeling_deepseek_v3.py、
#    deepseek_v32/modeling_deepseek_v32.py、deepseek_v4/modeling_deepseek_v4.py
```

---

## 1. 我们的模型（基线）

`src/model.py` + `src/attention.py`，默认配置 `configs/gpt_tinystories.yaml`：

- **类型**：decoder-only 自回归 Transformer（pre-norm 残差）
- **结构**：12 层，`d_model=768`，12 个 Q 头 / 4 个 KV 头（GQA），`d_ff=2048`，`ctx=1024`
- **位置**：RoPE（对整段 head_dim 做旋转，`rope_theta` 可配）
- **归一化**：RMSNorm（`eps=1e-6`）
- **FFN**：SwiGLU（`w3(silu(w1(x)) * w2(x))`）
- **输出**：`lm_head` 与 `tok_emb` **权重共享**（tie embeddings）
- **参数量**：主干 ~75.5M；Qwen 词表 151666 共享输出 ⇒ 总计 ~192M
- **训练**：bf16 AMP + 梯度累积 + 裁剪 + warmup/余弦 + 断点续训
- **推理**：KV 缓存 + temperature / top-k / top-p / 重复惩罚
- **多模态**：手写 ViT + 投影层（LLaVA 式 `<image>` 替换）

一句话：这是一个**规格对齐 Llama 系**、但规模缩小到单卡可训练的"教学级 dense decoder"。

---

## 2. Qwen3：几乎就是"放大 + 加 QK-Norm 的我们"

Qwen3 的每一块都能在我们的代码里找到对应物，差异是**规模**和**两个小改进**：

| 维度 | 我们 | Qwen3 | 差异性质 |
|---|---|---|---|
| 骨架 | decoder-only, RoPE, RMSNorm, SwiGLU, GQA | **完全相同** | ✅ 同源 |
| QK-Norm | ❌ 无 | ✅ `q_norm/k_norm = RMSNorm(head_dim)` 作用在 q/k 上（`modeling_qwen3.py:237`） | 🆕 缺一个稳定器 |
| RoPE theta | 可配（默认 `10000`） | **`1e6`**（长文外推），YaRN 扩到 128K/1M | 🔧 数值 |
| 注意力偏置 | `bias=False` | `attention_bias=False`（默认） | ✅ 一致 |
| GQA | `n_head=12, n_kv_head=4` | 8B：`32/8`；30B-A3B：`32/4` | 📈 规模 |
| 权重共享 | ✅ 共享 `lm_head`/`tok_emb` | 0.6B/1.7B/4B 共享，8B 及以上不共享 | 🔀 随规模取舍 |
| 词表 | Qwen 151666（复用其分词器） | 151936（BPE 151643 + 特殊符） | ✅ 几乎一致 |
| 上下文 | 1024（TinyStories）/ 2048（fineweb） | 32768 预训练，扩到 131072 / 256K / 1M | 📈 规模 |
| MoE 变体 | ❌ | 30B-A3B / 235B-A22B：128 专家、每 token 激活 8、**无共享专家** | 🆕 结构替换 |
| 思考模式 | ❌ | 混合 thinking / non-thinking（`</think>` id 151668） | 🆕 后训练层 |
| 训练/推理 | bf16、单卡 | FP8、张量并行、flash/sdpa/flex 后端、kernelized RoPE/RMSNorm | 🏭 工程 |

**教学结论**：Qwen3 是"我们的模型 × 足够数据 × 足够算力 + QK-Norm + MoE 可选分支"。
它证明了手写骨架的正确性：**主干没变，变的是规模与工程**。

---

## 3. DeepSeek：在同一骨架上做结构性替换

DeepSeek 保留了我们熟悉的 RMSNorm / SwiGLU / pre-norm 残差 / RoPE（部分维度），
但把三件核心部件做了重写：**注意力（MLA）、FFN（MoE）、预测目标（MTP）**。

### 3.1 V3（671B，MLA + DeepSeekMoE + MTP）

- **MLA 多头潜在注意力**（替换 GQA）：
  - Q 走低秩：`wq_a: dim→1536 → q_norm → wq_b → 128×192`（`DeepSeek-V3/inference/model.py:427`）
  - KV 只缓存**一个 512 维 latent + 一个 64 维 rope key**（`:443`），而非每头 K/V。
    缓存从每 token `128×(192+128)` 降到 `512+64=576` 维，**约 71×**
  - RoPE 只作用在 64 维"rope 切片"上（**partial / decoupled**），不是整段旋转（`:466`）
  - 推理时把 `q_nope` 吸收进 `wkv_b`（"absorb" 写法，`:481`）
- **DeepSeekMoE**（替换 dense SwiGLU FFN）：
  - 每层 **1 个共享专家 + 256 个路由专家**，每 token 激活 **8** 个（`config_671B.json:9-11`）
  - 前 3 层用 dense MLP（`n_dense_layers=3`）
  - 路由：sigmoid 打分 + 分组限制贪心（8 组选 4），`route_scale=2.5`
  - **aux-loss-free 负载均衡**：给每个专家一个 `float32` 选择偏置，**只影响 argmax 选谁**、
    不影响路由权重（`model.py:564,582`）——避免传统 aux loss 干扰主任务
- **MTP 多 token 预测**：额外挂 1 层预测"再下一个 token"（`num_nextn_predict_layers=1`），
  用于自推测解码（`README_WEIGHTS.md:36`）
- **精度**：FP8(e4m3) + 128×128 分块缩放；**无权重共享**（`tie_word_embeddings=False`）
- 规模：671B 总 / 37B 激活，61 层，`dim=7168`，词表 129280

### 3.2 V3.2-Exp（给 MLA 加"稀疏索引器"）

在 V3 之上加 **DeepSeek Sparse Attention (DSA)**：

- 多一个轻量 **Indexer**（`index_n_heads=64`, `index_head_dim=128`, `index_topk=2048`），
  用自己的 q/k 给每个 query 选出 **top-2048** 个最相关的 key，只对这些 key 做注意力
- 打分用 `fp8_index` 核 + Hadamard 旋转后再量化（`V3.2-Exp/inference/model.py:428,480`）
- Indexer 的 RoPE 是 **非交错**的（MLA 主分支是交错），并修过一次 bug（README:77）
- 其余沿用 V3：61 层、256/8 专家、MLA ranks 不变

价值：长上下文下把注意力从 O(n²) 压到 **O(n·topk)**，同时效果与 V3.1-Terminus 持平。

### 3.3 V4.1-Flash（CED 编码器-解码器 + CSA2 + Engram + DSpark）

V4/V4.1 是"新架构族"，与 V3 已不是同一档改造，按 `hf_config.json` / `hf_README.md`：

| 组件 | 内容 | 与我们/前代的关系 |
|---|---|---|
| 规模 | **552B** backbone，每 token 激活 **8B 预填充 / 16B 解码** | 比 V3 更"稀疏" |
| 层数 | **40 层 = 20 层 causal encoder + 20 层 decoder**（CED） | 🆕 不再是纯 decoder |
| 注意力 | **1 个 KV 头**（shared-KV MQA）、`head_dim=512`、`o_groups=8` 分组低秩输出 | MLA → MQA+分组输出 |
| 局部/全局 | 每层滑动窗口 `swa=128` + **CSA2** 压缩稀疏注意力（Full/Reindex/Reuse 三种静态模式跨层复用 KV 与索引） | DSA 的进化版 |
| 索引器 | 分层稀疏索引器：后层只在前层给的候选池（2048 块×8）里选，**512** topk | DSA 的分层版 |
| RoPE | 主分支 θ=10000，压缩分支 θ=160000 + YaRN factor 16，部分维度（64/512） | partial RoPE 延续 |
| MoE | **1 共享 + 384 路由**，每 token 激活 **6**，`sqrtsoftplus` 打分、`noaux_tc` | 专家数 256→384 |
| mHC | 单遍 mHC（Sinkhorn 20 次迭代到双随机矩阵）多流残差混合 | 🆕 残差流可学习混合 |
| Engram | 条件记忆，196B 参数，按 token n-gram 稀疏查表 | 🆕 外挂记忆 |
| DSpark | 半自回归草稿 + 置信度调度验证（推测解码） | MTP 的进化版 |
| MTP | `num_nextn_predict_layers=3` | 1→3 |
| 精度 | 权重 FP8(32×32 分块) + **专家/KV FP4**；global KV ≈ **890 B/token** | FP8 → FP4 |
| 视觉 | DeepSeek-ViT：32 层、patch 14、3×3 pixel-unshuffle，2 层 MLP 投影 | 与我们的手写 ViT 定位相同 |
| 上下文 | **1M** tokens（45T token 预训练） | 128K → 1M |

> ⚠️ **事实核对**：本地抓到的 `modeling_deepseek_v4.py` 是 transformers 的**通用 V4** 实现，
> 已实现 mHC / CSA+HCA / indexer / MoE 等，但**未实现 V4.1 独有的 CED 编解码拆分、
> CSA2 三模式、分层候选池、Engram、DSpark、DeepSeek-ViT**——这些只存在于 HF config/README。
> 对比时两者要区分。另：V3/V3.2 的 `inference/model.py` **并未实现 MTP**，只有 HF config
> 暴露 `num_mtp_layers`。

---

## 4. 一图总表

| 维度 | **mini-llm-lab（我们）** | **Qwen3（8B 为代表）** | **DeepSeek-V3** | **DeepSeek-V3.2** | **DeepSeek-V4.1-Flash** |
|---|---|---|---|---|---|
| 层数 | 12 | 36 | 61 | 61 | 40（20 enc+20 dec） |
| d_model | 768 | 4096 | 7168 | 7168 | 5120 |
| Q/KV 头 | 12 / 4 | 32 / 8 | 128 / 128(MLA) | 128 / 128(MLA) | 64 / **1** |
| 注意力 | **GQA** | GQA + **QK-Norm** | **MLA**（低秩 KV） | MLA + **DSA 稀疏** | MQA + **CSA2** 压缩稀疏 |
| 位置编码 | RoPE 全维 | RoPE θ=1e6 + YaRN | partial RoPE + YaRN | 同 V3 | partial RoPE 双 θ |
| 归一化 | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| FFN | **dense SwiGLU** | dense SwiGLU | **MoE 256+1/8** | MoE 256+1/8 | **MoE 384+1/6** |
| MTP | ❌ | ❌ | 1 层 | 1 层 | 3 层 + DSpark |
| 权重共享 | ✅ | 0.6–4B ✅ / 8B+ ❌ | ❌ | ❌ | ❌ |
| 上下文 | 1K–2K | 32K→1M | 128K | 128K+ | **1M** |
| 参数量 | ~192M | 0.6B–32B（+MoE） | 671B / 37B 激活 | 671B / 37B 激活 | 552B / 8–16B 激活 |
| 精度 | bf16 | FP8 | FP8 e4m3 | FP8 ue8m0 | FP8 + FP4 专家/KV |
| 多模态 | ✅ 手写 ViT | ❌（另 VL 系列） | ❌ | ❌ | ✅ DeepSeek-ViT |

---

## 5. 我们"缺"的东西 & 可练习的升级路线

按"从易到难、能在这个教学仓库里动手实现"排序：

1. **QK-Norm**（1 行级改动，Qwen3 做法）：在 `MultiHeadAttention` 里给 q/k 各加一个
   `RMSNorm(head_dim)`，能显著稳住大学习率训练。👉 最推荐的第一课。
2. **RoPE 换 θ=1e6 + YaRN**：把 `build_rope_cache` 的 `rope_theta` 调大，再实现 YaRN 缩放，
   就能把 `ctx_len` 从 1K 拉长（配合长文数据）。
3. **Partial RoPE**（DeepSeek 做法）：只旋转前 64 维，其余不旋转，省计算、利于外推。
4. **GQA 调参**：把 `n_kv_head` 继续调小（如 12→1，即 MQA），观察 KV 缓存与显存/质量的权衡。
5. **MTP 头**：再加一个预测 n+2 的 head，理解 DeepSeek 的推测解码收益。
6. **MoE**：把某个 Block 的 dense SwiGLU 换成"共享专家 + N 路由专家 + top-k 门控"，
   并加上 aux-loss-free 的选择偏置。
7. **MLA / 稀疏注意力 / FP8**：属于"前沿工程"，建议先读懂 `reference/` 里的实现再动手。

> 一句话总结：**我们缺的不是"骨架"，而是规模化、省算力（MLA/DSA/CSA2）、省参数（MoE）、
> 多目标（MTP）以及低精度训练（FP8/FP4）这几类工程改造。**

---

## 6. 相关文档

- 本项目模型结构逐项推导：[02-transformer.md](02-transformer.md)
- 训练原理与读曲线：[03-training.md](03-training.md)
- 可视化演示（3D 看权重如何变化）：见 `viz/` 与 [07-visualization.md](07-visualization.md)
- **更多模型（GLM / Qwen3.5-3.8 / MiMo / Kimi K2-K3）的逐层架构文档**：[models/README.md](models/README.md)
