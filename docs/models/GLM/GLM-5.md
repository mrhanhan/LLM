# GLM-5 · 架构说明（逐层）

> Zhipu AI / Z.ai (THUDM) · 发布 2026-02 · THUDM/GLM-5 `README.md` / `README_zh.md` + 技术报告 arXiv:2602.15763 + HF config `zai-org/GLM-5` / `zai-org/GLM-5-FP8` + HF transformers `modeling_glm_moe_dsa.py`（transformers 5.x）

**技术定位**：面向长程 Agent 工程的旗舰 MoE：744B 总参 / 40B 激活，78 层；注意力换成 DeepSeek 式 MLA，并叠加 DeepSeek Sparse Attention（DSA）轻量 indexer，每 query 只对 top-2048 个 token 做注意力；前三层 dense，其余 256 路由专家 + 1 共享专家、每 token 激活 8 个，sigmoid 无辅助损失路由 + MTP。

## 1. 参数与配置

| 项 | 值 |
|---|---|
| 总参数 | 744B（GLM-5.1/5.2 同量级） |
| 激活参数 | 40B（每 token 8 路由专家 + 1 共享专家） |
| 层数 | 78 |
| hidden | 6144 |
| Q 头 | 64 |
| KV 头 | 64（MLA 压缩 KV；非 GQA） |
| head_dim | 64（qk_rope）/ 256（qk_head_dim）/ 256（v_head_dim） |
| FFN/MoE 中间维 | dense 12288 / MoE 2048×256 + 共享 |
| 词表 | 154880 |
| 上下文 | 202752（≈200K；GLM-5.2 起 1M） |
| 权重共享 | false |

**完整配置**：

| 字段 | 值 |
|---|---|
| model_type | glm_moe_dsa（GlmMoeDsaForCausalLM） |
| num_hidden_layers | 78 |
| hidden_size | 6144 |
| num_attention_heads | 64 |
| num_key_value_heads | 64（MLA，非 GQA） |
| q_lora_rank / kv_lora_rank | 2048 / 512 |
| qk_nope_head_dim / qk_rope_head_dim | 192 / 64（qk_head_dim=256） |
| v_head_dim | 256 |
| head_dim | 64（= qk_rope_head_dim） |
| index_n_heads / index_head_dim | 32 / 128 |
| index_topk | 2048 |
| indexer_rope_interleave | true |
| rope_theta / rope_interleave | 1000000 / true |
| intermediate_size（dense） | 12288 |
| moe_intermediate_size | 2048 |
| n_routed_experts | 256 |
| n_shared_experts | 1 |
| num_experts_per_tok | 8 |
| first_k_dense_replace | 3 |
| moe_layer_freq | 1（除前 3 层外全 MoE） |
| n_group / topk_group | 1 / 1 |
| scoring_func / topk_method | sigmoid / noaux_tc |
| routed_scaling_factor | 2.5 |
| num_nextn_predict_layers | 1（MTP） |
| rms_norm_eps | 1e-5 |
| vocab_size | 154880 |
| max_position_embeddings | 202752 |
| attention_bias | false |
| tie_word_embeddings | false |
| indexer_types（full/shared） | ?：HF config 未显式给出；代码按 `indexer_types[layer_idx]`，默认每层 full indexer。GLM-5.2 起引入 IndexShare（每 4 个稀疏层共用 1 个 indexer） |

## 2. 技术方案与关键创新

**1. 从 355B 扩到 744B-A40B**　相比 GLM-4.5 的 355B-A32B，GLM-5 把参数扩到 744B、激活 40B，预训练数据从 23T 增到 28.5T；层数 78、hidden 6144、MoE 路由专家增至 256（每个中间维 2048）。

**2. MLA（Multi-head Latent Attention）**　注意力不是 GQA，而是 DeepSeek 式低秩压缩：query 先降到 `q_lora_rank=2048` 再展开，KV 压成 `kv_lora_rank=512` 的潜向量；qk 头维 192（nope）+ 64（rope）=256，v 头维 256。KV 缓存只存压缩潜量，显著省显存。

**3. DSA（DeepSeek Sparse Attention）**　每层（或共享层）用一个小 indexer 为每个 query 选出最相关的 `index_topk=2048` 个 token，再只在这 2048 个位置上做注意力。indexer 是独立轻量 Q/K 投影（32 头、头维 128、ReLU 打分、per-head 权重），README 称其“大幅降低部署成本同时保留长文能力”。

**4. 256 专家 MoE + loss-free 路由**　前三层 dense，其后 256 个路由专家 + 1 个共享专家，每 token 激活 8 个；sigmoid 打分 + `e_score_correction_bias` 无辅助损失选专家，权重归一化后乘 2.5。与 GLM-4.5 同一套路由范式。

**5. MTP 与 slime 异步 RL**　`num_nextn_predict_layers=1` 提供 MTP 投机解码；后训练用官方开源的异步 RL 基础设施 slime（THUDM/slime），提高 RL 吞吐与迭代粒度。

**6. GLM-5.1 / 5.2 / 5.3 演进**　5.1 强化长程 agentic 工程；5.2 提出 IndexShare（每 4 个稀疏注意力层复用一个 indexer，1M 上下文下每 token FLOPs 降 2.9×）并把上下文做到 1M；5.3 主干不变靠后训练提升，另有 5.3-Flash（320B-A18B）首次在 GLM 系列引入**稀疏 + 线性注意力混搭**与 Manifold-Constrained Hyper-Connections (mHC)。

## 3. 层级结构（可视化）

```mermaid
flowchart TD
  IN["输入 token B×T"] --> EMB["词嵌入"]
  EMB --> L0_2["层 0–2 ×3<br/>MLA + DSA（稀疏 MLA）<br/>SwiGLU（dense / 共享专家）"]
  L0_2 --> L3_77["层 3–77 ×75<br/>MLA + DSA（稀疏 MLA）<br/>GLM-5 MoE（256 路由 + 1 共享）"]
  L3_77 --> NORM["最终 RMSNorm"] --> HEAD["lm_head → logits"]
```

## 4. 逐层清单（共 78 层）

| 层 | Attention | FFN/MoE | 说明 |
|---|---|---|---|
| 0 | MLA + DSA（稀疏 MLA） | SwiGLU（dense / 共享专家） | 前 3 层 dense（first_k_dense_replace=3） |
| 1 | MLA + DSA（稀疏 MLA） | SwiGLU（dense / 共享专家） | 前 3 层 dense（first_k_dense_replace=3） |
| 2 | MLA + DSA（稀疏 MLA） | SwiGLU（dense / 共享专家） | 前 3 层 dense（first_k_dense_replace=3） |
| 3 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 4 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 5 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 6 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 7 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 8 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 9 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 10 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 11 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 12 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 13 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 14 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 15 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 16 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 17 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 18 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 19 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 20 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 21 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 22 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 23 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 24 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 25 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 26 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 27 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 28 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 29 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 30 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 31 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 32 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 33 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 34 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 35 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 36 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 37 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 38 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 39 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 40 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 41 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 42 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 43 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 44 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 45 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 46 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 47 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 48 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 49 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 50 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 51 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 52 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 53 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 54 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 55 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 56 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 57 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 58 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 59 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 60 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 61 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 62 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 63 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 64 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 65 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 66 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 67 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 68 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 69 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 70 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 71 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 72 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 73 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 74 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 75 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 76 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |
| 77 | MLA + DSA（稀疏 MLA） | GLM-5 MoE（256 路由 + 1 共享） | MoE：256 路由 + 1 共享，top-8 |

## 5. 关键模块：数学公式与代码

### 注意力 · MLA + DSA（稀疏 MLA）

先用 MLA 低秩压缩构造 Q/K/V 潜向量，再用 DSA indexer 为每个 query 选 top-2048 token，只在这些位置上做注意力；indexer 用 interleaved RoPE，主注意力用 qk_rope 的 64 维做 RoPE。

$$
c_t^{KV}=W^{DKV}h_t\in\mathbb{R}^{512},\quad k_t^{C}=\mathrm{RMSNorm}\!\left(W^{UK}c_t^{KV}\right)
$$

$$
q_t=\mathrm{expand}\!\left(\mathrm{RMSNorm}(W^{DQ}h_t)\right),\quad k_t=[k_t^{C};\,k_t^{R}],\;k_t^{R}=\mathrm{RoPE}(W^{KR}h_t)
$$

$$
s_{t,s}=\sum_{h}w_{t,h}\,\mathrm{ReLU}\!\left(q_{t,h}\!\cdot\!k_s\right)\cdot d^{-1/2}
$$

$$
\mathcal{I}_t=\mathrm{topk}_{s}\,s_{t,s}\;\;(k=2048),\qquad \mathrm{Attn}_t=\mathrm{softmax}_{s\in\mathcal{I}_t}\!\left(\frac{q_tk_s^{\top}}{\sqrt{256}}+M\right)v_s
$$

```python
# HF transformers modeling_glm_moe_dsa.py — GlmMoeDsaAttention + GlmMoeDsaIndexer
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))          # q_lora_rank = 2048
q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
key_states, value_states = self.expand_kv(k_pass, k_rot)
topk_indices = self.indexer(hidden_states, q_resid, position_embeddings, ...)  # DSA top-2048
# 之后把非 topk 位置 mask 掉，仅保留 2048 个 key 参与注意力
```

### 前馈/MoE · SwiGLU（dense / 共享专家）

前 3 层 dense FFN 与 MoE 的共享专家均为标准 SwiGLU（中间维 12288 / 共享 2048）。

$$
\mathrm{SwiGLU}(x)=W_{down}\big(\mathrm{SiLU}(W_{gate}x)\odot W_{up}x\big)
$$

```python
# HF transformers modeling_glm_moe_dsa.py — GlmMoeDsaMLP
return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 前馈/MoE · GLM-5 MoE（256 路由 + 1 共享）

sigmoid 打分 + `e_score_correction_bias` 选 top-8，权重归一化后乘 2.5；8 个专家的 SwiGLU 输出加权求和，再叠加恒开的共享专家。

$$
s=\sigma(W_g x),\quad \tilde{s}=s+b_{corr},\quad \mathcal{T}=\mathrm{top8}(\tilde{s})
$$

$$
y=\sum_{i\in\mathcal{T}}\frac{s_i}{\sum_{j\in\mathcal{T}}s_j}\cdot 2.5\,\cdot\,\mathrm{Expert}_i(x)+\mathrm{Shared}(x)
$$

```python
# HF transformers modeling_glm_moe_dsa.py — GlmMoeDsaTopkRouter + GlmMoeDsaMoE
scores = router_logits.sigmoid()
scores_for_choice = scores + self.e_score_correction_bias
topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
topk_weights = scores.gather(1, topk_indices) / (scores.gather(1, topk_indices).sum(-1, keepdim=True) + 1e-20)
# MoE 前向
_, topk_weights, topk_indices = self.gate(hidden_states)
hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
hidden_states = hidden_states + self.shared_experts(residuals)
```

### RMSNorm

$$
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^{2}+\epsilon}}\odot\gamma
$$

### MLA 低秩 KV 压缩

KV 只缓存 512 维潜向量 `c^{KV}` + 64 维带位置的 `k^R`，而非完整 K/V；这是 GLM-5 相对 GLM-4.5 GQA 的最大缓存/算力优化。

$$
\#\,\text{cached per token}=512+64=576\ \ (\text{vs. GQA 的 } H_{kv}\cdot 2d_h)
$$

$$
v_t=W^{UV}c_t^{KV},\qquad k_t^{C}=W^{UK}c_t^{KV}
$$

### DSA indexer 打分公式

indexer 有独立的 `wq_b`（q_lora_rank→32×128）与 `wk`（hidden→128），`weights_proj` 给出每头权重；对全序列打分后取 top-2048。

$$
q^{\,idx}=\mathrm{RoPE}\!\left(W_{q}^{idx}\,q_{resid}\right),\quad k^{\,idx}=\mathrm{RMSNorm}\!\left(W_{k}^{idx}h\right)
$$

$$
s_{t,s}=\sum_{h=1}^{32}w_{t,h}\cdot\mathrm{ReLU}\!\left(q^{\,idx}_{t,h}\cdot k^{\,idx}_{s}\right)\cdot 128^{-1/2}
$$

### RoPE（interleaved, θ=1e6）

GLM-5 使用 interleaved RoPE（相邻维成对旋转），只作用于 qk_rope 的 64 维；indexer 同样用 interleaved RoPE。

$$
\theta_i=\theta_0^{-2i/64},\quad \theta_0=10^{6}
$$

$$
\mathrm{RoPE}_{int}(x_{2i},x_{2i+1})=\begin{bmatrix}x_{2i}\cos m\theta_i-x_{2i+1}\sin m\theta_i\\ x_{2i}\sin m\theta_i+x_{2i+1}\cos m\theta_i\end{bmatrix}
$$

### MTP（多 token 预测）

`num_nextn_predict_layers=1`：主干后接 1 个 MTP 层（含 `eh_proj/enorm/hnorm`）用于投机解码；README 部署示例用 EAGLE/MTP。

$$
p(x_{t+1},x_{t+2}\mid x_{\le t})\;\approx\;p_{head}\!\left(\mathrm{MTP}(h_t)\right)
$$

## 6. 张量形状流

| 阶段 | 形状 | 说明 |
|---|---|---|
| 输入 token id | `[B, T]` |  |
| 词嵌入 | `[B, T, 6144]` |  |
| 层 0–2（dense） | `[B, T, 6144]` | MLA+DSA + SwiGLU |
| 层 3–77（MoE） | `[B, T, 6144]` | 8/256 路由 + 1 共享 |
| 最终 RMSNorm | `[B, T, 6144]` |  |
| lm_head（不共享） | `[B, T, 154880]` |  |
| MTP 头（1 层） | `[B, T, 154880]` | 投机解码用 |

## 7. 关键源码（引自 reference/）

**MLA 主注意力（低秩 Q/KV + interleaved RoPE）**　`https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py`

```python
q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
key_states, value_states = self.expand_kv(k_pass, k_rot)
```

**DSA indexer：为每个 query 选 top-2048 token**　`https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py`

```python
q = self.wq_b(q_resid).view(batch_size, seq_len, self.n_heads, self.head_dim)
q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * self.softmax_scale
scores = F.relu(scores)
weights = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)
index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
topk = min(self.index_topk, index_scores.shape[-1])            # 2048
return index_scores.topk(topk, dim=-1).indices.to(torch.int32)
```

## 8. 与 mini-llm-lab 的差异

GLM-5 与我们手写 mini-llm-lab 的差距是三代同堂式的：

- **注意力**：mini 是 MHA/GQA + 全量 RoPE；GLM-5 是 **MLA**（低秩压缩 KV，只存 512+64 维/ token）再加 **DSA**（每 query 只看 top-2048 token），复杂度从 O(T²) 降到近似 O(T·2048)；
- **FFN**：mini 单 SwiGLU；GLM-5 前三层 SwiGLU，其余 256 路由专家 + 1 共享、每 token 8 个；
- **规模**：6144 维 × 78 层、744B 总参，是我们 ~192M 的数千倍；
- **路由/位置**：sigmoid 无辅助损失 routing、interleaved RoPE θ=1e6、1 层 MTP。

要在 mini 里体会，可先加部分 RoPE + MoE（对齐 GLM-4.5），再单独实现 MLA+DSA（对齐 GLM-5）。

## 9. 3D 可视化

启动可视化网页后，在「架构浏览器」视图选择本模型（id：`glm-5`），可三维查看每一层并点开公式/代码：

```powershell
& ".venv\Scripts\python.exe" viz/server.py   # 打开 http://127.0.0.1:7861 → 架构浏览器
```

## 10. 参考资料

- [GLM-5: from Vibe Coding to Agentic Engineering (arXiv:2602.15763)](https://arxiv.org/abs/2602.15763)
- [THUDM/GLM-5](https://github.com/zai-org/GLM-5)
- [HF transformers modeling_glm_moe_dsa.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)
