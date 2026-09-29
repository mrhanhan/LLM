# 模型架构文档索引

每个模型一份逐层架构说明，由 `viz/architectures/*.json` 生成（`python scripts/gen_model_docs.py`），并可在可视化网页的「架构浏览器」中三维查看。

👉 **[整体对比（HTML）](compare.html)** —— 所有模型的参数/层数/注意力/MoE/上下文横向对比。

## DeepSeek

| 模型 | 文档 | 参数 | 说明 |
|---|---|---|---|
| DeepSeek-V4.1 | [DeepSeek-V4.1.md](DeepSeek/DeepSeek-V4.1.md) | 552B 骨架（+196B Engram） | 552B 骨架 / 每 token 激活 8B（prefill）–16B（decode）的原生多模态 MoE：采用 Causal Encoder-Decoder（CED，40 层 = 20 encoder + 20 decoder），注意力是 shared-KV MQA（仅 1 个 KV 头、head_dim 512、分组低秩输出）叠加 CSA2 压缩稀疏注意力与分层稀疏索引器，全局 KV 缓存仅约 890 B/token，支持 1M 上下文；另有 mHC 多流残差、Engram 条件记忆、DSpark 推测解码与 DeepSeek-ViT 视觉编码器。 |
| DeepSeek-V3 | [DeepSeek-V3.md](DeepSeek/DeepSeek-V3.md) | 671B（+14B MTP 权重） | 671B 总参 / 37B 激活的稀疏 MoE 大模型：61 层，注意力是 MLA（KV 压成 512 维 latent + 64 维 decoupled rope key），FFN 是 DeepSeekMoE（1 共享 + 256 路由专家、每 token 激活 8、前 3 层为 dense），首创 aux-loss-free 无辅助损失负载均衡与 MTP 多 token 预测，FP8 混合精度训练，128K 上下文。 |
| DeepSeek-V3.2 | [DeepSeek-V3.2.md](DeepSeek/DeepSeek-V3.2.md) | 671B | 在 V3 主干（61 层、671B/37B、MLA + DeepSeekMoE）上叠加 DeepSeek Sparse Attention (DSA)：一个轻量 lightning indexer（64 头、head_dim 128、index_topk 2048）为每个 query 选出最相关的 top-2048 个 key，只对这些 key 做注意力，把长上下文注意力从 O(n²) 压到 O(n·topk)，效果与 V3.1-Terminus 持平。 |

## GLM

| 模型 | 文档 | 参数 | 说明 |
|---|---|---|---|
| GLM-4 | [GLM-4.md](GLM/GLM-4.md) | 9B（另有 GLM-4-32B-0414 / Z1 32B / 9B 等） | ChatGLM 系 dense decoder-only：RMSNorm + SwiGLU + GQA，关键点是**部分旋转位置编码**（rotary 只作用于 head_dim 的一半），GLM-4-9B 用 32 个 Q 头对 2 个 KV 头（16:1）；GLM-4-32B-0414 换成 4 个 RMSNorm 的 sandwich 结构。 |
| GLM-4.5 | [GLM-4.5.md](GLM/GLM-4.5.md) | 355B（另 GLM-4.5-Air 106B） | 面向 Agent 的 MoE 基础模型：355B 总参 / 32B 激活，92 层，前三层 dense、其后每层 160 路由专家 + 1 共享专家（每 token 激活 8 个），GQA 96/8 + QK-Norm + 部分 RoPE，sigmoid 无辅助损失（loss-free）路由，并带 1 层 MTP。 |
| GLM-5.3-Flash | [GLM-5.3-Flash.md](GLM/GLM-5.3-Flash.md) | 320B-A18B | GLM 系列首个全新基座的多模态 Flash 模型（320B-A18B、45 层）：文本侧用**稀疏 + 线性注意力混搭**——34 层 KDA 门控线性注意力（O(1) 递归状态）与 11 层 deepseek_sparse_attention（MLA + DSA，每 query top-2048）按 3:1 交织；并用 Manifold-Constrained Hyper-Connections（mHC，hc_mult=4）提升缩放效率；MoE 288 路由 + 1 共享（top-8），原生 1M 上下文，30T token 多模态语料预训练。 |
| GLM-5.3 | [GLM-5.3.md](GLM/GLM-5.3.md) | 744B-A40B | GLM-5 系列的文本旗舰：与 GLM-5.2 同基座（744B-A40B、78 层），全部增益来自后训练。延续 MLA + DeepSeek Sparse Attention（DSA，每 query 只看 top-2048 token）与 IndexShare（每 4 个稀疏层共享一个 indexer），前 3 层 dense、其余 256 路由专家 + 1 共享专家（top-8），sigmoid 无辅助损失路由 + 1 层 MTP，原生 1M 上下文；支持 reasoning_effort = low / high / max。 |
| GLM-5 | [GLM-5.md](GLM/GLM-5.md) | 744B（GLM-5.1/5.2 同量级） | 面向长程 Agent 工程的旗舰 MoE：744B 总参 / 40B 激活，78 层；注意力换成 DeepSeek 式 MLA，并叠加 DeepSeek Sparse Attention（DSA）轻量 indexer，每 query 只对 top-2048 个 token 做注意力；前三层 dense，其余 256 路由专家 + 1 共享专家、每 token 激活 8 个，sigmoid 无辅助损失路由 + MTP。 |

## Kimi

| 模型 | 文档 | 参数 | 说明 |
|---|---|---|---|
| Kimi-K2 | [Kimi-K2.md](Kimi/Kimi-K2.md) | 1T | 1T 总参 / 32B 激活的稀疏 MoE 大模型：61 层，注意力用 MLA（KV 缓存压成一个 512 维 latent + 64 维 rope key），FFN 用 DeepSeekMoE（1 个共享专家 + 384 个路由专家、每 token 激活 8 个），以 MuonClip 优化器在 15.5T token 上零训练不稳定，主打 agentic 工具调用。 |
| Kimi-K2.5 | [Kimi-K2.5.md](Kimi/Kimi-K2.5.md) | 1T（+0.4B 视觉编码器） | 在 Kimi-K2-Base 之上继续预训练约 15T 图文混合 token 得到的原生多模态 agentic 模型：文本主干与 K2 同构（61 层、MLA + DeepSeekMoE 384+1/top-8、1T 总参 / 32B 激活），外加 400M MoonViT 视觉编码器，上下文 128K→256K，支持 instant/thinking 双模式与 Agent Swarm。 |
| Kimi-K3 | [Kimi-K3.md](Kimi/Kimi-K3.md) | 2.8T（+0.4B 视觉编码器） | 2.8T 总参 / 104B 激活的原生多模态 agentic 模型：93 层，注意力是 KDA 线性注意力（69 层）与 Gated MLA（24 层）的混合，残差用 Attention Residuals（块大小 12），FFN 用 Stable LatentMoE（latent 3584、2 共享 + 896 路由、每 token 激活 16），上下文 1M，激活函数 SiTU-GLU。 |

## MiMo

| 模型 | 文档 | 参数 | 说明 |
|---|---|---|---|
| MiMo | [MiMo.md](MiMo/MiMo.md) | 7B | 面向推理的 7B decoder-only：主干与 Qwen2 同构（RoPE + RMSNorm + SwiGLU + GQA + Q/K/V bias，θ=640000），训练侧用三阶段数据配方、合成推理数据与 MTP 目标，从预训练到后训练全程为推理服务。 |
| MiMo-VL | [MiMo-VL.md](MiMo/MiMo-VL.md) | 7B（LLM）+ 约 0.6B（ViT + projector） | 7B 视觉语言模型：MiMo-7B（Qwen2 式主干）作 LLM，前置原生分辨率 ViT（32 层、窗口注意力 + 2D-RoPE）与 2×2 空间合并的 MLP projector，用 <|image_pad|> 占位后替换为视觉 embedding；四阶段预训练 + MORL 混合在线 RL。 |

## Qwen

| 模型 | 文档 | 参数 | 说明 |
|---|---|---|---|
| Qwen3 | [Qwen3.md](Qwen/Qwen3.md) | 0.6B–32B（另 MoE 30B-A3B / 235B-A22B） | 标准 Llama 式 decoder-only（RoPE + RMSNorm + SwiGLU + GQA），在主干上加了 QK-Norm，并把 RoPE 底数升到 1e6；dense 0.6B–32B 与 MoE 30B-A3B / 235B-A22B 共一套模板。 |
| Qwen3-Next | [Qwen3-Next.md](Qwen/Qwen3-Next.md) | 80B（81.3B） | 超稀疏 MoE（80B 总 / 3B 激活，512 专家 top-10）叠加混合注意力：每 4 层里 3 层是 Gated DeltaNet（门控线性注意力、O(1) 递归状态），第 4 层回落到带 QK-Norm 的全注意力 GQA，兼顾长文吞吐与全局建模。 |
| Qwen3.5 | [Qwen3.5.md](Qwen/Qwen3.5.md) | 397B（权重 403.4B） | 原生多模态（早期融合）旗舰 MoE：397B 总 / 17B 激活，沿用 Qwen3-Next 的混合注意力（Gated DeltaNet 线性层 + 每 4 层一次全注意力 GQA），并加入 MTP 与 mRoPE 支撑 256K 长文与视觉 token。 |
| Qwen3.8 | [Qwen3.8.md](Qwen/Qwen3.8.md) | 2.4T（2.446T） | Qwen-Max 级开源旗舰：2.4T 总 / 95B 激活的稀疏 MoE，沿用 Qwen3.5 的混合注意力（Gated DeltaNet + 周期全注意力）与 256K 上下文，主打长程 agent 任务的可靠完成与可调推理深度。 |
