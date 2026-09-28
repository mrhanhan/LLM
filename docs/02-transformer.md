# 02 · Transformer 模型

> 对应代码：`src/model.py`（RMSNorm / SwiGLU / Block / GPT）、`src/attention.py`（RoPE / Attention）

本项目是一个 **decoder-only 的 GPT 式语言模型**：只做「给定前文，预测下一个 token」。
这一章按数据流从输入到输出拆解结构，最后逐项算清参数量。

---

## 1. 总体数据流

```
token id (B, T)
   │  tok_emb: Embedding(vocab, d_model)
   ▼
x (B, T, d_model)
   │  N × Block：  x = x + Attn(RMSNorm(x))
   │               x = x + MLP (RMSNorm(x))
   ▼
RMSNorm
   │  lm_head: Linear(d_model, vocab)   ← 与 tok_emb 权重共享
   ▼
logits (B, T, vocab)
```

`GPT.forward` 就是这条流水线：

```python
x = self.embed(idx)                    # 1. 查表取 embedding
for block in self.blocks:              # 2. N 层 Block
    x, _ = block(x, self.cos, self.sin, ...)
x = self.norm_f(x)                     # 3. 最终归一化
logits = self.lm_head(x)               # 4. 投影到词表
```

- `B` = batch，`T` = 序列长度，`d_model=768`。
- `logits[b, t]` 是「第 t 个位置的下一个 token 是词表里每个 token」的分数。
- 有 `targets` 时直接算交叉熵 loss（见 `docs/03-training.md`）。

---

## 2. Block：Pre-Norm 残差结构

```python
class Block(nn.Module):
    def forward(self, x, cos, sin, past_kv=None, offset=0):
        h, new_kv = self.attn(self.n1(x), cos, sin, past_kv=past_kv, offset=offset)
        x = x + h                               # 残差连接
        x = x + self.mlp(self.n2(x))            # 残差连接
        return x, new_kv
```

- **Pre-Norm**：先归一化再送进子层（`Norm → Attn`），比 Post-Norm（`Attn → Norm`）训练更稳定，
  是现代 LLM 的标准做法。
- **残差连接** `x = x + f(x)`：给梯度一条「高速公路」，让深层网络也能顺畅回传，
  是能堆到几十层还训得动的前提。
- 每层做两件事：**注意力**（token 之间互相看）+ **前馈**（每个 token 各自做非线性变换）。

---

## 3. RMSNorm

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight
```

对比 LayerNorm：LayerNorm 要减均值、除标准差，再学缩放和平移两组参数；
**RMSNorm 只除以均方根（RMS），不减均值、不加 bias**：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}} \cdot \gamma$$

少一次均值统计和一组 bias，更快、更省显存，效果几乎不掉，因此被 LLaMA、Qwen 等广泛采用。
每层两个 RMSNorm，各含 `d_model=768` 个可学习缩放参数。

---

## 4. 因果多头注意力（`MultiHeadAttention`）

注意力让每个位置「查询」序列中其它位置的信息。`forward` 的步骤：

### 4.1 Q/K/V 投影 + 拆多头

```python
q = self.q_proj(x).view(B, T, self.n_head,    hd).transpose(1, 2)   # (B, n_head,    T, hd)
k = self.k_proj(x).view(B, T, self.n_kv_head, hd).transpose(1, 2)   # (B, n_kv_head, T, hd)
v = self.v_proj(x).view(B, T, self.n_kv_head, hd).transpose(1, 2)
```

`head_dim = d_model / n_head = 768 / 12 = 64`。把 768 维拆成 12 个 64 维的「头」，
每个头独立做注意力，从不同子空间捕捉不同关系。

### 4.2 RoPE：旋转位置编码

Transformer 的注意力本身不区分顺序，必须显式注入位置信息（否则「猫追狗」和「狗追猫」一样）。
RoPE 的做法是把**位置编码成旋转角度**，再对 Q/K 的每一对维度做二维旋转：

```python
def build_rope_cache(head_dim, max_seq, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq).float(), inv_freq)     # (T, head_dim/2)
    return freqs.cos(), freqs.sin()
```

把 `head_dim=64` 的维度两两配对成 32 对，第 `i` 对用频率 $1/\theta^{2i/d}$。位置 `t` 的旋转
矩阵由 `cos(t·ω)`、`sin(t·ω)` 决定，作用在向量对 $(x_1,x_2)$ 上：

```python
x1, x2 = x[..., 0::2], x[..., 1::2]
out = torch.stack([x1*c - x2*s, x1*s + x2*c], dim=-1)    # 标准 2D 旋转
```

**直觉**：位置越大的 token，旋转角越大。两个 token 的注意力打分本质上依赖它们的**相对旋转角**，
从而天然编码**相对位置**——这正是语言最需要的（「相邻」比「绝对第几个」更重要）。
`offset` 参数用于增量解码时把新 token 旋转到正确的位置。

### 4.3 因果 mask：为什么必须

Decoder 是自回归的：预测第 `t` 个 token 时，**只能看到 `t` 及之前的内容，不能偷看未来**。
如果不加 mask，训练时模型会直接把「答案」（下一个位置的真实 token）通过注意力抄过来，loss 假性降到 0，
一推理就废。

实现用上三角布尔矩阵把未来位置置为 `-inf`：

```python
mask = torch.triu(torch.ones(tq, tk, dtype=torch.bool), diagonal=tk - tq + 1)
att = att.masked_fill(mask, float("-inf"))
```

`softmax` 后这些位置的概率变为 0。测试 `tests/test_attention.py::test_causal_masking`
通过「篡改最后一个 token，检查前 9 个位置输出不变」来验证因果性正确。

### 4.4 缩放点积与输出投影

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)    # 打分并缩放
att = torch.softmax(att, dim=-1)                   # 归一化成权重
out = (att @ v).transpose(1, 2).reshape(B, T, -1)  # 加权求和并拼回
return self.o_proj(out), new_kv
```

除以 `sqrt(head_dim)` 是为了防止点积随维度增大而爆炸，导致 softmax 梯度消失。

---

## 5. GQA：用更少的 KV 头省显存

标准多头注意力里 Q/K/V 头数相同。`GQA`（Grouped-Query Attention）让 **V/K 头数少于 Q 头数**，
多个 Q 头共享一组 K/V：

```python
self.n_head, self.n_kv_head = 12, 4      # 每 3 个 Q 头共享 1 组 KV
...
if self.n_kv_head != self.n_head:
    rep = self.n_head // self.n_kv_head
    k = k.repeat_interleave(rep, dim=1)  # 计算时把 KV 复制到与 Q 同数
    v = v.repeat_interleave(rep, dim=1)
```

- **省在哪**：推理时的 KV 缓存。缓存大小 ∝ `n_kv_head`，从 12 降到 4，KV 缓存直接省 2/3。
- **代价**：表达能力略降，但实测几乎无损。
- 本项目默认 `n_kv_head=4`，是 LLaMA-2/3 系列的主流配置。

**KV 缓存**：自回归生成每步只新增一个 token，若每步都重算整段会很浪费。缓存历史 K/V：

```python
if past_kv is not None:
    k = torch.cat([pk, k], dim=2)     # 拼接历史
    v = torch.cat([pv, v], dim=2)
new_kv = (k, v)                        # 返回给下一步使用
```

显存估算（bf16，每 token）：`2 × n_layer × n_kv_head × head_dim × 2B`
= `2 × 12 × 4 × 64 × 2` = **12 KB/token**；1024 上下文约 **12.6 MB**。
若不用 GQA（12 个 KV 头），则是 3 倍。

---

## 6. SwiGLU 前馈网络

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))
```

$$\text{SwiGLU}(x) = W_3\big(\text{SiLU}(W_1 x) \odot W_2 x\big)$$

- `w1`、`w2` 把 `d_model` 升到 `d_ff`，`w3` 再降回 `d_model`。
- 门控 `SiLU(w1(x)) * w2(x)` 让网络能动态决定「放行多少信息」，比普通两层 MLP 更强，
  是 LLaMA、Qwen 等采用的 FFN 变体。
- 默认 `d_ff=2048`（为普通 MLP 的 4×768=3072 的约 2/3，因为 SwiGLU 多了 `w2` 一路，总参数相当）。

---

## 7. 权重共享（Weight Tying）

输入 embedding 表 `tok_emb` 的形状是 `(vocab, d_model)`，输出投影 `lm_head` 的形状是
`(d_model, vocab)`，两者形状互为转置（内容也近似互为「逆映射」），因此可以共享同一份权重：

```python
if cfg.tie_embeddings:
    self.lm_head.weight = self.tok_emb.weight      # 同一个 Parameter 对象
```

- **好处**：省下整整一份 `vocab × d_model` 参数。以 Qwen 词表计，省下约 **116.5M** 参数。
- 训练时两者一起更新；`num_params()` 中只计一次（PyTorch 会自动对重复参数去重）。
- 测试 `test_weight_tying` 用 `data_ptr()` 相等来验证确实共享同一块内存。

---

## 8. 参数量逐项计算

沿用默认配置 `d_model=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2048`，
`head_dim = 768/12 = 64`，词表 `vocab=151666`。

### 8.1 单层参数

| 子层 | 形状 | 参数量 |
|---|---|---|
| `q_proj` | 768 × (12×64=768) | 589,824 |
| `k_proj` | 768 × (4×64=256) | 196,608 |
| `v_proj` | 768 × 256 | 196,608 |
| `o_proj` | 768 × 768 | 589,824 |
| **注意力小计** | | **1,572,864** |
| `w1` (gate) | 768 × 2048 | 1,572,864 |
| `w2` (up) | 768 × 2048 | 1,572,864 |
| `w3` (down) | 2048 × 768 | 1,572,864 |
| **MLP 小计** | | **4,718,592** |
| `n1`/`n2` RMSNorm | 768 + 768 | 1,536 |
| **单层合计** | | **6,292,992** |

### 8.2 整体参数

| 组件 | 计算 | 参数量 |
|---|---|---|
| N 层 Block | 6,292,992 × 12 | 75,515,904 |
| `norm_f` | 768 | 768 |
| **主干（不含词表）** | | **75,516,672 ≈ 75.5M** |
| 词表 embedding（与 `lm_head` 共享） | 151666 × 768 | 116,479,488 ≈ 116.5M |
| **总计** | | **191,996,160 ≈ 192.0M** |

> 这就是设计文档里「主干约 80M、含 Qwen 词表约 200M」的来源。若关掉权重共享，
> 总计会变成约 308.5M。跑训练脚本时控制台会打印实际值：
> `模型参数量 192.0M（不含词表 75.5M）`。

### 8.3 换个 16k BPE 词表

同一主干换 `vocab=16384`：embedding = 16384 × 768 ≈ 12.6M，总计约 **88.1M**。
对比之下可见，词表大小几乎完全决定了 embedding 的参数量。

---

## 9. 初始化与优化器分组（`GPT` 的辅助方法）

```python
def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)      # 小方差正态
```

所有 Linear/Embedding 用 `std=0.02` 的小方差正态初始化，帮助残差网络稳定起步。

`configure_optimizers` 把参数分成两组交给 AdamW：

```python
if p.dim() >= 2:  decay.append(p)       # 权重矩阵 → 做 weight decay
else:             no_decay.append(p)    # norm 的 γ、bias → 不衰减
```

**为什么 norm 参数不衰减**：weight decay 会把参数往 0 拉，而 RMSNorm 的 `γ` 本身就该是
1 附近，衰减它没有意义还可能伤害训练，所以单独分组、`weight_decay=0`。CUDA 可用时用
`fused=True` 融合实现加速。

---

## 10. 小结

- 一条主流水线：Embedding → N×(Norm+Attention+残差 / Norm+FFN+残差) → Norm → 输出投影。
- RMSNorm 更快、Pre-Norm + 残差让深层可训；SwiGLU 是更强的 FFN。
- RoPE 用旋转编码相对位置，因果 mask 保证不偷看未来，GQA 大幅缩小 KV 缓存。
- 权重共享把 embedding 与输出投影合并，省下约 116.5M 参数。
- 默认配置：主干 ≈ 75.5M，含 Qwen 词表总计 ≈ 192M。
