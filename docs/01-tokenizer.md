# 01 · 分词器（Tokenizer）

> 对应代码：`src/tokenizer.py`、`src/data.py`
> 运行脚本：`scripts/prepare_data_part1.py`、`scripts/train_tokenizer.py`、`scripts/prepare_data_part2.py`

语言模型本质是一个「在离散符号序列上做下一 token 预测」的模型，它不认识文字，只认整数 id。
**分词器就是「文本 ⇄ 整数序列」的翻译器**，它决定了模型的「字母表」有多大、一句话会变成多长。
这一章从原理讲到本项目三套分词器的实现与对比方法。

---

## 1. 为什么需要分词

神经网络只处理数字。把一句话喂给模型前，必须先切成若干 **token**，每个 token 映射成一个整数 id：

```
"小猫在跑"  →  ["小", "猫", "在", "跑"]  →  [1024, 2048, 30, 55]
```

切分粒度的选择（**词表设计**）是一组权衡：

| 粒度 | 词表大小 | 序列长度 | 问题 |
|---|---|---|---|
| 词（word） | 极大（百万级） | 短 | 大量未登录词 OOV，中文尤其严重 |
| 字符（char） | 小（几千） | 长 | 序列太长，模型要记更多步 |
| 子词（subword / BPE） | 中等（几万） | 适中 | 兼顾两者，业界主流 |

本项目三种都实现了：**字节级 BPE**（教学核心）、**字符级**（最简单的对照）、
**Qwen 子词分词器**（默认使用，效果最好）。

---

## 2. 字符级分词器（`CharTokenizer`）

最简单：一个字符 = 一个 id。训练时扫一遍语料，把所有出现过的字符收集起来：

```python
class CharTokenizer(_BaseTokenizer):
    def encode(self, text, add_bos=False, add_eos=False):
        unk = self.special_id("unk")
        ids = [self._char_to_id.get(c, unk) for c in text]   # 没见过的字符 → <unk>
        ...
```

- 词表 = 特殊符号 + 语料中出现过的所有字符（中文常用字约 3000~6000）。
- 优点：直观、词表小；缺点：序列长，且**没见过的字符会退化成 `<unk>`，不可逆**。
- 用途：作为理解分词的最小对照实验。

---

## 3. 字节级 BPE（`BPETokenizer`，教学核心）

BPE（Byte Pair Encoding）原本是一种数据压缩算法，GPT-2 之后成为 LLM 的主流分词方案：

1. **预切分**：先用正则把文本切成大致独立的「词块」（英文单词、单个汉字、数字、符号、空白）；
2. **转字节**：每个词块用 UTF-8 编码成字节序列，初始符号就是这 256 个「字节」；
3. **统计 + 合并**：反复统计相邻符号对的频率，把**最高频的一对合并成一个新符号**；
4. 重复 `vocab_size - 初始符号数` 次，得到一个 `merges` 表；
5. **编码**：对输入字节序列，按 merges 的先后顺序（rank 小者优先）不断合并。

### 3.1 预切分正则

```python
_PRE_TOKEN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[\u4e00-\u9fff\u3400-\u4dbf]| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]|\s+"
)
```

注意 `\w` 在 Unicode 下会把汉字也算进去，所以**汉字要显式列出**（`\u4e00-\u9fff` 等）。
这样「一个汉字」会被单独切出来，BPE 合并时才可能把常用词（如「我们」）合并成一个 token。

### 3.2 merges 表的合并过程

训练循环（`BPETokenizer.train`）的核心：

```python
for step in range(target_merges):
    pair_freq = Counter()
    for w, f in zip(words, freqs):          # 每个词 × 其出现频率
        for p, c in _get_pairs(w).items():
            pair_freq[p] += c * f           # 统计相邻符号对的总频次
    best_pair, best_count = pair_freq.most_common(1)[0]
    if best_count < min_frequency:
        break                                # 低于阈值就停止，避免为噪声建表
    new_id = byte_offset + 256 + len(merges)
    merges.append(best_pair)
    vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]   # 新符号 = 两个子串拼接
    words = [_merge(w, best_pair, new_id) for w in words]       # 在原词里就地替换
```

一个简化例子（假设语料里 `a b` 出现最多）：

```
初始：  "ab" -> [a, b]        "abc" -> [a, b, c]
merge1 (a,b)->"ab":  "ab" -> [ab]   "abc" -> [ab, c]
merge2 (ab,c)->"abc": "abc" -> [abc]
```

`merges` 的顺序就是**合并优先级**（rank）。编码一个新句子时，只要按这个顺序做合并即可：

```python
pair = min((p for p in pairs if p in self.merge_ranks),
           key=lambda p: self.merge_ranks[p], default=None)   # rank 最小者优先
```

### 3.3 词表布局（id 怎么分配）

`BPETokenizer` 的 id 布局是固定的三段：

```
[ 特殊符号 0..S-1 ][ 256 个字节 S..S+255 ][ merges 新符号 S+256 .. ]
       S = len(specials) = 5                         共 len(merges) 个
vocab_size = (S + 256) + len(merges)
```

- 初始符号用「全局 id」`byte_offset + 字节值`，编码与训练时保持一致；
- merge 产生的新符号 id = `base_size + merge_rank`。

### 3.4 为什么字节级 BPE 不会 OOV

因为**任何 UTF-8 文本最终都能拆成字节**，而 256 个字节都在词表里。即使一个汉字或 emoji
从未在训练语料出现过，它也只是被拆成多个字节 token，而不是变成 `<unk>`。这就是字节级的最大好处，
也天然适合中文。

### 3.5 `decode(encode(x)) == x` 的意义

字节级 BPE 是**可逆**的：`encode` 只是把字节序列做无损合并，`decode` 时把每个 token 的字节
拼回去再按 UTF-8 解码即可（见 `decode` 里的 `bytearray` 缓冲逻辑）。

```
原文 x --encode--> id 序列 --decode--> 原文 x    （必须完全相等）
```

这个性质是分词的**正确性底线**：如果往返不相等，说明分词器丢了信息，训练时的标签就和原文对不上。
本项目 `tests/test_tokenizer.py` 用 `assert tok.decode(tok.encode(s)) == s` 专门守住这条底线。

> 细节：特殊符号在 `decode` 时会被还原成它们的字面串（如 `<eos>`），且它们不参与 BPE 合并。

---

## 4. 特殊符号

`SPECIALS` 是一张名称到字面串的表：

```python
SPECIALS = {"pad": "<pad>", "bos": "<bos>", "eos": "<eos>", "unk": "<unk>", "image": "<image>"}
```

| 符号 | 作用 |
|---|---|
| `<pad>` | 批处理时把短序列补齐到等长 |
| `<bos>` | 序列开始 |
| `<eos>` | 序列结束；预处理时**每篇末尾追加一个 `<eos>`**，让模型学会在哪里停 |
| `<unk>` | 未登录字符（字节级 BPE 实际用不到，字符级会用） |
| `<image>` | 视觉占位符，为后续多模态阶段预留 |

特殊符号在词表里占用最前面的几个 id，**不参与 BPE 合并**。

---

## 5. Qwen 分词器（`QwenTokenizer`，默认使用）

`QwenTokenizer` 把 `Qwen/Qwen2.5-0.5B` 的分词器文件**下载到项目内**，并用
`AutoTokenizer.from_pretrained(本地路径)` 加载，**不读 HF 缓存**：

```python
def download_qwen_tokenizer(dest_dir):        # src/data.py
    snapshot_download(repo_id="Qwen/Qwen2.5-0.5B",
                      local_dir=dest_dir,
                      allow_patterns=_TOKENIZER_FILES)   # 只拉分词器相关文件
```

加载时额外注册 `<image>` 特殊符号并**持久化回本地目录**，保证重启后仍然存在：

```python
@classmethod
def load(cls, local_dir, add_image_token=True):
    hf = AutoTokenizer.from_pretrained(local_dir)
    if add_image_token and "<image>" not in hf.get_vocab():
        hf.add_special_tokens({"additional_special_tokens": ["<image>"]})
        hf.save_pretrained(local_dir)
    return cls(hf, hf.convert_tokens_to_ids("<image>"))
```

**真实词表大小：151666**（含 `<image>`；`<eos>`/`<bos>` id 都是 151643，`<image>` 是 151665）。

> 注意：`configs/gpt_tinystories.yaml` 里写的 `vocab_size: 151936` 只是名义默认值；
> `scripts/train_gpt.py` 和 `scripts/chat.py` 会在运行时用 `cfg.model.vocab_size = tok.vocab_size`
> 覆盖它，真正生效的是 151666。

---

## 6. 编码完成后的数据落盘

`scripts/prepare_data_part2.py` 把文本编码成二进制：

```python
def build_text_bin(tokenizer, texts, out_path, val_bin_path=None,
                   max_docs=None, val_ratio=0.01):
    eos = tokenizer.special_id("eos")
    ids = []
    for i, t in enumerate(texts):
        ids.extend(tokenizer.encode(t))
        ids.append(eos)                       # 每篇补 <eos>
    n_val = int(len(ids) * val_ratio)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    np.array(train_ids, dtype=np.uint32).tofile(out_path)   # 必须 uint32
```

- **为什么用 `uint32`**：Qwen 词表 151666 > 65535，`uint16` 会溢出。
- **为什么先全量编码**：分词是 CPU 瓶颈，训练时反复分词会很慢；一次性编码成 memmap，
  训练时只做随机切片（`load_bin` 用 `np.memmap`，不占大内存）。
- **语言字段**：`iter_texts` 优先取中文字段 `story_zh`，避免把英文原文喂给中文语料。

---

## 7. Qwen vs 手写 BPE：对比实验方法

本项目两套分词器可以通过 `configs/gpt_bpe.yaml` 切换，用来做定性/定量对比。
推荐的实验步骤：

**步骤 1：用同一批语料训练一个 16k BPE 词表**（纯 Python 较慢，建议先取子集）：

```powershell
# 内部等价于 BPETokenizer.train(iter_texts(files), vocab_size=16384, min_frequency=2)
& ".venv\Scripts\python.exe" scripts/train_tokenizer.py --max_bytes 3000000
# 产物：data/tokenizer/bpe_16k/{merges.json, vocab.json, specials.json}
```

> 说明：目标词表由 `--vocab_size`（默认 16384）控制；若语料子集较小，`min_frequency=2`
> 会在高频对被合并完后提前停止，实际词表可能小于目标。

**步骤 2：用 BPE 分词器重新编码一份数据**

```powershell
& ".venv\Scripts\python.exe" scripts/prepare_data_part2.py --tokenizer_kind bpe `
  --tokenizer_dir data/tokenizer/bpe_16k `
  --out data/processed/text/bpe_train.bin --val data/processed/text/bpe_val.bin --max_docs 2000
```

`prepare_data_part2.py` 的 `--tokenizer_kind {qwen,bpe,char}` 默认 `qwen`，因此原有 Qwen
流程不受影响；切换为 `bpe` 后，编码与特殊符号都来自 BPE 词表，id 语义与
`configs/gpt_bpe.yaml` 完全一致。

**步骤 3：对比这些指标**

| 指标 | Qwen2.5 | 手写 BPE 16k | 说明 |
|---|---|---|---|
| 词表大小 | 151666 | 16384 | 词表越大 embedding 越占参数 |
| 压缩率（字/token） | 高 | 中 | 同一句话的 token 数越少越省算力 |
| OOV 率 | 0 | 0 | 两者都是字节级 |
| `decode(encode(x))==x` | ✅ | ✅ | 正确性底线 |
| 词表训练耗时 | 0（预训练） | 较慢（纯 Python） | 教学代价 |
| embedding 参数量 | 151666×768 ≈ 116.5M | 16384×768 ≈ 12.6M | 相差近 10 倍 |

**结论**：Qwen 词表大、压缩率高、效果最好，所以**默认使用**；手写 BPE 词表小、可完全掌控，
适合观察 BPE 的合并过程，并用 `gpt_bpe.yaml` 做「词表大小 → 参数量」的对比。

> ✅ 已接通：`scripts/train_tokenizer.py` 负责训练并保存 BPE 词表，
> `scripts/prepare_data_part2.py --tokenizer_kind bpe` 负责用 BPE 重新编码数据，
> `configs/gpt_bpe.yaml` 已指向 `data/processed/text/bpe_{train,val}.bin`，
> 直接 `scripts/train_gpt.py --config configs/gpt_bpe.yaml` 即可训练。

---

## 8. 小结

- 分词 = 文本与整数 id 之间的可逆翻译；粒度选择是一组「词表大小 vs 序列长度」的权衡。
- 字节级 BPE 通过「统计高频相邻对 → 合并」构建词表，字节级保证永不 OOV、天然支持中文。
- `decode(encode(x)) == x` 是分词器的正确性底线，必须由测试守住。
- 本项目默认用 Qwen 分词器（词表 151666），同时保留手写 BPE / 字符级用于教学与对比。
