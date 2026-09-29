# 07 · 3D 权重可视化（模型检视 / 实时训练 / 架构浏览器）

`viz/` 是一个本地网页应用：**Python 后端 + Three.js 前端**，把模型内部的权重与
激活渲染成可交互的 3D 图形。设计参考 [bbycroft.net/llm](https://bbycroft.net/llm)
的"**竖直层叠架构 + 可展开权重矩阵**"范式：整模型像一座**书架**从上到下逐层堆叠，
点开某层即可看到该层的权重矩阵由**带数值、颜色与连线的方块**组成。

- 后端：`viz/server.py`（FastAPI + WebSocket）
- 数据层：`viz/graph.py`（统一模型图）、`viz/weights.py`（权重网格 / 纹理 / 元素查询）
- 统计：`viz/stats.py`（权重统计、更新幅度、激活 hook）、`viz/neurons.py`（神经元点云 PCA）
- 运行时：`viz/runtime.py`（tiny GPT + 内置语料、逐 token 推理）、`viz/datasets.py`（数据集注册表）、`viz/train_engine.py`（simple / hifi 训练引擎）
- 架构：`viz/arch.py` + `viz/architectures/*.json`
- 前端：`viz/static/`（`index.html` + `style.css` + `js/{main,shelf,matrix,links,panel,cloud,colors,api,arch,kv,toplabels}.js` + vendored Three.js/KaTeX）

> 旧的 `viz/topology.py` 与 `viz/static/app.js` 已在本轮重构中删除，
> 其能力分别并入 `viz/graph.py` / `viz/stats.py` 与模块化的 `viz/static/js/`。

## 1. 启动

```powershell
# 默认进入「实时训练」：内置 tiny 小模型（d_model=64 / 4 层，Qwen 词表），几秒即可看到权重变化
& ".venv\Scripts\python.exe" viz/server.py
# 浏览器打开 http://127.0.0.1:7861
```

可选参数：

```powershell
# 换个端口 / 调训练超参
& ".venv\Scripts\python.exe" viz/server.py --port 8000 --lr 2e-3 --batch 32 --block 48

# 冒烟构建（不启动服务，只验证数据层，打印一个 dict 后退出 0）
& ".venv\Scripts\python.exe" viz/server.py --check

# 检视真实 checkpoint 的权重分布（约 1.9 亿参数；关闭实时训练，切换到「真实模型检视」模式）
& ".venv\Scripts\python.exe" viz/server.py --ckpt out/gpt/latest.pt --config configs/gpt_tinystories.yaml
```

## 2. 三个模式

顶栏在三种模式间切换；模式决定 3D 主视图的数据来源：

| 模式 | 数据源 | 说明 |
|---|---|---|
| **真实模型检视** | 本地 checkpoint（`--ckpt`） | 静态权重网格 + 逐 token 推理流动，主力视图 |
| **实时训练** | 内置 tiny / `--ckpt` 192M / 随机初始化 192M | 实时权重变化、loss 曲线与 L3 流带；训练目标、数据集、超参与引擎由左栏选择（见 3.9） |
| **架构浏览器** | `viz/architectures/*.json` | `viz/architectures/` 下的全部真实模型规格（当前 17 个）的**仅结构 / 公式 / 源码**（无权重） |

> 未用 `--ckpt` 启动时，服务端 `has_ckpt=false`，前端会自动禁用顶栏「真实模型检视」
> 按钮并给出提示；此时直接请求 `source=ckpt` 的接口返回 `400` 及友好错误信息。


> 「推理播放」不是独立模式，而是检视 / 训练模式下均可触发的动作
> （检视用真实模型、训练用小模型）。

两套权重数据源各自独立加载、互不干扰；真实 checkpoint **仅在检视模式懒加载**，
默认进入轻量的实时训练模式。

## 3. 界面与交互

```
┌──────────────────────────────────────────────────────────────┐
│ 顶栏  brand │ 模式[检视·训练·架构] │ 模型/checkpoint │ 状态        │
├───────────┬──────────────────────────────────┬───────────────┤
│ 左栏       │                                  │ 右抽屉(标签页)  │
│ 训练控制    │     3D 书架主视图 (WebGL)         │ 矩阵/结果/      │
│ / 架构选择  │  层托盘·矩阵板·连线·流带·缓存条    │ KV/TopN        │
├───────────┴──────────────────────────────────┴───────────────┤
│ 底栏  播放器(▶/⏭) │ loss │ 注意力 │ topN │ 悬停数值            │
└──────────────────────────────────────────────────────────────┘
```

### 3.1 竖直书架（布局 C1+C2）

- 每一层是一块**薄板（slab）+ 层标签**，沿 Y 轴从上到下堆叠：
  `词嵌入 → 层0…层N → 最终 Norm + 输出头`。
- **点击某层**展开该层（同一时刻**只展开一层**，其余层降到低透明度），
  再点一次收起；薄板透明度做缓动过渡。
- 相机操作：**左键拖拽旋转**、**右键 / 中键拖拽平移**、`Shift`（或 `Ctrl`）+ 左键平移、
  滚轮缩放；键盘 `WASD` / 方向键平移；屏幕左侧 HUD（`▲ ◀ ⌖ ▶ ▼`）平移与一键复位。

### 3.2 层内：单排流水线（M1）+ 分块聚合下钻（M2）

- 展开后，该层的矩阵按 **M1 单排流水线** 排布：
  `Norm1 → Q K V O → Norm2 → W1 W2 W3`（含最终 Norm / 输出头）。
- 方块语义按矩阵元素数 **≤ 4096（约 64×64，`cube_threshold`）** 判定：
  - **小矩阵**用 `InstancedMesh` 真 3D 立方块（一方块 = 一权重）；
  - **大矩阵**用服务端生成的发散色阶 **PNG 热力图纹理**贴到平面上，默认按 N×N
    分块聚合（`tiles` 默认 64、上限 256），可下钻到元素级。
- **悬停**矩阵板：由射线命中的 UV 换算格 / 块行列索引，去抖动后经
  `/api/matrix/{name}/cell` 取值，显示在底栏（`i=… j=… value=…`）。
- **点板**打开右侧抽屉（见 3.4）。

### 3.3 连接（L1 / L2 / L3）

- **L1 结构连线**（常态）：层与层之间的 `spine` 连线（蓝，粗），
  以及展开层内部的 `inner` 连线（绿），线粗与亮度随权重范数实时变化。
- **L2 神经元连线**：仅在下钻到**小矩阵**时按需开启，展示 out / in 神经元之间的
  强连接（见 3.5）。
- **L3 数据流带**：推理时相邻层托盘之间叠加半透明流带，颜色 / 透明度随该层
  代表矩阵的**激活值**推进并在 token 间缓动。

### 3.4 右抽屉（标签页：矩阵 / 结果 / KV Cache / TopN）

点击矩阵板后右侧抽屉打开，顶部四个标签页：

- **矩阵**：矩阵全名、标签、`shape = out × in`、role 与元素数；2D 热力图（服务端
  `/api/matrix/{name}/grid` 取分块网格，客户端按发散色阶着色）；色阶图例与
  「适配 / 块 / 元素」三档网格粒度（192 / 64 / 256）。对小矩阵另有「点云」与
  「神经元连线」按钮（见 3.5）。
- **结果**：推理生成文本（prompt 灰、续写亮，多行自动换行可滚动），以及推理参数
  `温度 / Top-K / Top-P / 最大 token / TopN / 种子`（留空用默认）。
- **KV Cache**：逐层缓存长度、估算显存与 K / V 热力图，随推理逐 token 增长。
- **TopN**：输出层每一步的 topN 候选词概率条形图（黄色为选中词）。

### 3.5 神经元点云

- 在抽屉里对小矩阵点「点云」：后端 `/api/neurons` 采样最多 160 个输入/输出神经元，
  权重向量经 **PCA 投到 3D**，点颜色 = 权重范数；
- 点「神经元连线」：`|w_ij|` 越大线越粗，暖色 = 正权重、冷色 = 负权重。

### 3.6 颜色

- **发散色阶**：正 = 暖红、负 = 冷蓝、近零 = 暗（`colors.js` / `weights.py` 的
  `diverging_rgb` 前后端一致）。
- 归一化可选「**全局**（所有矩阵共用一个 absmax，默认）/ **本矩阵**」，
  由 `/api/matrix/{name}/png?norm=global|matrix` 决定；图例标注当前量程。
- 权重态（`norm` / `delta`）与激活态（`act`）各有一套映射与图例。

### 3.7 训练 / 推理联动

- 点「开始训练」→ 后端线程按间隔把每个矩阵的
  `norm / mean / max / delta`（delta = 相对上一步的更新幅度）经 WebSocket 的
  `tick` 推给前端；前端刷新矩阵板颜色 / 自发光与连线粗细，并滚动 **loss 曲线**。
- **训练目标可选**（左栏「模型」下拉）：
  - `live`：内置 tiny 小模型（d_model=64 / 4 层，Qwen 词表），几秒见效；
  - `ckpt`：在 `--ckpt` 加载的 **192M** 权重上继续微调（与「真实模型检视」共享同一模型，
    训练能直接改变所见权重）；
  - `scratch`：按 `configs/gpt_tinystories.yaml` 的 `model` 段随机初始化一个同结构
    **192M** 模型从头训练。
  `ckpt` / `scratch` 使用 Qwen 分词器；训练数据由左栏数据集下拉选择（见 3.9），
  超参取自左栏；有 GPU 时自动用 cuda。
- **训练一启动即可见**：折叠状态下每块**层薄板**按其**层内矩阵聚合**的 norm / delta
  持续变色（不必先点开某层）；首个 `tick` 还会**自动展开当前变化最大的一层**，
  使其矩阵板实时刷新。
- 点「推理 ▶ / 单步 ⏭」→ 后端逐 token 生成，每步用 forward hook 读取各模块
  **激活值**，并以 `token` 消息推给前端：L3 流带点亮推进、注意力分布条刷新、
  右侧抽屉显示**生成文本** + **输出层 topN 概率** + **逐层 KV cache**；
  3D 场景里每层旁出现**缓存长条**（长度 ∝ 已缓存 token 数、颜色 ∝ 数值），
  输出头旁浮出 **topN 词标签**，左下角出现 **KV 占用浮层**（总长度 / 估算显存 / 逐层柱状）。
- 推理参数（温度 / Top-K / Top-P / 最大 token / TopN / 种子）在抽屉「结果」页配置，
  经 `POST /api/infer` 透传给 `stream_generate`。

### 3.8 架构浏览器

把 `viz/architectures/` 下的全部真实模型规格（当前 17 个，含 GLM-4/4.5/5/5.3/5.3-Flash、Qwen3/3.5/3.8/3-Next、MiMo/MiMo-VL、
Kimi K2/K2.5/K3、DeepSeek V3/V3.2/V4.1）**逐层**三维铺开：每个方块 = 一层，
颜色 = 该层注意力 / FFN 类型。顶部下拉选择模型，**点击任意一层或模块**，右侧抽屉用
KaTeX 渲染该层公式并附对应源码。数据来自 `viz/architectures/<id>.json`，与
`docs/models/**/*.md` 同源生成，不会不一致。

完整逐层文档见 [models/README.md](models/README.md)；所有模型横向对比
（参数 / 层数 / 注意力 / MoE / 上下文，可排序过滤）见
[models/compare.html](models/compare.html)，或在服务里访问 `/compare`。

### 3.9 训练配置

左栏「实时训练」面板可在发起训练前配置：

- **模型**（`train-target`）：`live` 内置 tiny（d_model=64 / 4 层，Qwen 词表）、
  `ckpt` 在 `--ckpt` 的 192M 权重上继续微调、`scratch` 按 config 随机初始化 192M 从头训练。
- **训练方式**（`train-mode`）：`pretrain`（next-token）或 `sft`（只监督助手回复）；
  数据集下拉按方式过滤。
- **训练数据集**（`train-dataset`）：来自 `GET /api/datasets` 的写死注册表（见下），
  文件缺失的项标「（文件缺失）」并禁用。
- **超参**：最大步数 `max_steps`、学习率 `lr`、批大小 `batch_size`、梯度累积 `grad_accum`。
- **高保真模式**（`hp-hifi`）：勾选用 `hifi` 引擎，否则用 `simple` 引擎。

数据集注册表（`viz/datasets.py` 的 `DATASETS`，写死不扫描目录）：

| id | 标签 | 方式 | 文件 |
|---|---|---|---|
| `corpus_qwen` | 内置中文语料（最快） | pretrain | 内置 `viz.runtime.CORPUS`（无文件） |
| `tiny_stories` | TinyStories | pretrain | `data/processed/text/train.bin` |
| `poetry` | 中文诗词 | pretrain | `data/processed/text/poetry.train.bin` |
| `advertise` | AdvertiseGen | pretrain | `data/processed/text/advertise.train.bin` |
| `fineweb_cmn` | FineWeb 中文（大文件） | pretrain | `data/processed/text/fineweb_cmn.train.bin` |
| `sft_chat` | Alpaca+Firefly 多轮 | sft | `data/processed/sft/train.npz` |
| `sft_poetry` | 诗词续写 | sft | `data/processed/sft/poetry.train.npz` |
| `sft_advertise` | 广告文案 | sft | `data/processed/sft/advertise.train.npz` |

所有数据集都用 **Qwen 分词器**；`corpus_qwen` 是唯一内置项（无需数据文件）。

两种训练引擎（`viz/train_engine.py`）：

- **`simple`**（默认，`SimpleEngine`）：朴素 fp32 循环，可随时中断，逐 step 回调；
  不启用 bf16，最快看到权重变化。
- **`hifi`**（`HifiTrainer` / `HifiSFTTrainer`）：复用 `src.trainer.Trainer` / `SFTTrainer`，
  bf16 + warmup（`warmup_steps = max(1, max_steps // 20)`），通过子类注入 `on_step`
  回调与可中断线程。192M + `hifi` + 大 SFT 序列可能吃显存，可下调 `--sft-max-len`
  （默认 256）。

保存：停止训练时把权重存到
`out/viz/<target>-<mode>-<dataset>-<YYYYmmdd-HHMMSS>.pt`
（如 `out/viz/live-pretrain-corpus_qwen-20260929-101530.pt`）。

实时刷新分级节流（前端 `main.js`）：

- **3D 主视图每 tick 刷新**：`shelf.updateValues` + `links.update` + loss 曲线；
- **右抽屉约 500ms 节流**：`panel.refreshLive()`；
- **展开层纹理每 20 tick 刷新一次**：`shelf.refreshTextures(step)`。

## 4. HTTP / WebSocket 协议

### 4.1 HTTP 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | `index.html` |
| GET | `/api/model?source=ckpt\|live\|scratch` | 模型图（layers / modules / matrices / connections）+ 全局色阶 `global_absmax` |
| GET | `/api/matrix/{name}/grid?source=&tiles=64` | 分块聚合二维网格（每块均值），返回 `{shape, grid, values}` |
| GET | `/api/matrix/{name}/png?source=&tiles=&norm=global\|matrix` | 发散色阶 PNG 纹理（客户端当贴图用） |
| GET | `/api/matrix/{name}/stats?source=` | 该矩阵 `{shape, vmin, vmax, absmax, mean, std}` |
| GET | `/api/matrix/{name}/cell?i=&j=&source=` | 单元素数值 |
| GET | `/api/matrix/{name}/patch?r=&c=&h=&w=&source=` | 元素子块（下钻） |
| GET | `/api/neurons?matrix=&source=&n=&top=` | PCA 神经元点云 |
| GET | `/api/arch/list` · `/api/arch/{id}` | 架构浏览器列表 / 详情 |
| GET | `/compare` | 模型横向对比页（`docs/models/compare.html`） |
| GET | `/api/datasets` | 数据集注册表（`{id, label, kind, tokenizer, available, note}`） |
| POST | `/api/train/start` · `/api/train/stop` | 触发 / 停止训练；start body `{target: live\|ckpt\|scratch, mode: pretrain\|sft, dataset, engine: simple\|hifi, params:{max_steps, lr, batch_size, grad_accum}}`（依次默认 `live` / `pretrain` / `corpus_qwen` / `simple`）；stop 返回保存路径 `{saved}` |
| POST | `/api/infer` | 触发逐 token 推理，body `{source, prompt, max_new_tokens, temperature, top_k, top_p, top_n, seed}` |

### 4.2 WebSocket `ws://host/ws`

| type | 内容 |
|---|---|
| `init` | `{has_ckpt, source}`（轻量握手；完整模型图由前端另行请求 `/api/model?source=`） |
| `tick` | `{step, loss, lr, target, mode, dataset, engine, total_steps, values:{matrixName:{norm, mean, max, n, delta}}}` |
| `token` | `{token, id, text, values:{matrixName:{act}}, attn:[...], topn:[{id,token,prob}], kv:{n_layer,total_len,total_bytes,layers:[{layer,len,bytes,heads,head_dim,k:{rows,cols,values},v:{...}}]}}` |
| `status` | `{training: bool}` |

> WS 连接建立后立即收到一条轻量 `init`（含 `has_ckpt`，用于禁用/启用「真实模型检视」），
> 随后是被动推送的 `tick` / `token` / `status`。

## 5. 性能策略

- 大矩阵走**纹理平面**（`DataTexture` / `CanvasTexture`），`tiles` 默认 64、上限 256；
- 小矩阵走 **`InstancedMesh`** 立方块，面积阈值 `cube_threshold=4096`（可配）；
- 连线只需 `spine`（始终）与展开层的 `inner`；连接用 top-k / LOD 控制；
- 同一时刻只展开一层，几何量有界；
- checkpoint **懒加载**，仅检视模式触发；实时训练默认用内置 tiny 小模型（约 985 万参数，Qwen 词表）。

## 6. 相关文档

- 模型结构：[02-transformer.md](02-transformer.md)
- 与 Qwen3 / DeepSeek 的架构对比：[06-architecture-compare.md](06-architecture-compare.md)
- **真实模型逐层架构文档**：[models/README.md](models/README.md)
- 架构 JSON schema 与新增模型：`viz/architectures/schema.md`、`scripts/gen_model_docs.py`
