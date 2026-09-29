# 07 · 3D 权重可视化（模型检视 / 实时训练 / 架构浏览器）

`viz/` 是一个本地网页应用：**Python 后端 + Three.js 前端**，把模型内部的权重与
激活渲染成可交互的 3D 图形。设计参考 [bbycroft.net/llm](https://bbycroft.net/llm)
的"**竖直层叠架构 + 可展开权重矩阵**"范式：整模型像一座**书架**从上到下逐层堆叠，
点开某层即可看到该层的权重矩阵由**带数值、颜色与连线的方块**组成。

- 后端：`viz/server.py`（FastAPI + WebSocket）
- 数据层：`viz/graph.py`（统一模型图）、`viz/weights.py`（权重网格 / 纹理 / 元素查询）
- 统计：`viz/stats.py`（权重统计、更新幅度、激活 hook）、`viz/neurons.py`（神经元点云 PCA）
- 运行时：`viz/runtime.py`（小字符级 GPT、实时训练线程、逐 token 推理）
- 架构：`viz/arch.py` + `viz/architectures/*.json`
- 前端：`viz/static/`（`index.html` + `style.css` + `js/{main,shelf,matrix,links,panel,cloud,colors,api,arch}.js` + vendored Three.js/KaTeX）

> 旧的 `viz/topology.py` 与 `viz/static/app.js` 已在本轮重构中删除，
> 其能力分别并入 `viz/graph.py` / `viz/stats.py` 与模块化的 `viz/static/js/`。

## 1. 启动

```powershell
# 默认进入「实时训练」：内置字符级小模型（d_model=64 / 4 层），几秒即可看到权重变化
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
| **实时训练** | 内置字符级小模型 | 实时权重变化、loss 曲线与 L3 流带 |
| **架构浏览器** | `viz/architectures/*.json` | `viz/architectures/` 下的全部真实模型规格（当前 17 个）的**仅结构 / 公式 / 源码**（无权重） |

> 「推理播放」不是独立模式，而是检视 / 训练模式下均可触发的动作
> （检视用真实模型、训练用小模型）。

两套权重数据源各自独立加载、互不干扰；真实 checkpoint **仅在检视模式懒加载**，
默认进入轻量的实时训练模式。

## 3. 界面与交互

```
┌──────────────────────────────────────────────────────────────┐
│ 顶栏  brand │ 模式[检视·训练·架构] │ 模型/checkpoint │ 状态        │
├───────────┬──────────────────────────────────┬───────────────┤
│ 左栏       │                                  │ 右抽屉(2D)     │
│ 训练控制    │     3D 书架主视图 (WebGL)         │ 选中矩阵热力图  │
│ / 架构选择  │  层托盘·矩阵板·连线·流带          │ 图例/统计/公式/代码│
├───────────┴──────────────────────────────────┴───────────────┤
│ 底栏  推理播放器(▶/⏭) │ 训练 loss 曲线 │ 注意力分布 │ 悬停数值      │
└──────────────────────────────────────────────────────────────┘
```

### 3.1 竖直书架（布局 C1+C2）

- 每一层是一块**薄板（slab）+ 层标签**，沿 Y 轴从上到下堆叠：
  `词嵌入 → 层0…层N → 最终 Norm + 输出头`。
- **点击某层**展开该层（同一时刻**只展开一层**，其余层降到低透明度），
  再点一次收起；薄板透明度做缓动过渡。
- 旋转 / 缩放 / 平移由 `OrbitControls` 提供。

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

### 3.4 右抽屉（2D 热力图）

点击矩阵板后右侧抽屉展示：

- 矩阵全名、标签、`shape = out × in`、role 与元素数；
- 2D 热力图（服务端 `/api/matrix/{name}/grid` 取分块网格，客户端按发散色阶着色）；
- **色阶图例**（min/max 数值）与「适配 / 块 / 元素」三档网格粒度切换
  （192 / 64 / 256，tiles 越大越接近逐元素）。

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
- 点「推理 ▶ / 单步 ⏭」→ 后端逐 token 生成，每步用 forward hook 读取各模块
  **激活值**，以 `token` 消息推给前端：L3 流带点亮推进、注意力分布条同步刷新、
  底部显示已生成文本。

### 3.8 架构浏览器

把 `viz/architectures/` 下的全部真实模型规格（当前 17 个，含 GLM-4/4.5/5/5.3/5.3-Flash、Qwen3/3.5/3.8/3-Next、MiMo/MiMo-VL、
Kimi K2/K2.5/K3、DeepSeek V3/V3.2/V4.1）**逐层**三维铺开：每个方块 = 一层，
颜色 = 该层注意力 / FFN 类型。顶部下拉选择模型，**点击任意一层或模块**，右侧抽屉用
KaTeX 渲染该层公式并附对应源码。数据来自 `viz/architectures/<id>.json`，与
`docs/models/**/*.md` 同源生成，不会不一致。

完整逐层文档见 [models/README.md](models/README.md)；所有模型横向对比
（参数 / 层数 / 注意力 / MoE / 上下文，可排序过滤）见
[models/compare.html](models/compare.html)，或在服务里访问 `/compare`。

## 4. HTTP / WebSocket 协议

### 4.1 HTTP 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | `index.html` |
| GET | `/api/model?source=ckpt\|live` | 模型图（layers / modules / matrices / connections）+ 全局色阶 `global_absmax` |
| GET | `/api/matrix/{name}/grid?source=&tiles=64` | 分块聚合二维网格（每块均值），返回 `{shape, grid, values}` |
| GET | `/api/matrix/{name}/png?source=&tiles=&norm=global\|matrix` | 发散色阶 PNG 纹理（客户端当贴图用） |
| GET | `/api/matrix/{name}/stats?source=` | 该矩阵 `{shape, vmin, vmax, absmax, mean, std}` |
| GET | `/api/matrix/{name}/cell?i=&j=&source=` | 单元素数值 |
| GET | `/api/matrix/{name}/patch?r=&c=&h=&w=&source=` | 元素子块（下钻） |
| GET | `/api/neurons?matrix=&source=&n=&top=` | PCA 神经元点云 |
| GET | `/api/arch/list` · `/api/arch/{id}` | 架构浏览器列表 / 详情 |
| GET | `/compare` | 模型横向对比页（`docs/models/compare.html`） |
| POST | `/api/train/start` · `/api/train/stop` | 触发小模型实时训练 |
| POST | `/api/infer` | 触发逐 token 推理，body `{source, prompt, max_new_tokens}` |

### 4.2 WebSocket `ws://host/ws`

| type | 内容 |
|---|---|
| `init` | `{source, device, global_absmax, model, layers, matrices, connections, cube_threshold}` |
| `tick` | `{step, loss, lr, values:{matrixName:{norm, mean, max, n, delta}}}` |
| `token` | `{token, id, text, values:{matrixName:{act}}, attn:[...]}`（`attn` 为最后最多 24 个位置的注意力） |
| `status` | `{training: bool}` |

> WS 连接建立后立即收到一条 `init`（内容与 `/api/model` 的图一致），
> 随后是被动推送的 `tick` / `token` / `status`。

## 5. 性能策略

- 大矩阵走**纹理平面**（`DataTexture` / `CanvasTexture`），`tiles` 默认 64、上限 256；
- 小矩阵走 **`InstancedMesh`** 立方块，面积阈值 `cube_threshold=4096`（可配）；
- 连线只需 `spine`（始终）与展开层的 `inner`；连接用 top-k / LOD 控制；
- 同一时刻只展开一层，几何量有界；
- checkpoint **懒加载**，仅检视模式触发；实时训练默认用几十万参数的小模型。

## 6. 相关文档

- 模型结构：[02-transformer.md](02-transformer.md)
- 与 Qwen3 / DeepSeek 的架构对比：[06-architecture-compare.md](06-architecture-compare.md)
- **真实模型逐层架构文档**：[models/README.md](models/README.md)
- 架构 JSON schema 与新增模型：`viz/architectures/schema.md`、`scripts/gen_model_docs.py`
