# viz 梯度地图 设计（Spec）

**日期：** 2026-09-29
**状态：** 已与用户对齐（两轮 AskUserQuestion）

## 目标

训练过程中把**梯度**当作与权重并列的一等可视化对象：抽屉里能看元素级梯度热力图，3D 书架能按梯度给矩阵上色，底栏显示逐层梯度流；并提供「权重 / 梯度 / 更新量 ΔW」三态切换与「全局 / 局部」色标切换。

## 决策（用户访谈）

1. 呈现位置：**抽屉热力图 + 3D 板材着色 + 逐层梯度流 HUD**（底栏）。点云/连线不做。
2. 颜色语义：**有符号（发散色阶）/ 幅值（顺序色阶）可切**。
3. 数据来源：**仅训练过程中**（`p.grad`）。不对 ckpt 做单次反向。
4. 计算范围：**全部矩阵都算**（大模型下也全量）。
5. 共存方式：**权重 / 梯度 / 更新量 ΔW 三态切换**（顶栏一个全局开关，3D 与抽屉同步）。
6. ΔW 语义：**两者都要** —— 元素级有符号差值 `W − W_prev` 画地图；标量比值 `‖ΔW‖/‖W‖`（现有 `delta`）用于满板/薄板上色。
7. 色标归一化：**全局 / 局部可切**。
8. 采集时机：**每 tick（`log_interval`）全量采集**所有矩阵的梯度网格 + 统计。
9. 梯度流 HUD：**底栏**，与 loss/attn 并列。

## 数据模型

- `mode ∈ {"weight","grad","delta"}`
- `norm ∈ {"global","local"}`；`global` 用该 mode/source 的全局 `absmax`，`local` 用该矩阵 `absmax`。
- 梯度只在训练中出现：非训练态 `grad` 视为 0，前端显示中性/空。

## 后端接口（在现有端点上加 `mode`）

- `GET /api/matrix/{name}/grid?source=&tiles=&mode=&norm=`
- `GET /api/matrix/{name}/png?source=&tiles=&mode=&norm=`
- `GET /api/matrix/{name}/stats?source=&mode=`
- `/api/model` 返回 `global_absmax`（weight，兼容旧字段）+ `global`（`{weight,grad,delta}`）。
- `tick` 的 `values[name]` 增 `grad = {norm, absmean, absmax}`；`delta`（标量比值）保留。

## 缓存与并发

- 梯度是瞬态的（下一步 `zero_grad` 即消失），所以在训练线程每个 tick 把梯度**池化成小网格**并连同标量统计写入 **锁保护** 的缓存（`tiles=128`，约 111 矩阵 ≈ 7MB），HTTP 只读缓存，避免与反传竞争。
- ΔW 地图按需从 `W − WeightTracker.prev` 计算（同一 tick 窗口内稳定）；全局 ΔW absmax 惰性计算并按 tick 失效。
- 梯度网格每 tick 全量重算（用户选择），`192M` 下是主要开销；`tiles` 与 `log_interval` 可调。

## 色阶一致性

`weight`/`grad`/`delta` 三种地图与 3D 板材共用同一发散色阶公式（`viz/weights.diverging_rgb` 与 JS `divergingRGB`）；有符号地图用发散色阶，幅值模式用顺序色阶（新增，两侧公式必须一致）。

## 非目标

- 不做 ckpt 单次反向的静态梯度。
- 不做点云/连线按梯度着色。
- 不改 `src/` 训练语义。
