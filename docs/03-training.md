# 03 · 训练原理

> 对应代码：`src/trainer.py`、`src/utils.py`、`scripts/train_gpt.py`
> 配置：`configs/gpt_tinystories.yaml`（`TrainConfig`）

这一章把训练循环拆开讲：模型在学什么、学习率怎么走、显存不够怎么省、
精度怎么混、怎么从 loss 曲线判断模型状态、以及断点续训到底存了什么。

---

## 1. 模型在学什么：下一 token 预测 + 交叉熵

语言模型的训练目标只有一个：**给定前文，预测下一个 token**。

数据采样时，输入 `x` 与标签 `y` 只差一个位置（右移一位）：

```python
# src/data.py::get_batch
x = data[i     : i + ctx_len]        # 输入
y = data[i + 1 : i + 1 + ctx_len]    # 标签：每个位置的下一个 token
```

模型对每个位置输出 `vocab` 个分数（logits），与真实下一个 token 做**交叉熵**：

```python
# src/model.py::GPT.forward
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
)
```

- 交叉熵衡量「模型给正确答案的概率有多低」：正确答案概率越接近 1，loss 越接近 0。
- 随机瞎猜时的理论 loss ≈ `ln(vocab)`。Qwen 词表 151666，`ln(151666) ≈ 11.93`，
  所以训练刚开始 loss 大约在 11~12，随训练下降。
- 一个 `ctx_len=1024` 的样本里**同时**有 1024 个预测任务（每个位置都预测下一个），
  所以语言模型能从一段文本里榨出非常密集的监督信号。
- `ignore_index=-100` 让某些位置（如多模态里的图像占位）不参与 loss，文本阶段用不到。

---

## 2. 训练循环总览

`Trainer.train` 的核心（简化）：

```python
for step in range(self.start_step, cfg.max_steps):
    lr = get_lr(step, cfg)
    for g in self.opt.param_groups:
        g["lr"] = lr                                  # 1. 按调度器设学习率
    self.opt.zero_grad(set_to_none=True)
    for _ in range(cfg.grad_accum):                   # 2. 梯度累积
        x, y = self._next_batch()
        loss = self._forward_loss(x, y) / cfg.grad_accum
        loss.backward()
    if cfg.grad_clip > 0:                             # 3. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
    self.opt.step()                                   # 4. 更新参数
```

配合周期性评估与保存：

```python
if step % cfg.log_interval == 0:   print + logger.log(train loss, lr)
if step % cfg.eval_interval == 0:  val_loss = self._estimate_val()
if step % cfg.save_interval == 0:  self.save("out/gpt/latest.pt"); plot_loss(...)
```

每个 epoch 没有固定边界——训练按 **step 数**推进，每个 step 从数据里随机切一段片段。

---

## 3. 学习率调度：warmup + 余弦退火

```python
# src/utils.py::get_lr
if step < cfg.warmup_steps:                                   # 线性 warmup
    return cfg.lr * (step + 1) / cfg.warmup_steps
if step >= cfg.max_steps:
    return cfg.min_lr
ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * ratio))  # 余弦退火
```

- **warmup（热身）**：前 `200` 步学习率从接近 0 线性升到 `lr=6e-4`。
  训练初期参数是随机的，梯度方向噪声大，一上来用大学习率容易把网络「带飞」甚至发散；
  先小步试探再加速更稳。
- **余弦退火**：warmup 之后学习率按余弦曲线从 `6e-4` 平滑降到 `min_lr=6e-5`，
  后期小步精调，帮助收敛到更平缓的解。默认 `max_steps=20000`。

曲线形状：`/‾‾‾\`（先升后缓慢下降）。

---

## 4. 梯度累积：用时间换显存

**问题**：大 batch 训练更稳，但显存装不下。
**解法**：把一个大 batch 拆成 `grad_accum` 个 micro-batch 顺序前向+反向，
梯度**累加**起来，攒够后再更新一次参数：

```python
loss = self._forward_loss(x, y) / cfg.grad_accum   # 每个 micro-batch 的 loss 先除以累积步数
loss.backward()                                     # 梯度累加到 .grad 上
# ... 循环 grad_accum 次 ...
self.opt.step()                                     # 一次更新 ≈ 等效大 batch
```

- **等效 batch = `batch_size × grad_accum`**。默认 `8 × 8 = 64`。
- 除以 `grad_accum` 是为了让累积后的梯度等于「大 batch 的平均梯度」，而不是总和。
- 代价：micro-batch 之间串行，速度变慢——**用时间换显存**。

> 打印/记录时乘回去：`loss.item() * cfg.grad_accum`，恢复成「单个 token 的 loss」，
> 便于和 val loss、理论值 `ln(vocab)` 比较。见 `src/trainer.py` 的日志行。

**显存不足时**：`batch_size` 减半、`grad_accum` 加倍（如 4 × 16），等效 batch 不变。
物理显存占用由 `batch_size` 决定，训练效果由乘积决定。

---

## 5. 梯度裁剪

```python
if cfg.grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)   # 默认 1.0
```

把所有参数的梯度拼成一个向量，若其 L2 范数超过 `1.0`，就等比缩放到 `1.0`。

- **为什么需要**：偶尔会出现「一个异常 batch 导致梯度过大」，一步更新就把参数推到坏区域，
  loss 突然飙升甚至 NaN。裁剪给更新量设了上限，防止这种翻车。
- 裁剪发生在 `opt.step()` 之前、所有 micro-batch 梯度累积之后。

---

## 6. bf16 混合精度

```python
# src/trainer.py::_autocast_ctx
if train_cfg.dtype == "bf16" and torch.cuda.is_available():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
return contextlib.nullcontext()
```

- **混合精度**：前向/反向的计算用 bf16（16 位）跑，**主权重仍保持 fp32**。
  矩阵乘法用 bf16 更快、显存减半；权重更新用 fp32 保证精度。
- **为什么不用 GradScaler**：fp16 动态范围小、需要梯度缩放来防下溢；而 bf16 的动态范围
  和 fp32 一样大，**不需要 GradScaler**，实现更简单。
- 仅在 CUDA 可用时启用；CPU 上自动退回 fp32（用 `nullcontext`）。

---

## 7. torch.compile（可选）

配置 `compile: false` 默认关闭。开启时：

```python
compiled = torch.compile(model)
with torch.no_grad():
    dummy = torch.randint(0, 2, (1, n), device=self.device)
    compiled(dummy, targets=dummy)            # 先用 dummy batch 触发「惰性编译」
self.model = compiled
```

- `torch.compile` 是**惰性编译**：真正的编译发生在第一次前向。所以这里先用一个极小的
  dummy batch 跑一次，把编译错误**提前暴露**，才能被 `try/except` 捕获。
- 失败时打印原因并回退 eager 模型，训练照常进行。
- Windows 上通常缺 triton，失败属于**正常现象**，不是 bug。

---

## 8. 如何读 `loss.png` 判断欠拟合 / 过拟合

训练每隔 `eval_interval` 步在验证集上算平均 loss（`_estimate_val`，默认 `eval_iters=100`
个 batch 取平均），并把 train/val loss 追加到 `out/gpt/metrics.jsonl`。保存时用
`plot_loss` 画出 `out/gpt/loss.png`（蓝线=train，带点的线=val）。

常见形态：

| 曲线形态 | 诊断 | 对策 |
|---|---|---|
| train、val 都很高且几乎不降 | **欠拟合**（模型太小 / lr 太小 / 步数不够 / 数据太难） | 加大模型、提高 lr、多训、检查数据 |
| train 持续降，val 先降后升 | **过拟合**（模型开始背训练集） | 加数据、减训练步数、加 dropout/weight decay |
| train、val 都平稳下降并趋于平台 | **健康收敛** | 正常；val 略高于 train 属正常 |
| loss 反复剧烈震荡 | lr 太大 / batch 太小 | 降 lr、加大等效 batch |
| loss 变 NaN | 数值爆炸 | 降 lr、确认梯度裁剪开启、检查数据 |

> 本项目用的是故事语料，任务是故事续写，`dropout=0`，所以 train/val 曲线通常比较贴近；
> 若 val 明显比 train 高，多半是数据分布差异或过拟合。

---

## 9. Checkpoint 内容与续训原理

`Trainer.save` 保存的字典：

```python
ckpt = {
    "model": self.raw_model.state_dict(),   # 模型权重（用未 compile 的原始模型）
    "optimizer": self.opt.state_dict(),     # AdamW 的动量等状态
    "step": getattr(self, "_current_step", 0),   # 已训练步数
    "cfg": self.raw_model.cfg,              # ModelConfig（校验结构一致）
    "train_cfg": self.cfg,                  # TrainConfig
}
torch.save(ckpt, path)
```

默认写到 `out/gpt/latest.pt`，每 `save_interval` 步覆盖一次。

**续训**：

```python
# src/trainer.py::load
self.raw_model.load_state_dict(ckpt["model"])
self.opt.load_state_dict(ckpt["optimizer"])    # 恢复优化器状态（失败可忽略）
self.start_step = int(ckpt.get("step", 0))
self._current_step = self.start_step
```

`scripts/train_gpt.py --resume out/gpt/latest.pt` 会调用 `load`，然后从 `start_step`
继续循环到 `max_steps`。

- **为什么要存优化器状态**：AdamW 有动量（一阶/二阶），不恢复的话续训初期会「抖」一下。
- **步数的坑**：训练循环从 `start_step` 开始。若 `start_step >= max_steps`，循环不执行，
  会直接进入收尾保存。这时必须同步 `_current_step`，否则 checkpoint 的步数会被写回 0，
  导致下次又从零训起（`test_resume_finished_checkpoint_preserves_step` 专门测这个）。
- **想多训**：`--set train.max_steps=40000` 把总步数调大，再 `--resume`。

> 当前实现**没有**保存随机数生成器（RNG）状态，所以续训不能做到与「一次训到底」逐位一致；
> 也不会自动保存学习率调度器的「步数之外的额外状态」——好在余弦调度只需 `step` 即可重建。

---

## 10. 优化器与权重衰减

`GPT.configure_optimizers` 使用 AdamW，并把参数分成两组：

```python
decay    = [p for p in params if p.dim() >= 2]   # 权重矩阵：weight_decay=0.1
no_decay = [p for p in params if p.dim() <  2]   # norm γ、bias：weight_decay=0.0
AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())
```

- **AdamW**：对每个参数自适应调整学习率，并正确地把 weight decay 与梯度更新解耦。
- **betas=(0.9, 0.95)**：第二动量用了 0.95（比默认 0.999 小），是 LLM 训练的常见配置，
  对稀疏/长序列梯度响应更快。
- **分组衰减**：权重矩阵做衰减防过拟合；norm 的缩放参数和 bias 不衰减。

---

## 11. 快速上手命令

```powershell
# 开始训练（默认 configs/gpt_tinystories.yaml）
& ".venv\Scripts\python.exe" scripts/train_gpt.py

# 覆盖超参
& ".venv\Scripts\python.exe" scripts/train_gpt.py --set train.lr=3e-4 train.batch_size=4 train.grad_accum=16

# 从 checkpoint 续训（可同时调大总步数）
& ".venv\Scripts\python.exe" scripts/train_gpt.py --resume out/gpt/latest.pt --set train.max_steps=40000
```

产物：`out/gpt/latest.pt`、`out/gpt/metrics.jsonl`、`out/gpt/loss.png`。

---

## 12. 小结

- 目标是下一 token 预测，损失是交叉熵；乱猜的 loss ≈ `ln(vocab) ≈ 11.9`。
- 学习率先 warmup 再余弦退火；梯度累积用时间换显存，等效 batch = `batch_size × grad_accum`。
- 梯度裁剪（1.0）防梯度爆炸；bf16 autocast 混合精度且不需要 GradScaler。
- 从 `loss.png` 的 train/val 走势判断欠拟合、过拟合或健康收敛。
- checkpoint 存「权重 + 优化器状态 + 步数 + 配置」，`--resume` 从该步数继续。
