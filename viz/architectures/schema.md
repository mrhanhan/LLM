# 架构 JSON schema（`viz/architectures/<id>.json`）

一份 JSON 同时驱动：
- Markdown 文档：`scripts/gen_model_docs.py` → `docs/models/<family>/<name>.md`
- 3D 网页「架构浏览器」：`viz/server.py` 的 `/api/arch/<id>`

## 字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | ✅ | 唯一 id，等于文件名（如 `kimi-k3`）|
| `family` | ✅ | 家族目录名：`GLM` / `Qwen` / `MiMo` / `Kimi` |
| `name` | ✅ | 版本名，等于文件名（如 `Kimi-K3`）|
| `vendor` | | 厂商 |
| `released` | | 发布年月 `YYYY-MM` |
| `banner` | | 顶部提示（如"官方未开源建模代码"）|
| `source` | | 来源：官方仓库 / HF config / 论文 |
| `summary` | ✅ | 一句话技术定位 |
| `vision` | | 视觉/多模态方案（有则填；缺失表示纯文本模型）。例：`"MoonViT（约 400M…）"` |
| `params` | | 总览表：`total/active/layers/hidden/heads/kv_heads/head_dim/ffn/vocab/ctx/tie` |
| `config_table` | | `[["字段","值"], ...]` 完整配置表 |
| `innovations` | | `[{"title","text"}]` 关键创新（3–6 条）|
| `signal_flow` | | `[{"stage","shape","note"}]` 张量形状流 |
| `layers` | ✅ | `{"n": <层数>, "groups":[{"from","to","attn","ffn","note"}]}`；`from`/`to` 闭区间，逐层展开 |
| `attn_specs` | | `{"<key>": {"label","desc","tex","code"}}`，key 对应 groups 的 `attn` |
| `ffn_specs` | | 同上，key 对应 `ffn` |
| `extra_specs` | | `[{"label","desc","tex","code"}]` 额外公式（RMSNorm/RoPE/MLA 等）|
| `code` | | `[{"name","ref","lang","code"}]` 关键源码，`ref` 形如 `reference/...:行号` |
| `papers` | | `[{"title","url"}]` |
| `diff_mini` | | Markdown：与本项目 `mini-llm-lab` 的差异 |

## 约定

- **每一层都要能被枚举**：用 `groups` 覆盖 `0..n-1`；异构层（如滑动窗口/稀疏层）用多个 group 或逐层 group 表达。
- `tex` 里如果有多条公式，用换行分隔（生成器会拆成多个 `$$` 块）。**不要在 `tex` 里写 `\\n` 当公式内换行**——需要对齐请用 KaTeX 的 `\begin{aligned}...\end{aligned}`。
- `code` 字符串自带 ```` ```python ```` 围栏（第 5 节渲染用），`extra_specs[].code` 同理；顶层 `code[].code` 不含围栏。
- 引用真实源码时给 `reference/` 路径 + 行号；无开源代码时以官方 README/技术报告/HF `config.json` 为依据并在 `source` 注明。
- 写完后自检：
  ```powershell
  & ".venv\Scripts\python.exe" scripts/gen_model_docs.py <id>
  ```
  应打印 `(N 层)` 且 N>0，然后检查 `docs/models/<family>/<name>.md` 无 `\n` 残留、表格/公式正常。

## 最小示例

```json
{
  "id": "demo", "family": "Qwen", "name": "Demo",
  "summary": "一句话",
  "params": {"total": "1B", "layers": 2},
  "layers": {"n": 2, "groups": [{"from": 0, "to": 1, "attn": "gqa", "ffn": "swiglu"}]},
  "attn_specs": {"gqa": {"label": "GQA", "tex": "\\mathrm{softmax}(QK^\\top/\\sqrt{d})V"}},
  "ffn_specs": {"swiglu": {"label": "SwiGLU", "tex": "W_3(\\mathrm{SiLU}(W_1x)\\odot W_2x)"}}
}
```
