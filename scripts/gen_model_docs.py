"""从 `viz/architectures/*.json` 生成每个模型的 Markdown 说明文档。

设计：**一份架构 JSON 同时驱动文档与 3D 网页**，二者不会不一致。
JSON schema（字段全部可选，除 id/name/layers 外）：

{
  "id": "qwen3",
  "family": "Qwen",                # 决定输出目录 docs/models/<family>/
  "name": "Qwen3",                 # 文件名 <name>.md
  "vendor": "Alibaba Qwen",
  "released": "2025-04",
  "banner": "可选提示（如：官方未开建模代码）",
  "source": "来源说明",
  "summary": "一句话技术定位",
  "params": {"total": "8B", "active": "8B(dense)", "layers": 36, ...},
  "config_table": [["字段", "值"], ...],
  "innovations": [{"title": "...", "text": "..."}],
  "signal_flow": [{"stage": "输入", "shape": "[B,T]", "note": "..."}],
  "layers": {"groups": [{"from": 0, "to": 35, "attn": "gqa", "ffn": "swiglu", "note": ""}]},
  "attn_specs": {"gqa": {"label": "GQA+QK-Norm", "tex": "...", "desc": "...", "code": "```python ...```"}},
  "ffn_specs":  {"swiglu": {"label": "SwiGLU", "tex": "...", "desc": "...", "code": "..."}},
  "code": [{"name": "...", "ref": "reference/...:12", "lang": "python", "code": "..."}],
  "papers": [{"title": "...", "url": "..."}],
  "diff_mini": "Markdown 文本"
}

用法：
    python scripts/gen_model_docs.py            # 生成全部
    python scripts/gen_model_docs.py qwen3       # 只生成某个 id
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viz.arch import ARCH_DIR, expand_layers, label_of, runs  # noqa: E402

OUT_DIR = ROOT / "docs" / "models"


def emit_tex(a, tex: str) -> None:
    """把公式拆成独立的 $$ 块；`\\n` 或真实换行都当分隔符。"""
    for part in re.split(r"\\n|\n", tex or ""):
        part = part.strip()
        if part:
            a("$$")
            a(part)
            a("$$")
            a("")


def mermaid(spec: dict, rows: list[dict]) -> str:
    lines = ["```mermaid", "flowchart TD",
             '  IN["输入 token B×T"] --> EMB["词嵌入"]']
    prev = "EMB"
    for (a, b, ak, fk) in runs(rows):
        node = f'L{a}_{b}["层 {a}–{b} ×{b - a + 1}<br/>{label_of(spec.get("attn_specs", {}), ak)}<br/>{label_of(spec.get("ffn_specs", {}), fk)}"]'
        lines.append(f"  {prev} --> {node}")
        prev = f"L{a}_{b}"
    lines.append(f'  {prev} --> NORM["最终 RMSNorm"] --> HEAD["lm_head → logits"]')
    lines.append("```")
    return "\n".join(lines)


def render(spec: dict) -> str:
    rows = expand_layers(spec)
    p = spec.get("params", {})
    out: list[str] = []
    a = out.append

    a(f"# {spec['name']} · 架构说明（逐层）")
    a("")
    meta = " · ".join(x for x in [
        spec.get("vendor"), f"发布 {spec.get('released')}" if spec.get("released") else None,
        spec.get("source"),
    ] if x)
    a(f"> {meta}")
    a("")
    if spec.get("banner"):
        a(f"> ⚠️ {spec['banner']}")
        a("")
    a(f"**技术定位**：{spec.get('summary', '')}")
    a("")

    # 1. 参数总览
    a("## 1. 参数与配置")
    a("")
    keys = [("总参数", "total"), ("激活参数", "active"), ("层数", "layers"),
            ("hidden", "hidden"), ("Q 头", "heads"), ("KV 头", "kv_heads"),
            ("head_dim", "head_dim"), ("FFN/MoE 中间维", "ffn"),
            ("词表", "vocab"), ("上下文", "ctx"), ("权重共享", "tie")]
    a("| 项 | 值 |")
    a("|---|---|")
    for label, k in keys:
        if k in p:
            a(f"| {label} | {p[k]} |")
    a("")
    if spec.get("config_table"):
        a("**完整配置**：")
        a("")
        a("| 字段 | 值 |")
        a("|---|---|")
        for kv in spec["config_table"]:
            a(f"| {kv[0]} | {kv[1]} |")
        a("")

    if spec.get("vision"):
        a("**视觉 / 多模态方案**：")
        a("")
        a(spec["vision"])
        a("")

    # 2. 技术方案
    if spec.get("innovations"):
        a("## 2. 技术方案与关键创新")
        a("")
        for i, inv in enumerate(spec["innovations"], 1):
            a(f"**{i}. {inv.get('title', '')}**　{inv.get('text', '')}")
            a("")

    # 3. 层级结构（Mermaid）
    a("## 3. 层级结构（可视化）")
    a("")
    a(mermaid(spec, rows))
    a("")

    # 4. 逐层清单
    a(f"## 4. 逐层清单（共 {len(rows)} 层）")
    a("")
    a("| 层 | Attention | FFN/MoE | 说明 |")
    a("|---|---|---|---|")
    for r in rows:
        a(f"| {r['idx']} | {label_of(spec.get('attn_specs', {}), r['attn'])} | "
          f"{label_of(spec.get('ffn_specs', {}), r['ffn'])} | {r['note']} |")
    a("")

    # 5. 关键模块公式 + 代码
    a("## 5. 关键模块：数学公式与代码")
    a("")
    for group_name, group in (("注意力", spec.get("attn_specs", {})),
                              ("前馈/MoE", spec.get("ffn_specs", {}))):
        used = {r["attn" if group_name == "注意力" else "ffn"] for r in rows}
        for key, s in group.items():
            if key not in used and key not in spec.get("show_specs", []):
                continue
            a(f"### {group_name} · {s.get('label', key)}")
            a("")
            if s.get("desc"):
                a(s["desc"])
                a("")
            if s.get("tex"):
                emit_tex(a, s["tex"])
            if s.get("code"):
                a(s["code"])
                a("")

    for s in spec.get("extra_specs", []):
        a(f"### {s.get('label', s.get('id', ''))}")
        a("")
        if s.get("desc"):
            a(s["desc"])
            a("")
        if s.get("tex"):
            emit_tex(a, s["tex"])
        if s.get("code"):
            a(s["code"])
            a("")

    # 6. 张量形状流
    if spec.get("signal_flow"):
        a("## 6. 张量形状流")
        a("")
        a("| 阶段 | 形状 | 说明 |")
        a("|---|---|---|")
        for s in spec["signal_flow"]:
            a(f"| {s.get('stage')} | `{s.get('shape')}` | {s.get('note', '')} |")
        a("")

    # 7. 关键源码
    if spec.get("code"):
        a("## 7. 关键源码（引自 reference/）")
        a("")
        for c in spec["code"]:
            a(f"**{c.get('name', '')}**　`{c.get('ref', '')}`")
            a("")
            a(f"```{c.get('lang', 'python')}")
            a(c.get("code", "").rstrip())
            a("```")
            a("")

    # 8. 与 mini-llm-lab 的差异
    if spec.get("diff_mini"):
        a("## 8. 与 mini-llm-lab 的差异")
        a("")
        a(spec["diff_mini"].rstrip())
        a("")

    # 9. 3D 可视化 & 参考
    a("## 9. 3D 可视化")
    a("")
    a("启动可视化网页后，在「架构浏览器」视图选择本模型（id："
      f"`{spec['id']}`），可三维查看每一层并点开公式/代码：")
    a("")
    a("```powershell")
    a('& ".venv\\Scripts\\python.exe" viz/server.py   # 打开 http://127.0.0.1:7861 → 架构浏览器')
    a("```")
    a("")
    if spec.get("papers"):
        a("## 10. 参考资料")
        a("")
        for pp in spec["papers"]:
            a(f"- [{pp.get('title', pp.get('url', ''))}]({pp.get('url', '')})")
        a("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="只生成指定 id；缺省生成全部")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    files = sorted(ARCH_DIR.glob("*.json"))
    made = []
    for f in files:
        spec = json.loads(f.read_text(encoding="utf-8"))
        if args.ids and spec.get("id") not in args.ids:
            continue
        family = spec.get("family", "Other")
        dest = OUT_DIR / family / f"{spec['name']}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(render(spec), encoding="utf-8")
        made.append((spec, dest))
        print(f"[gen] {dest.relative_to(ROOT)}  ({len(expand_layers(spec))} 层)")

    if not args.ids:
        _write_index([s for s, _ in made])


def _write_index(specs: list[dict]) -> None:
    lines = ["# 模型架构文档索引", "",
             "每个模型一份逐层架构说明，由 `viz/architectures/*.json` 生成"
             "（`python scripts/gen_model_docs.py`），并可在可视化网页的"
             "「架构浏览器」中三维查看。", "",
             "👉 **[整体对比（HTML）](compare.html)** —— 所有模型的参数/层数/注意力/MoE/上下文横向对比。", ""]
    by_family: dict[str, list[dict]] = {}
    for s in specs:
        by_family.setdefault(s.get("family", "Other"), []).append(s)
    for fam in sorted(by_family):
        lines.append(f"## {fam}")
        lines.append("")
        lines.append("| 模型 | 文档 | 参数 | 说明 |")
        lines.append("|---|---|---|---|")
        for s in sorted(by_family[fam], key=lambda x: x.get("released", "")):
            p = s.get("params", {})
            lines.append(f"| {s['name']} | [{s['name']}.md]({fam}/{s['name']}.md) | "
                         f"{p.get('total', '-')} | {s.get('summary', '')} |")
        lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[gen] {OUT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
