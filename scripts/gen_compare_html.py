"""由 `viz/architectures/*.json` 生成一张自包含的「整体对比」HTML。

产物：`docs/models/compare.html`（无外部依赖，双击即可打开；也可经可视化
服务 `/compare` 访问）。支持按家族过滤、关键字搜索、点表头排序、可视化条形图。
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viz.arch import ARCH_DIR, expand_layers, load  # noqa: E402

OUT = ROOT / "docs" / "models" / "compare.html"

_UNIT = {"t": 1000.0, "b": 1.0, "m": 0.001, "k": 0.000001}


def parse_num(s: str) -> float | None:
    """把 '2.8T' / '552B' / '192M' / '1048576' 解析成以 B(十亿) 为单位的数值。"""
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*([TBMK])", str(s))
    if m:
        return float(m.group(1)) * _UNIT[m.group(2).lower()]
    m = re.search(r"(\d+(?:\.\d+)?)", str(s))
    if not m:
        return None
    val = float(m.group(1))
    return val / 1e9 if val > 1e6 else val  # 纯数字且很大 → 当成 token 数，换回 B 不合理，另用


def parse_ctx(s: str) -> int | None:
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*([MKB]?)", str(s).upper())
    if not m:
        return None
    n = float(m.group(1))
    unit = m.group(2)
    if unit == "M":
        n *= 1_000_000
    elif unit == "K":
        n *= 1_000
    return int(n)


def short(label: str) -> str:
    """对比表里用短名：去掉第一个括号及其后的细节（完整解释放悬浮提示）。"""
    s = re.split(r"[（(]", label, 1)[0].strip()
    return s or label


def _type_entries(spec: dict, specs_key: str, key_name: str) -> list[dict]:
    """每个去重后的注意力/FFN 类型：短名 s、全名 f、详细解释 d、使用层数 c。

    用与文档同一套 `expand_layers` 展开后逐层统计，避免 groups 重叠时重复计数。
    """
    counts: dict[str, int] = {}
    for r in expand_layers(spec):
        k = r.get(key_name)
        counts[k] = counts.get(k, 0) + 1
    specs = spec.get(specs_key, {})
    out = []
    for k in counts:
        s = specs.get(k, {})
        out.append({"s": short(s.get("label", k)), "f": s.get("label", k),
                    "d": s.get("desc", ""), "c": counts[k]})
    return out


def build_data() -> list[dict]:
    rows: list[dict] = []
    for f in sorted(ARCH_DIR.glob("*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        p = s.get("params", {})
        rows.append({
            "id": s["id"], "family": s.get("family", "Other"), "name": s["name"],
            "vendor": s.get("vendor", ""), "released": s.get("released", ""),
            "summary": s.get("summary", ""),
            "total": p.get("total", ""), "total_b": parse_num(p.get("total", "")),
            "active": p.get("active", ""),
            "layers": s.get("layers", {}).get("n", 0),
            "hidden": p.get("hidden", ""), "heads": p.get("heads", ""),
            "kv": p.get("kv_heads", ""), "head_dim": p.get("head_dim", ""),
            "ffn": p.get("ffn", ""), "vocab": p.get("vocab", ""),
            "ctx": p.get("ctx", ""), "ctx_tokens": parse_ctx(p.get("ctx", "")),
            "tie": p.get("tie", ""),
            "attn": _type_entries(s, "attn_specs", "attn"),
            "ffn_labels": _type_entries(s, "ffn_specs", "ffn"),
            "vision": s.get("vision", ""),
            "doc": f"{s.get('family', 'Other')}/{s['name']}.md",
        })
    # 基线：本项目的手写模型
    rows.append({
        "id": "mini-llm-lab", "family": "mini-llm-lab（本项目）", "name": "mini-llm-lab",
        "vendor": "本仓库", "released": "",
        "summary": "从零手写的 decoder-only 中文 GPT：RoPE + RMSNorm + SwiGLU + GQA + 权重共享。",
        "total": "~192M（主干 75.5M）", "total_b": 0.192, "active": "全激活",
        "layers": 12, "hidden": "768", "heads": "12", "kv": "4", "head_dim": "64",
        "ffn": "2048", "vocab": "151666", "ctx": "1024 / 2048", "ctx_tokens": 2048,
        "tie": "共享",
        "attn": [{"s": "GQA", "f": "GQA（分组查询注意力）", "c": 12,
                  "d": "分组查询注意力：12 个 Q 头共享 4 组 KV 头，KV 缓存降到 MHA 的 1/3；配合 RoPE 旋转位置编码与因果 mask。"}],
        "ffn_labels": [{"s": "SwiGLU", "f": "SwiGLU 门控前馈", "c": 12,
                        "d": "门控前馈 w3(silu(w1(x))·w2(x))，中间维 d_ff=2048；比普通 MLP 表达更强。"}],
        "vision": "手写 ViT（patch 16 → 64 个 <image> token / d_vision 384 / 6 层）+ 投影层，LLaVA 式替换占位符",
        "doc": "../02-transformer.md",
    })
    return rows


HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>大模型架构整体对比 · mini-llm-lab</title>
<style>
  :root{{--bg:#0b1020;--panel:#121a30;--line:#233152;--fg:#dfe6f3;--mut:#8fa3c8;--accent:#4f8cff;}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}}
  header{{padding:22px 26px 14px;border-bottom:1px solid var(--line);background:linear-gradient(#0e1730,#0b1020)}}
  h1{{margin:0 0 6px;font-size:20px}}
  .sub{{color:var(--mut);font-size:13px}}
  .wrap{{padding:18px 26px 60px}}
  .controls{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:14px}}
  input,select{{background:#0e1730;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font-size:13px}}
  input{{min-width:220px}}
  .chip{{padding:3px 10px;border-radius:999px;border:1px solid var(--line);background:#0e1730;color:var(--mut);cursor:pointer;font-size:12px;user-select:none}}
  .chip.on{{background:#1c47a8;color:#fff;border-color:#2f63d8}}
  table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
  th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);font-size:12.5px;vertical-align:top;overflow-wrap:anywhere;word-break:break-word}}
  th{{background:#0e1730;color:var(--mut);cursor:pointer;white-space:nowrap;position:sticky;top:0}}
  th:hover{{color:#fff}}
  tr:hover td{{background:#16203a}}
  .name{{font-weight:700;color:#fff;white-space:nowrap}}
  .fam{{display:inline-block;padding:1px 7px;border-radius:6px;font-size:11px;border:1px solid var(--line);color:var(--mut)}}
  .tag{{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;margin:0 3px 3px 0;
    border:1px solid transparent;white-space:normal;max-width:100%;overflow-wrap:anywhere;line-height:1.5}}
  .col-tags{{max-width:170px}}
  .col-vis{{max-width:260px;color:#cfe0ff}}
  .tag{{cursor:help}}
  #tip{{position:fixed;display:none;max-width:380px;padding:9px 12px;border-radius:9px;background:#0b1226;
    border:1px solid #2f63d8;color:#dfe6f3;font-size:12px;line-height:1.6;z-index:99;pointer-events:none;
    box-shadow:0 8px 26px #000b}}
  #tip b{{color:#fff}}
  #tip .tipd{{margin-top:5px;color:#cfe0ff}}
  #tip .tipmeta{{color:var(--mut);font-weight:400}}
  .bar{{height:8px;border-radius:4px;background:#1c47a8;display:inline-block;vertical-align:middle}}
  .barwrap{{width:130px;background:#0e1730;border-radius:4px;display:inline-block;vertical-align:middle;margin-left:6px}}
  .num{{color:var(--fg);font-variant-numeric:tabular-nums;white-space:nowrap}}
  a{{color:var(--accent);text-decoration:none}} a:hover{{text-decoration:underline}}
  .muted{{color:var(--mut)}} .nowrap{{white-space:nowrap}}
  .legend{{margin:10px 0 0;color:var(--mut);font-size:12px}}
</style></head><body>
<header>
  <h1>大模型架构整体对比</h1>
  <div class="sub">共 <b id="count"></b> 个模型 · 数据来自 <code>viz/architectures/*.json</code>（与 <code>docs/models/**</code> 同源）。
    点表头排序，点家族标签过滤。</div>
</header>
<div class="wrap">
  <div class="controls">
    <input id="q" placeholder="搜索模型 / 厂商 / 技术…"/>
    <span id="famChips"></span>
    <select id="sort">
      <option value="total_b">按总参数排序</option>
      <option value="layers">按层数排序</option>
      <option value="ctx_tokens">按上下文排序</option>
      <option value="family">按家族排序</option>
    </select>
  </div>
  <table><thead><tr>
    <th data-k="name">模型</th><th data-k="family">家族</th><th data-k="total_b">总参</th>
    <th data-k="layers">层数</th><th data-k="hidden">hidden</th><th data-k="heads">Q/KV</th>
    <th class="col-tags">注意力</th><th class="col-tags">FFN / MoE</th><th data-k="ctx_tokens">上下文</th>
    <th>权重共享</th><th class="col-vis">视觉方案</th><th>文档</th>
  </tr></thead><tbody id="tb"></tbody></table>
  <div class="legend">注意力/FFN 标签颜色仅为区分类型；数值条形长度按各列最大值归一（参数用 log 缩放）。</div>
</div>
<div id="tip"></div>
<script>
const DATA = {data};
let TIPS = [];
const famColors = {{}};
const palette = ["#4f8cff","#38bdf8","#34d399","#fbbf24","#f472b6","#a78bfa","#f87171","#22d3ee","#84cc16","#fb923c"];
function colorFor(k){{ if(!(k in famColors)) famColors[k]=palette[Object.keys(famColors).length%palette.length]; return famColors[k]; }}
function tag(o){{ const c=colorFor(o.f); const id=TIPS.push(o)-1; return `<span class="tag" data-tip="${{id}}" style="background:${{c}}22;border-color:${{c}}66;color:${{c}}">${{o.s}}</span>`; }}
let sortKey="total_b", asc=false, q="", famFilter=new Set();
const fams=[...new Set(DATA.map(d=>d.family))];
document.getElementById("famChips").innerHTML = fams.map(f=>`<span class="chip" data-f="${{f}}">${{f}}</span>`).join(" ");
document.querySelectorAll(".chip").forEach(c=>c.onclick=()=>{{ const f=c.dataset.f; c.classList.toggle("on"); famFilter.has(f)?famFilter.delete(f):famFilter.add(f); render(); }});
document.getElementById("q").oninput=e=>{{q=e.target.value.toLowerCase();render();}};
document.getElementById("sort").onchange=e=>{{sortKey=e.target.value;asc=(sortKey==="family");render();}};
document.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{{ const k=th.dataset.k; asc=(sortKey===k)?!asc:false; sortKey=k; render(); }});
function fmtTotal(d){{ return d.total || (d.total_b? d.total_b+"B":""); }}
function bar(v,mx,log){{ if(!v||!mx)return ""; let r=log?Math.log10(v+1)/Math.log10(mx+1):v/mx; return `<span class="barwrap"><span class="bar" style="width:${{Math.max(3,r*100)}}%"></span></span>`;}}
function render(){{
  TIPS=[];
  const maxTotal=Math.max(...DATA.map(d=>d.total_b||0));
  const maxLayers=Math.max(...DATA.map(d=>d.layers||0));
  const maxCtx=Math.max(...DATA.map(d=>d.ctx_tokens||0));
  let rows=DATA.filter(d=>{{
    if(famFilter.size && !famFilter.has(d.family)) return false;
    if(!q) return true;
    return (d.name+" "+d.vendor+" "+d.summary+" "+d.attn.map(x=>x.f).join(" ")+" "+d.ffn_labels.map(x=>x.f).join(" ")+" "+(d.vision||"")).toLowerCase().includes(q);
  }});
  rows.sort((a,b)=>{{ let x=a[sortKey],y=b[sortKey];
    if(typeof x==="string") x=x.toLowerCase(),y=(y||"").toLowerCase();
    if(x==null)x=-1; if(y==null)y=-1;
    return (x<y?-1:x>y?1:0)*(asc?1:-1); }});
  document.getElementById("count").textContent=rows.length;
  document.getElementById("tb").innerHTML=rows.map(d=>`<tr>
    <td class="name">${{d.name}}</td>
    <td><span class="fam">${{d.family}}</span></td>
    <td class="nowrap"><span class="num">${{fmtTotal(d)}}</span>${{bar(d.total_b,maxTotal,true)}}</td>
    <td class="nowrap"><span class="num">${{d.layers}}</span>${{bar(d.layers,maxLayers,false)}}</td>
    <td class="num">${{d.hidden}}</td>
    <td class="num nowrap">${{d.heads}}/${{d.kv}}</td>
    <td>${{d.attn.map(tag).join("")}}</td>
    <td>${{d.ffn_labels.map(tag).join("")}}</td>
    <td class="nowrap"><span class="num">${{d.ctx}}</span>${{bar(d.ctx_tokens,maxCtx,true)}}</td>
    <td class="num">${{d.tie}}</td>
    <td class="col-vis">${{d.vision ? d.vision : '<span class="muted">—（纯文本）</span>'}}</td>
    <td><a href="${{d.doc}}">打开</a></td></tr>`).join("");
}}
render();

const tip = document.getElementById("tip");
function moveTip(e){{ const w=tip.offsetWidth, h=tip.offsetHeight, p=14;
  let x=e.clientX+p, y=e.clientY+p;
  if(x+w>innerWidth) x=e.clientX-w-p;
  if(y+h>innerHeight) y=e.clientY-h-p;
  tip.style.left=x+"px"; tip.style.top=y+"px"; }}
document.addEventListener("mousemove", e=>{{
  const t=e.target.closest ? e.target.closest(".tag") : null;
  if(!t){{ tip.style.display="none"; return; }}
  const o=TIPS[+t.dataset.tip];
  if(!o){{ tip.style.display="none"; return; }}
  tip.innerHTML="";
  const b=document.createElement("b"); b.textContent=o.f; tip.appendChild(b);
  if(o.c){{ const m=document.createElement("span"); m.className="tipmeta"; m.textContent=" · 本模型 "+o.c+" 层"; tip.appendChild(m); }}
  if(o.d){{ const d=document.createElement("div"); d.className="tipd"; d.textContent=o.d; tip.appendChild(d); }}
  tip.style.display="block"; moveTip(e);
}});
</script>
</body></html>"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = build_data()
    OUT.write_text(HTML.format(data=json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print(f"[gen] {OUT.relative_to(ROOT)}  ({len(data)} 行)")


if __name__ == "__main__":
    main()
