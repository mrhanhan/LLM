"""架构规格加载：`viz/architectures/*.json` 同时供文档生成器与 3D 网页使用。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "viz" / "architectures"


def list_models() -> list[dict]:
    out = []
    for f in sorted(ARCH_DIR.glob("*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        out.append({
            "id": s.get("id"), "family": s.get("family"), "name": s.get("name"),
            "vendor": s.get("vendor"), "released": s.get("released"),
            "summary": s.get("summary", ""), "params": s.get("params", {}),
        })
    return out


def load(model_id: str) -> dict | None:
    f = ARCH_DIR / f"{model_id}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def expand_layers(spec: dict) -> list[dict]:
    """把 groups 展开成每一层一行：{idx, attn, ffn, note}。"""
    layer_spec = spec.get("layers", {})
    n = layer_spec.get("n")
    if n is None:
        n = max((g.get("to", 0) for g in layer_spec.get("groups", [])), default=-1) + 1
    rows = [{"idx": i, "attn": "?", "ffn": "?", "note": ""} for i in range(n)]
    for g in layer_spec.get("groups", []):
        a, b = g.get("from", 0), g.get("to", n - 1)
        for i in range(max(0, a), min(b, n - 1) + 1):
            rows[i]["attn"] = g.get("attn", rows[i]["attn"])
            rows[i]["ffn"] = g.get("ffn", rows[i]["ffn"])
            if g.get("note"):
                rows[i]["note"] = g["note"]
    return rows


def runs(rows: list[dict]) -> list[tuple[int, int, str, str]]:
    """把连续同类型的层合并成 (from, to, attn, ffn)。"""
    out = []
    for r in rows:
        if out and out[-1][2] == r["attn"] and out[-1][3] == r["ffn"] and out[-1][1] == r["idx"] - 1:
            out[-1] = (out[-1][0], r["idx"], r["attn"], r["ffn"])
        else:
            out.append((r["idx"], r["idx"], r["attn"], r["ffn"]))
    return out


def label_of(specs: dict, key: str) -> str:
    return specs.get(key, {}).get("label", key)


def public(spec: dict) -> dict:
    """给前端的版本：附带展开好的逐层 rows。"""
    out = dict(spec)
    out["rows"] = expand_layers(spec)
    return out
