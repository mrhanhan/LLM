# 教学注释：校验 viz/architectures/*.json 能被 3D 网页与文档生成器正确消费。
import importlib.util
import json
from pathlib import Path

from viz.arch import ARCH_DIR, expand_layers, list_models, load, public

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"qwen3", "qwen3.5", "qwen3.8", "qwen3-next", "glm-4", "glm-4.5", "glm-5",
            "glm-5.3", "glm-5.3-flash", "mimo", "mimo-vl", "kimi-k2", "kimi-k2.5", "kimi-k3",
            "deepseek-v3", "deepseek-v3.2", "deepseek-v4.1"}


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_model_docs", ROOT / "scripts" / "gen_model_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_expected_models_present():
    ids = {m["id"] for m in list_models()}
    assert EXPECTED <= ids, f"缺少：{EXPECTED - ids}"


def test_each_spec_is_wellformed():
    for f in sorted(ARCH_DIR.glob("*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        assert s["id"] == f.stem, f"{f.name} 的 id 应为 {f.stem}"
        assert s.get("family") and s.get("name")
        rows = expand_layers(s)
        assert len(rows) == s["layers"]["n"] > 0, f"{s['id']} 层数对不上"
        for r in rows:
            assert r["attn"] in s.get("attn_specs", {}), (s["id"], "attn", r["attn"])
            assert r["ffn"] in s.get("ffn_specs", {}), (s["id"], "ffn", r["ffn"])


def test_generator_renders_without_stray_newlines():
    gen = _load_generator()
    for f in sorted(ARCH_DIR.glob("*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        md = gen.render(s)
        assert "\\n" not in md, f"{s['id']} 公式里残留 \\n"
        assert f"共 {s['layers']['n']} 层" in md
        # 逐层表每层一行
        assert md.count("| " + "0 |") >= 1


def test_public_payload_has_rows():
    s = public(load("kimi-k3"))
    assert len(s["rows"]) == s["layers"]["n"] == 93
    assert s["rows"][0]["attn"] in s["attn_specs"]


def test_compare_page_data():
    spec = importlib.util.spec_from_file_location(
        "gen_compare_html", ROOT / "scripts" / "gen_compare_html.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.build_data()
    n_json = len(list(ARCH_DIR.glob("*.json")))
    assert len(rows) == n_json + 1  # 含 mini-llm-lab 基线
    assert all(r["doc"] and r["layers"] > 0 for r in rows)
    assert any(r["id"] == "mini-llm-lab" and r["total_b"] > 0 for r in rows)
