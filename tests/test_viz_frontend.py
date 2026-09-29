import shutil
import subprocess
from pathlib import Path

STATIC = Path("viz/static")
JS = ["api.js", "colors.js", "shelf.js", "matrix.js", "links.js",
      "panel.js", "cloud.js", "arch.js", "kv.js", "toplabels.js", "main.js"]


def test_static_files_exist():
    for f in JS:
        assert (STATIC / "js" / f).exists(), f"缺少 viz/static/js/{f}"
    assert (STATIC / "index.html").exists()
    assert not (STATIC / "app.js").exists(), "旧 app.js 应已删除"


def test_js_syntax_ok(tmp_path):
    for f in JS:
        p = STATIC / "js" / f
        tmp = tmp_path / (f + ".mjs")
        shutil.copyfile(p, tmp)
        r = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f} 语法错误：{r.stderr}"


def test_index_has_training_controls():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for dom_id in ["train-target", "train-mode", "train-dataset",
                   "hp-max-steps", "hp-lr", "hp-batch", "hp-accum", "hp-hifi"]:
        assert f'id="{dom_id}"' in html, f"缺少 #{dom_id}"
