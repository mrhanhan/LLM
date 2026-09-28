# 教学注释：验证手写 BPE 能从 jsonl 语料训练并保存，随后能被 prepare_data_part2 用
# --tokenizer_kind bpe 编码成 .bin，且保存/加载后分词往返一致。
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.tokenizer import BPETokenizer

CORPUS = ["从前有座山，山里有庙。", "小猫在草地上跑。", "太阳 sun 月亮 moon。"] * 20


def _write_raw(tmp_path):
    raw = tmp_path / "jsonl"
    raw.mkdir()
    (raw / "a.jsonl").write_text(
        "\n".join(json.dumps({"story_zh": s}, ensure_ascii=False) for s in CORPUS),
        encoding="utf-8",
    )
    return raw


def test_train_tokenizer_script(tmp_path):
    raw = _write_raw(tmp_path)
    out = tmp_path / "bpe"
    r = subprocess.run(
        [sys.executable, "scripts/train_tokenizer.py",
         "--raw_dir", str(raw), "--out", str(out),
         "--vocab_size", "400", "--max_bytes", "100000"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    tok = BPETokenizer.load(str(out))
    assert tok.decode(tok.encode("甲乙")) == "甲乙"


def test_prepare_data_part2_bpe(tmp_path):
    raw = _write_raw(tmp_path)
    tp = tmp_path / "bpe"
    subprocess.run(
        [sys.executable, "scripts/train_tokenizer.py",
         "--raw_dir", str(raw), "--out", str(tp),
         "--vocab_size", "400", "--max_bytes", "100000"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    tbin = tmp_path / "train.bin"
    vbin = tmp_path / "val.bin"
    r = subprocess.run(
        [sys.executable, "scripts/prepare_data_part2.py",
         "--tokenizer_kind", "bpe", "--tokenizer_dir", str(tp),
         "--raw_dir", str(raw), "--out", str(tbin), "--val", str(vbin)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    arr = np.fromfile(str(tbin), dtype=np.uint32)
    assert len(arr) > 0
