# 教学注释：Qwen 分词器从项目内目录加载；<image> 特殊符号需可用。
import shutil
from pathlib import Path
import pytest
from src.tokenizer import QwenTokenizer

QWEN_DIR = "data/tokenizer/qwen2.5-0.5b"


@pytest.mark.skipif(not Path(QWEN_DIR).exists(), reason="需先运行 scripts/prepare_data_part1.py 下载分词器")
def test_qwen_roundtrip_and_image_token():
    tok = QwenTokenizer.load(QWEN_DIR)
    s = "你好，世界！Hello 123。"
    assert tok.decode(tok.encode(s)) == s
    assert 0 <= tok.special_id("image") < tok.vocab_size
    ids = tok.encode("测试", add_bos=True, add_eos=True)
    assert len(ids) == len(tok.encode("测试")) + 2
