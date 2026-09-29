"""下载 Qwen 分词器与中文 TinyStories 到 data/（走 hf-mirror）。"""
import argparse
import sys
from pathlib import Path

# 教学注释：Windows 控制台默认 cp1252，打印中文会 UnicodeEncodeError，强制 stdout 用 utf-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import download_qwen_tokenizer, download_tinystories
from src.hfenv import setup_hf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 Qwen 分词器与中文 TinyStories")
    parser.add_argument("--endpoint", default="https://hf-mirror.com",
                        help="HF 镜像端点；传空字符串则直连 huggingface.co")
    parser.add_argument("--proxy", default=None,
                        help="HTTP(S) 代理地址，例如 http://127.0.0.1:7890")
    args = parser.parse_args()
    setup_hf(args.endpoint or None, args.proxy or None)
    d1 = download_qwen_tokenizer("data/tokenizer/qwen2.5-0.5b")
    print("tokenizer ->", d1)
    files = download_tinystories("data/raw/text/tinystories_zh")
    print(f"tinystories jsonl: {len(files)} 个")
