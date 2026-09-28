"""下载真实中文儿童图文数据并建立索引。"""
import argparse
import sys
from pathlib import Path

# 教学注释：Windows 控制台默认 cp1252，打印中文会 UnicodeEncodeError，强制 stdout 用 utf-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import download_children_captions, build_caption_index


def main(max_items=None):
    raw = download_children_captions("data/raw/image/children_caption")
    n = build_caption_index(raw, "data/processed/image/children/index.jsonl", max_items=max_items)
    print(f"下载到 {raw}，建立索引 {n} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载中文儿童图文数据并建索引")
    parser.add_argument("--max_items", type=int, default=None, help="索引条数上限，便于快速跑通")
    args = parser.parse_args()
    main(max_items=args.max_items)
