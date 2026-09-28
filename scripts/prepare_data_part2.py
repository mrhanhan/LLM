"""把 TinyStories 中文语料编码成 data/processed/text/{train,val}.bin。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import iter_texts, build_text_bin
from src.tokenizer import QwenTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", default="data/tokenizer/qwen2.5-0.5b")
    ap.add_argument("--raw_dir", default="data/raw/text/tinystories_zh/jsonl")
    ap.add_argument("--out", default="data/processed/text/train.bin")
    ap.add_argument("--val", default="data/processed/text/val.bin")
    ap.add_argument("--max_docs", type=int, default=None)
    args = ap.parse_args()

    tok = QwenTokenizer.load(args.tokenizer_dir)
    files = sorted(str(p) for p in Path(args.raw_dir).glob("*.jsonl"))
    print(f"语料文件 {len(files)} 个，开始编码…")
    n_train, n_val = build_text_bin(
        tok, iter_texts(files), args.out, val_bin_path=args.val, max_docs=args.max_docs
    )
    print(f"完成：train={n_train} tokens, val={n_val} tokens")


if __name__ == "__main__":
    main()
