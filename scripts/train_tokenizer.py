"""训练手写字节级 BPE 分词器并保存到 data/tokenizer/bpe_16k/。

教学用途：语料子集上真训一个 16k 词表，用于与 Qwen 分词器做对比实验。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data import iter_texts
from src.tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/text/tinystories_zh/jsonl")
    ap.add_argument("--out", default="data/tokenizer/bpe_16k")
    ap.add_argument("--vocab_size", type=int, default=16384)
    ap.add_argument("--max_bytes", type=int, default=20_000_000)
    args = ap.parse_args()

    files = sorted(str(p) for p in Path(args.raw_dir).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"未找到 jsonl 语料：{args.raw_dir}")
    print(f"从 {len(files)} 个文件训练 BPE，目标词表 {args.vocab_size}，"
          f"最多读取 {args.max_bytes} 字节…")
    tok = BPETokenizer.train(iter_texts(files), vocab_size=args.vocab_size,
                             max_bytes=args.max_bytes)
    tok.save(args.out)
    print(f"完成：词表 {tok.vocab_size}，已保存到 {args.out}")


if __name__ == "__main__":
    main()
