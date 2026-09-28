"""把 TinyStories 中文语料编码成 data/processed/text/{train,val}.bin。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 教学注释：Windows 控制台/管道默认 cp1252，打印中文会 UnicodeEncodeError，强制 stdout 用 utf-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data import iter_texts, build_text_bin


def build_tokenizer(kind: str, tokenizer_dir: str):
    """按类型加载分词器：qwen（默认）| bpe | char。

    教学注释：三种分词器都实现同一套接口（encode/decode/vocab_size/special_id），
    预处理脚本只关心接口，不关心具体实现，所以这里可以按配置切换。
    """
    if kind == "qwen":
        from src.tokenizer import QwenTokenizer
        return QwenTokenizer.load(tokenizer_dir)
    if kind == "bpe":
        from src.tokenizer import BPETokenizer
        return BPETokenizer.load(tokenizer_dir)
    from src.tokenizer import CharTokenizer
    return CharTokenizer.load(tokenizer_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_kind", default="qwen", choices=["qwen", "bpe", "char"])
    ap.add_argument("--tokenizer_dir", default="data/tokenizer/qwen2.5-0.5b")
    ap.add_argument("--raw_dir", default="data/raw/text/tinystories_zh/jsonl")
    ap.add_argument("--out", default="data/processed/text/train.bin")
    ap.add_argument("--val", default="data/processed/text/val.bin")
    ap.add_argument("--max_docs", type=int, default=None)
    args = ap.parse_args()

    tok = build_tokenizer(args.tokenizer_kind, args.tokenizer_dir)
    files = sorted(str(p) for p in Path(args.raw_dir).glob("*.jsonl"))
    print(f"分词器 {args.tokenizer_kind}（词表 {tok.vocab_size}），语料文件 {len(files)} 个，开始编码…")
    n_train, n_val = build_text_bin(
        tok, iter_texts(files), args.out, val_bin_path=args.val, max_docs=args.max_docs
    )
    print(f"完成：train={n_train} tokens, val={n_val} tokens")


if __name__ == "__main__":
    main()
