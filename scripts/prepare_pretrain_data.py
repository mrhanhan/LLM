"""准备预训练语料：采样 fineweb-2 中文文本并编码成 uint32 二进制。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import load_config
from src.data import FINEWEB_REPO, build_text_bin, read_fineweb_cached
from src.hfenv import setup_hf
from src.tokenizer import QwenTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpt_fineweb.yaml")
    ap.add_argument("--endpoint", default="", help="空=直连 HF；可传 https://hf-mirror.com")
    ap.add_argument("--proxy", default=None, help="HTTP 代理，如 http://127.0.0.1:7890")
    ap.add_argument("--max_bytes", type=int, default=1_000_000_000, help="预训练文本字节上限")
    ap.add_argument("--out", default="data/processed/text/fineweb_cmn.train.bin")
    ap.add_argument("--val", default="data/processed/text/fineweb_cmn.val.bin")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_hf(args.endpoint or cfg.data.hf_endpoint, args.proxy or cfg.data.proxy)
    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)

    train_txt = read_fineweb_cached(FINEWEB_REPO, "cmn_Hani", "train",
                                    "000_00000.parquet", args.max_bytes,
                                    "data/raw/text/fineweb_cmn/train.jsonl.gz")
    val_txt = read_fineweb_cached(FINEWEB_REPO, "cmn_Hani", "test",
                                  "000_00000.parquet", max(2_000_000, args.max_bytes // 200),
                                  "data/raw/text/fineweb_cmn/val.jsonl.gz")
    n_train, _ = build_text_bin(tok, train_txt, args.out, val_bin_path=None, val_ratio=0.0)
    n_val, _ = build_text_bin(tok, val_txt, args.val, val_bin_path=None, val_ratio=0.0)
    print(f"完成：train={n_train} tokens, val={n_val} tokens")


if __name__ == "__main__":
    main()
