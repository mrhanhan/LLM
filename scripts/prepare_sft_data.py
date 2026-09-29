"""准备 SFT 数据：下载 alpaca-zh + firefly 子集，归一化/合成多轮，编码成 npz。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import load_config
from src.hfenv import setup_hf
from src.sft_data import iter_sft_examples, save_sft_npz
from src.tokenizer import QwenTokenizer


def download_sft(dest_dir: str, firefly_items: int, proxy: str | None):
    """下载 alpaca-zh（整文件）与 firefly 前 N 条（流式，不整下 1.17GB）。"""
    from huggingface_hub import HfFileSystem

    root = Path(dest_dir)
    root.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    alp = root / "alpaca_gpt4_data_zh.json"
    if not alp.exists():
        with fs.open("datasets/shibing624/alpaca-zh/alpaca_gpt4_data_zh.json", "rb") as f:
            alp.write_bytes(f.read())
    ff = root / "firefly_subset.jsonl"
    if not ff.exists():
        with fs.open("datasets/YeungNLP/firefly-train-1.1M/firefly-train-1.1M.jsonl", "rb") as f:
            import io
            with ff.open("w", encoding="utf-8") as out:
                for i, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
                    if i >= firefly_items:
                        break
                    out.write(line)
    return [str(alp), str(ff)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_zh.yaml")
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--firefly_items", type=int, default=40000)
    ap.add_argument("--out", default="data/processed/sft/train.npz")
    ap.add_argument("--val", default="data/processed/sft/val.npz")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_hf(args.endpoint, args.proxy or cfg.data.proxy)
    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
    paths = download_sft("data/raw/sft", args.firefly_items, args.proxy or cfg.data.proxy)

    # 先全部读成列表，切分 train/val（固定顺序，保证可复现）
    examples = list(iter_sft_examples(paths, synth_prob=0.5, seed=0))
    n_val = max(1, int(len(examples) * args.val_ratio))
    train_ex, val_ex = examples[:-n_val], examples[-n_val:]
    n_tr = save_sft_npz(tok, train_ex, args.out, cfg.data.max_len)
    n_va = save_sft_npz(tok, val_ex, args.val, cfg.data.max_len)
    print(f"完成：train={n_tr}, val={n_va}（共 {len(examples)} 段对话）")


if __name__ == "__main__":
    main()
