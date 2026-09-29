"""准备 ModelScope 数据集：中文诗词集 + AdvertiseGen。

- 预训练：把文本用 Qwen 分词器编码成 uint32 .bin（train + val 切分）；
- SFT：诗词 = 前半→后半 续写；AdvertiseGen = 商品信息→广告文案，
  复用 src.sft_data 编码成 npz（labels 已按 chat 模板只监督 assistant）。

用法：
    & ".venv\\Scripts\\python.exe" scripts/prepare_modelscope_data.py --dataset all
    # 只做预训练 bin、跳过 SFT
    ... --dataset poetry --skip_sft
    # 只做 SFT、跳过 bin
    ... --dataset advertise --skip_bin
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import load_config
from src.data import (
    MODELSCOPE_DATASETS,
    build_text_bin,
    download_modelscope_dataset,
    iter_advertise_sft,
    iter_advertise_texts,
    iter_poetry_sft,
    iter_poetry_texts,
)
from src.sft_data import save_sft_npz
from src.tokenizer import QwenTokenizer


def _text_iter(name: str):
    return iter_poetry_texts if name == "poetry" else iter_advertise_texts


def _sft_iter(name: str):
    return iter_poetry_sft if name == "poetry" else iter_advertise_sft


def prepare_pretrain(tok, name, csv_paths, bin_dir, val_ratio):
    pieces = _text_iter(name)

    def gen():
        for p in csv_paths:
            if Path(p).exists():
                yield from pieces(str(p))

    train_bin = Path(bin_dir) / f"{name}.train.bin"
    val_bin = Path(bin_dir) / f"{name}.val.bin"
    n_train, n_val = build_text_bin(tok, gen(), str(train_bin),
                                    val_bin_path=str(val_bin), val_ratio=val_ratio)
    print(f"[{name}] 预训练：train={n_train} val={n_val} tokens -> {train_bin} / {val_bin}")


def prepare_sft(tok, name, csv_path, sft_dir, max_len, val_ratio, max_items):
    examples = list(_sft_iter(name)(csv_path, max_items=max_items))
    if not examples:
        print(f"[{name}] SFT：无可用样本，跳过")
        return
    n_val = max(1, int(len(examples) * val_ratio))
    train_ex, val_ex = examples[:-n_val], examples[-n_val:]
    train_npz = Path(sft_dir) / f"{name}.train.npz"
    val_npz = Path(sft_dir) / f"{name}.val.npz"
    n_tr = save_sft_npz(tok, train_ex, str(train_npz), max_len)
    n_va = save_sft_npz(tok, val_ex, str(val_npz), max_len)
    print(f"[{name}] SFT：train={n_tr} val={n_va}（共 {len(examples)} 条）-> {train_npz} / {val_npz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["poetry", "advertise", "all"], default="all")
    ap.add_argument("--config", default="configs/gpt_tinystories.yaml")
    ap.add_argument("--raw_dir", default="data/raw/modelscope")
    ap.add_argument("--bin_dir", default="data/processed/text")
    ap.add_argument("--sft_dir", default="data/processed/sft")
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--max_items", type=int, default=None, help="每个数据集 SFT 最多取多少条")
    ap.add_argument("--skip_bin", action="store_true", help="跳过预训练 .bin")
    ap.add_argument("--skip_sft", action="store_true", help="跳过 SFT .npz")
    ap.add_argument("--revision", default="master")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
    names = ["poetry", "advertise"] if args.dataset == "all" else [args.dataset]

    for name in names:
        spec = MODELSCOPE_DATASETS[name]
        local = download_modelscope_dataset(
            spec["repo_id"], str(Path(args.raw_dir) / name), args.revision)
        train_csv = Path(local) / spec["texts"]["train"]
        val_csv = Path(local) / spec["texts"]["val"]
        if not args.skip_bin:
            prepare_pretrain(tok, name, [train_csv, val_csv], args.bin_dir, args.val_ratio)
        if not args.skip_sft:
            prepare_sft(tok, name, str(train_csv), args.sft_dir,
                        cfg.data.max_len, args.val_ratio, args.max_items)

    print("完成。")


if __name__ == "__main__":
    main()
