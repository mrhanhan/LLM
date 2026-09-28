"""训练文本 GPT：读取 yaml 配置，支持命令行覆盖与 --resume。"""
import argparse
import sys
from pathlib import Path

# 教学注释：Windows 控制台默认 cp1252，打印中文会 UnicodeEncodeError，强制 stdout 用 utf-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, apply_overrides
from src.data import load_bin
from src.model import GPT
from src.tokenizer import QwenTokenizer, BPETokenizer, CharTokenizer
from src.trainer import Trainer
from src.utils import human_params, set_seed


def build_tokenizer(cfg):
    kind = cfg.data.tokenizer_kind
    if kind == "qwen":
        return QwenTokenizer.load(cfg.data.tokenizer_dir)
    if kind == "bpe":
        return BPETokenizer.load(cfg.data.tokenizer_dir)
    return CharTokenizer.load(cfg.data.tokenizer_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpt_tinystories.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint 路径")
    ap.add_argument("--set", nargs="*", default=[], help="覆盖项，如 train.lr=3e-4")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    set_seed(cfg.train.seed)

    tok = build_tokenizer(cfg)
    cfg.model.vocab_size = tok.vocab_size  # 词表大小必须与分词器一致
    print(f"分词器 {cfg.data.tokenizer_kind}，词表 {tok.vocab_size}")

    train_data = load_bin(cfg.data.train_bin)
    val_data = load_bin(cfg.data.val_bin)
    model = GPT(cfg.model)
    print(f"模型参数量 {human_params(model.num_params())}（不含词表 "
          f"{human_params(model.num_params(non_embedding=True))}）")

    trainer = Trainer(model, cfg.train, train_data, val_data, tokenizer=tok)
    if args.resume:
        step = trainer.load(args.resume)
        print(f"从 {args.resume} 恢复，已完成 {step} 步")
    trainer.train()


if __name__ == "__main__":
    main()
