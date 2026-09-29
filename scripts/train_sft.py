"""SFT 训练：从预训练 checkpoint 初始化，用对话数据做指令微调。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.config import apply_overrides, load_config
from src.model import GPT
from src.sft_data import SFTDataset
from src.tokenizer import QwenTokenizer
from src.trainer import SFTTrainer
from src.utils import human_params, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_zh.yaml")
    ap.add_argument("--init", default="out/gpt_pretrain/latest.pt", help="预训练权重")
    ap.add_argument("--resume", default=None, help="从 SFT checkpoint 续训")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    set_seed(cfg.train.seed)
    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
    cfg.model.vocab_size = tok.vocab_size

    model = GPT(cfg.model)
    if args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = args.init
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"])
        print(f"已从 {ckpt_path} 加载权重")
    else:
        print(f"[warn] 未找到 {ckpt_path}，将从头训练")
    print(f"模型参数量 {human_params(model.num_params())}")

    train_ds = SFTDataset(cfg.data.sft_train)
    val_ds = SFTDataset(cfg.data.sft_val) if Path(cfg.data.sft_val).exists() else None
    print(f"SFT 样本：train={len(train_ds)}, val={len(val_ds) if val_ds else 0}")
    trainer = SFTTrainer(model, cfg.train, train_ds, tok, max_len=cfg.data.max_len, val_ds=val_ds)
    if args.resume:
        trainer.load(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
