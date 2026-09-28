"""命令行聊天：加载训练好的 checkpoint 续写。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.config import load_config
from src.model import GPT
from src.tokenizer import QwenTokenizer
from src.generate import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpt_tinystories.yaml")
    ap.add_argument("--ckpt", default="out/gpt/latest.pt")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
    cfg.model.vocab_size = tok.vocab_size
    model = GPT(cfg.model)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print("模型已加载。输入文字开始续写，输入 /quit 退出。")
    while True:
        prompt = input("你> ").strip()
        if prompt in ("/quit", "/exit", ""):
            break
        print("AI> " + generate(model, tok, prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, top_p=args.top_p))


if __name__ == "__main__":
    main()
