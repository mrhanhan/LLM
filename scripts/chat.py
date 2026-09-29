"""命令行聊天：--mode chat 走对话模板多轮，--mode base 走原来的续写。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.config import load_config
from src.generate import generate
from src.model import GPT
from src.tokenizer import QwenTokenizer


def load_model(cfg_path, ckpt, tok):
    cfg = load_config(cfg_path)
    cfg.model.vocab_size = tok.vocab_size
    model = GPT(cfg.model)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chat", "base"], default="chat")
    ap.add_argument("--config", default="configs/sft_zh.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    if args.ckpt is None:
        args.ckpt = "out/sft/latest.pt" if args.mode == "chat" else "out/gpt/latest.pt"
    tok = QwenTokenizer.load("data/tokenizer/qwen2.5-0.5b")
    model = load_model(args.config, args.ckpt, tok)

    if args.mode == "base":
        print("模型已加载（续写模式）。输入 /quit 退出。")
        while True:
            prompt = input("你> ").strip()
            if prompt in ("/quit", "/exit", ""):
                break
            print("AI> " + generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_p=args.top_p))
        return

    from src.chat_format import (render_messages, stop_ids, strip_special_text)
    print("模型已加载（对话模式）。输入 /reset 清空历史，/quit 退出。")
    messages = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    while True:
        user = input("你> ").strip()
        if user in ("/quit", "/exit", ""):
            break
        if user == "/reset":
            messages = []
            print("[已清空对话历史]")
            continue
        messages.append({"role": "user", "content": user})
        prompt = render_messages(tok, messages, add_generation_prompt=True)
        reply = generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_p=args.top_p,
                         device=device, stop_ids=stop_ids(tok))
        reply = strip_special_text(tok, reply)
        messages.append({"role": "assistant", "content": reply})
        print("AI> " + reply)


if __name__ == "__main__":
    main()
