"""自动评测：固定留出问题 → 生成 → 检查非空/不重复/正常停止/多轮引用。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_zh.yaml")
    ap.add_argument("--ckpt", default="out/sft/latest.pt")
    ap.add_argument("--out", default="out/sft/eval.md")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    import torch
    from src.chat_eval import MULTI_TURN, PROMPTS, check_output, recall_ok
    from src.chat_format import render_messages, stop_ids, strip_special_text
    from src.config import load_config
    from src.generate import generate
    from src.model import GPT
    from src.tokenizer import QwenTokenizer

    tok = QwenTokenizer.load("data/tokenizer/qwen2.5-0.5b")
    cfg = load_config(args.config)
    cfg.model.vocab_size = tok.vocab_size
    model = GPT(cfg.model)
    if Path(args.ckpt).exists():
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
    else:
        print(f"[warn] 未找到 {args.ckpt}，使用随机权重")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stop = stop_ids(tok)

    def ask(messages):
        prompt = render_messages(tok, messages, add_generation_prompt=True)
        raw = generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                       temperature=0.7, top_p=0.9, device=device, stop_ids=stop,
                       add_bos=False)
        n_new = len(tok.encode(raw))
        return strip_special_text(tok, raw), n_new < args.max_new_tokens

    lines, n_ok = ["# SFT 对话评测", ""], 0
    for q in PROMPTS:
        out, stopped = ask([{"role": "user", "content": q}])
        res = check_output(out, stopped)
        n_ok += int(res["ok"])
        lines.append(f"- **Q**: {q}\n  - A: {out}\n  - {res}")

    messages = []
    for q, kw in MULTI_TURN:
        messages.append({"role": "user", "content": q})
        out, _ = ask(messages)
        messages.append({"role": "assistant", "content": out})
        ok = recall_ok(out, kw)
        n_ok += int(ok)
        lines.append(f"- **多轮 Q**: {q}\n  - A: {out}\n  - recall({kw})={ok}")

    total = len(PROMPTS) + len(MULTI_TURN)
    lines.insert(1, f"\n通过 {n_ok}/{total}。\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"评测完成：{n_ok}/{total} 通过，结果写入 {args.out}")


if __name__ == "__main__":
    main()
