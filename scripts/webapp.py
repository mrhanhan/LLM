"""Gradio 网页：文本聊天 + 图片描述/VQA。

教学注释：这是把前面两条链路（文本 GPT、MiniVLM）包一层可视化界面，
方便直接交互观察效果。命令行版分别是 scripts/chat.py 与 src/vlm_generate.py。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 教学注释：Windows 控制台默认 cp1252，打印中文会 UnicodeEncodeError，强制 stdout 用 utf-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import yaml
import torch
import gradio as gr

from src.config import ModelConfig, load_config
from src.model import GPT
from src.vlm import build_vlm
from src.tokenizer import QwenTokenizer
from src.generate import generate
from src.vlm_generate import generate_caption, answer_question
from src.data import image_to_tensor

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(args):
    tok = QwenTokenizer.load(args.tokenizer_dir)

    text_cfg = load_config(args.text_config)
    text_cfg.model.vocab_size = tok.vocab_size
    text_model = GPT(text_cfg.model)
    if Path(args.text_ckpt).exists():
        text_model.load_state_dict(torch.load(args.text_ckpt, map_location="cpu", weights_only=False)["model"])
    else:
        print(f"[warn] 未找到文本 checkpoint：{args.text_ckpt}（将使用随机权重）")

    vlm = None
    if Path(args.vlm_ckpt).exists():
        raw = yaml.safe_load(Path(args.vlm_config).read_text(encoding="utf-8"))
        vlm_cfg = dict(raw["model"])
        vlm_cfg["vocab_size"] = tok.vocab_size
        vlm = build_vlm(ModelConfig(**vlm_cfg), raw.get("vlm", {}),
                        gpt_ckpt=None, image_token_id=tok.special_id("image"), device=DEV)
        vlm.load_state_dict(torch.load(args.vlm_ckpt, map_location="cpu", weights_only=False)["model"])
    else:
        print(f"[warn] 未找到 VLM checkpoint：{args.vlm_ckpt}（图片页不可用）")
    return tok, text_model, vlm


def load_chat_model(args, tok):
    """加载 SFT 对话模型（存在才用）。"""
    if not Path(args.sft_ckpt).exists():
        return None
    cfg = load_config(args.sft_config)
    cfg.model.vocab_size = tok.vocab_size
    m = GPT(cfg.model)
    m.load_state_dict(torch.load(args.sft_ckpt, map_location="cpu", weights_only=False)["model"])
    print(f"已加载对话模型：{args.sft_ckpt}")
    return m


def build_demo(tok, chat_model, vlm):
    """构建 Gradio 界面（与启动分离，便于 --check 冒烟测试）。"""
    def chat_reply(message, history, max_tokens, temperature, top_p):
        from src.chat_format import (history_to_messages, render_messages,
                                     stop_ids, strip_special_text)
        if chat_model is None:
            return "未找到对话模型，请先训练：python scripts/train_sft.py"
        messages = history_to_messages(history)
        messages.append({"role": "user", "content": message})
        prompt = render_messages(tok, messages, add_generation_prompt=True)
        out = generate(chat_model, tok, prompt, max_new_tokens=int(max_tokens),
                       temperature=float(temperature), top_p=float(top_p), device=DEV,
                       stop_ids=stop_ids(tok), add_bos=False)
        return strip_special_text(tok, out)

    def caption_fn(image, question):
        if vlm is None:
            return "未找到 VLM checkpoint，请先训练：python scripts/train_vlm.py"
        if image is None:
            return "请上传图片"
        # 把图片缩放到训练时的尺寸：side = grid * patch_size
        side = vlm.vision.patch_embed.grid * vlm.vision.patch_embed.proj.kernel_size[0]
        img = image.convert("RGB").resize((side, side))
        tensor = image_to_tensor(img)
        if question and question.strip():
            return answer_question(vlm, tok, tensor, question.strip(), device=DEV)
        return generate_caption(vlm, tok, tensor, device=DEV)

    with gr.Blocks(title="mini-llm-lab") as demo:
        gr.Markdown("# mini-llm-lab（从零手写 LLM + VLM）")
        with gr.Tab("文本对话"):
            gr.ChatInterface(
                chat_reply,
                additional_inputs=[
                    gr.Slider(16, 512, value=256, step=16, label="生成长度"),
                    gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="temperature"),
                    gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p"),
                ],
            )
        with gr.Tab("图片描述 / VQA"):
            image = gr.Image(type="pil", label="上传图片")
            question = gr.Textbox(label="问题（留空 = 生成描述）", placeholder="例如：图中有几个图形？")
            out2 = gr.Textbox(label="回答")
            gr.Button("运行").click(caption_fn, [image, question], out2)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", default="data/tokenizer/qwen2.5-0.5b")
    ap.add_argument("--text_config", default="configs/gpt_tinystories.yaml")
    ap.add_argument("--text_ckpt", default="out/gpt/latest.pt")
    ap.add_argument("--sft_config", default="configs/sft_zh.yaml")
    ap.add_argument("--sft_ckpt", default="out/sft/latest.pt")
    ap.add_argument("--vlm_config", default="configs/vlm_children.yaml")
    ap.add_argument("--vlm_ckpt", default="out/vlm/latest.pt")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--check", action="store_true", help="只构建模型与界面并退出（冒烟测试）")
    args = ap.parse_args()

    tok, text_model, vlm = load_models(args)
    demo = build_demo(tok, load_chat_model(args, tok), vlm)
    if args.check:
        print(f"[check] 界面构建成功；VLM 可用：{vlm is not None}")
        return
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
