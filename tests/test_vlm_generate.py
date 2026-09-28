# 教学注释：推理时复用训练同款 prompt 模板，才与训练分布一致。
import torch
from src.config import ModelConfig
from src.model import GPT
from src.vision import VisionEncoder
from src.vlm import VisionProjector, MiniVLM
from src.tokenizer import CharTokenizer
from src.vlm_generate import generate_caption, answer_question


def build(vocab_size=50, image_token_id=0):
    gpt = GPT(ModelConfig(vocab_size=vocab_size, d_model=32, n_layer=1, n_head=4,
                          n_kv_head=2, d_ff=64, ctx_len=128))
    vit = VisionEncoder(d_vision=32, depth=1, n_head=4, img_size=32, patch_size=16)
    return MiniVLM(gpt, vit, VisionProjector(32, 32), image_token_id=image_token_id)


def test_generate_caption_and_qa_return_strings():
    # 关键：image 特殊符号排在前面（id=0），字符 id 从 len(specials)=5 开始，避免冲突。
    specials = {"image": 0, "bos": 1, "eos": 2, "pad": 3, "unk": 4}
    tok = CharTokenizer(["你", "好", "红", "色", "圆", "形", "有", "没", "个"], specials)
    # GPT 词表必须等于分词器词表，否则生成的 id 无法被 tokenizer.decode 映射
    model = build(vocab_size=tok.vocab_size, image_token_id=0)
    img = torch.randn(3, 32, 32)
    cap = generate_caption(model, tok, img, max_new_tokens=8, device="cpu")
    ans = answer_question(model, tok, img, "有几个图形？", max_new_tokens=4, device="cpu")
    assert isinstance(cap, str) and isinstance(ans, str)
