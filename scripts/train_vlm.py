"""训练 MiniVLM：合成数据快速验证 -> 真实儿童图文 + VQA。

教学注释：LLM 权重从文本阶段的 checkpoint 迁移（gpt_ckpt），
ViT 与投影层随机初始化，用图文数据把视觉特征对齐到语言空间。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import yaml

from src.config import load_config, apply_overrides
from src.data import SyntheticImageDataset, CaptionDataset
from src.tokenizer import QwenTokenizer
from src.trainer import VLMTrainer
from src.utils import set_seed, human_params
from src.vlm import build_vlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm_children.yaml")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--data_kind", default=None, choices=["synthetic", "children"])
    ap.add_argument("--use_qa", action="store_true", help="用 VQA 样本训练（问题->答案）")
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.data_kind:
        raw["data_kind"] = args.data_kind
    if args.use_qa:
        raw["use_qa"] = True
    cfg = apply_overrides(load_config(args.config), args.set)
    set_seed(cfg.train.seed)

    tok = QwenTokenizer.load(cfg.data.tokenizer_dir)
    cfg.model.vocab_size = tok.vocab_size  # 词表必须与分词器一致

    vlm_cfg = raw.get("vlm", {})
    model = build_vlm(cfg.model, vlm_cfg, gpt_ckpt=raw.get("gpt_ckpt"),
                      image_token_id=tok.special_id("image"))
    n_img = model.vision.num_patches
    print(f"MiniVLM 参数量 {human_params(model.num_params())}，图像 patch 数 {n_img}")

    data_kind = raw.get("data_kind", "synthetic")
    use_qa = bool(raw.get("use_qa", False))
    if data_kind == "synthetic":
        dataset = SyntheticImageDataset(num_samples=raw.get("num_samples", 20000),
                                        img_size=vlm_cfg.get("img_size", 128), seed=cfg.train.seed)
    else:
        dataset = CaptionDataset("data/processed/image/children/index.jsonl",
                                 img_size=vlm_cfg.get("img_size", 128))

    trainer = VLMTrainer(model, cfg.train, dataset, tok, num_image_tokens=n_img, use_qa=use_qa)
    if args.resume:
        print(f"从 {args.resume} 恢复，已完成 {trainer.load(args.resume)} 步")
    trainer.train()


if __name__ == "__main__":
    main()
