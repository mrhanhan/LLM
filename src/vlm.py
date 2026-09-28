"""多模态模型：ViT 视觉特征 -> MLP 投影 -> 替换 <image> 占位符的 embedding（LLaVA 式）。

教学注释：语言模型只认识 token embedding。做法是——先用 <image> 这个特殊 token
在文本序列里"占位"，得到 embedding 后，把占位位置的向量替换成图片的视觉特征。
这样对语言模型来说，图片和文字就是同一条序列，无需改动它的注意力结构。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model import GPT
from src.vision import VisionEncoder


class VisionProjector(nn.Module):
    """把视觉特征投影到语言模型维度（两层 MLP + GELU）。

    不同模态的特征空间不同（ViT 输出 d_vision 维、LLM 需要 d_model 维），
    投影层就是把它们"对齐"的桥梁。
    """

    def __init__(self, d_vision: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_vision, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MiniVLM(nn.Module):
    def __init__(self, gpt: GPT, vision: VisionEncoder, projector: VisionProjector, image_token_id: int):
        super().__init__()
        self.gpt = gpt
        self.vision = vision
        self.projector = projector
        self.image_token_id = image_token_id

    def replace_image_embeds(self, input_ids, pixel_values):
        """先把 token 转成 embedding，再用视觉特征填充 <image> 所在位置。"""
        embeds = self.gpt.embed(input_ids)                       # (B, T, d_model)
        vis = self.projector(self.vision(pixel_values))          # (B, N, d_model)
        mask = input_ids == self.image_token_id                  # (B, T)
        # 每行恰有 N 个 True，按行优先展开后与 vis.reshape(-1, d) 一一对应
        assert mask.sum(1).eq(vis.size(1)).all(), "图像 token 数量与 patch 数不一致"
        embeds = embeds.clone()
        embeds[mask] = vis.reshape(-1, vis.size(-1)).to(embeds.dtype)
        return embeds

    def forward(self, input_ids, pixel_values, labels=None):
        embeds = self.replace_image_embeds(input_ids, pixel_values)
        return self.gpt(inputs_embeds=embeds, targets=labels)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, train_cfg):
        """把**全部**参数（LLM + ViT + 投影层）分成衰减/不衰减两组。"""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        try:
            return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                     betas=(train_cfg.beta1, train_cfg.beta2),
                                     fused=torch.cuda.is_available())
        except TypeError:
            return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                     betas=(train_cfg.beta1, train_cfg.beta2))


def build_vlm(model_cfg, vlm_cfg: dict, gpt_ckpt: str | None = None, image_token_id: int = 0,
              device: str = "cpu") -> MiniVLM:
    """组装 MiniVLM；若给了 gpt_ckpt，则把 Plan 1 训练好的 LLM 权重迁移进来。

    教学注释：这就是"先教它说话，再给它装眼睛"——LLM 权重从文本阶段迁移，
    ViT 与投影层随机初始化，用图文数据把两者对齐。
    """
    gpt = GPT(model_cfg)
    if gpt_ckpt:
        ckpt = torch.load(gpt_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = gpt.load_state_dict(state, strict=False)
        print(f"LLM 权重迁移：missing={len(missing)} unexpected={len(unexpected)}")
    vision = VisionEncoder(
        d_vision=vlm_cfg.get("d_vision", 384),
        depth=vlm_cfg.get("depth", 6),
        n_head=vlm_cfg.get("n_head", 6),
        img_size=vlm_cfg.get("img_size", 128),
        patch_size=vlm_cfg.get("patch_size", 16),
    )
    projector = VisionProjector(vision.patch_embed.proj.out_channels, model_cfg.d_model)
    model = MiniVLM(gpt, vision, projector, image_token_id)
    return model.to(device)
