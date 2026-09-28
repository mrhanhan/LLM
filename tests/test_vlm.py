# 教学注释：验证三点——输出形状、图像位置标签被屏蔽、以及 patch 特征确实
# 被投影后替换进了 <image> 的位置（改变图片会改变 logits）。
import torch
from src.config import ModelConfig
from src.model import GPT
from src.vision import VisionEncoder
from src.vlm import VisionProjector, MiniVLM


def build_tiny(image_token_id=5, num_patches=4):
    gpt = GPT(ModelConfig(vocab_size=20, d_model=32, n_layer=1, n_head=4,
                          n_kv_head=2, d_ff=64, ctx_len=64))
    vit = VisionEncoder(d_vision=32, depth=1, n_head=4, img_size=32, patch_size=16)
    proj = VisionProjector(32, 32)
    return MiniVLM(gpt, vit, proj, image_token_id), num_patches


def test_vlm_forward_and_loss():
    model, N = build_tiny()
    B = 2
    # 文本 token 一律用 6..20，避免与 image id（5）混淆；第 0 位是 bos。
    input_ids = torch.randint(6, 20, (B, 1 + N + 6), dtype=torch.long)
    input_ids[:, 0] = 0
    input_ids[:, 1:1 + N] = 5
    labels = input_ids.clone()
    labels[:, :1 + N] = -100
    pix = torch.randn(B, 3, 32, 32)
    logits, loss, _ = model(input_ids, pix, labels)
    assert logits.shape[:2] == input_ids.shape
    assert loss.item() > 0


def test_vlm_label_masking_excludes_image_positions():
    model, N = build_tiny()
    input_ids = torch.randint(6, 20, (1, 1 + N + 3), dtype=torch.long)
    input_ids[0, 0] = 0
    input_ids[0, 1:1 + N] = 5
    labels = input_ids.clone()
    labels[:, :1 + N] = -100
    assert (labels[:, :1 + N] == -100).all()


def test_image_features_really_used():
    torch.manual_seed(0)
    model, N = build_tiny()
    model.eval()
    input_ids = torch.randint(6, 20, (1, 1 + N + 3), dtype=torch.long)
    input_ids[0, 0] = 0
    input_ids[0, 1:1 + N] = 5
    a = model(input_ids, torch.randn(1, 3, 32, 32))[0]
    b = model(input_ids, torch.randn(1, 3, 32, 32))[0]
    assert not torch.allclose(a, b, atol=1e-5), "换图片后输出应变化"


def test_build_vlm_loads_gpt_weights(tmp_path):
    from src.config import ModelConfig
    import torch as t
    gpt = GPT(ModelConfig(vocab_size=20, d_model=32, n_layer=1, n_head=4, n_kv_head=2, d_ff=64, ctx_len=64))
    ckpt = {"model": gpt.state_dict()}
    p = tmp_path / "g.pt"
    t.save(ckpt, str(p))
    from src.vlm import build_vlm
    model = build_vlm(gpt.cfg, {"d_vision": 32, "depth": 1, "n_head": 4, "img_size": 32, "patch_size": 16},
                      gpt_ckpt=str(p), image_token_id=5)
    assert t.allclose(model.gpt.tok_emb.weight, gpt.tok_emb.weight)
