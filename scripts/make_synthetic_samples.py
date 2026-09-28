"""生成若干合成图文样本并保存为 png + 文本，肉眼确认图与描述一致。"""
import sys
from pathlib import Path

# 教学注释：Windows 控制台默认 cp1252，打印中文会 UnicodeEncodeError，强制 stdout 用 utf-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import SyntheticImageDataset

OUT = Path("data/processed/image/synthetic_preview")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ds = SyntheticImageDataset(num_samples=6, img_size=128, seed=42)
    for i in range(len(ds)):
        item = ds[i]
        img = (item["image"] * 0.5 + 0.5).clamp(0, 1) * 255
        from PIL import Image
        import numpy as np
        Image.fromarray(img.permute(1, 2, 0).numpy().astype("uint8")).save(OUT / f"{i}.png")
        (OUT / f"{i}.txt").write_text(item["caption"], encoding="utf-8")
    print(f"已保存 {len(ds)} 组样本到 {OUT}")
