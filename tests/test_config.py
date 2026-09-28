# 教学注释：配置系统测试——加载 YAML、命令行覆盖、保存回读一致性。
import textwrap
from pathlib import Path
from src.config import load_config, apply_overrides, save_config


def test_load_config_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  n_layer: 4\ntrain:\n  lr: 0.001\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.model.n_layer == 4
    assert cfg.train.lr == 0.001
    # 未提供的字段用默认值
    assert cfg.model.d_model == 768
    assert cfg.data.tokenizer_kind == "qwen"


def test_apply_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lr: 0.001\n", encoding="utf-8")
    cfg = load_config(str(p))
    cfg = apply_overrides(cfg, ["train.lr=3e-4", "model.n_layer=2", "train.compile=true"])
    assert cfg.train.lr == 3e-4
    assert cfg.model.n_layer == 2
    assert cfg.train.compile is True


def test_save_and_reload(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  d_model: 256\n", encoding="utf-8")
    cfg = load_config(str(p))
    out = tmp_path / "saved.yaml"
    save_config(cfg, str(out))
    cfg2 = load_config(str(out))
    assert cfg2.model.d_model == 256
