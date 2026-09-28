"""配置系统：YAML <-> dataclass，支持命令行覆盖。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """模型结构超参（教学注释：这些数字决定参数量与显存占用）。"""
    vocab_size: int = 151936      # Qwen2.5 词表大小；用自定义 BPE 时改为 16384
    d_model: int = 768            # 隐藏维度
    n_layer: int = 12             # Transformer 层数
    n_head: int = 12              # 查询头数
    n_kv_head: int = 4            # KV 头数（GQA：少于 n_head 可省显存）
    d_ff: int = 2048              # SwiGLU 中间维度
    ctx_len: int = 1024           # 最大上下文长度
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True   # 输入 embedding 与输出投影权重共享


@dataclass
class TrainConfig:
    batch_size: int = 8           # 单次前向的 micro-batch
    grad_accum: int = 8           # 梯度累积步数（等效 batch = batch_size*grad_accum）
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    max_steps: int = 20000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 100
    save_interval: int = 1000
    log_interval: int = 20
    out_dir: str = "out/gpt"
    dtype: str = "bf16"           # bf16 | fp32
    compile: bool = False         # Windows 上默认关闭
    seed: int = 42


@dataclass
class DataConfig:
    tokenizer_kind: str = "qwen"  # qwen | bpe | char
    tokenizer_dir: str = "data/tokenizer/qwen2.5-0.5b"
    train_bin: str = "data/processed/text/train.bin"
    val_bin: str = "data/processed/text/val.bin"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        data=DataConfig(**raw.get("data", {})),
    )


def _coerce(value: str, target_type):
    """把字符串命令行的值转成目标字段类型。"""
    if target_type is bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """支持形如 'train.lr=3e-4' 的命令行覆盖。"""
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=")[0]:
            raise ValueError(f"非法覆盖参数: {item!r}，应形如 train.lr=3e-4")
        key, value = item.split("=", 1)
        section, attr = key.split(".", 1)
        obj = getattr(cfg, section)
        if not hasattr(obj, attr):
            raise AttributeError(f"未知配置项: {section}.{attr}")
        target_type = type(getattr(obj, attr))
        setattr(obj, attr, _coerce(value, target_type))
    return cfg


def save_config(cfg: Config, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
