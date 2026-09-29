import numpy as np
import pytest

from viz.datasets import DATASETS, list_datasets, load_pretrain, resolve_spec


class _FakeTok:
    def encode(self, text):
        return [1, 2, 3]


def test_registry_shape_and_kinds():
    rows = list_datasets()
    ids = {r["id"] for r in rows}
    assert {"corpus_qwen", "tiny_stories", "poetry", "sft_chat"} <= ids
    for r in rows:
        assert r["kind"] in ("pretrain", "sft")
        assert r["tokenizer"] == "qwen"
        assert isinstance(r["available"], bool)


def test_resolve_spec_enforces_kind():
    assert resolve_spec("poetry", "pretrain")["id"] == "poetry"
    assert resolve_spec("sft_chat", "sft")["id"] == "sft_chat"
    with pytest.raises(ValueError):
        resolve_spec("poetry", "sft")
    with pytest.raises(ValueError):
        resolve_spec("sft_chat", "pretrain")
    with pytest.raises(KeyError):
        resolve_spec("nope", "pretrain")


def test_load_pretrain_builtin_uses_tokenizer():
    arr = load_pretrain(DATASETS["corpus_qwen"], _FakeTok())
    assert arr.dtype == np.int64 and arr.ndim == 1
    assert arr.tolist()[:3] == [1, 2, 3]
