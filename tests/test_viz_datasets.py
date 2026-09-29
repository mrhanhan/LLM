import numpy as np
import pytest
import torch

from viz.datasets import (
    DATASETS, list_datasets, load_pretrain, resolve_spec,
    PretrainBatches, SFTBatches,
)


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


def test_pretrain_batches_shapes():
    data = np.arange(1000, dtype=np.int64)
    b = PretrainBatches(data, batch_size=4, block=8, device="cpu")
    x, y = b.next()
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_sft_batches_masks_padding(tmp_path):
    npz = tmp_path / "toy.npz"
    np.savez(npz,
             ids=np.array([5, 6, 7, 8, 9, 10], dtype=np.int32),
             labels=np.array([-100, 6, 7, -100, 9, 10], dtype=np.int32),
             offsets=np.array([0, 3, 6], dtype=np.int64))
    b = SFTBatches(str(npz), batch_size=2, max_len=16, pad_id=0, device="cpu")
    inputs, labels = b.next()
    assert inputs.shape == labels.shape
    assert labels.dtype == torch.long and inputs.dtype == torch.long
    assert int((labels == -100).sum()) >= 2
