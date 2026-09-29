import gzip

import pyarrow as pa
import pyarrow.parquet as pq

from src.data import iter_parquet_text, read_fineweb_cached


def _make_parquet(path, texts):
    pq.write_table(pa.table({"text": texts}), path)


def test_iter_parquet_text_respects_byte_budget(tmp_path):
    p = tmp_path / "a.parquet"
    _make_parquet(p, ["你好世界", "再见了", "第三句"])
    got = list(iter_parquet_text(pq.ParquetFile(p), max_bytes=6))  # "你好世界"=12 字节
    assert got == ["你好世界"]


def test_iter_parquet_text_no_budget_reads_all(tmp_path):
    p = tmp_path / "b.parquet"
    _make_parquet(p, ["a", "b"])
    assert list(iter_parquet_text(pq.ParquetFile(p))) == ["a", "b"]


def test_read_fineweb_cached_uses_cache_without_network(tmp_path):
    cache = tmp_path / "train.jsonl.gz"
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        f.write("第一行\n第二行\n")
    out = list(read_fineweb_cached("n/a", "cmn_Hani", "train", "x.parquet",
                                   max_bytes=None, cache_path=str(cache)))
    assert out == ["第一行", "第二行"]
