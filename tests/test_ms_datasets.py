from pathlib import Path

from src.data import (
    iter_advertise_sft,
    iter_advertise_texts,
    iter_poetry_sft,
    iter_poetry_texts,
    split_poem,
)

POEMS = (
    "text1\n"
    "云髻高梳鬓不分，扫除虚室事元君。新糊白纸屏风上，尽画蓬莱五色云。\n"
    "春眠不觉晓，处处闻啼鸟。\n"
)

ADVERTS = (
    "content,summary\n"
    "\"上衣 牛仔布 白色\",\"简约而不简单的牛仔外套，十分百搭。\"\n"
    "\"连衣裙 红色\",\"浪漫红裙，尽显优雅。\"\n"
)


def _write(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_iter_poetry_texts(tmp_path):
    path = _write(tmp_path, "poetry.csv", POEMS)
    texts = list(iter_poetry_texts(path))
    assert len(texts) == 2
    assert texts[0].startswith("云髻高梳鬓不分")


def test_split_poem_keeps_content(tmp_path):
    text = "云髻高梳鬓不分，扫除虚室事元君。新糊白纸屏风上，尽画蓬莱五色云。"
    front, back = split_poem(text)
    assert front + back == text
    assert front and back
    assert split_poem("没有标点的句子") is None


def test_iter_poetry_sft(tmp_path):
    path = _write(tmp_path, "poetry.csv", POEMS)
    ex = list(iter_poetry_sft(path))
    assert ex
    for e in ex:
        msgs = e["messages"]
        assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
        assert msgs[0]["content"].startswith("请补全古诗：")
        assert msgs[1]["content"]


def test_iter_advertise_texts_and_sft(tmp_path):
    path = _write(tmp_path, "advertise.csv", ADVERTS)
    texts = list(iter_advertise_texts(path))
    assert len(texts) == 2
    assert "商品信息：" in texts[0] and "广告文案：" in texts[0]

    ex = list(iter_advertise_sft(path))
    assert len(ex) == 2
    user = ex[0]["messages"][0]["content"]
    assert "商品信息：" in user and "请写一段广告文案：" in user
    assert ex[0]["messages"][1]["content"].startswith("简约而不简单")


def test_max_items(tmp_path):
    path = _write(tmp_path, "advertise.csv", ADVERTS)
    assert len(list(iter_advertise_sft(path, max_items=1))) == 1
