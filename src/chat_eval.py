"""对话评测的纯逻辑与固定题库（不依赖模型/网络，便于单测）。"""
from __future__ import annotations

from collections import Counter

PROMPTS = [
    "你好，请介绍一下你自己。", "什么是人工智能？", "给我讲一个一句话的笑话。",
    "把“今天天气很好”翻译成英文。", "列出三种水果。", "1 加 1 等于几？",
    "用一句话解释什么是太阳。", "推荐一本适合初学者读的书。", "什么是水循环？",
    "写一句鼓励人的话。", "苹果和香蕉哪个通常更甜？", "中国首都是哪里？",
    "用三个词形容大海。", "为什么要多喝水？", "把“谢谢你”改写得更正式一些。",
    "简单说说地球为什么有四季。", "什么是计算机？", "给“小猫”写一个比喻句。",
]
MULTI_TURN = [("我叫小明，请记住。", "小明"), ("我刚才说我叫什么？", "小明")]


def check_output(text: str, stopped: bool) -> dict:
    """检查单条回答：非空、无明显重复、能正常停止。

    教学注释：小模型最容易出的两种毛病是"空/极短"和"复读循环"，
    用最高频二元组占比来近似检测复读。
    """
    t = (text or "").strip()
    non_empty = len(t) >= 2
    grams = [t[i:i + 2] for i in range(len(t) - 1)]
    rep = (max(Counter(grams).values()) / len(grams)) if grams else 1.0
    no_repeat = rep < 0.5
    return {"non_empty": non_empty, "no_repeat": no_repeat, "stopped": stopped,
            "ok": non_empty and no_repeat and stopped}


def recall_ok(text: str, keyword: str) -> bool:
    """多轮召回：回答里是否出现前文的关键词。"""
    return bool(keyword) and keyword in (text or "")
