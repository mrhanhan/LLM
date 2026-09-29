from src.chat_format import (SYSTEM_PROMPT, build_sft_example, has_supervision,
                             history_to_messages, im_end_id, im_start_id,
                             render_messages, stop_ids, strip_special_text)
from src.tokenizer import QwenTokenizer

TOK = QwenTokenizer.load("data/tokenizer/qwen2.5-0.5b")


def _index_of(seq: list[int], sub: list[int]) -> int | None:
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return None


def test_im_tokens_single_id():
    assert TOK.encode("<|im_start|>") == [im_start_id(TOK)]
    assert TOK.encode("<|im_end|>") == [im_end_id(TOK)]


def test_render_messages_contains_roles_and_generation_prompt():
    s = render_messages(TOK, [{"role": "user", "content": "你好"}],
                        add_generation_prompt=True)
    assert s.startswith(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    assert "<|im_start|>user\n你好<|im_end|>\n" in s
    assert s.endswith("<|im_start|>assistant\n")


def test_sft_example_only_labels_assistant():
    msgs = [{"role": "user", "content": "你好"},
            {"role": "assistant", "content": "很高兴见到你"}]
    ids, labels = build_sft_example(TOK, msgs, max_len=256)
    assert len(ids) == len(labels)
    assert has_supervision(labels)
    # 监督目标（labels 的值 = 要预测的 token）：assistant 回答 + 轮末 <|im_end|>
    targets = [l for l in labels if l != -100]
    assert TOK.encode("很高兴见到你")[0] in targets
    assert im_end_id(TOK) in targets
    # 新对齐（labels[t] == ids[t+1]）：第一个 assistant body token 的目标
    # 挂在它的前一个位置（assistant 头部最后一个 token）上。
    p = _index_of(ids, TOK.encode("很高兴见到你"))
    assert p is not None and p > 0
    assert labels[p - 1] != -100 and labels[p - 1] == ids[p]
    # 用户段所有位置都不监督（用前缀长度精确定位用户内容起点）
    user_start = len(TOK.encode(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"))
    for k in range(len(TOK.encode("你好"))):
        assert labels[user_start + k] == -100
    # 序列最后是 "\n"(-100)；labels 左移一位后末位补 -100
    assert labels[-1] == -100


def test_sft_example_labels_are_next_token():
    """回归：每个监督位 labels[i] 都必须等于 ids[i+1]（下一 token 目标）。"""
    msgs = [{"role": "user", "content": "你好"},
            {"role": "assistant", "content": "很高兴见到你"}]
    ids, labels = build_sft_example(TOK, msgs, max_len=256)
    assert len(ids) == len(labels)
    assert has_supervision(labels)
    for i, l in enumerate(labels):
        if l != -100:
            assert i + 1 < len(ids), "监督位不能落在最后一个位置（否则不是左移）"
            assert l == ids[i + 1], f"位置 {i} 的标签不是下一 token"


def test_sft_example_truncates_keeping_last_turns():
    msgs = [{"role": "user", "content": "第一轮问题" * 30},
            {"role": "assistant", "content": "第一轮回答" * 30},
            {"role": "user", "content": "最后一问"},
            {"role": "assistant", "content": "最后答案"}]
    ids, labels = build_sft_example(TOK, msgs, max_len=64)
    assert len(ids) <= 64 and len(ids) == len(labels)
    assert has_supervision(labels)
    # 被保留的最后答案：其首 token 的目标是它的前一位
    p = _index_of(ids, TOK.encode("最后答案"))
    assert p is not None and p > 0
    assert labels[p - 1] != -100 and labels[p - 1] == ids[p]


def test_stop_ids_and_strip_special():
    assert im_end_id(TOK) in stop_ids(TOK)
    assert strip_special_text(TOK, "你好<|im_end|>") == "你好"
    assert strip_special_text(TOK, "<|endoftext|>再见") == "再见"


def test_history_to_messages_both_formats():
    pairs = [["你好", "嗨"]]                       # gradio 元组形式
    assert history_to_messages(pairs) == [{"role": "user", "content": "你好"},
                                          {"role": "assistant", "content": "嗨"}]
    dicts = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert history_to_messages(dicts) == dicts


def test_no_duplicate_or_custom_system():
    msgs = [{"role": "system", "content": "自定义"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "嗨"}]
    s = render_messages(TOK, msgs)
    assert s.count("<|im_start|>system") == 1
    assert SYSTEM_PROMPT in s and "自定义" not in s
    sys_ids = TOK.encode(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    ids, _ = build_sft_example(TOK, msgs)
    assert ids[:len(sys_ids)] == sys_ids
    assert "自定义" not in TOK.decode(ids)


def test_truncation_keeps_system_prefix():
    msgs = [{"role": "user", "content": "第一轮问题" * 30},
            {"role": "assistant", "content": "第一轮回答" * 30},
            {"role": "user", "content": "最后一问"},
            {"role": "assistant", "content": "最后答案"}]
    ids, labels = build_sft_example(TOK, msgs, max_len=64)
    sys_ids = TOK.encode(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    assert len(ids) <= 64 and len(ids) == len(labels)
    assert ids[:len(sys_ids)] == sys_ids
    assert has_supervision(labels)


def test_overlong_single_assistant_keeps_im_end():
    msgs = [{"role": "user", "content": "问"},
            {"role": "assistant", "content": "答" * 500}]
    ids, labels = build_sft_example(TOK, msgs, max_len=64)
    assert len(ids) <= 64 and len(ids) == len(labels)
    targets = [l for l in labels if l != -100]
    assert im_end_id(TOK) in targets
    assert TOK.encode("答")[0] in targets
