from src.chat_format import (SYSTEM_PROMPT, build_sft_example, has_supervision,
                             history_to_messages, im_end_id, im_start_id,
                             render_messages, stop_ids, strip_special_text)
from src.tokenizer import QwenTokenizer

TOK = QwenTokenizer.load("data/tokenizer/qwen2.5-0.5b")


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
    sup_ids = [ids[i] for i, l in enumerate(labels) if l != -100]
    assert has_supervision(labels)
    assert TOK.encode("很高兴见到你")[0] in sup_ids
    assert im_end_id(TOK) in sup_ids
    # 用户段所有位置都不监督（用前缀长度精确定位用户内容起点）
    user_start = len(TOK.encode(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"))
    for k in range(len(TOK.encode("你好"))):
        assert labels[user_start + k] == -100
    # 序列最后是 "\n"(-100)，其前是助手轮末 <|im_end|>(被监督)
    assert labels[-1] == -100 and labels[-2] != -100


def test_sft_example_truncates_keeping_last_turns():
    msgs = [{"role": "user", "content": "第一轮问题" * 30},
            {"role": "assistant", "content": "第一轮回答" * 30},
            {"role": "user", "content": "最后一问"},
            {"role": "assistant", "content": "最后答案"}]
    ids, labels = build_sft_example(TOK, msgs, max_len=64)
    assert len(ids) <= 64 and len(ids) == len(labels)
    assert has_supervision(labels)
    assert TOK.encode("最后答案")[0] in [ids[i] for i, l in enumerate(labels) if l != -100]


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
