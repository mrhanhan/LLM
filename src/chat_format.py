"""对话模板：复用 Qwen 的 <|im_start|>/<|im_end|>，构造 SFT 样本（只监督助手）。

教学注释：Qwen2.5 分词器词表里已有 <|im_start|>(151644) 与 <|im_end|>(151645)，
所以既不扩表也不改 embedding；id 一律用 convert_tokens_to_ids 动态取，避免写死。
"""
from __future__ import annotations

SYSTEM_PROMPT = "你是一个乐于助人的中文助手。"


def _ids(tok, text: str) -> list[int]:
    return tok.encode(text)


def im_start_id(tok) -> int:
    if hasattr(tok, "hf"):
        return tok.hf.convert_tokens_to_ids("<|im_start|>")
    return _ids(tok, "<|im_start|>")[0]


def im_end_id(tok) -> int:
    if hasattr(tok, "hf"):
        return tok.hf.convert_tokens_to_ids("<|im_end|>")
    return _ids(tok, "<|im_end|>")[0]


def render_messages(tok, messages: list[dict], add_generation_prompt: bool = False) -> str:
    """把消息列表渲染成 Qwen 对话格式字符串。"""
    out = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


def _segment(tok, role: str, content: str):
    """返回 (ids, labels)：仅 assistant 段的 content 与其后 <|im_end|> 计入损失。"""
    head = _ids(tok, f"<|im_start|>{role}\n")
    body = _ids(tok, content)
    end = [im_end_id(tok)]
    nl = _ids(tok, "\n")
    ids = head + body + end + nl
    if role == "assistant":
        labels = [-100] * len(head) + body + end + [-100] * len(nl)
    else:
        labels = [-100] * len(ids)
    return ids, labels


def build_sft_example(tok, messages: list[dict], max_len: int = 2048):
    """构造 SFT 的 (input_ids, labels)；超长则从最早的消息开始丢弃，保留最近若干轮。

    教学注释：与 render_messages 一样自动补 system 段，保证训练与推理的序列前缀一致。
    """
    msgs = list(messages)
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    segs = [_segment(tok, m["role"], m["content"]) for m in msgs]
    out_ids: list[int] = []
    out_labels: list[int] = []
    for ids, labels in reversed(segs):
        if out_ids and len(ids) + len(out_ids) > max_len:
            break
        out_ids = ids + out_ids
        out_labels = labels + out_labels
    if len(out_ids) > max_len:                      # 单条消息就超长：保守截断
        out_ids = out_ids[:max_len]
        out_labels = out_labels[:max_len]
    return out_ids, out_labels


def has_supervision(labels: list[int]) -> bool:
    return any(l != -100 for l in labels)


def stop_ids(tok) -> list[int]:
    """生成时的停止 token：正常情况下模型应输出 <|im_end|>，eos 兜底。"""
    return [im_end_id(tok), tok.special_id("eos")]


def strip_special_text(tok, text: str) -> str:
    """去掉回答里可能残留的特殊 token 字符串（显示用）。"""
    for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>",
                  "<image>", "<pad>", "<bos>", "<eos>", "<unk>"):
        text = text.replace(token, "")
    return text.strip()


def history_to_messages(history) -> list[dict]:
    """把 Gradio 的 history（元组或 dict 形式）转成 messages 列表。"""
    msgs: list[dict] = []
    for item in history or []:
        if isinstance(item, dict):
            msgs.append({"role": item["role"], "content": item["content"]})
        else:
            user, bot = item[0], item[1]
            if user:
                msgs.append({"role": "user", "content": user})
            if bot:
                msgs.append({"role": "assistant", "content": bot})
    return msgs
