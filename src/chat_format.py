"""对话模板：复用 Qwen 的 <|im_start|>/<|im_end|>，构造 SFT 样本（只监督助手）。

教学注释：Qwen2.5 分词器词表里已有 <|im_start|>(151644) 与 <|im_end|>(151645)，
所以既不扩表也不改 embedding；id 一律用 convert_tokens_to_ids 动态取，避免写死。
"""
from __future__ import annotations

SYSTEM_PROMPT = "你是一个乐于助人的中文助手。"

_TURN_END = "<|im_end|>\n"


def _ids(tok, text: str) -> list[int]:
    return tok.encode(text)


def _turn_head(role: str) -> str:
    """一轮消息的角色前缀行；render 与训练共用，避免格式漂移。"""
    return f"<|im_start|>{role}\n"


def _turn_text(role: str, content: str) -> str:
    """完整渲染一轮消息；render_messages 与 build_sft_example 共用同一格式。"""
    return _turn_head(role) + content + _TURN_END


def im_start_id(tok) -> int:
    if hasattr(tok, "hf"):
        return tok.hf.convert_tokens_to_ids("<|im_start|>")
    return _ids(tok, "<|im_start|>")[0]


def im_end_id(tok) -> int:
    if hasattr(tok, "hf"):
        return tok.hf.convert_tokens_to_ids("<|im_end|>")
    return _ids(tok, "<|im_end|>")[0]


def render_messages(tok, messages: list[dict], add_generation_prompt: bool = False) -> str:
    """把消息列表渲染成 Qwen 对话格式字符串。

    固定的 SYSTEM_PROMPT 是训练与推理唯一的 system 前缀；messages 里若出现
    system 角色会被跳过，避免产生第二个（且与训练不一致的）system 轮。
    """
    out = [_turn_text("system", SYSTEM_PROMPT)]
    for m in messages:
        if m["role"] == "system":
            continue
        out.append(_turn_text(m["role"], m["content"]))
    if add_generation_prompt:
        out.append(_turn_head("assistant"))
    return "".join(out)


def _segment(tok, role: str, content: str):
    """返回 (ids, labels)：仅 assistant 段的 content 与其后 <|im_end|> 计入损失。"""
    head = _ids(tok, _turn_head(role))
    body = _ids(tok, content)
    end = [im_end_id(tok)]
    nl = _ids(tok, "\n")
    ids = head + body + end + nl
    if role == "assistant":
        labels = [-100] * len(head) + body + end + [-100] * len(nl)
    else:
        labels = [-100] * len(ids)
    return ids, labels


def _system_segment(tok):
    """固定 system 前缀的 (ids, labels)（全部 -100），训练与推理共用。"""
    ids = _ids(tok, _turn_text("system", SYSTEM_PROMPT))
    return ids, [-100] * len(ids)


def build_sft_example(tok, messages: list[dict], max_len: int = 2048):
    """构造 SFT 的 (input_ids, labels)；超长时始终保留 system 前缀，从最早的轮次丢弃。

    教学注释：system 前缀与 render_messages 完全一致，保证训练/推理分布一致；
    若单条消息仍超长，则保留其尾部，确保结尾的助手内容及其 <|im_end|> 被监督。
    """
    sys_ids, sys_labels = _system_segment(tok)
    segs = [_segment(tok, m["role"], m["content"])
            for m in messages if m["role"] != "system"]
    keep = max_len - len(sys_ids)
    body_ids: list[int] = []
    body_labels: list[int] = []
    for ids, labels in reversed(segs):
        if body_ids and len(ids) + len(body_ids) > keep:
            break
        body_ids = ids + body_ids
        body_labels = labels + body_labels
    if len(body_ids) > keep:                        # 单条消息就超长：保留尾部
        if keep > 0:
            body_ids = body_ids[-keep:]
            body_labels = body_labels[-keep:]
        else:
            body_ids = []
            body_labels = []
    out_ids = sys_ids + body_ids
    out_labels = sys_labels + body_labels
    if len(out_ids) > max_len:                      # 前缀本身已超长：保守截断
        out_ids = out_ids[:max_len]
        out_labels = out_labels[:max_len]
    # 对齐约定（与 GPT.forward 一致）：logits[t] 预测的是 token t+1，而交叉熵直接比较
    # logits[t] 与 targets[t]，所以 labels[t] 必须是 ids[t+1]。上面的 labels 仍与 ids
    # 逐位对齐，这里整体左移一位并给末位补 -100（ignore_index）：每个被监督 token 的
    # 目标于是挂在它前一个位置上——第一个 assistant body token 的目标就落在它的前一位。
    out_labels = out_labels[1:] + [-100]
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
