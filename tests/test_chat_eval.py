from src.chat_eval import check_output, recall_ok


def test_check_output_flags_empty_and_repetition():
    assert check_output("", stopped=True)["ok"] is False
    rep = "哈哈哈哈" * 40
    assert check_output(rep, stopped=True)["no_repeat"] is False
    assert check_output(rep, stopped=True)["ok"] is False


def test_check_output_allows_short_answers():
    # 短回答（二元组不足 4 个）不做复读判定，正常停止即通过
    out = check_output("北京", stopped=True)
    assert out == {"non_empty": True, "no_repeat": True, "stopped": True, "ok": True}
    assert check_output("北京", stopped=False)["ok"] is False


def test_check_output_requires_stop():
    assert check_output("这是一个正常的回答。", stopped=False)["ok"] is False
    assert check_output("这是一个正常的回答。", stopped=True)["ok"] is True


def test_recall_ok():
    assert recall_ok("北京的简称是京。", "北京") is True
    assert recall_ok("不知道。", "北京") is False
