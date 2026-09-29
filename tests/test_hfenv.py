import os
from src.hfenv import setup_hf


def test_setup_hf_proxy_direct(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "HF_ENDPOINT", "NO_PROXY"):
        monkeypatch.delenv(k, raising=False)
    setup_hf("", "http://127.0.0.1:7890")
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert "HF_ENDPOINT" not in os.environ          # 直连：不能残留镜像端点
    assert "127.0.0.1" in os.environ["NO_PROXY"]


def test_setup_hf_endpoint_no_proxy(monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    setup_hf("https://hf-mirror.com", None)
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert "HTTP_PROXY" not in os.environ
