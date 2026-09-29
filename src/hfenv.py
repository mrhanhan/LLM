"""统一配置 HuggingFace 访问：代理直连或镜像端点。

教学注释：本机通过 HTTP 代理（默认 127.0.0.1:7890）直连 huggingface.co；
若无代理则可用 hf-mirror.com 回退。集中在这里设置，避免各脚本各写一份。
"""
from __future__ import annotations

import os

_LOCAL = "127.0.0.1,localhost,::1"


def setup_hf(endpoint: str | None = None, proxy: str | None = None) -> None:
    """设置 HF_ENDPOINT 与 HTTP(S)_PROXY。

    - proxy 非空：走代理；endpoint 为空则直连官方 HF（清除镜像端点）。
    - endpoint 非空：使用该镜像端点。
    """
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        existing = os.environ.get("NO_PROXY", "")
        parts = [p for p in existing.split(",") if p]
        for host in _LOCAL.split(","):
            if host not in parts:
                parts.append(host)
        os.environ["NO_PROXY"] = ",".join(parts)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        # 直连（或仅走代理）：移除镜像端点，否则会打到 hf-mirror
        os.environ.pop("HF_ENDPOINT", None)
