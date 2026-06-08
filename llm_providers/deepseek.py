"""
DeepSeek Provider
=================
通过 OpenAI 兼容接口调用 DeepSeek 大模型。
"""

import logging
from typing import Optional

import requests

from ..config import (
    LLM_API_KEY, LLM_API_BASE, LLM_TIMEOUT,
)
from .base import LLMProvider
from . import register_provider

logger = logging.getLogger(__name__)


class DeepSeekProvider(LLMProvider):
    """DeepSeek Chat API Provider."""

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def model_name(self) -> str:
        return "deepseek-chat"

    def _chat_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        url = f"{LLM_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            logger.warning(f"[deepseek] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"[deepseek] 超时 ({LLM_TIMEOUT}s)")
            return None
        except Exception as e:
            logger.warning(f"[deepseek] 异常: {e}")
            return None


# 自动注册
register_provider("deepseek", DeepSeekProvider)
