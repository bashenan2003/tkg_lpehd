"""
GPT-3.5 Turbo Provider
======================
通过 OpenAI 兼容接口调用 GPT-3.5 Turbo。
默认使用 OpenAI 官方 API, 也可通过设置 OPENAI_API_BASE 使用代理。
"""

import os
import logging
from typing import Optional

import requests

from ..config import LLM_TIMEOUT
from .base import LLMProvider
from . import register_provider

logger = logging.getLogger(__name__)

# GPT-3.5 配置 (环境变量可覆盖)
GPT35_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT35_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
GPT35_MODEL = os.environ.get("GPT35_MODEL", "gpt-3.5-turbo")


class GPT35TurboProvider(LLMProvider):
    """OpenAI GPT-3.5 Turbo Provider."""

    @property
    def name(self) -> str:
        return "gpt35_turbo"

    @property
    def model_name(self) -> str:
        return GPT35_MODEL

    def is_available(self) -> bool:
        return bool(GPT35_API_KEY)

    def _chat_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        if not GPT35_API_KEY:
            logger.warning("[gpt35_turbo] OPENAI_API_KEY 未设置")
            return None

        url = f"{GPT35_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {GPT35_API_KEY}",
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
            logger.warning(f"[gpt35_turbo] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"[gpt35_turbo] 超时 ({LLM_TIMEOUT}s)")
            return None
        except Exception as e:
            logger.warning(f"[gpt35_turbo] 异常: {e}")
            return None


register_provider("gpt35_turbo", GPT35TurboProvider)
