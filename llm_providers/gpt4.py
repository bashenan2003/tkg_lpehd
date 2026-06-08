"""
GPT-4 Provider (Baseline — 真实 OpenAI API)
=============================================
论文 Baseline: GPT-4 + Original Prompt + Stochastic Decoding (temp=0.7).

配置方式:
  环境变量 OPENAI_API_KEY  —  OpenAI API 密钥
  环境变量 OPENAI_API_BASE —  默认 https://api.openai.com/v1
  环境变量 GPT4_MODEL     —  默认 gpt-4

使用:
  TKG_LLM_PROVIDER=gpt4 python -m tkg_lpehd.main --dataset ICEWS14
"""

import os
import logging
from typing import Optional

import requests

from .base import LLMProvider
from . import register_provider

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
GPT4_MODEL = os.environ.get("GPT4_MODEL", "gpt-4")
GPT4_TIMEOUT = int(os.environ.get("GPT4_TIMEOUT", "30"))


class GPT4Provider(LLMProvider):
    """
    GPT-4 Provider — 通过 OpenAI 官方 API 调用。

    论文 Baseline 参数:
      - model: gpt-4
      - temperature: 0.7 (随机解码)
      - prompt: Appendix B 原始版
    """

    @property
    def name(self) -> str:
        return "gpt4"

    @property
    def model_name(self) -> str:
        return GPT4_MODEL

    def is_available(self) -> bool:
        if not OPENAI_API_KEY:
            logger.info("GPT-4 不可用: 未设置 OPENAI_API_KEY 环境变量")
            return False
        return True

    def _chat_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        url = f"{OPENAI_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GPT4_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=GPT4_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.warning(
                    f"OpenAI API 返回 {resp.status_code}: {resp.text[:300]}"
                )
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"OpenAI API 超时 ({GPT4_TIMEOUT}s)")
            return None
        except Exception as e:
            logger.warning(f"OpenAI API 异常: {e}")
            return None


register_provider("gpt4", GPT4Provider)
