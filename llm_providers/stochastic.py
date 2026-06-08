"""
Stochastic Decoding Provider
=============================
论文 Baseline 随机解码: temperature=0.7, top_p=0.9.

对比确定性解码 (temp=0) 验证 LLM 稳定性 (论文 Section 5.6)。

配置方式:
  环境变量 STOCH_MODEL — 默认 gpt-4
  环境变量 OPENAI_API_KEY — OpenAI API 密钥

使用:
  TKG_LLM_PROVIDER=stochastic python -m tkg_lpehd.main --dataset ICEWS14
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
STOCH_MODEL = os.environ.get("STOCH_MODEL", "gpt-4")
STOCH_TIMEOUT = int(os.environ.get("STOCH_TIMEOUT", "30"))


class StochasticProvider(LLMProvider):
    """
    随机解码 Provider。

    论文设定:
      - temperature = 0.7 (引入多样性)
      - top_p = 0.9 (nucleus sampling)
      - 后端默认 GPT-4 (可通过 STOCH_MODEL 切换)

    用途: LLM 稳定性实验中作为 Baseline 解码策略,
    测试温度/采样方式变化对推理结果的影响。
    """

    @property
    def name(self) -> str:
        return "stochastic"

    @property
    def model_name(self) -> str:
        return f"stoch({STOCH_MODEL})"

    def is_available(self) -> bool:
        if not OPENAI_API_KEY:
            logger.info("Stochastic 不可用: 未设置 OPENAI_API_KEY")
            return False
        return True

    def _chat_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.7,  # 默认随机温度
    ) -> Optional[str]:
        # 强制使用随机解码参数
        url = f"{OPENAI_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": STOCH_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=STOCH_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.warning(
                    f"Stochastic API 返回 {resp.status_code}: {resp.text[:300]}"
                )
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"Stochastic API 超时 ({STOCH_TIMEOUT}s)")
            return None
        except Exception as e:
            logger.warning(f"Stochastic API 异常: {e}")
            return None


register_provider("stochastic", StochasticProvider)
