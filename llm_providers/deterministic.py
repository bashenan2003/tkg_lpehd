"""
Deterministic Decoding Provider
===============================
确定性解码包装器: 强制 temperature=0, 禁用所有随机采样,
使用 greedy decoding 确保相同输入始终产生相同输出。

这是论文中的 "Deterministic Decoding" 策略:
- temperature = 0 (消除采样随机性)
- 可选: 多次调用取多数投票 (自一致性)
"""

import os
import logging
from typing import Optional

import requests

from ..config import (
    LLM_API_KEY, LLM_API_BASE, LLM_TIMEOUT,
)
from .base import LLMProvider
from . import register_provider

logger = logging.getLogger(__name__)

DET_MODEL = os.environ.get("DET_MODEL", "deepseek-chat")
# 自一致性投票次数 (1 = 单次 greedy)
SELF_CONSISTENCY_N = int(os.environ.get("DET_SELF_CONSISTENCY", "1"))


class DeterministicProvider(LLMProvider):
    """
    确定性解码 Provider。

    特性:
    - temperature 强制为 0 (greedy decoding)
    - 可选自一致性: 用 temp=0.3 采样 N 次, 取多数结果
    """

    @property
    def name(self) -> str:
        return "deterministic"

    @property
    def model_name(self) -> str:
        return f"det({DET_MODEL})"

    def is_available(self) -> bool:
        return bool(LLM_API_KEY)

    def _chat_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        # 确定性模式: 强制 temperature=0
        if SELF_CONSISTENCY_N <= 1:
            return self._single_call(system_prompt, user_prompt, max_tokens, 0.0)

        # 自一致性模式: 多次采样, 投票
        return self._self_consistency(system_prompt, user_prompt, max_tokens)

    def _single_call(self, system: str, user: str, max_tokens: int, temp: float) -> Optional[str]:
        """单次 API 调用。"""
        url = f"{LLM_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DET_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": 1.0,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            return None
        except Exception:
            return None

    def _self_consistency(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        """自一致性: N 次采样 (temp=0.3), 取最频繁结果。"""
        from collections import Counter
        results = []
        for _ in range(SELF_CONSISTENCY_N):
            r = self._single_call(system, user, max_tokens, 0.3)
            if r is not None:
                results.append(r.strip())
        if not results:
            return None
        # 投票: 取出现最多的结果
        counter = Counter(results)
        best, count = counter.most_common(1)[0]
        logger.info(f"[deterministic] 自一致性: {count}/{len(results)} 票")
        return best


register_provider("deterministic", DeterministicProvider)
