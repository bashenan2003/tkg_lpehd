"""
Prompt Paraphrasing Provider
=============================
Prompt 改写增强: 用一个轻量 LLM 对原始 prompt 进行语义等效改写,
生成多个变体, 再由底层 LLM 分别推理, 融合结果以提高鲁棒性。

工作流程:
  1. 将原始 system + user prompt 发给 paraphrasing 模型
  2. 模型输出改写后的 prompt (保持语义不变)
  3. 将改写后的 prompt 发给底层推理模型
  4. 返回推理结果

这相当于论文中的 "Prompt Paraphrasing" 增强策略。
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

# Paraphrasing 使用 DeepSeek (或可配置为其他轻量模型)
PARA_MODEL = os.environ.get("PARA_MODEL", "deepseek-chat")


class PromptParaphrasingProvider(LLMProvider):
    """
    Prompt 改写包装器。

    先用 paraphrasing 指令让 LLM 改写原始 prompt,
    再将改写后的 prompt 发给同一 LLM 进行推理。
    """

    @property
    def name(self) -> str:
        return "prompt_paraphrasing"

    @property
    def model_name(self) -> str:
        return f"paraphrase({PARA_MODEL})"

    def is_available(self) -> bool:
        return bool(LLM_API_KEY)

    def _chat_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        # ---- Phase 1: 改写 prompt ----
        para_system = (
            "You are a prompt optimizer. Rewrite the user's prompt to be "
            "clearer, more structured, and more likely to elicit an accurate "
            "response from an LLM. Preserve ALL constraints, formulas, and "
            "output format requirements exactly. Do NOT add new tasks or "
            "change the meaning. Return ONLY the rewritten prompt text."
        )
        para_user = (
            f"Original system instruction:\n{system_prompt}\n\n"
            f"Original task:\n{user_prompt}\n\n"
            f"Rewrite the task prompt to improve clarity and accuracy."
        )

        rewritten = self._raw_call(para_system, para_user, max_tokens=1024, temperature=0.3)
        if rewritten is None:
            logger.warning("[paraphrasing] 改写失败, 使用原始 prompt")
            rewritten = user_prompt

        # ---- Phase 2: 用改写后的 prompt 推理 ----
        return self._raw_call(system_prompt, rewritten, max_tokens, temperature)

    def _raw_call(self, system: str, user: str, max_tokens: int, temperature: float) -> Optional[str]:
        """底层 HTTP 调用。"""
        url = f"{LLM_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": PARA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            return None
        except Exception:
            return None


register_provider("prompt_paraphrasing", PromptParaphrasingProvider)
