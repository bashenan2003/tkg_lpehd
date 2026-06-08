"""
Llama2 7B Provider
==================
通过本地 Ollama 服务调用 Llama2 7B 模型。

前置条件:
  ollama pull llama2:7b     # 下载 Llama2 7B 模型 (~3.8 GB)
  ollama serve              # 启动 Ollama 服务 (默认 http://localhost:11434)

配置:
  OLLAMA_HOST 环境变量可覆盖服务地址
"""

import os
import logging
from typing import Optional

import requests

from .base import LLMProvider
from . import register_provider

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("LLAMA2_MODEL", "llama2:7b")


class Llama2Provider(LLMProvider):
    """Llama2 7B via Ollama Provider."""

    @property
    def name(self) -> str:
        return "llama2_7b"

    @property
    def model_name(self) -> str:
        return OLLAMA_MODEL

    def is_available(self) -> bool:
        """探测 Ollama 服务是否可达。"""
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _chat_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        # Ollama chat API: /api/chat
        url = f"{OLLAMA_HOST}/api/chat"
        # 将 system prompt 合并到 user prompt (Llama2 无原生 system role)
        combined = f"System: {system_prompt}\n\nUser: {user_prompt}"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": combined},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()["message"]["content"]
            logger.warning(f"[llama2_7b] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except requests.exceptions.Timeout:
            logger.warning("[llama2_7b] 超时 (120s)")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning(f"[llama2_7b] Ollama 服务不可达 ({OLLAMA_HOST})")
            return None
        except Exception as e:
            logger.warning(f"[llama2_7b] 异常: {e}")
            return None


register_provider("llama2_7b", Llama2Provider)
