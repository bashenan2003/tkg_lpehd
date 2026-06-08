"""
LLM 客户端模块 (多 Provider 路由)
==================================
所有 LLM prompt 统一经由此模块发送, 根据配置切换底层 Provider。

Provider 切换方式:
  1. config.py 中设置 LLM_PROVIDER
  2. 环境变量 TKG_LLM_PROVIDER 覆盖
  3. 函数调用时传入 provider 参数

支持: deepseek / gpt35_turbo / llama2_7b / prompt_paraphrasing / deterministic
"""

import os
import time
import logging
from typing import Optional

from .config import is_real_llm_enabled, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# 延迟导入避免循环依赖
_provider_cache = None


def _get_provider(name: str = None):
    """获取当前 Provider 实例 (带缓存)。"""
    global _provider_cache
    from .llm_providers import get_provider
    provider_name = name or os.environ.get("TKG_LLM_PROVIDER", None)
    if provider_name is None:
        if _provider_cache is not None:
            return _provider_cache
    inst = get_provider(provider_name)
    if provider_name is None:
        _provider_cache = inst
    return inst


def _current_provider_name() -> str:
    """当前使用的 Provider 名称。"""
    try:
        return _get_provider().name
    except Exception:
        return "unknown"


# ------------------------------------------------------------------
# 便捷接口 (保持向后兼容)
# ------------------------------------------------------------------
def llm_chat(system_prompt: str, user_prompt: str, **kwargs) -> Optional[str]:
    """发送 system + user 消息, 返回模型回复文本。

    额外 kwarg:
      provider: str  — 临时指定 provider 名称
      max_tokens: int
      temperature: float
    """
    if not is_real_llm_enabled():
        return None  # 模拟模式 / Phase A 确定性阶段

    prov = kwargs.pop("provider", None)
    provider = _get_provider(prov) if prov else _get_provider()

    max_tokens = kwargs.pop("max_tokens", LLM_MAX_TOKENS)
    temperature = kwargs.pop("temperature", LLM_TEMPERATURE)

    return provider.chat(system_prompt, user_prompt, max_tokens, temperature)


def llm_ask(user_prompt: str, **kwargs) -> Optional[str]:
    """发送单条 user 消息 (无 system prompt)。"""
    return llm_chat("", user_prompt, **kwargs)


def llm_chat_with_retry(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
    **kwargs,
) -> Optional[str]:
    """带重试的 LLM 调用, 失败后等待 1s 重试。"""
    for attempt in range(max_retries + 1):
        result = llm_chat(system_prompt, user_prompt, **kwargs)
        if result is not None:
            return result
        if attempt < max_retries:
            time.sleep(1.0)
    return None


# ------------------------------------------------------------------
# 诊断
# ------------------------------------------------------------------
def get_active_provider_info() -> dict:
    """获取当前 Provider 的诊断信息。"""
    try:
        p = _get_provider()
        return {
            "provider": p.name,
            "model": p.model_name,
            "available": p.is_available(),
        }
    except Exception as e:
        return {"provider": "error", "error": str(e)}


def switch_provider(name: str):
    """运行时切换 Provider (清除缓存)。"""
    global _provider_cache
    _provider_cache = None
    os.environ["TKG_LLM_PROVIDER"] = name
