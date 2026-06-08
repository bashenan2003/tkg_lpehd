"""
LLM Provider 注册中心
=====================
所有 LLM 后端统一通过此模块注册和切换。

支持的 Provider:
- deepseek       : DeepSeek API (默认)
- gpt35_turbo    : OpenAI GPT-3.5 Turbo
- llama2_7b      : Llama2 7B (本地 Ollama)
- prompt_paraphrasing : Prompt 改写增强 (包装器)
- deterministic  : 确定性解码 (温度=0 强制)

使用方式:
  from tkg_lpehd.llm_providers import get_provider
  provider = get_provider("deepseek")
  result = provider.chat("system", "user")
"""

from .base import LLMProvider

_PROVIDER_REGISTRY: dict = {}


def register_provider(name: str, factory):
    """注册 Provider 工厂函数。"""
    _PROVIDER_REGISTRY[name.lower()] = factory


def get_provider(name: str = None) -> LLMProvider:
    """根据名称获取 LLM Provider 实例。"""
    from ..config import LLM_PROVIDER
    name = (name or LLM_PROVIDER).lower()
    if name not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown LLM provider: '{name}'. "
            f"Available: {list(_PROVIDER_REGISTRY.keys())}"
        )
    return _PROVIDER_REGISTRY[name]()


def list_providers() -> list:
    """列出所有已注册的 Provider。"""
    return list(_PROVIDER_REGISTRY.keys())


# 延迟导入避免循环依赖 (各 provider 文件在被导入时自行 register)
def _auto_register_all():
    """自动注册所有内置 Provider。"""
    import importlib
    import os
    _here = os.path.dirname(__file__)
    for fname in os.listdir(_here):
        if fname.endswith(".py") and fname not in ("__init__.py", "base.py"):
            mod_name = f".{fname[:-3]}"
            try:
                importlib.import_module(mod_name, __package__)
            except ImportError:
                pass  # 依赖缺失时静默跳过, 该 provider 不可用


_auto_register_all()
