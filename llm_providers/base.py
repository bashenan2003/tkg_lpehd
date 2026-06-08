"""
LLM Provider 基类
=================
所有 LLM 后端统一接口。
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """LLM Provider 抽象基类。

    每个子类必须实现:
    - name         : provider 短名
    - model_name   : 实际模型标识符
    - _chat_impl() : 底层调用逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 短名 (如 'deepseek', 'gpt35_turbo')。"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """实际模型名 (如 'deepseek-chat', 'gpt-3.5-turbo')。"""
        ...

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Optional[str]:
        """发送 system + user 消息, 返回模型回复文本。失败返回 None。"""
        try:
            return self._chat_impl(
                system_prompt, user_prompt, max_tokens, temperature
            )
        except Exception as e:
            logger.warning(f"[{self.name}] 调用失败: {e}")
            return None

    @abstractmethod
    def _chat_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        """子类实现: 实际的 API / 模型调用。"""
        ...

    def is_available(self) -> bool:
        """检测 Provider 是否可用。子类可覆盖。"""
        return True

    def __repr__(self):
        return f"<{self.name}: {self.model_name}>"
