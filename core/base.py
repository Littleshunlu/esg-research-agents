"""
Agent基类定义
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentResponse:
    """Agent统一响应格式"""
    success: bool
    data: Any = None
    message: str = ""


@dataclass
class AgentConfig:
    """Agent配置"""
    model: str = "deepseek-chat"
    api_base: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 60


class BaseAgent:
    """
    所有Agent的基类

    提供统一的初始化、配置管理和日志接口
    """

    def __init__(self, name: str, config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self._status = "idle"

    @property
    def status(self) -> str:
        return self._status

    def _set_status(self, status: str):
        self._status = status

    async def run(self, **kwargs) -> AgentResponse:
        """执行Agent主逻辑（子类实现）"""
        raise NotImplementedError("子类必须实现 run 方法")

    def reset(self):
        """重置Agent状态"""
        self._status = "idle"

    def info(self) -> dict:
        """获取Agent信息"""
        return {
            "name": self.name,
            "status": self._status,
            "config_keys": list(self.config.keys()),
        }
