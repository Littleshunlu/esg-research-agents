"""
LLM客户端 - 统一的大模型调用接口
支持OpenAI兼容API（DeepSeek、混元、Mimo等）
"""

import json
import aiohttp
from typing import Optional


class LLMClient:
    """
    大语言模型客户端

    支持OpenAI兼容API格式的模型调用：
    - DeepSeek
    - 混元 (Hunyuan)
    - Mimo
    - 其他OpenAI兼容API
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.api_base = config.get("api_base", "https://api.deepseek.com/v1")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "deepseek-chat")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 60)

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送聊天请求

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 采样温度
            max_tokens: 最大输出token数

        Returns:
            模型生成的文本响应
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await resp.text()
                    raise Exception(f"API请求失败 ({resp.status}): {error_text}")

    def chat_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        同步版本的聊天请求（方便非异步环境使用）
        """
        import requests

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API请求失败 ({response.status_code}): {response.text}")
