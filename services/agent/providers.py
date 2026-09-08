from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import httpx

from .state import ChatMessage


COMPANION_SYSTEM_PROMPT = """你是“数字心屿”中的 AI 情绪陪伴助手小安。
请用自然、温暖、克制的中文回应，优先倾听与澄清用户感受。
不要进行医疗诊断，不提供药物建议，也不要声称可以替代专业帮助。
回复控制在 120 个汉字以内，不要输出动作标签或内部推理过程。"""


class CompanionProviderError(RuntimeError):
    """云端 Provider 请求失败或返回内容无效。"""


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"
    temperature: float = 0.75
    max_tokens: int = 250
    timeout_seconds: float = 30.0


class CompanionProvider(Protocol):
    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        """根据已审查的上下文生成回复。"""
        ...


class FakeCompanionProvider:
    """无密钥、无网络时的可预测开发与 CI Provider。"""

    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        user_text = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        if not user_text:
            return "我在这里，你可以慢慢说。"
        if any(keyword in user_text for keyword in ("累", "疲惫", "压力")):
            return "听起来你今天承受了不少压力。愿意和我说说，哪一件事最让你疲惫吗？"
        return f"我听到你说“{user_text}”。此刻你最希望我陪你聊哪一部分？"


class OpenAICompatibleCompanionProvider:
    """适配 DashScope 等 OpenAI Chat Completions 兼容服务。"""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not config.api_key.strip():
            raise ValueError("api_key 不能为空")
        self.config = config
        self._client = client

    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": COMPANION_SYSTEM_PROMPT},
                *(message.model_dump() for message in messages),
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        try:
            if self._client is not None:
                response = await self._client.post(
                    "/chat/completions", json=payload, headers=headers
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self.config.base_url.rstrip("/"),
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                ) as client:
                    response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise CompanionProviderError("云端陪伴模型调用失败") from exc
        if not content:
            raise CompanionProviderError("云端陪伴模型返回了空内容")
        return content


class FallbackCompanionProvider:
    """云端不可用时自动回退，保证体验链路仍可运行。"""

    def __init__(self, primary: CompanionProvider, fallback: CompanionProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        try:
            return await self._primary.generate(messages)
        except CompanionProviderError:
            return await self._fallback.generate(messages)
