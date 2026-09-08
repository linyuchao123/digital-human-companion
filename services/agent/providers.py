from __future__ import annotations

from typing import Protocol, Sequence

from .state import ChatMessage


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
