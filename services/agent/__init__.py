"""数字心屿智能体工作流公共协议。"""

from .emotion import EmotionAnalyzer
from .providers import (
    CompanionProvider,
    CompanionProviderError,
    FakeCompanionProvider,
    FallbackCompanionProvider,
    OpenAICompatibleCompanionProvider,
    OpenAICompatibleConfig,
    create_companion_provider,
)
from .state import (
    AgentEvent,
    AgentEventType,
    AgentState,
    AvatarCommand,
    ChatMessage,
    EmotionContext,
    RiskLevel,
    SafetyDecision,
    ToolCallRecord,
)
from .safety import SafetyTriage
from .workflow import DigitalXinyuWorkflow

__all__ = [
    "AgentEvent",
    "AgentEventType",
    "AgentState",
    "AvatarCommand",
    "ChatMessage",
    "CompanionProvider",
    "CompanionProviderError",
    "DigitalXinyuWorkflow",
    "EmotionAnalyzer",
    "EmotionContext",
    "FakeCompanionProvider",
    "FallbackCompanionProvider",
    "OpenAICompatibleCompanionProvider",
    "OpenAICompatibleConfig",
    "RiskLevel",
    "SafetyDecision",
    "SafetyTriage",
    "ToolCallRecord",
    "create_companion_provider",
]
