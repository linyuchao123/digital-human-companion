from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentEventType(StrEnum):
    RUN_STARTED = "agent.run.started"
    RUN_COMPLETED = "agent.run.completed"
    NODE_STARTED = "agent.node.started"
    NODE_COMPLETED = "agent.node.completed"
    TOOL_STARTED = "agent.tool.started"
    TOOL_COMPLETED = "agent.tool.completed"
    ROUTE_SELECTED = "agent.route.selected"
    GUARD_TRIGGERED = "agent.guard.triggered"
    METRIC_UPDATED = "agent.metric.updated"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class SafetyDecision(BaseModel):
    risk_level: RiskLevel = RiskLevel.LOW
    reason: str = ""
    requires_safe_response: bool = False
    allow_memory_write: bool = True
    allow_web_search: bool = False


class EmotionContext(BaseModel):
    emotion: Literal["Neutral", "Happy", "Sad", "Anxiety", "Concerned"] = "Neutral"
    valence: float = Field(default=0, ge=-1, le=1)
    arousal: float = Field(default=0, ge=-1, le=1)
    label: str = "平静"


class KnowledgeSnippet(BaseModel):
    content: str = Field(min_length=1, max_length=1200)
    source: str = Field(min_length=1, max_length=200)
    score: float = Field(default=0, ge=0, le=1)


class MemoryRecord(BaseModel):
    id: str = Field(min_length=1)
    user_id: int
    content: str = Field(min_length=1, max_length=500)
    category: Literal["preference", "profile", "goal", "context"] = "context"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolCallRecord(BaseModel):
    name: str = Field(min_length=1)
    reason: str = ""
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    elapsed_ms: float | None = Field(default=None, ge=0)
    error_code: str | None = None


class AvatarCommand(BaseModel):
    emotion: str = "Neutral"
    intensity: float = Field(default=0.3, ge=0, le=1)
    motion: Literal["Idle", "Listen", "Think", "Respond", "Comfort"] = "Idle"
    mouth_open: float = Field(default=0, ge=0, le=1)
    gaze_x: float = Field(default=0, ge=-1, le=1)
    gaze_y: float = Field(default=0, ge=-1, le=1)
    speaking_state: Literal["idle", "listening", "thinking", "speaking"] = "idle"


class AgentEvent(BaseModel):
    type: AgentEventType
    trace_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    node: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    trace_id: str
    thread_id: str
    session_id: str
    turn_id: int
    user_id: int | None
    guest_id: str | None
    user_text: str
    messages: list[ChatMessage]
    intent: str
    safety: SafetyDecision
    emotion_context: EmotionContext
    retrieved_memories: list[MemoryRecord]
    retrieved_knowledge: list[KnowledgeSnippet]
    tool_calls: list[ToolCallRecord]
    draft_response: str
    final_response: str
    avatar_command: AvatarCommand
    node_timings_ms: dict[str, float]
    cancelled: bool
    memory_consent: bool
    errors: list[str]
    execution_path: list[str]
