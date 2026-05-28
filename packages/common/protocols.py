from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Timestamp(BaseModel):
    iso_utc: Optional[str] = None
    epoch_ms: Optional[int] = None


class Constraints(BaseModel):
    deadline_ms: int = 60000
    time_budget_ms: int = 55000
    safety_level: Literal["normal", "strict"] = "normal"


class UserInfo(BaseModel):
    user_id: str
    display_name: Optional[str] = None
    age_range: Optional[str] = None
    x_profile: Dict[str, Any] = Field(default_factory=dict)


class ClientInfo(BaseModel):
    device_id: Optional[str] = None
    app_id: Optional[str] = None
    ip: Optional[str] = None
    x_meta: Dict[str, Any] = Field(default_factory=dict)


class VadInfo(BaseModel):
    speech_start_ms: Optional[int] = None
    speech_end_ms: Optional[int] = None


class TurnInfo(BaseModel):
    utterance_id: Optional[str] = None
    input_mode: Literal["voice", "text", "multimodal"] = "voice"
    barge_in: bool = False
    vad: Optional[VadInfo] = None


class AsrWord(BaseModel):
    w: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    conf: Optional[float] = None


class AsrWerHint(BaseModel):
    domain: Optional[str] = None
    hotwords: List[str] = Field(default_factory=list)


class AsrInfo(BaseModel):
    text: str
    language: str = "zh-CN"
    confidence: Optional[float] = None
    words: List[AsrWord] = Field(default_factory=list)
    wer_hint: Optional[AsrWerHint] = None


class EmotionSignal(BaseModel):
    enabled: bool = True
    x_features: Dict[str, Any] = Field(default_factory=dict)


class EmotionSignals(BaseModel):
    voice: EmotionSignal = Field(default_factory=EmotionSignal)
    text: EmotionSignal = Field(default_factory=EmotionSignal)
    vision: EmotionSignal = Field(default_factory=lambda: EmotionSignal(enabled=False))


class EmotionInfo(BaseModel):
    primary: Optional[str] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    confidence: Optional[float] = None
    signals: EmotionSignals = Field(default_factory=EmotionSignals)


class VisionBlendshape(BaseModel):
    name: str
    score: float


class VisionAU(BaseModel):
    name: str
    intensity: float


class VisionLandmark(BaseModel):
    i: int
    x: float
    y: float
    z: float
    presence: Optional[float] = None
    visibility: Optional[float] = None


class HeadPose(BaseModel):
    pitch: float
    yaw: float
    roll: float


class Gaze(BaseModel):
    x: float
    y: float
    z: float


class VisionFaceFeatures(BaseModel):
    landmarks_478: List[VisionLandmark] = Field(default_factory=list)
    blendshapes_52: List[VisionBlendshape] = Field(default_factory=list)
    au_15: List[VisionAU] = Field(default_factory=list)
    head_pose: Optional[HeadPose] = None
    gaze: Optional[Gaze] = None


class VisionFeatures(BaseModel):
    face: VisionFaceFeatures = Field(default_factory=VisionFaceFeatures)


class AuSchema(BaseModel):
    source: Literal["dataset"] = "dataset"
    schema_id: str = "dataset_au_v1"
    au_names: List[str]
    va_names: List[str]
    expression_8_names: List[str]


class AuMapping(BaseModel):
    name: str = "name2auweight"
    version: str = "v1"
    au_schema: AuSchema


class SymmetryConfig(BaseModel):
    enabled: bool = True
    method: Literal["left_right_mean"] = "left_right_mean"


class SmoothingConfig(BaseModel):
    enabled: bool = True
    type: Literal["iir_1st_order"] = "iir_1st_order"
    alpha: float = 0.7


class VisionProcessing(BaseModel):
    au_mapping: AuMapping
    symmetry: SymmetryConfig = Field(default_factory=SymmetryConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)


class VaSummary(BaseModel):
    valence: float
    arousal: float
    confidence: float = 0.0


class ExpressionSummary(BaseModel):
    label: str
    probs: Dict[str, float] = Field(default_factory=dict)


class VisionSummary(BaseModel):
    va: Optional[VaSummary] = None
    expression_8: Optional[ExpressionSummary] = None


class VisionQuality(BaseModel):
    tracking_state: Literal["tracked", "no_face", "unknown"] = "unknown"
    confidence: float = 0.0
    dropped_frames: int = 0


class VisionModelInfo(BaseModel):
    name: str = "face_landmarker.task"
    with_blendshapes: bool = True


class VisionInfo(BaseModel):
    enabled: bool = True
    provider: str = "mediapipe"
    model: VisionModelInfo = Field(default_factory=VisionModelInfo)
    mode: Literal["live_stream", "image", "video"] = "live_stream"
    frame_rate_fps: int = 15
    window_ms: int = 1000
    face_count: int = 1
    features: VisionFeatures = Field(default_factory=VisionFeatures)
    processing: VisionProcessing
    summary: VisionSummary = Field(default_factory=VisionSummary)
    quality: VisionQuality = Field(default_factory=VisionQuality)
    x_ext: Dict[str, Any] = Field(default_factory=dict)


class MemoryReadConfig(BaseModel):
    enabled: bool = True
    top_k: int = 6
    query: str = ""
    results: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryState(BaseModel):
    summary: str = ""
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    x_memory_state: Dict[str, Any] = Field(default_factory=dict)


class MemoryInfo(BaseModel):
    read: MemoryReadConfig = Field(default_factory=MemoryReadConfig)
    state: MemoryState = Field(default_factory=MemoryState)


class PerceptionToLLM(BaseModel):
    protocol: Literal["perception_to_llm"] = "perception_to_llm"
    version: str = "1.0"
    trace_id: str
    session_id: str
    turn_id: int
    timestamp: Timestamp = Field(default_factory=Timestamp)
    locale: str = "zh-CN"
    user: Optional[UserInfo] = None
    client: Optional[ClientInfo] = None
    constraints: Constraints = Field(default_factory=Constraints)
    turn: TurnInfo = Field(default_factory=TurnInfo)
    asr: AsrInfo
    emotion: EmotionInfo = Field(default_factory=EmotionInfo)
    vision: Optional[VisionInfo] = None
    memory: MemoryInfo = Field(default_factory=MemoryInfo)
    x_ext: Dict[str, Any] = Field(default_factory=dict)


class RenderVoice(BaseModel):
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion: str = "neutral"


class RenderAvatar(BaseModel):
    expression: str = "neutral"
    gesture: Optional[str] = None


class RenderInfo(BaseModel):
    voice: RenderVoice = Field(default_factory=RenderVoice)
    avatar: RenderAvatar = Field(default_factory=RenderAvatar)


class Action(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class SafetyPolicy(BaseModel):
    risk_level: Literal["low", "medium", "high"] = "low"
    handoff: bool = False
    recommendations: List[str] = Field(default_factory=list)


class PolicyInfo(BaseModel):
    safety: SafetyPolicy = Field(default_factory=SafetyPolicy)


class AssistantInfo(BaseModel):
    text: str


class LLMToDriver(BaseModel):
    protocol: Literal["llm_to_driver"] = "llm_to_driver"
    version: str = "1.0"
    trace_id: str
    session_id: str
    turn_id: int
    assistant: AssistantInfo
    actions: List[Action] = Field(default_factory=list)
    render: RenderInfo = Field(default_factory=RenderInfo)
    policy: PolicyInfo = Field(default_factory=PolicyInfo)
    x_ext: Dict[str, Any] = Field(default_factory=dict)

