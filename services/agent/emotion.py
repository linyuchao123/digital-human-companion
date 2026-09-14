from __future__ import annotations

from .state import EmotionContext, RiskLevel, SafetyDecision


class EmotionAnalyzer:
    """首版确定性情绪分析器，后续可替换为分类模型或多模态融合。"""

    _RULES = (
        (("开心", "高兴", "快乐", "好消息", "太棒"), "Happy", 0.75, 0.4, "喜悦"),
        (("难过", "伤心", "悲伤", "失落", "痛苦", "想哭"), "Sad", -0.6, -0.2, "悲伤"),
        (("焦虑", "紧张", "压力", "烦躁", "害怕", "睡不着"), "Anxiety", -0.4, 0.55, "焦虑"),
    )

    def analyze(self, text: str, safety: SafetyDecision) -> EmotionContext:
        if safety.risk_level is RiskLevel.HIGH:
            return EmotionContext(
                emotion="Concerned",
                valence=-0.9,
                arousal=0.45,
                label="紧急关切",
            )
        for keywords, emotion, valence, arousal, label in self._RULES:
            if any(keyword in text for keyword in keywords):
                return EmotionContext(
                    emotion=emotion,
                    valence=valence,
                    arousal=arousal,
                    label=label,
                )
        return EmotionContext()
