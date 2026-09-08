from __future__ import annotations

import re
from dataclasses import dataclass

from .state import RiskLevel, SafetyDecision


@dataclass(frozen=True)
class SafetyRule:
    pattern: re.Pattern[str]
    risk_level: RiskLevel
    reason: str


class SafetyTriage:
    """在任何模型和工具调用前执行的确定性安全分诊。"""

    def __init__(self) -> None:
        self._rules = (
            SafetyRule(
                re.compile(r"想死|不想活|轻生|自杀|结束生命|活不下去|伤害自己"),
                RiskLevel.HIGH,
                "self_harm_signal",
            ),
            SafetyRule(
                re.compile(r"绝望|撑不住|彻底崩溃|没有活下去的意义"),
                RiskLevel.MEDIUM,
                "distress_signal",
            ),
        )

    def evaluate(self, text: str) -> SafetyDecision:
        normalized = "".join(text.lower().split())
        for rule in self._rules:
            if rule.pattern.search(normalized):
                is_high = rule.risk_level is RiskLevel.HIGH
                return SafetyDecision(
                    risk_level=rule.risk_level,
                    reason=rule.reason,
                    requires_safe_response=is_high,
                    allow_memory_write=False,
                    allow_web_search=False,
                )
        return SafetyDecision()
