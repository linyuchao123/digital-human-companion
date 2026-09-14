import unittest

from pydantic import ValidationError

from services.agent import (
    AgentEvent,
    AgentEventType,
    AvatarCommand,
    RiskLevel,
    SafetyDecision,
    SafetyTriage,
)


class AgentStateModelTests(unittest.TestCase):
    def test_agent_event_is_json_serializable(self):
        event = AgentEvent(
            type=AgentEventType.NODE_COMPLETED,
            trace_id="trace-1",
            session_id="session-1",
            node="safety_triage",
            data={"risk_level": RiskLevel.LOW},
        )

        payload = event.model_dump(mode="json")

        self.assertEqual(payload["type"], "agent.node.completed")
        self.assertEqual(payload["data"]["risk_level"], "low")
        self.assertIn("timestamp", payload)

    def test_avatar_command_rejects_out_of_range_values(self):
        with self.assertRaises(ValidationError):
            AvatarCommand(intensity=1.1)

    def test_high_risk_decision_disables_memory_by_default_when_requested(self):
        decision = SafetyDecision(
            risk_level=RiskLevel.HIGH,
            requires_safe_response=True,
            allow_memory_write=False,
        )

        self.assertTrue(decision.requires_safe_response)
        self.assertFalse(decision.allow_memory_write)

    def test_safety_triage_blocks_high_risk_text_before_agent_execution(self):
        decision = SafetyTriage().evaluate("我已经不想活了")

        self.assertEqual(decision.risk_level, RiskLevel.HIGH)
        self.assertTrue(decision.requires_safe_response)
        self.assertFalse(decision.allow_memory_write)
        self.assertFalse(decision.allow_web_search)

    def test_safety_triage_allows_normal_companion_chat(self):
        decision = SafetyTriage().evaluate("今天工作有点累，想和你聊聊")

        self.assertEqual(decision.risk_level, RiskLevel.LOW)
        self.assertFalse(decision.requires_safe_response)
        self.assertTrue(decision.allow_memory_write)


if __name__ == "__main__":
    unittest.main()
