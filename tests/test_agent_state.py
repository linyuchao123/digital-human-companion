import unittest

from pydantic import ValidationError

from services.agent import AgentEvent, AgentEventType, AvatarCommand, RiskLevel, SafetyDecision


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


if __name__ == "__main__":
    unittest.main()
