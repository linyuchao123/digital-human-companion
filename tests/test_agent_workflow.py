import unittest

from services.agent import DigitalXinyuWorkflow, RiskLevel


class FailingProvider:
    async def generate(self, messages):
        raise AssertionError("高风险路径不应调用普通对话模型")


class DigitalXinyuWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_chat_runs_through_companion_and_avatar_nodes(self):
        workflow = DigitalXinyuWorkflow()

        result = await workflow.run(
            user_text="今天工作特别累",
            trace_id="trace-normal",
            session_id="session-normal",
        )

        self.assertEqual(result["safety"].risk_level, RiskLevel.LOW)
        self.assertEqual(
            result["execution_path"],
            ["safety_triage", "intent_router", "companion", "avatar_director"],
        )
        self.assertEqual(set(result["node_timings_ms"]), set(result["execution_path"]))
        self.assertTrue(all(value >= 0 for value in result["node_timings_ms"].values()))
        self.assertEqual(result["intent"], "emotional_support")
        self.assertEqual(result["avatar_command"].motion, "Respond")
        self.assertTrue(result["final_response"])

    async def test_high_risk_chat_bypasses_companion_provider(self):
        workflow = DigitalXinyuWorkflow(provider=FailingProvider())

        result = await workflow.run(
            user_text="我不想活了",
            trace_id="trace-risk",
            session_id="session-risk",
        )

        self.assertEqual(result["safety"].risk_level, RiskLevel.HIGH)
        self.assertEqual(
            result["execution_path"],
            ["safety_triage", "safe_response", "avatar_director"],
        )
        self.assertEqual(set(result["node_timings_ms"]), set(result["execution_path"]))
        self.assertEqual(result["avatar_command"].motion, "Comfort")
        self.assertNotIn("draft_response", result)


if __name__ == "__main__":
    unittest.main()
