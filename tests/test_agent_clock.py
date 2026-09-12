import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from services.agent.clock import clock_answer, is_clock_query
from services.agent.workflow import DigitalXinyuWorkflow


class NeverGenerate:
    async def generate(self, messages):
        raise AssertionError("时间工具与安全路径不应调用大模型")


class ClockTests(unittest.TestCase):
    def test_queries_do_not_match_unrelated_conversation(self):
        for text in ("现在几点了？", "今天星期几", "明天几号", "纽约几点", "北京时间几点"):
            self.assertTrue(is_clock_query(text), text)
        for text in ("我每天几点睡觉比较好", "今天心情很差", "提醒我明天几点开会", "为什么时间过得很慢"):
            self.assertFalse(is_clock_query(text), text)

    def test_timezone_date_rollover_and_weekday(self):
        now = datetime(2026, 12, 31, 18, 30, tzinfo=timezone.utc)
        self.assertEqual(clock_answer("现在几点", now), "现在是北京时间 02:30。")
        self.assertEqual(clock_answer("明天几号", now), "按北京时间，明天是2027年1月2日，星期六。")
        self.assertEqual(clock_answer("纽约几点", now), "现在是纽约当地时间 13:30。")


class ClockWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_tool_bypasses_provider_and_records_history(self):
        with patch("services.agent.workflow.clock_answer", return_value="现在是北京时间 12:34。"):
            result = await DigitalXinyuWorkflow(provider=NeverGenerate()).run(
                user_text="现在几点", trace_id="clock", session_id="clock-session",
                user_id=1, memory_consent=True,
            )
        self.assertEqual(result["intent"], "clock_query")
        self.assertEqual(result["final_response"], "现在是北京时间 12:34。")
        self.assertIn("clock_tool", result["execution_path"])
        self.assertNotIn("companion", result["execution_path"])
        self.assertEqual(result["tool_calls"][0].name, "current_datetime")
        self.assertEqual(result["messages"][-1].content, result["final_response"])

    async def test_high_risk_keeps_safety_priority(self):
        result = await DigitalXinyuWorkflow(provider=NeverGenerate()).run(
            user_text="现在几点，我想自杀", trace_id="safe-clock", session_id="safe-session",
        )
        self.assertIn("safe_response", result["execution_path"])
        self.assertNotIn("clock_tool", result["execution_path"])
