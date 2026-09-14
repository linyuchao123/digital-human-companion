import unittest
from services.agent.activities import requests_activities
from services.agent.workflow import DigitalXinyuWorkflow


class NoModel:
    async def generate(self, messages):
        raise AssertionError("固定活动工具不应调用模型")


class ActivitiesTests(unittest.IsolatedAsyncioTestCase):
    async def test_planner_produces_bounded_cards_and_tool_record(self):
        result = await DigitalXinyuWorkflow(provider=NoModel()).run(
            user_text="有点无聊，推荐几个小活动", trace_id="activities", session_id="activities",
        )
        self.assertEqual(result["intent"], "activity_plan")
        self.assertEqual(len(result["activities"]), 3)
        self.assertIn("activity_planner", result["execution_path"])
        self.assertEqual(result["tool_calls"][0].name, "companion_activity_plan")
        self.assertEqual(len({card.id for card in result["activities"]}), 3)
        self.assertTrue(all(1 <= card.minutes <= 30 for card in result["activities"]))

    async def test_safety_never_offers_cards(self):
        result = await DigitalXinyuWorkflow(provider=NoModel()).run(
            user_text="我想自杀，给我推荐活动", trace_id="safe-activities", session_id="safe-activities",
        )
        self.assertIn("safe_response", result["execution_path"])
        self.assertEqual(result["activities"], [])

    def test_does_not_hijack_music_request(self):
        self.assertFalse(requests_activities("推荐几首周杰伦的歌曲"))
        self.assertFalse(requests_activities("我今天参加了学校活动"))
