import json
import unittest

from apps.api import integrated_server


class CaptureWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, data: str):
        self.messages.append(json.loads(data))


class AgentWebSocketFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_api_key = integrated_server.QWEN_API_KEY
        integrated_server.QWEN_API_KEY = ""
        integrated_server._agent_workflow = None
        integrated_server._agent_provider_name = "offline"

    def tearDown(self):
        integrated_server.QWEN_API_KEY = self.original_api_key
        integrated_server._agent_workflow = None

    async def test_websocket_chat_uses_agent_and_emits_trace(self):
        state = integrated_server.SessionState("guest-session")
        websocket = CaptureWebSocket()

        await integrated_server._trigger_llm("今天工作很累", state, websocket)

        message_types = [message["type"] for message in websocket.messages]
        self.assertEqual(message_types, ["llm_thinking", "agent_trace", "llm_reply"])
        trace = websocket.messages[1]
        self.assertEqual(trace["provider"], "offline")
        self.assertEqual(
            trace["execution_path"],
            [
                "safety_triage",
                "intent_router",
                "emotion_analyzer",
                "companion",
                "avatar_director",
            ],
        )
        self.assertEqual(len(state.agent_messages), 2)

    async def test_websocket_chat_keeps_context_between_turns(self):
        state = integrated_server.SessionState("guest-context")

        await integrated_server._trigger_llm("第一轮消息", state, CaptureWebSocket())
        await integrated_server._trigger_llm("第二轮消息", state, CaptureWebSocket())

        self.assertEqual(len(state.agent_messages), 4)
        self.assertEqual(state.agent_messages[-2]["content"], "第二轮消息")


if __name__ == "__main__":
    unittest.main()
