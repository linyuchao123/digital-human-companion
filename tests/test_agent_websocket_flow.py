import json
import tempfile
import unittest
from pathlib import Path

from apps.api import integrated_server


class CaptureWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, data: str):
        self.messages.append(json.loads(data))


class AgentWebSocketFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = integrated_server.DB_PATH
        self.original_deepseek_api_key = integrated_server.DEEPSEEK_API_KEY
        self.original_api_key = integrated_server.QWEN_API_KEY
        integrated_server.DB_PATH = Path(self.temp_dir.name) / "users.db"
        integrated_server._init_db()
        integrated_server.DEEPSEEK_API_KEY = ""
        integrated_server.QWEN_API_KEY = ""
        integrated_server._agent_workflow = None
        integrated_server._agent_provider_name = "offline"
        integrated_server._agent_knowledge_provider_name = "uninitialized"

    def tearDown(self):
        integrated_server.DEEPSEEK_API_KEY = self.original_deepseek_api_key
        integrated_server.QWEN_API_KEY = self.original_api_key
        integrated_server.DB_PATH = self.original_db_path
        integrated_server._agent_workflow = None
        self.temp_dir.cleanup()

    async def test_websocket_chat_uses_agent_and_emits_trace(self):
        state = integrated_server.SessionState("guest-session")
        websocket = CaptureWebSocket()

        await integrated_server._trigger_llm("今天工作很累", state, websocket)

        message_types = [message["type"] for message in websocket.messages]
        self.assertEqual(message_types[0], "llm_thinking")
        self.assertEqual(message_types[-2:], ["agent_trace", "llm_reply"])
        events = [
            message["event"]
            for message in websocket.messages
            if message["type"] == "agent_event"
        ]
        self.assertEqual(events[0]["type"], "agent.run.started")
        self.assertEqual(events[-1]["type"], "agent.run.completed")
        completed_nodes = [
            event["node"]
            for event in events
            if event["type"] == "agent.node.completed"
        ]
        trace = next(
            message for message in websocket.messages if message["type"] == "agent_trace"
        )
        self.assertEqual(completed_nodes, trace["execution_path"])
        self.assertEqual(trace["provider"], "offline")
        self.assertEqual(trace["knowledge_provider"], "bm25_with_fallback")
        self.assertEqual(
            trace["execution_path"],
            [
                "safety_triage",
                "intent_router",
                "emotion_analyzer",
                "knowledge_retriever",
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

    async def test_authorized_user_memory_is_persisted_and_retrieved(self):
        conn = integrated_server._get_db()
        try:
            conn.execute(
                "INSERT INTO user_memory_settings(user_id,enabled,updated_at) VALUES(?,?,?)",
                (12, 1, "2026-09-08T00:00:00"),
            )
            conn.commit()
        finally:
            conn.close()
        state = integrated_server.SessionState("memory-session")
        state.user_id = 12
        state.memory_consent = integrated_server._memory_enabled_for_user(12)

        first_socket = CaptureWebSocket()
        await integrated_server._trigger_llm(
            "我喜欢睡前听轻音乐", state, first_socket
        )
        second_socket = CaptureWebSocket()
        await integrated_server._trigger_llm(
            "睡前听什么音乐合适", state, second_socket
        )

        first_trace = next(
            message for message in first_socket.messages if message["type"] == "agent_trace"
        )
        second_trace = next(
            message for message in second_socket.messages if message["type"] == "agent_trace"
        )
        self.assertIn("memory_writer", first_trace["execution_path"])
        self.assertIn("memory_retriever", second_trace["execution_path"])
        self.assertNotIn("memory_writer", second_trace["execution_path"])
        self.assertEqual(len(second_trace["memories"]), 1)

        forget_socket = CaptureWebSocket()
        await integrated_server._trigger_llm(
            "请忘掉我喜欢睡前听轻音乐", state, forget_socket
        )
        forget_trace = next(
            message for message in forget_socket.messages if message["type"] == "agent_trace"
        )
        self.assertIn("memory_forgetter", forget_trace["execution_path"])
        self.assertEqual(forget_trace["tool_calls"][-1]["name"], "long_term_memory_delete")
        conn = integrated_server._get_db()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM user_memories WHERE user_id=?", (12,)
            ).fetchone()[0]
            run_count = conn.execute(
                "SELECT COUNT(*) FROM agent_runs WHERE user_id=?", (12,)
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 0)
        self.assertEqual(run_count, 3)


if __name__ == "__main__":
    unittest.main()
