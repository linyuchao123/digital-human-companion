import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import integrated_server


class AgentChatApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = integrated_server.DB_PATH
        self.original_api_key = integrated_server.QWEN_API_KEY
        integrated_server.DB_PATH = Path(self.temp_dir.name) / "users.db"
        integrated_server.QWEN_API_KEY = ""
        integrated_server._init_db()
        integrated_server._agent_workflow = None
        self.client = TestClient(integrated_server.app)

    def tearDown(self):
        self.client.close()
        integrated_server.DB_PATH = self.original_db_path
        integrated_server.QWEN_API_KEY = self.original_api_key
        integrated_server._agent_workflow = None
        self.temp_dir.cleanup()

    def test_chat_returns_agent_path_and_avatar_command(self):
        response = self.client.post(
            "/api/agent/chat",
            json={"text": "今天工作很累", "session_id": "demo-session"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session_id"], "demo-session")
        self.assertEqual(payload["provider"], "offline")
        self.assertEqual(payload["safety"]["risk_level"], "low")
        self.assertEqual(payload["avatar"]["motion"], "Respond")
        self.assertEqual(payload["emotion"]["emotion"], "Neutral")
        self.assertEqual(
            payload["execution_path"],
            [
                "safety_triage",
                "intent_router",
                "emotion_analyzer",
                "knowledge_retriever",
                "companion",
                "avatar_director",
            ],
        )
        self.assertEqual(set(payload["node_timings_ms"]), set(payload["execution_path"]))

    def test_chat_rejects_empty_text(self):
        response = self.client.post("/api/agent/chat", json={"text": ""})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
