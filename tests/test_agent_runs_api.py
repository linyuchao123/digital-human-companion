import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import integrated_server


class AgentRunsApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = integrated_server.DB_PATH
        integrated_server.DB_PATH = Path(self.temp_dir.name) / "users.db"
        integrated_server._init_db()
        self.client = TestClient(integrated_server.app)
        first = self.client.post(
            "/api/auth/register",
            json={"username": "trace-user", "password": "test-password"},
        ).json()
        second = self.client.post(
            "/api/auth/register",
            json={"username": "other-trace-user", "password": "test-password"},
        ).json()
        self.user_id = first["user_id"]
        self.other_user_id = second["user_id"]
        self.headers = {"X-Auth-Token": first["token"]}

    def tearDown(self):
        self.client.close()
        integrated_server.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def _insert_run(self, trace_id: str, user_id: int):
        conn = integrated_server._get_db()
        try:
            conn.execute(
                """INSERT INTO agent_runs(
                       trace_id,session_id,user_id,provider,risk_level,emotion,
                       execution_path,tool_calls,node_timings_ms,total_latency_ms,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trace_id,
                    "session-1",
                    user_id,
                    "offline",
                    "low",
                    "Neutral",
                    json.dumps(["safety_triage", "companion"]),
                    json.dumps([]),
                    json.dumps({"safety_triage": 0.1, "companion": 1.2}),
                    1.3,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def test_requires_authentication_and_returns_only_owned_runs(self):
        self._insert_run("owned-trace", self.user_id)
        self._insert_run("other-trace", self.other_user_id)

        unauthorized = self.client.get("/api/agent/runs")
        response = self.client.get("/api/agent/runs", headers=self.headers)

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["trace_id"] for item in response.json()["runs"]], ["owned-trace"])
        self.assertNotIn("user_text", response.json()["runs"][0])
        self.assertNotIn("response", response.json()["runs"][0])

    def test_limit_is_bounded(self):
        for index in range(3):
            self._insert_run(f"trace-{index}", self.user_id)

        response = self.client.get("/api/agent/runs?limit=2", headers=self.headers)

        self.assertEqual(len(response.json()["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
