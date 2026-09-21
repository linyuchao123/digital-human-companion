import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.api import integrated_server as server
from apps.api.admin_dashboard import save_model_usage


class AdminDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original = server.DB_PATH
        server.DB_PATH = Path(self.temp.name) / "users.db"
        self.env = patch.dict(os.environ, {
            "ADMIN_USERNAMES": "owner",
            "DEEPSEEK_INPUT_PRICE_CNY_PER_MILLION": "1",
            "DEEPSEEK_OUTPUT_PRICE_CNY_PER_MILLION": "2",
        })
        self.env.start()
        server._init_db()
        self.client = TestClient(server.app)

    def tearDown(self):
        self.client.close()
        self.env.stop()
        server.DB_PATH = self.original
        self.temp.cleanup()

    def register(self, username):
        response = self.client.post(
            "/api/auth/register",
            json={"username": username, "password": "test-password"},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_admin_routes_are_hidden_and_server_authorized(self):
        owner = self.register("owner")
        member = self.register("member")
        owner_headers = {"X-Auth-Token": owner["token"]}
        member_headers = {"X-Auth-Token": member["token"]}

        self.assertEqual(self.client.get("/admin").status_code, 401)
        self.assertEqual(self.client.get("/admin", headers=member_headers).status_code, 403)
        self.assertEqual(self.client.get("/api/admin/summary", headers=member_headers).status_code, 403)
        self.assertEqual(self.client.get("/admin", headers=owner_headers).status_code, 200)
        self.assertEqual(self.client.get("/api/profile", headers=owner_headers).json()["role"], "admin")
        self.assertEqual(self.client.get("/api/profile", headers=member_headers).json()["role"], "user")

    def test_per_user_aggregates_and_usage_do_not_expose_content(self):
        owner = self.register("owner")
        member = self.register("member")
        headers = {"X-Auth-Token": owner["token"]}
        with server._get_db() as conn:
            member_id = conn.execute("SELECT id FROM users WHERE username='member'").fetchone()[0]
            conn.execute(
                "INSERT INTO chat_sessions(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                ("member-session", member_id, "private title", "2026-09-21T10:00:00", "2026-09-21T10:00:00"),
            )
            conn.execute(
                "INSERT INTO chat_messages(session_id,role,content,ts) VALUES(?,?,?,?)",
                ("member-session", "user", "private message", "2026-09-21T10:01:00"),
            )
            conn.commit()
        save_model_usage(
            server._get_db,
            [{"provider": "deepseek", "model": "test", "request_kind": "chat",
              "input_tokens": 120, "output_tokens": 30, "status": "success"}],
            user_id=member_id, session_id="member-session", trace_id="trace-1",
        )

        response = self.client.get("/api/admin/users", headers=headers)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        member_row = next(row for row in body["users"] if row["username"] == "member")
        self.assertEqual(member_row["dialogue_rounds"], 1)
        self.assertEqual(member_row["model_calls"], 1)
        self.assertEqual(member_row["input_tokens"] + member_row["output_tokens"], 150)
        self.assertGreater(member_row["estimated_cost_cny"], 0)
        serialized = response.text
        self.assertNotIn("private message", serialized)
        self.assertNotIn("private title", serialized)
