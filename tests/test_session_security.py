import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import integrated_server


class SessionAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = integrated_server.DB_PATH
        integrated_server.DB_PATH = Path(self.temp_dir.name) / "users.db"
        integrated_server._init_db()
        self.client = TestClient(integrated_server.app)

    def tearDown(self):
        self.client.close()
        integrated_server.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def _register(self, username: str) -> str:
        response = self.client.post(
            "/api/auth/register",
            json={"username": username, "password": "test-password"},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["token"]

    def _create_session(self, token: str) -> str:
        response = self.client.post(
            "/api/sessions",
            headers={"X-Auth-Token": token},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["session_id"]

    def test_user_cannot_delete_another_users_session_messages(self):
        owner_token = self._register("owner")
        attacker_token = self._register("attacker")
        session_id = self._create_session(owner_token)

        integrated_server._db_save_message(session_id, "user", "private message")

        response = self.client.delete(
            f"/api/sessions/{session_id}",
            headers={"X-Auth-Token": attacker_token},
        )

        self.assertEqual(response.status_code, 404)
        with integrated_server._get_db() as conn:
            session_count = conn.execute(
                "SELECT COUNT(*) FROM chat_sessions WHERE id=?", (session_id,)
            ).fetchone()[0]
            message_count = conn.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id=?", (session_id,)
            ).fetchone()[0]
        self.assertEqual(session_count, 1)
        self.assertEqual(message_count, 1)


if __name__ == "__main__":
    unittest.main()
