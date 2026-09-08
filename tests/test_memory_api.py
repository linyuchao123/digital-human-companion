import tempfile
import time
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import integrated_server


class MemoryApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = integrated_server.DB_PATH
        integrated_server.DB_PATH = Path(self.temp_dir.name) / "users.db"
        integrated_server._init_db()
        self.client = TestClient(integrated_server.app)
        response = self.client.post(
            "/api/auth/register",
            json={"username": "memory-user", "password": "test-password"},
        )
        self.token = response.json()["token"]
        self.user_id = response.json()["user_id"]
        self.headers = {"X-Auth-Token": self.token}

    def tearDown(self):
        self.client.close()
        integrated_server.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_memory_is_disabled_by_default_and_requires_auth(self):
        unauthorized = self.client.get("/api/memory/settings")
        response = self.client.get("/api/memory/settings", headers=self.headers)

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(response.json(), {"enabled": False, "count": 0})

    def test_user_can_enable_memory_explicitly(self):
        response = self.client.put(
            "/api/memory/settings",
            headers=self.headers,
            json={"enabled": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"enabled": True})
        self.assertTrue(self.client.get(
            "/api/memory/settings", headers=self.headers
        ).json()["enabled"])

    def test_user_can_list_and_delete_own_memories(self):
        memory_id = str(uuid.uuid4())
        conn = integrated_server._get_db()
        try:
            conn.execute(
                """INSERT INTO user_memories(id,user_id,content,category,created_at)
                   VALUES(?,?,?,?,?)""",
                (
                    memory_id,
                    self.user_id,
                    "喜欢睡前听音乐",
                    "preference",
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        listed = self.client.get("/api/memories", headers=self.headers)
        deleted = self.client.delete("/api/memories", headers=self.headers)

        self.assertEqual(listed.json()["memories"][0]["id"], memory_id)
        self.assertEqual(deleted.json()["deleted"], 1)
        self.assertEqual(
            self.client.get("/api/memories", headers=self.headers).json()["memories"],
            [],
        )

    def test_user_can_delete_one_own_memory_but_not_another_users_memory(self):
        other = self.client.post(
            "/api/auth/register",
            json={"username": "another-memory-user", "password": "test-password"},
        ).json()
        own_memory_id = str(uuid.uuid4())
        other_memory_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        conn = integrated_server._get_db()
        try:
            conn.executemany(
                """INSERT INTO user_memories(id,user_id,content,category,created_at)
                   VALUES(?,?,?,?,?)""",
                (
                    (own_memory_id, self.user_id, "我喜欢跑步", "preference", now),
                    (other_memory_id, other["user_id"], "我喜欢游泳", "preference", now),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        forbidden = self.client.delete(
            f"/api/memories/{other_memory_id}", headers=self.headers
        )
        deleted = self.client.delete(
            f"/api/memories/{own_memory_id}", headers=self.headers
        )

        self.assertEqual(forbidden.status_code, 404)
        self.assertEqual(deleted.json(), {"ok": True, "deleted": 1})
        other_headers = {"X-Auth-Token": other["token"]}
        other_memories = self.client.get("/api/memories", headers=other_headers).json()
        self.assertEqual(other_memories["memories"][0]["id"], other_memory_id)


if __name__ == "__main__":
    unittest.main()
