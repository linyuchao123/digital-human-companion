import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import integrated_server as server


class ChatSessionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.original = server.DB_PATH
        server.DB_PATH = Path(self.folder.name) / "sessions.db"
        server._init_db()
        self.client = TestClient(server.app)
        self.headers = []
        for name in ("sessions-a", "sessions-b"):
            result = self.client.post("/api/auth/register", json={"username": name, "password": "test-password"}).json()
            self.headers.append({"X-Auth-Token": result["token"]})

    def tearDown(self):
        self.client.close()
        server.DB_PATH = self.original
        self.folder.cleanup()

    def main(self, index=0):
        return self.client.post("/api/sessions/main", headers=self.headers[index])

    def test_main_session_is_idempotent_and_account_scoped(self):
        first = self.main().json()
        self.assertEqual(first["session_id"], self.main().json()["session_id"])
        self.assertNotEqual(first["session_id"], self.main(1).json()["session_id"])
        sessions = self.client.get("/api/sessions", headers=self.headers[0]).json()["sessions"]
        self.assertEqual(sum(item["is_main"] for item in sessions), 1)
        self.assertEqual(self.client.get(f'/api/sessions/{first["session_id"]}/messages', headers=self.headers[1]).status_code, 404)
        self.assertEqual(self.client.delete(f'/api/sessions/{first["session_id"]}', headers=self.headers[0]).status_code, 409)

    def test_upgrade_selects_latest_nonempty_session_and_is_repeatable(self):
        user_id = self.client.get("/api/profile", headers=self.headers[0]).json()["id"]
        with server._get_db() as conn:
            for sid, updated in (("old", "2026-01-01"), ("latest", "2026-02-01"), ("empty", "2026-03-01")):
                conn.execute("INSERT INTO chat_sessions(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", (sid,user_id,sid,updated,updated))
            for sid in ("old", "latest"):
                conn.execute("INSERT INTO chat_messages(session_id,role,content,ts) VALUES(?,?,?,?)", (sid,"user",sid,"2026"))
            conn.commit()
        server._init_db(); server._init_db()
        self.assertEqual(self.main().json()["session_id"], "latest")

    def test_blank_new_session_is_reused_and_settings_persist(self):
        self.main()
        first = self.client.post("/api/sessions", headers=self.headers[0]).json()
        second = self.client.post("/api/sessions", headers=self.headers[0]).json()
        self.assertEqual(first["session_id"], second["session_id"])
        sid = first["session_id"]
        changed = self.client.patch(f"/api/sessions/{sid}", headers=self.headers[0], json={
            "title": "工作压力与睡眠困扰", "dialogue_mode": "emotional", "emotion_style": "gentle"
        }).json()
        self.assertEqual((changed["title_manual"], changed["dialogue_mode"], changed["emotion_style"]), (1,"emotional","gentle"))
        self.assertEqual(self.client.patch(f"/api/sessions/{sid}", headers=self.headers[1], json={"title":"越权"}).status_code, 404)

    def test_message_pagination_clear_and_delete(self):
        main = self.main().json()["session_id"]
        with server._get_db() as conn:
            for index in range(7):
                conn.execute("INSERT INTO chat_messages(session_id,role,content,ts) VALUES(?,?,?,?)", (main,"user",str(index),f"2026-{index}"))
            conn.commit()
        recent = self.client.get(f"/api/sessions/{main}/messages?limit=3", headers=self.headers[0]).json()
        older = self.client.get(f'/api/sessions/{main}/messages?limit=3&before_id={recent["next_before_id"]}', headers=self.headers[0]).json()
        self.assertEqual([m["content"] for m in recent["messages"]], ["4","5","6"])
        self.assertEqual([m["content"] for m in older["messages"]], ["1","2","3"])
        self.assertFalse({m["id"] for m in recent["messages"]} & {m["id"] for m in older["messages"]})
        self.assertEqual(self.client.post(f"/api/sessions/{main}/clear", headers=self.headers[0]).status_code, 200)
        self.assertEqual(self.client.get(f"/api/sessions/{main}/messages", headers=self.headers[0]).json()["messages"], [])

    def test_session_list_pagination_has_no_duplicates(self):
        self.main()
        user_id = self.client.get("/api/profile", headers=self.headers[0]).json()["id"]
        with server._get_db() as conn:
            for index in range(8):
                conn.execute("INSERT INTO chat_sessions(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", (f"s{index}",user_id,str(index),"2026",f"2026-01-{index+1:02d}"))
            conn.commit()
        first = self.client.get("/api/sessions?limit=4", headers=self.headers[0]).json()
        second = self.client.get(f'/api/sessions?limit=4&cursor={first["next_cursor"]}', headers=self.headers[0]).json()
        first_ids = {item["id"] for item in first["sessions"]}
        second_ids = {item["id"] for item in second["sessions"]}
        self.assertFalse(first_ids & second_ids)


if __name__ == "__main__":
    unittest.main()
