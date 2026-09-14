import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from apps.api import integrated_server as server


class KnowledgeHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.original=server.DB_PATH
        server.DB_PATH=Path(self.temp.name)/'test.db'
        server._init_db()
        conn=server._get_db()
        conn.execute("INSERT INTO chat_sessions VALUES('owned',1,'测试','2026','2026')")
        conn.commit();conn.close()
        self.client=TestClient(server.app)

    def tearDown(self):
        self.client.close();server.DB_PATH=self.original;self.temp.cleanup()

    def test_snapshot_survives_reopen_and_remains_owner_scoped(self):
        refs=[{'id':'old-document','source':'原始来源','excerpt':'当轮知识快照','url':'https://www.nhs.uk/'}]
        server._db_save_message('owned','assistant','知识性句子[1]','平静',refs)
        server._init_db()  # 可重复迁移，不覆盖历史。
        with patch.object(server,'_get_user_id_from_request',return_value=1):
            messages=self.client.get('/api/sessions/owned/messages').json()['messages']
        self.assertEqual(messages[0]['knowledge_sources'],refs)
        with patch.object(server,'_get_user_id_from_request',return_value=2):
            self.assertEqual(self.client.get('/api/sessions/owned/messages').status_code,404)

    def test_legacy_and_corrupt_snapshot_do_not_break_history(self):
        server._db_save_message('owned','user','输入',knowledge_sources=[{'source':'private','excerpt':'not saved'}])
        server._db_save_message('owned','assistant','旧消息')
        conn=server._get_db()
        conn.execute("UPDATE chat_messages SET knowledge_sources='broken' WHERE role='assistant'")
        conn.commit();conn.close()
        with patch.object(server,'_get_user_id_from_request',return_value=1):
            messages=self.client.get('/api/sessions/owned/messages').json()['messages']
        self.assertTrue(all(m['knowledge_sources']==[] for m in messages))
