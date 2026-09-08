import sqlite3
import tempfile
import unittest
from pathlib import Path

from services.agent import InMemoryMemoryStore, NullMemoryStore, SQLiteMemoryStore


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_is_isolated_by_user(self):
        store = InMemoryMemoryStore()
        await store.remember(1, "用户喜欢在睡前听轻音乐", "preference")

        owner_results = await store.search(1, "睡前喜欢什么")
        other_results = await store.search(2, "睡前喜欢什么")

        self.assertEqual(len(owner_results), 1)
        self.assertEqual(owner_results[0].category, "preference")
        self.assertEqual(other_results, [])

    async def test_user_can_delete_all_memories(self):
        store = InMemoryMemoryStore()
        await store.remember(7, "正在准备研究生考试", "goal")
        await store.remember(7, "偏好简短建议", "preference")

        deleted = await store.forget_all(7)

        self.assertEqual(deleted, 2)
        self.assertEqual(await store.search(7, "研究生考试"), [])

    async def test_null_store_never_returns_memory(self):
        self.assertEqual(await NullMemoryStore().search(1, "任意内容"), [])


class SQLiteMemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memory.db"

        def connection_factory():
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

        self.connection_factory = connection_factory
        conn = self.connection_factory()
        conn.execute("""CREATE TABLE user_memories(
            id TEXT PRIMARY KEY,user_id INTEGER,content TEXT,category TEXT,created_at TEXT
        )""")
        conn.close()
        self.store = SQLiteMemoryStore(self.connection_factory)

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_memory_survives_store_recreation_and_is_deduplicated(self):
        first = await self.store.remember(3, "用户喜欢跑步", "preference")
        second = await SQLiteMemoryStore(self.connection_factory).remember(
            3, "用户喜欢跑步", "preference"
        )
        results = await SQLiteMemoryStore(self.connection_factory).search(3, "喜欢跑步")

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(results), 1)

    async def test_sqlite_memory_remains_user_isolated(self):
        await self.store.remember(3, "正在准备面试", "goal")

        self.assertEqual(await self.store.search(4, "准备面试"), [])


if __name__ == "__main__":
    unittest.main()
