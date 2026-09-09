import sqlite3
import tempfile
import unittest
from pathlib import Path

from services.agent import InMemoryMemoryStore, NullMemoryStore, SQLiteMemoryStore
from services.agent.memory import extract_forget_query, extract_memory_candidate


class MemoryExtractionTests(unittest.TestCase):
    def test_extracts_stable_information_by_category(self):
        cases = {
            "我喜欢睡前听轻音乐": "preference",
            "我的目标是成为一名AI应用开发工程师": "goal",
            "我叫小林": "profile",
            "我每天早上七点起床": "context",
        }

        for text, category in cases.items():
            with self.subTest(text=text):
                candidate = extract_memory_candidate(text)
                self.assertIsNotNone(candidate)
                self.assertEqual(candidate.category, category)

    def test_rejects_casual_questions_and_sensitive_information(self):
        rejected = (
            "今天天气不错",
            "你记得我喜欢什么吗？",
            "不要记住我喜欢跑步",
            "我的银行卡是123456",
            "我被诊断为焦虑症",
        )

        for text in rejected:
            with self.subTest(text=text):
                self.assertIsNone(extract_memory_candidate(text))

    def test_extracts_specific_forget_query_but_rejects_delete_all(self):
        self.assertEqual(extract_forget_query("请忘掉我喜欢跑步"), "我喜欢跑步")
        self.assertEqual(extract_forget_query("删除关于睡前音乐的记忆"), "睡前音乐")
        self.assertIsNone(extract_forget_query("忘掉所有记忆"))


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

    async def test_user_can_forget_only_matching_memories(self):
        store = InMemoryMemoryStore()
        await store.remember(7, "我喜欢跑步", "preference")
        await store.remember(7, "我喜欢睡前听轻音乐", "preference")

        deleted = await store.forget_matching(7, "我喜欢跑步")

        self.assertEqual(deleted, 1)
        self.assertEqual(await store.search(7, "跑步"), [])
        self.assertEqual(len(await store.search(7, "睡前音乐")), 1)

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

    async def test_sqlite_forget_matching_only_deletes_owned_match(self):
        await self.store.remember(3, "我喜欢跑步", "preference")
        await self.store.remember(3, "我喜欢游泳", "preference")
        await self.store.remember(4, "我喜欢跑步", "preference")

        deleted = await self.store.forget_matching(3, "我喜欢跑步")

        self.assertEqual(deleted, 1)
        self.assertEqual(await self.store.search(3, "跑步"), [])
        self.assertEqual(len(await self.store.search(3, "游泳")), 1)
        self.assertEqual(len(await self.store.search(4, "跑步")), 1)


if __name__ == "__main__":
    unittest.main()
