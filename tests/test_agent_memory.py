import unittest

from services.agent import InMemoryMemoryStore, NullMemoryStore


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


if __name__ == "__main__":
    unittest.main()
