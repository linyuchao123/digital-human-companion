import unittest

from services.agent import BuiltInKnowledgeRetriever, NullKnowledgeRetriever


class KnowledgeRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_builtin_retriever_returns_sourced_sleep_knowledge(self):
        retriever = BuiltInKnowledgeRetriever()

        results = await retriever.retrieve("最近压力很大而且总是睡不着")

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.source for result in results))
        self.assertTrue(all(0 < result.score <= 1 for result in results))

    async def test_builtin_retriever_ignores_unrelated_chat(self):
        results = await BuiltInKnowledgeRetriever().retrieve("今天中午吃什么")

        self.assertEqual(results, [])

    async def test_null_retriever_always_returns_empty_result(self):
        results = await NullKnowledgeRetriever().retrieve("任意问题")

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
