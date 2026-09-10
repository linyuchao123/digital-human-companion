import json
import tempfile
import unittest
from pathlib import Path

from services.agent import (
    BM25KnowledgeRetriever,
    BuiltInKnowledgeRetriever,
    FallbackKnowledgeRetriever,
    NullKnowledgeRetriever,
    create_knowledge_retriever,
)


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

    async def test_bm25_retriever_returns_ranked_sourced_documents(self):
        corpus = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "psychology.json"
        retriever = BM25KnowledgeRetriever(corpus)

        results = await retriever.retrieve("最近总是焦虑紧张，怎样回到当下", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn("接触", results[0].content)
        self.assertIn("WHO", results[0].source)
        self.assertEqual(results[0].document_id, "who-grounding")
        self.assertTrue(results[0].source_url.startswith("https://www.who.int/"))
        self.assertEqual(results[0].score, 1.0)
        self.assertGreaterEqual(results[0].score, results[1].score)

    async def test_bm25_retriever_returns_empty_for_unrelated_query(self):
        corpus = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "psychology.json"
        retriever = BM25KnowledgeRetriever(corpus)

        results = await retriever.retrieve("量子芯片编译器", top_k=3)

        self.assertEqual(results, [])

    async def test_bm25_retriever_rejects_duplicate_document_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "knowledge.json"
            corpus.write_text(json.dumps([
                {"id": "same", "content": "焦虑支持", "source": "source", "keywords": []},
                {"id": "same", "content": "睡眠支持", "source": "source", "keywords": []},
            ]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "ID 为空或重复"):
                BM25KnowledgeRetriever(corpus)

    async def test_factory_falls_back_when_corpus_is_missing(self):
        retriever, mode = create_knowledge_retriever(Path("/missing/knowledge.json"))

        results = await retriever.retrieve("我最近很焦虑")

        self.assertEqual(mode, "builtin_fallback")
        self.assertGreaterEqual(len(results), 1)

    async def test_fallback_retriever_recovers_from_primary_failure(self):
        class FailingRetriever:
            async def retrieve(self, query, top_k=3):
                raise OSError("index unavailable")

        retriever = FallbackKnowledgeRetriever(
            FailingRetriever(),
            BuiltInKnowledgeRetriever(),
        )

        results = await retriever.retrieve("我最近很焦虑")

        self.assertGreaterEqual(len(results), 1)

    async def test_null_retriever_always_returns_empty_result(self):
        results = await NullKnowledgeRetriever().retrieve("任意问题")

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
