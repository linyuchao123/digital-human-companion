import json
import tempfile
import unittest
from pathlib import Path

from services.agent import (
    BM25KnowledgeRetriever,
    BuiltInKnowledgeRetriever,
    FallbackKnowledgeRetriever,
    HybridKnowledgeRetriever,
    NullKnowledgeRetriever,
    create_knowledge_retriever,
)


class KnowledgeRetrieverTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _corpus_path():
        return Path(__file__).resolve().parents[1] / "data" / "knowledge" / "psychology.json"

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
        corpus = self._corpus_path()
        retriever = BM25KnowledgeRetriever(corpus)

        results = await retriever.retrieve("最近总是焦虑紧张，怎样回到当下", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn("接触", results[0].content)
        self.assertIn("WHO", results[0].source)
        self.assertEqual(results[0].document_id, "who-grounding")
        self.assertTrue(results[0].source_url.startswith("https://www.who.int/"))
        self.assertEqual(results[0].score, 1.0)
        self.assertGreaterEqual(results[0].score, results[1].score)
        self.assertEqual(
            len({item.document_id for item in results}),
            len(results),
        )

    async def test_bm25_retriever_returns_empty_for_unrelated_query(self):
        corpus = self._corpus_path()
        retriever = BM25KnowledgeRetriever(corpus)

        results = await retriever.retrieve("量子芯片编译器", top_k=3)

        self.assertEqual(results, [])

    async def test_bm25_retriever_recalls_colloquial_crisis_expression(self):
        retriever = BM25KnowledgeRetriever(self._corpus_path())

        results = await retriever.retrieve("我不想活了，继续下去没有意义", top_k=1)

        self.assertEqual(results[0].document_id, "who-immediate-danger")

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

    async def test_factory_enables_hybrid_retrieval_with_injected_encoder(self):
        class SemanticEncoder:
            def encode(self, texts):
                return [[1.0, 0.0] for _ in texts]

        retriever, mode = create_knowledge_retriever(
            self._corpus_path(),
            embedding_encoder=SemanticEncoder(),
        )

        results = await retriever.retrieve("焦虑时怎样回到当下", top_k=1)

        self.assertEqual(mode, "hybrid_with_fallback")
        self.assertEqual(results[0].document_id, "who-grounding")

    async def test_factory_uses_bm25_when_local_embedding_model_is_missing(self):
        retriever, mode = create_knowledge_retriever(
            self._corpus_path(),
            embedding_model_path=Path("/missing/local-embedding-model"),
        )

        results = await retriever.retrieve("焦虑时怎样回到当下", top_k=1)

        self.assertEqual(mode, "bm25_with_fallback")
        self.assertEqual(results[0].document_id, "who-grounding")

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

    async def test_hybrid_retriever_can_recall_semantic_only_match(self):
        class SemanticEncoder:
            def encode(self, texts):
                vectors = []
                for text in texts:
                    if "孤独" in text or "被世界遗忘" in text:
                        vectors.append([1.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0])
                return vectors

        retriever = HybridKnowledgeRetriever(
            BM25KnowledgeRetriever(self._corpus_path()),
            SemanticEncoder(),
        )

        results = await retriever.retrieve("感觉自己被世界遗忘了", top_k=1)

        self.assertEqual(results[0].document_id, "who-social-connection")
        self.assertEqual(results[0].score, 1.0)

    async def test_hybrid_retriever_falls_back_when_query_encoding_fails(self):
        class FailingQueryEncoder:
            calls = 0

            def encode(self, texts):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("encoder unavailable")
                return [[1.0, 0.0] for _ in texts]

        retriever = HybridKnowledgeRetriever(
            BM25KnowledgeRetriever(self._corpus_path()),
            FailingQueryEncoder(),
        )

        results = await retriever.retrieve("焦虑紧张，怎样回到当下", top_k=1)

        self.assertEqual(results[0].document_id, "who-grounding")


if __name__ == "__main__":
    unittest.main()
