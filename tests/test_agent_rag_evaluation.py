import json
import tempfile
import unittest
from pathlib import Path

from services.agent import KnowledgeSnippet
from services.agent.rag_evaluation import (
    RetrievalEvalCase,
    evaluate_retriever,
    load_retrieval_eval_cases,
)


class StaticRetriever:
    async def retrieve(self, query, top_k=3):
        document_ids = ["wrong", "expected"] if "hit" in query else ["wrong"]
        return [
            KnowledgeSnippet(
                document_id=document_id,
                content="测试内容",
                source="测试来源",
                score=1 / rank,
            )
            for rank, document_id in enumerate(document_ids[:top_k], start=1)
        ]


class AgentRagEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluation_computes_hit_rate_and_mrr(self):
        cases = [
            RetrievalEvalCase("hit case", ("expected",)),
            RetrievalEvalCase("miss case", ("expected",)),
        ]

        report = await evaluate_retriever(StaticRetriever(), cases, top_k=3)

        self.assertEqual(report.hit_rate, 0.5)
        self.assertEqual(report.mean_reciprocal_rank, 0.25)
        self.assertEqual(len(report.failures), 1)

    async def test_eval_case_loader_rejects_empty_expected_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.json"
            path.write_text(
                json.dumps([{"query": "test", "expected_document_ids": []}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "缺少问题或期望文档"):
                load_retrieval_eval_cases(path)


if __name__ == "__main__":
    unittest.main()
