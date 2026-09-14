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
    def test_mixed_query_dataset_is_valid_and_distinct(self):
        root=Path(__file__).resolve().parents[1]
        cases=load_retrieval_eval_cases(root/'eval/rag/mixed_query_cases.json')
        ids={d['id'] for d in json.loads((root/'data/knowledge/psychology.json').read_text(encoding='utf-8'))}
        old=load_retrieval_eval_cases(root/'eval/rag/natural_language_cases.json')
        self.assertEqual(len(cases),24)
        self.assertEqual(sum(c.expect_no_results for c in cases),12)
        self.assertEqual(len({c.query for c in cases}),24)
        self.assertFalse({c.query for c in cases}&{c.query for c in old})
        self.assertTrue(all(set(c.expected_document_ids)<=ids for c in cases))

    async def test_negative_cases_report_false_positives_separately(self):
        report=await evaluate_retriever(StaticRetriever(),[
            RetrievalEvalCase('hit case',('expected',)),
            RetrievalEvalCase('unrelated question',(),expect_no_results=True)])
        self.assertEqual(report.hit_rate,1)
        self.assertEqual(report.mean_reciprocal_rank,0.5)
        self.assertEqual(report.negative_case_count,1)
        self.assertEqual(report.false_positive_rate,1)
        self.assertEqual(report.failures[0]['failure_type'],'false_positive')

    async def test_all_negative_empty_retriever_has_no_division_error(self):
        class Empty:
            async def retrieve(self,query,top_k=3):return []
        report=await evaluate_retriever(Empty(),[RetrievalEvalCase('unrelated',(),True)])
        self.assertEqual(report.false_positive_rate,0)
        self.assertEqual(report.hit_rate,0)
        self.assertEqual(report.failures,())

    def test_natural_language_dataset_ids_exist(self):
        root=Path(__file__).resolve().parents[1]
        cases=load_retrieval_eval_cases(root/'eval/rag/natural_language_cases.json')
        ids={d['id'] for d in json.loads((root/'data/knowledge/psychology.json').read_text(encoding='utf-8'))}
        self.assertEqual(len(cases),16)
        self.assertEqual(sum(c.expect_no_results for c in cases),4)
        self.assertTrue(all(set(c.expected_document_ids)<=ids for c in cases))
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
