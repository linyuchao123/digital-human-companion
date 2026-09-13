from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .knowledge import KnowledgeRetriever


@dataclass(frozen=True)
class RetrievalEvalCase:
    query: str
    expected_document_ids: tuple[str, ...]
    expect_no_results: bool = False


@dataclass(frozen=True)
class RetrievalEvalReport:
    case_count: int
    top_k: int
    hit_rate: float
    mean_reciprocal_rank: float
    failures: tuple[dict[str, object], ...]
    negative_case_count: int = 0
    false_positive_rate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_retrieval_eval_cases(path: Path) -> list[RetrievalEvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("RAG 评测集必须是非空数组")
    cases: list[RetrievalEvalCase] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"RAG 评测集第 {index + 1} 条不是对象")
        query = str(item.get("query", "")).strip()
        expected = item.get("expected_document_ids", [])
        negative = item.get('expect_no_results', False)
        if type(negative) is not bool or not query or not isinstance(expected, list) or (not expected and not negative) or (expected and negative):
            raise ValueError(f"RAG 评测集第 {index + 1} 条缺少问题或期望文档")
        cases.append(RetrievalEvalCase(
            query=query,
            expected_document_ids=tuple(str(value) for value in expected),
            expect_no_results=negative,
        ))
    return cases


async def evaluate_retriever(
    retriever: KnowledgeRetriever,
    cases: Sequence[RetrievalEvalCase],
    *,
    top_k: int = 3,
) -> RetrievalEvalReport:
    if not cases:
        raise ValueError("RAG 评测集不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    hit_count = 0
    reciprocal_rank_total = 0.0
    failures: list[dict[str, object]] = []
    negative_count = 0; false_positives = 0
    for case in cases:
        results = list(await retriever.retrieve(case.query, top_k=top_k))
        returned_ids = [item.document_id for item in results if item.document_id]
        if case.expect_no_results:
            negative_count += 1
            if results:
                false_positives += 1
                failures.append({'query':case.query, 'failure_type':'false_positive', 'returned_document_ids':returned_ids})
            continue
        expected_ids = set(case.expected_document_ids)
        rank = next(
            (
                index
                for index, document_id in enumerate(returned_ids, start=1)
                if document_id in expected_ids
            ),
            None,
        )
        if rank is None:
            failures.append({
                "query": case.query,
                "expected_document_ids": list(case.expected_document_ids),
                "returned_document_ids": returned_ids,
            })
        else:
            hit_count += 1
            reciprocal_rank_total += 1 / rank
    case_count = len(cases)
    positive_count = case_count - negative_count
    return RetrievalEvalReport(
        case_count=case_count,
        top_k=top_k,
        hit_rate=hit_count / positive_count if positive_count else 0.0,
        mean_reciprocal_rank=reciprocal_rank_total / positive_count if positive_count else 0.0,
        failures=tuple(failures),
        negative_case_count=negative_count,
        false_positive_rate=false_positives / negative_count if negative_count else 0.0,
    )
