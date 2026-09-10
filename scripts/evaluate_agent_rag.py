#!/usr/bin/env python3
"""评测数字心屿智能体的心理知识检索质量。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.agent import create_knowledge_retriever
from services.agent.rag_evaluation import (
    evaluate_retriever,
    load_retrieval_eval_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测智能体 RAG 检索质量")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "knowledge" / "psychology.json",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "eval" / "rag" / "cases.json",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, default=0.85)
    parser.add_argument("--min-mrr", type=float, default=0.7)
    parser.add_argument(
        "--embedding-model",
        type=Path,
        help="可选的本地 Sentence Transformers 模型目录；不可用时降级为 BM25",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


async def run() -> int:
    args = parse_args()
    retriever, provider = create_knowledge_retriever(
        args.corpus,
        embedding_model_path=args.embedding_model,
    )
    cases = load_retrieval_eval_cases(args.cases)
    report = await evaluate_retriever(retriever, cases, top_k=args.top_k)
    payload = report.to_dict()
    payload["provider"] = provider
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    passed = (
        report.hit_rate >= args.min_hit_rate
        and report.mean_reciprocal_rank >= args.min_mrr
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
