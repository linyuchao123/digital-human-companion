from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .state import KnowledgeSnippet


class KnowledgeRetriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 3) -> Sequence[KnowledgeSnippet]:
        """返回经过来源标注的相关知识片段。"""
        ...


class NullKnowledgeRetriever:
    async def retrieve(self, query: str, top_k: int = 3) -> Sequence[KnowledgeSnippet]:
        return []


@dataclass(frozen=True)
class _IndexedDocument:
    document_id: str
    content: str
    source: str
    source_url: str | None
    tokens: Counter[str]
    length: int


class BM25KnowledgeRetriever:
    """从可审查的本地 JSON 语料中执行离线 BM25 检索。"""

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path
        self._documents = self._load_documents(corpus_path)
        self._document_frequency: Counter[str] = Counter()
        for document in self._documents:
            self._document_frequency.update(document.tokens.keys())
        self._average_length = (
            sum(document.length for document in self._documents) / len(self._documents)
        )

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def list_documents(self, limit: int = 50) -> list[KnowledgeSnippet]:
        safe_limit = min(max(limit, 1), 500)
        return [
            KnowledgeSnippet(
                document_id=document.document_id,
                content=document.content,
                source=document.source,
                source_url=document.source_url,
                score=0,
            )
            for document in self._documents[:safe_limit]
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        normalized = text.strip().lower()
        tokens: list[str] = []
        for segment in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalized):
            if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
                if len(segment) == 1:
                    tokens.append(segment)
                else:
                    tokens.extend(
                        segment[index : index + 2]
                        for index in range(len(segment) - 1)
                    )
            else:
                tokens.append(segment)
        return tokens

    @classmethod
    def _load_documents(cls, corpus_path: Path) -> list[_IndexedDocument]:
        if not corpus_path.is_file():
            raise FileNotFoundError(f"心理知识语料不存在: {corpus_path}")
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("心理知识语料必须是非空数组")
        documents: list[_IndexedDocument] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"心理知识语料第 {index + 1} 条不是对象")
            document_id = str(item.get("id", "")).strip()
            content = str(item.get("content", "")).strip()
            source = str(item.get("source", "")).strip()
            source_url = str(item.get("source_url", "")).strip() or None
            keywords = item.get("keywords", [])
            if not document_id or document_id in seen_ids:
                raise ValueError(f"心理知识语料第 {index + 1} 条 ID 为空或重复")
            if not content or not source:
                raise ValueError(f"心理知识语料第 {index + 1} 条缺少内容或来源")
            if not isinstance(keywords, list):
                raise ValueError(f"心理知识语料第 {index + 1} 条 keywords 必须是数组")
            seen_ids.add(document_id)
            tokens = cls._tokenize(" ".join((content, *map(str, keywords))))
            if not tokens:
                raise ValueError(f"心理知识语料第 {index + 1} 条无法建立索引")
            documents.append(_IndexedDocument(
                document_id=document_id,
                content=content,
                source=source,
                source_url=source_url,
                tokens=Counter(tokens),
                length=len(tokens),
            ))
        return documents

    def _score(self, query_tokens: set[str], document: _IndexedDocument) -> float:
        score = 0.0
        count = len(self._documents)
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            frequency = document.tokens.get(token, 0)
            if not frequency:
                continue
            document_frequency = self._document_frequency[token]
            inverse_frequency = math.log(
                1 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * document.length / self._average_length
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        return score

    async def retrieve(self, query: str, top_k: int = 3) -> Sequence[KnowledgeSnippet]:
        if top_k <= 0:
            return []
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []
        ranked = sorted(
            (
                (self._score(query_tokens, document), document)
                for document in self._documents
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        positive = [(score, document) for score, document in ranked if score > 0]
        if not positive:
            return []
        maximum = positive[0][0]
        return [
            KnowledgeSnippet(
                document_id=document.document_id,
                content=document.content,
                source=document.source,
                source_url=document.source_url,
                score=min(1.0, score / maximum),
            )
            for score, document in positive[:top_k]
        ]


class FallbackKnowledgeRetriever:
    """主检索器无结果或异常时回退，避免知识服务阻断陪伴对话。"""

    def __init__(
        self,
        primary: KnowledgeRetriever,
        fallback: KnowledgeRetriever,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    async def retrieve(self, query: str, top_k: int = 3) -> Sequence[KnowledgeSnippet]:
        try:
            results = await self._primary.retrieve(query, top_k)
        except Exception:
            results = []
        if results:
            return results
        return await self._fallback.retrieve(query, top_k)


@dataclass(frozen=True)
class _BuiltInEntry:
    keywords: tuple[str, ...]
    content: str
    source: str = "数字心屿内置心理教育知识"


class BuiltInKnowledgeRetriever:
    """无需向量模型的保底知识检索，适合本地演示与云端降级。"""

    _ENTRIES = (
        _BuiltInEntry(
            keywords=("焦虑", "紧张", "压力"),
            content="焦虑明显时，可以尝试缓慢呼吸并把注意力带回当下；若持续影响生活，建议寻求专业帮助。",
        ),
        _BuiltInEntry(
            keywords=("睡不着", "失眠", "睡眠"),
            content="规律作息、睡前减少强光和咖啡因、将床主要用于睡眠，有助于建立稳定的睡眠节律。",
        ),
        _BuiltInEntry(
            keywords=("孤独", "孤单", "没人"),
            content="孤独感需要被看见。与可信任的人保持低压力联系，通常比强迫自己立即振作更可行。",
        ),
    )

    async def retrieve(self, query: str, top_k: int = 3) -> Sequence[KnowledgeSnippet]:
        matches = []
        for entry in self._ENTRIES:
            hit_count = sum(keyword in query for keyword in entry.keywords)
            if hit_count:
                matches.append(KnowledgeSnippet(
                    content=entry.content,
                    source=entry.source,
                    score=min(1, hit_count / len(entry.keywords)),
                ))
        return sorted(matches, key=lambda item: item.score, reverse=True)[:top_k]


def create_knowledge_retriever(
    corpus_path: Path,
) -> tuple[KnowledgeRetriever, str]:
    """创建本地知识检索器，语料不可用时保留内置降级能力。"""
    fallback = BuiltInKnowledgeRetriever()
    try:
        primary = BM25KnowledgeRetriever(corpus_path)
    except (OSError, ValueError):
        return fallback, "builtin_fallback"
    return FallbackKnowledgeRetriever(primary, fallback), "bm25_with_fallback"
