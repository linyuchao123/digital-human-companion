from __future__ import annotations

from dataclasses import dataclass
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
