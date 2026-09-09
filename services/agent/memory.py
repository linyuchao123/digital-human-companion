from __future__ import annotations

import sqlite3
import uuid
from collections import defaultdict
from typing import Callable, Literal, NamedTuple, Protocol, Sequence

from .state import MemoryRecord


class MemoryCandidate(NamedTuple):
    content: str
    category: Literal["preference", "profile", "goal", "context"]


_SENSITIVE_TERMS = (
    "密码", "验证码", "身份证", "银行卡", "信用卡", "家庭住址", "详细地址",
    "手机号", "电话号码", "护照", "病历", "诊断", "抑郁症", "焦虑症",
)
_REJECTION_TERMS = ("不要记", "别记", "不许记", "忘掉", "删除记忆", "清除记忆")
_CATEGORY_MARKERS = (
    ("preference", ("我喜欢", "我不喜欢", "我偏好", "我习惯", "我最爱", "我讨厌")),
    ("goal", ("我的目标", "我打算", "我计划", "我正在准备", "我想要", "我希望")),
    ("profile", ("我叫", "我的名字", "请叫我", "我是一个", "我是名", "我有一只")),
    ("context", ("我每天", "我每周", "我通常", "我经常", "请记住", "记住我")),
)


def extract_memory_candidate(text: str) -> MemoryCandidate | None:
    """只提取用户主动表达、相对稳定且非敏感的长期信息。"""
    content = " ".join(text.strip().split())
    if not 4 <= len(content) <= 500:
        return None
    if content.endswith(("?", "？")):
        return None
    if any(term in content for term in _REJECTION_TERMS):
        return None
    if any(term in content for term in _SENSITIVE_TERMS):
        return None
    for category, markers in _CATEGORY_MARKERS:
        if any(marker in content for marker in markers):
            return MemoryCandidate(content=content, category=category)
    return None


class MemoryStore(Protocol):
    async def search(self, user_id: int, query: str, limit: int = 5) -> Sequence[MemoryRecord]:
        """仅检索指定用户拥有的长期记忆。"""
        ...

    async def remember(
        self,
        user_id: int,
        content: str,
        category: str = "context",
    ) -> MemoryRecord:
        """为指定用户保存一条长期记忆。"""
        ...

    async def forget_all(self, user_id: int) -> int:
        """删除指定用户的全部长期记忆。"""
        ...

    async def forget_matching(self, user_id: int, query: str, limit: int = 20) -> int:
        """仅删除指定用户与查询相关的长期记忆。"""
        ...


class NullMemoryStore:
    async def search(self, user_id: int, query: str, limit: int = 5) -> Sequence[MemoryRecord]:
        return []

    async def remember(
        self, user_id: int, content: str, category: str = "context"
    ) -> MemoryRecord:
        raise RuntimeError("长期记忆存储未启用")

    async def forget_all(self, user_id: int) -> int:
        return 0

    async def forget_matching(self, user_id: int, query: str, limit: int = 20) -> int:
        return 0


class InMemoryMemoryStore:
    """测试和本地开发使用的用户隔离记忆存储。"""

    def __init__(self) -> None:
        self._records: dict[int, list[MemoryRecord]] = defaultdict(list)

    async def search(self, user_id: int, query: str, limit: int = 5) -> Sequence[MemoryRecord]:
        if not query.strip():
            return []
        query_terms = {query[index:index + 2] for index in range(max(1, len(query) - 1))}
        scored = []
        for record in self._records[user_id]:
            score = sum(term in record.content for term in query_terms)
            if score:
                scored.append((score, record))
        return [record for _, record in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]

    async def remember(
        self, user_id: int, content: str, category: str = "context"
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            category=category,
        )
        self._records[user_id].append(record)
        return record

    async def forget_all(self, user_id: int) -> int:
        deleted = len(self._records[user_id])
        self._records.pop(user_id, None)
        return deleted

    async def forget_matching(self, user_id: int, query: str, limit: int = 20) -> int:
        terms = _bigrams(query)
        if not terms:
            return 0
        threshold = max(1, (len(terms) * 3 + 3) // 4)
        matches = [
            record for record in self._records[user_id]
            if _match_score(record.content, terms) >= threshold
        ][:limit]
        matched_ids = {record.id for record in matches}
        self._records[user_id] = [
            record for record in self._records[user_id] if record.id not in matched_ids
        ]
        return len(matches)


class SQLiteMemoryStore:
    """使用项目数据库持久化长期记忆，不持有跨请求数据库连接。"""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connection_factory = connection_factory

    async def search(self, user_id: int, query: str, limit: int = 5) -> Sequence[MemoryRecord]:
        if not query.strip():
            return []
        conn = self._connection_factory()
        try:
            rows = conn.execute(
                """SELECT id,user_id,content,category,created_at FROM user_memories
                   WHERE user_id=? ORDER BY created_at DESC LIMIT 100""",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        terms = {query[index:index + 2] for index in range(max(1, len(query) - 1))}
        scored = []
        for row in rows:
            score = sum(term in row["content"] for term in terms)
            if score:
                scored.append((score, MemoryRecord(**dict(row))))
        return [record for _, record in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]

    async def remember(
        self, user_id: int, content: str, category: str = "context"
    ) -> MemoryRecord:
        conn = self._connection_factory()
        try:
            existing = conn.execute(
                """SELECT id,user_id,content,category,created_at FROM user_memories
                   WHERE user_id=? AND content=? LIMIT 1""",
                (user_id, content),
            ).fetchone()
            if existing:
                return MemoryRecord(**dict(existing))
            record = MemoryRecord(
                id=str(uuid.uuid4()), user_id=user_id, content=content, category=category
            )
            conn.execute(
                """INSERT INTO user_memories(id,user_id,content,category,created_at)
                   VALUES(?,?,?,?,?)""",
                (
                    record.id,
                    record.user_id,
                    record.content,
                    record.category,
                    record.created_at.isoformat(),
                ),
            )
            conn.commit()
            return record
        finally:
            conn.close()

    async def forget_all(self, user_id: int) -> int:
        conn = self._connection_factory()
        try:
            cursor = conn.execute("DELETE FROM user_memories WHERE user_id=?", (user_id,))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    async def forget_matching(self, user_id: int, query: str, limit: int = 20) -> int:
        terms = _bigrams(query)
        if not terms:
            return 0
        conn = self._connection_factory()
        try:
            rows = conn.execute(
                "SELECT id,content FROM user_memories WHERE user_id=? ORDER BY created_at DESC LIMIT 100",
                (user_id,),
            ).fetchall()
            threshold = max(1, (len(terms) * 3 + 3) // 4)
            matched_ids = [
                row["id"] for row in rows
                if _match_score(row["content"], terms) >= threshold
            ][:limit]
            if not matched_ids:
                return 0
            placeholders = ",".join("?" for _ in matched_ids)
            cursor = conn.execute(
                f"DELETE FROM user_memories WHERE user_id=? AND id IN ({placeholders})",
                (user_id, *matched_ids),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()


def _bigrams(text: str) -> set[str]:
    content = "".join(text.strip().split())
    if not content:
        return set()
    return {content[index:index + 2] for index in range(max(1, len(content) - 1))}


def _match_score(content: str, terms: set[str]) -> int:
    return sum(term in content for term in terms)


def extract_forget_query(text: str) -> str | None:
    content = " ".join(text.strip().split())
    if any(term in content for term in ("全部", "所有", "清空")):
        return None
    for marker in ("不要再记得", "删除关于", "忘掉", "忘记"):
        if marker in content:
            query = content.split(marker, 1)[1].strip(" ：:，,。.!！")
            if query.endswith("的记忆"):
                query = query[:-3].strip()
            return query if len(query) >= 2 else None
    return None
