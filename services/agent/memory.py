from __future__ import annotations

import sqlite3
import uuid
from collections import defaultdict
from typing import Callable, Protocol, Sequence

from .state import MemoryRecord


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


class NullMemoryStore:
    async def search(self, user_id: int, query: str, limit: int = 5) -> Sequence[MemoryRecord]:
        return []

    async def remember(
        self, user_id: int, content: str, category: str = "context"
    ) -> MemoryRecord:
        raise RuntimeError("长期记忆存储未启用")

    async def forget_all(self, user_id: int) -> int:
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
