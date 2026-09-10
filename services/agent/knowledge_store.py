from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path


class KnowledgeCorpusStore:
    """对版本化 JSON 知识语料执行进程内串行、原子写入。"""

    _write_lock = threading.Lock()

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path

    def _load(self) -> list[dict[str, object]]:
        payload = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("知识语料必须是数组")
        return payload

    def _save(self, documents: list[dict[str, object]]) -> None:
        temporary = self.corpus_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(documents, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.corpus_path)

    def add(
        self,
        *,
        content: str,
        source: str,
        source_url: str | None = None,
        keywords: list[str] | None = None,
        category: str = "custom",
    ) -> str:
        normalized_content = content.strip()
        normalized_source = source.strip()
        normalized_url = source_url.strip() if source_url else None
        normalized_keywords = [
            item.strip() for item in (keywords or []) if item.strip()
        ]
        if not normalized_content or len(normalized_content) > 1200:
            raise ValueError("知识内容长度必须为 1 到 1200 个字符")
        if not normalized_source or len(normalized_source) > 200:
            raise ValueError("来源名称长度必须为 1 到 200 个字符")
        if normalized_url and not normalized_url.startswith("https://"):
            raise ValueError("来源链接必须使用 https://")
        document_id = f"custom-{uuid.uuid4()}"
        document: dict[str, object] = {
            "id": document_id,
            "keywords": normalized_keywords,
            "content": normalized_content,
            "source": normalized_source,
            "category": category.strip() or "custom",
        }
        if normalized_url:
            document["source_url"] = normalized_url
        with self._write_lock:
            documents = self._load()
            documents.append(document)
            self._save(documents)
        return document_id

    def delete_custom(self, document_id: str) -> bool:
        if not document_id.startswith("custom-"):
            raise PermissionError("内置权威知识不能通过管理接口删除")
        with self._write_lock:
            documents = self._load()
            remaining = [item for item in documents if item.get("id") != document_id]
            if len(remaining) == len(documents):
                return False
            self._save(remaining)
        return True
