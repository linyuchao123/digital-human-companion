import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import integrated_server


class RagApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_corpus_path = integrated_server.KNOWLEDGE_CORPUS_PATH
        self.original_admin_token = integrated_server.RAG_ADMIN_TOKEN
        self.original_embedding_model_path = integrated_server.RAG_EMBEDDING_MODEL_PATH
        source = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "psychology.json"
        self.corpus_path = Path(self.temp_dir.name) / "psychology.json"
        self.corpus_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        integrated_server.KNOWLEDGE_CORPUS_PATH = self.corpus_path
        integrated_server.RAG_ADMIN_TOKEN = "test-admin-token"
        integrated_server.RAG_EMBEDDING_MODEL_PATH = ""
        integrated_server._invalidate_rag_caches()
        self.client = TestClient(integrated_server.app)

    def tearDown(self):
        self.client.close()
        integrated_server.KNOWLEDGE_CORPUS_PATH = self.original_corpus_path
        integrated_server.RAG_ADMIN_TOKEN = self.original_admin_token
        integrated_server.RAG_EMBEDDING_MODEL_PATH = self.original_embedding_model_path
        integrated_server._invalidate_rag_caches()
        self.temp_dir.cleanup()

    def test_stats_reports_versioned_bm25_corpus(self):
        response = self.client.get("/api/rag/stats")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["count"], 10)
        self.assertIn("psychology.json", payload["db_path"])
        self.assertTrue(payload["mutable"])

    def test_list_returns_sourced_documents(self):
        response = self.client.get("/api/rag/list?limit=2")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["documents"]), 2)
        self.assertEqual(payload["total"], 10)
        self.assertTrue(
            payload["documents"][0]["metadata"]["source_url"].startswith("https://")
        )

    def test_search_uses_same_retriever_as_agent(self):
        response = self.client.post(
            "/api/rag/search",
            json={"query": "焦虑紧张时怎样回到当下", "top_k": 3},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "bm25_with_fallback")
        self.assertEqual(payload["results"][0]["id"], "who-grounding")
        self.assertEqual(payload["results"][0]["similarity"], 1.0)

    def test_search_rejects_empty_query(self):
        response = self.client.post("/api/rag/search", json={"query": ""})

        self.assertEqual(response.status_code, 400)

    def test_write_requires_admin_token(self):
        response = self.client.post(
            "/api/rag/add",
            json={"content": "测试知识", "source": "测试来源"},
        )

        self.assertEqual(response.status_code, 403)

    def test_write_is_disabled_when_server_has_no_admin_token(self):
        integrated_server.RAG_ADMIN_TOKEN = ""

        stats = self.client.get("/api/rag/stats")
        response = self.client.post(
            "/api/rag/add",
            json={"content": "测试知识", "source": "测试来源"},
        )

        self.assertFalse(stats.json()["mutable"])
        self.assertEqual(response.status_code, 503)

    def test_admin_can_add_and_delete_custom_document(self):
        headers = {"X-RAG-Admin-Token": "test-admin-token"}
        added = self.client.post(
            "/api/rag/add",
            headers=headers,
            json={
                "content": "遇到压力时先暂停片刻",
                "source": "管理员测试",
                "source_url": "https://example.com/knowledge",
                "keywords": ["压力"],
            },
        )

        self.assertEqual(added.status_code, 200)
        document_id = added.json()["id"]
        self.assertTrue(document_id.startswith("custom-"))
        deleted = self.client.delete(
            f"/api/rag/delete/{document_id}",
            headers=headers,
        )
        self.assertEqual(deleted.status_code, 200)

    def test_admin_cannot_delete_curated_document(self):
        response = self.client.delete(
            "/api/rag/delete/who-grounding",
            headers={"X-RAG-Admin-Token": "test-admin-token"},
        )

        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
