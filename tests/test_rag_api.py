import unittest

from fastapi.testclient import TestClient

from apps.api import integrated_server


class RagApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(integrated_server.app)

    def tearDown(self):
        self.client.close()

    def test_stats_reports_versioned_bm25_corpus(self):
        response = self.client.get("/api/rag/stats")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["count"], 10)
        self.assertIn("psychology.json", payload["db_path"])
        self.assertFalse(payload["mutable"])

    def test_list_returns_sourced_documents(self):
        response = self.client.get("/api/rag/list?limit=2")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["documents"]), 2)
        self.assertEqual(payload["total"], 10)
        self.assertTrue(
            payload["documents"][0]["metadata"]["source_url"].startswith("https://")
        )

    def test_search_uses_same_bm25_index_as_agent(self):
        response = self.client.post(
            "/api/rag/search",
            json={"query": "焦虑紧张时怎样回到当下", "top_k": 3},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "bm25")
        self.assertEqual(payload["results"][0]["id"], "who-grounding")
        self.assertEqual(payload["results"][0]["similarity"], 1.0)

    def test_search_rejects_empty_query(self):
        response = self.client.post("/api/rag/search", json={"query": ""})

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
