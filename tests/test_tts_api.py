import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.api import integrated_server


class TtsApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(integrated_server.app)

    def tearDown(self):
        self.client.close()

    def test_status_reports_dependency_missing_without_exposing_credentials(self):
        with patch.object(integrated_server, "HAS_TTS", False):
            response = self.client.get("/api/status")

        payload = response.json()
        self.assertFalse(payload["modules"]["tts_cosyvoice"])
        self.assertEqual(payload["tts"]["reason"], "dependency_missing")
        self.assertNotIn("api_key", str(payload).lower())

    def test_status_reports_missing_credential_separately(self):
        with (
            patch.object(integrated_server, "HAS_TTS", True),
            patch.object(integrated_server, "QWEN_API_KEY", ""),
        ):
            response = self.client.get("/api/status")

        self.assertEqual(response.json()["tts"]["reason"], "credential_missing")

    def test_unavailable_tts_returns_service_unavailable_with_fallback(self):
        with patch.object(integrated_server, "HAS_TTS", False):
            response = self.client.post("/api/tts", json={"text": "你好"})

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error"], "tts_unavailable")
        self.assertEqual(payload["fallback"], "browser_speech_synthesis")
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_tts_rejects_query_string_get_to_keep_text_out_of_access_logs(self):
        response = self.client.get("/api/tts", params={"text": "私密对话"})

        self.assertEqual(response.status_code, 405)

    def test_tts_rejects_text_over_limit_before_provider_call(self):
        response = self.client.post("/api/tts", json={"text": "语" * 501})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
