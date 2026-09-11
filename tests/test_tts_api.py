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
            response = self.client.get("/api/tts", params={"text": "你好"})

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error"], "tts_unavailable")
        self.assertEqual(payload["fallback"], "browser_speech_synthesis")


if __name__ == "__main__":
    unittest.main()
