import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.api import integrated_server


class AsrStatusTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(integrated_server.app)

    def tearDown(self):
        self.client.close()

    def test_missing_dependency_reports_browser_fallback(self):
        with patch.object(integrated_server, "ASR_DEPENDENCY_AVAILABLE", False):
            payload = self.client.get("/api/status").json()

        self.assertFalse(payload["modules"]["asr_funasr"])
        self.assertFalse(payload["asr"]["available"])
        self.assertEqual(payload["asr"]["provider"], "browser_speech_recognition")

    def test_installed_dependency_reports_lazy_loading(self):
        with (
            patch.object(integrated_server, "ASR_DEPENDENCY_AVAILABLE", True),
            patch.object(integrated_server, "HAS_ASR", False),
            patch.object(integrated_server, "_asr_model", None),
            patch.object(integrated_server, "ASR_LAST_ERROR", None),
        ):
            payload = self.client.get("/api/status").json()

        self.assertTrue(payload["modules"]["asr_funasr"])
        self.assertTrue(payload["asr"]["available"])
        self.assertFalse(payload["asr"]["ready"])
        self.assertIn("首次录音", payload["asr"]["message"])

    def test_loaded_model_reports_ready(self):
        with (
            patch.object(integrated_server, "ASR_DEPENDENCY_AVAILABLE", True),
            patch.object(integrated_server, "HAS_ASR", True),
            patch.object(integrated_server, "_asr_model", object()),
        ):
            payload = self.client.get("/api/status").json()

        self.assertTrue(payload["asr"]["ready"])
        self.assertEqual(payload["asr"]["provider"], "funasr_paraformer")


if __name__ == "__main__":
    unittest.main()
