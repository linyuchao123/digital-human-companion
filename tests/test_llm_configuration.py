import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.api import integrated_server


class LlmConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(integrated_server.app)

    def tearDown(self):
        self.client.close()

    def test_deepseek_is_primary_and_qwen_is_secondary(self):
        with (
            patch.object(integrated_server, "DEEPSEEK_API_KEY", "deepseek-secret"),
            patch.object(integrated_server, "QWEN_API_KEY", "qwen-secret"),
        ):
            response = self.client.get("/api/status")

        payload = response.json()
        self.assertTrue(payload["modules"]["deepseek_api"])
        self.assertTrue(payload["modules"]["qwen_api"])
        self.assertEqual(
            payload["modules"]["agent_provider"],
            "deepseek_with_qwen_fallback",
        )
        self.assertNotIn("deepseek-secret", str(payload))
        self.assertNotIn("qwen-secret", str(payload))

    def test_qwen_is_used_when_deepseek_is_missing(self):
        with (
            patch.object(integrated_server, "DEEPSEEK_API_KEY", ""),
            patch.object(integrated_server, "QWEN_API_KEY", "qwen-secret"),
        ):
            provider_name = integrated_server._configured_agent_provider_name()

        self.assertEqual(provider_name, "qwen_with_offline_fallback")

    def test_no_cloud_key_uses_offline_provider(self):
        with (
            patch.object(integrated_server, "DEEPSEEK_API_KEY", ""),
            patch.object(integrated_server, "QWEN_API_KEY", ""),
        ):
            provider_name = integrated_server._configured_agent_provider_name()

        self.assertEqual(provider_name, "offline")

    def test_status_reports_the_current_server_port(self):
        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["port"], 80)


if __name__ == "__main__":
    unittest.main()
