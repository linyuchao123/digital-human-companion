import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.api import integrated_server as server


class LiveTalkingAdapterTests(unittest.TestCase):
    def test_base_url_validation_rejects_credentials_and_query(self):
        for value in ("", "ftp://127.0.0.1:8010", "http://u:p@127.0.0.1:8010", "http://127.0.0.1:8010?x=1"):
            with patch.object(server, "LIVETALKING_BASE_URL", value):
                self.assertEqual(server._validated_livetalking_base_url(), "")

    def test_catalog_exposes_unconfigured_wav2lip_without_claiming_ready(self):
        with patch.object(server, "LIVETALKING_BASE_URL", ""):
            response = TestClient(server.app).get("/api/avatar/catalog")
        self.assertEqual(response.status_code, 200)
        wav = next(item for item in response.json()["avatars"] if item["id"] == "wav2lip")
        self.assertFalse(wav["available"])
        self.assertEqual(wav["reason"], "service_not_configured")

    def test_offer_fails_actionably_when_service_not_configured(self):
        with patch.object(server, "LIVETALKING_BASE_URL", ""):
            response = TestClient(server.app).post(
                "/api/avatar/livetalking/offer",
                json={"sdp": "v=0", "type": "offer", "avatar": "wav2lip256_avatar1"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "livetalking_unavailable")


if __name__ == "__main__":
    unittest.main()
