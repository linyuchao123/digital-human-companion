import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.api import integrated_server
from services.tts import SynthesizedAudio, TtsVoice


class _FakeSystemProvider:
    def list_voices(self):
        return (
            TtsVoice(id="Tingting", name="Tingting", locale="zh_CN", provider="macos_say"),
            TtsVoice(id="Meijia", name="Meijia", locale="zh_TW", provider="macos_say"),
            TtsVoice(
                id="Sandy (中文（中国大陆）)",
                name="Sandy (中文（中国大陆）)",
                locale="zh_CN",
                provider="macos_say",
            ),
        )

    def synthesize(self, text, *, voice, rate=185):
        return SynthesizedAudio(
            content=b"RIFF" + b"\x00" * 64,
            media_type="audio/wav",
            provider="macos_say",
            voice=voice,
        )


class _FakeQwen3Provider:
    def list_voices(self):
        return (
            TtsVoice(id="Chelsie", name="Chelsie · 二次元少女", locale="zh-CN", provider="qwen3_tts"),
            TtsVoice(id="Momo", name="Momo · 活泼俏皮", locale="zh-CN", provider="qwen3_tts"),
        )

    def synthesize(self, text, *, voice, rate=185):
        return SynthesizedAudio(
            content=b"RIFF" + b"\x00" * 64,
            media_type="audio/wav",
            provider="qwen3_tts",
            voice=voice,
        )

class TtsApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(integrated_server.app)

    def tearDown(self):
        self.client.close()

    def test_status_reports_dependency_missing_without_exposing_credentials(self):
        with (
            patch.object(integrated_server, "HAS_TTS", False),
            patch.object(integrated_server, "TTS_PROVIDER", "cosyvoice"),
        ):
            response = self.client.get("/api/status")

        payload = response.json()
        self.assertFalse(payload["modules"]["tts_cosyvoice"])
        self.assertEqual(payload["tts"]["reason"], "dependency_missing")
        self.assertNotIn("api_key", str(payload).lower())

    def test_status_reports_missing_credential_separately(self):
        with (
            patch.object(integrated_server, "HAS_TTS", True),
            patch.object(integrated_server, "QWEN_API_KEY", ""),
            patch.object(integrated_server, "TTS_PROVIDER", "cosyvoice"),
        ):
            response = self.client.get("/api/status")

        self.assertEqual(response.json()["tts"]["reason"], "credential_missing")

    def test_unavailable_tts_returns_service_unavailable_with_fallback(self):
        with patch.object(integrated_server, "TTS_PROVIDER", "browser"):
            response = self.client.post("/api/tts", json={"text": "你好"})

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error"], "tts_unavailable")
        self.assertEqual(payload["fallback"], "browser_speech_synthesis")
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_voice_catalog_lists_server_side_voices(self):
        with (
            patch.object(integrated_server, "TTS_PROVIDER", "macos_say"),
            patch.object(integrated_server, "_system_tts_checked", True),
            patch.object(integrated_server, "_system_tts_provider", _FakeSystemProvider()),
        ):
            response = self.client.get("/api/tts/voices")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["default_voice"], "macos_say:Tingting")
        self.assertEqual(len(payload["voices"]), 3)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_system_tts_returns_wav_and_provider_headers(self):
        with (
            patch.object(integrated_server, "TTS_PROVIDER", "macos_say"),
            patch.object(integrated_server, "_system_tts_checked", True),
            patch.object(integrated_server, "_system_tts_provider", _FakeSystemProvider()),
        ):
            response = self.client.post(
                "/api/tts",
                json={"text": "你好", "voice": "macos_say:Meijia", "rate": 200},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b"RIFF"))
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-tts-provider"], "macos_say")
        self.assertEqual(response.headers["x-tts-voice"], "Meijia")

    def test_system_tts_encodes_chinese_voice_name_in_response_header(self):
        with (
            patch.object(integrated_server, "TTS_PROVIDER", "macos_say"),
            patch.object(integrated_server, "_system_tts_checked", True),
            patch.object(integrated_server, "_system_tts_provider", _FakeSystemProvider()),
        ):
            response = self.client.post(
                "/api/tts",
                json={"text": "你好", "voice": "macos_say:Sandy (中文（中国大陆）)"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("%E4%B8%AD%E6%96%87", response.headers["x-tts-voice"])

    def test_tts_rejects_voice_outside_server_catalog(self):
        with (
            patch.object(integrated_server, "TTS_PROVIDER", "macos_say"),
            patch.object(integrated_server, "_system_tts_checked", True),
            patch.object(integrated_server, "_system_tts_provider", _FakeSystemProvider()),
        ):
            response = self.client.post(
                "/api/tts",
                json={"text": "你好", "voice": "macos_say:$(whoami)"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_voice")

    def test_qwen3_character_voice_is_default_and_can_synthesize(self):
        with (
            patch.object(integrated_server, "TTS_PROVIDER", "qwen3_tts"),
            patch.object(
                integrated_server,
                "_get_qwen3_tts_provider",
                return_value=_FakeQwen3Provider(),
            ),
        ):
            catalog_response = self.client.get("/api/tts/voices")
            audio_response = self.client.post(
                "/api/tts",
                json={"text": "你好呀", "voice": "qwen3_tts:Chelsie"},
            )

        catalog = catalog_response.json()
        self.assertEqual(catalog["default_voice"], "qwen3_tts:Chelsie")
        self.assertEqual(catalog["voices"][1]["name"], "Momo · 活泼俏皮")
        self.assertEqual(audio_response.status_code, 200)
        self.assertEqual(audio_response.headers["x-tts-provider"], "qwen3_tts")
        self.assertEqual(audio_response.headers["x-tts-voice"], "Chelsie")

    def test_tts_rejects_query_string_get_to_keep_text_out_of_access_logs(self):
        response = self.client.get("/api/tts", params={"text": "私密对话"})

        self.assertEqual(response.status_code, 405)

    def test_tts_rejects_text_over_limit_before_provider_call(self):
        response = self.client.post("/api/tts", json={"text": "语" * 501})

        self.assertEqual(response.status_code, 422)

    def test_frontend_checks_capability_and_uses_private_post_request(self):
        html = (
            Path(__file__).resolve().parents[1] / "integrated.html"
        ).read_text(encoding="utf-8")

        self.assertIn("await _canUseServerTTS()", html)
        self.assertIn("fetch('/api/status',{cache:'no-store'})", html)
        self.assertIn("fetch('/api/tts/voices',{cache:'no-store'})", html)
        self.assertIn('id="tts-voice-select"', html)
        self.assertIn("voice:voice||null", html)
        self.assertIn("method:'POST'", html)
        self.assertNotIn("/api/tts?text=", html)


if __name__ == "__main__":
    unittest.main()
