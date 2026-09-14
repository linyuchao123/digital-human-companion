import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.asr import cloud_provider as cloud
from apps.api import integrated_server as server


class CloudAsrTests(unittest.TestCase):
    def test_cloud_status_is_not_claimed_as_verified_ready(self):
        with patch.dict(os.environ, {"ASR_PROVIDER": "qwen"}), patch.object(cloud, "api_key", return_value="test"):
            status = server._asr_status_payload()
        self.assertTrue(status["available"])
        self.assertFalse(status["ready"])
        self.assertEqual(status["provider"], "qwen_cloud")

    def test_http_error_does_not_expose_response_or_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            path.write_bytes(b"RIFF-test")
            response = Mock(status_code=401, text="private response")
            with patch.object(cloud, "api_key", return_value="private-key"), patch.object(cloud.httpx, "Client") as client:
                client.return_value.__enter__.return_value.post.return_value = response
                with self.assertRaisesRegex(cloud.CloudAsrError, "^http_401$"):
                    cloud.transcribe(path)

    def test_payload_and_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            path.write_bytes(b"RIFF-test")
            response = Mock(status_code=200)
            response.json.return_value = {"output": {"text": "我不是不开心。"}}
            with patch.dict(os.environ, {"ASR_API_KEY": "test-secret", "ASR_CLOUD_MODEL": "qwen-audio-3.0-asr-flash"}), patch.object(cloud.httpx, "Client") as client:
                client.return_value.__enter__.return_value.post.return_value = response
                self.assertEqual(cloud.transcribe(path, "小安"), "我不是不开心。")
                request = client.return_value.__enter__.return_value.post.call_args.kwargs
                self.assertEqual(request["json"]["parameters"]["vocabulary"], {"小安": 2})
                self.assertTrue(request["json"]["input"]["messages"][0]["content"][0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))

    def test_forbidden_endpoint(self):
        with patch.dict(os.environ, {"ASR_CLOUD_URL": "https://example.com"}):
            with self.assertRaises(cloud.CloudAsrError):
                cloud.endpoint()

    def test_explicit_fallback(self):
        with patch.dict(os.environ, {"ASR_PROVIDER": "qwen", "ASR_FALLBACK_LOCAL": "true"}), patch.object(cloud, "transcribe", side_effect=cloud.CloudAsrError("http_401")), patch.object(server, "ASR_DEPENDENCY_AVAILABLE", True), patch.object(server, "_run_asr", return_value="原话"):
            self.assertEqual(server._recognize_audio("audio.wav"), ("原话", "funasr_paraformer", "http_401"))

    def test_can_disable_fallback(self):
        with patch.dict(os.environ, {"ASR_PROVIDER": "qwen", "ASR_FALLBACK_LOCAL": "false"}), patch.object(cloud, "transcribe", side_effect=cloud.CloudAsrError("http_401")):
            with self.assertRaises(cloud.CloudAsrError):
                server._recognize_audio("audio.wav")
