import subprocess
import unittest
from unittest.mock import patch

from services.tts import MacOSSayProvider, Qwen3TtsProvider, TtsVoice
from services.tts.providers import TtsProviderError


class MacOSSayProviderTests(unittest.TestCase):
    def setUp(self):
        self.voices = (
            TtsVoice("Tingting", "Tingting", "zh_CN", "macos_say"),
            TtsVoice("Flo (中文（中国大陆）)", "Flo (中文（中国大陆）)", "zh_CN", "macos_say"),
        )

    def test_discovers_only_chinese_system_voices(self):
        output = "Tingting            zh_CN    # 你好\nAlice               it_IT    # Ciao\n"
        with patch("services.tts.providers.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
            provider = MacOSSayProvider(executable="/usr/bin/say")

        self.assertEqual([voice.id for voice in provider.list_voices()], ["Tingting"])

    def test_synthesis_uses_argument_list_and_returns_wav(self):
        provider = MacOSSayProvider(executable="/usr/bin/say", voices=self.voices)

        def create_wav(arguments, **kwargs):
            output_path = arguments[arguments.index("-o") + 1]
            with open(output_path, "wb") as output:
                output.write(b"RIFF" + b"\x00" * 64)
            return subprocess.CompletedProcess(arguments, 0)

        with patch("services.tts.providers.subprocess.run", side_effect=create_wav) as run:
            audio = provider.synthesize("你好", voice="Tingting", rate=190)

        self.assertEqual(audio.media_type, "audio/wav")
        self.assertEqual(audio.provider, "macos_say")
        self.assertIn("Tingting", run.call_args.args[0])

    def test_rejects_voice_outside_allowlist_before_subprocess(self):
        provider = MacOSSayProvider(executable="/usr/bin/say", voices=self.voices)

        with patch("services.tts.providers.subprocess.run") as run:
            with self.assertRaisesRegex(TtsProviderError, "允许列表"):
                provider.synthesize("你好", voice="$(unsafe)")

        run.assert_not_called()


class _FakeResponse:
    def __init__(self, *, payload=None, content=b"", content_type="application/json"):
        self._payload = payload
        self.content = content
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttpClient:
    def __init__(self, *, audio_url="https://result.aliyuncs.com/test.wav"):
        self.audio_url = audio_url
        self.request_json = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, *, headers, json):
        self.request_json = json
        return _FakeResponse(payload={"output": {"audio": {"url": self.audio_url}}})

    def get(self, url):
        return _FakeResponse(content=b"RIFF" + b"\x00" * 64, content_type="audio/wav")


class Qwen3TtsProviderTests(unittest.TestCase):
    def test_lists_character_voices_and_sends_cute_instruction(self):
        client = _FakeHttpClient()
        provider = Qwen3TtsProvider(api_key="secret", client_factory=lambda **kwargs: client)

        audio = provider.synthesize("你好呀", voice="Chelsie")

        self.assertEqual(audio.provider, "qwen3_tts")
        self.assertEqual(audio.media_type, "audio/wav")
        self.assertEqual([voice.id for voice in provider.list_voices()], ["Chelsie", "Momo", "Cherry"])
        self.assertIn("可爱俏皮", client.request_json["input"]["instructions"])
        self.assertNotIn("secret", str(client.request_json))

    def test_rejects_unknown_voice_before_network_request(self):
        factory = unittest.mock.Mock()
        provider = Qwen3TtsProvider(api_key="secret", client_factory=factory)

        with self.assertRaisesRegex(TtsProviderError, "允许列表"):
            provider.synthesize("你好", voice="Unknown")

        factory.assert_not_called()

    def test_rejects_audio_download_outside_aliyun_domain(self):
        client = _FakeHttpClient(audio_url="https://example.com/untrusted.wav")
        provider = Qwen3TtsProvider(api_key="secret", client_factory=lambda **kwargs: client)

        with self.assertRaisesRegex(TtsProviderError, "阿里云域名"):
            provider.synthesize("你好", voice="Momo")


if __name__ == "__main__":
    unittest.main()
