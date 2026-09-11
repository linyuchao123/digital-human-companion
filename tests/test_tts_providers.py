import subprocess
import unittest
from unittest.mock import patch

from services.tts import MacOSSayProvider, TtsVoice
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


if __name__ == "__main__":
    unittest.main()
