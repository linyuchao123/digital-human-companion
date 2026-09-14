import os
import unittest
from unittest.mock import Mock, patch

from apps.api import integrated_server as server


class AsrQualityTests(unittest.TestCase):
    def test_hotwords_are_bounded_deduplicated_and_not_paths(self):
        with patch.dict(os.environ, {"ASR_HOTWORDS": "小安,小安，数字心屿 https://example.com /tmp/words " + "啊" * 21}):
            self.assertEqual(server._asr_hotwords(), "小安 数字心屿")

    def test_empty_config_disables_hotwords(self):
        with patch.dict(os.environ, {"ASR_HOTWORDS": ""}):
            self.assertEqual(server._asr_hotwords(), "")

    def test_recognition_preserves_original_words_and_merges_vad(self):
        model = Mock()
        model.generate.return_value = [{"text": "我不是不开心。"}]
        with patch.object(server, "_get_asr_model", return_value=model), patch.dict(os.environ, {"ASR_HOTWORDS": "小安"}):
            self.assertEqual(server._run_asr("test.wav"), "我不是不开心。")
        self.assertEqual(model.generate.call_args.kwargs, {
            "input": "test.wav", "batch_size_s": 30,
            "merge_vad": True, "merge_length_s": 15, "hotword": "小安",
        })
