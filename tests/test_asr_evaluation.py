import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from services.asr.evaluation import tokenize, error_counts, summarize


class AsrEvaluationTests(unittest.TestCase):
    def test_chinese_normalization_and_counts(self):
        self.assertEqual(tokenize("你好， DeepSeek ２！"), ["你", "好", "deepseek", "2"])
        result = error_counts("帮我推荐周杰伦的歌曲", "帮我推荐好听的歌曲")
        self.assertEqual(result["reference_tokens"], 10)
        self.assertEqual(result["errors"], 3)
        self.assertAlmostEqual(result["cer"], .3)

    def test_insert_delete_substitute_and_micro_average(self):
        deletion = error_counts("你好呀", "你好")
        insertion = error_counts("你好", "你真好")
        substitution = error_counts("天气", "天晴")
        self.assertEqual(deletion["deletions"], 1)
        self.assertEqual(insertion["insertions"], 1)
        self.assertEqual(substitution["substitutions"], 1)
        report = summarize([{**deletion, "bucket": "安静"}, {**insertion, "bucket": "噪声"}])
        self.assertEqual(report["cases"], 2)
        self.assertAlmostEqual(report["cer"], 2 / 5)
        self.assertEqual(set(report["buckets"]), {"安静", "噪声"})

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            error_counts("", "文本")
        with self.assertRaises(ValueError):
            summarize([])

    def test_manifest_uses_production_transcriber_without_text_leak(self):
        from scripts.manual_asr_smoke import load_cases, run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.wav").write_bytes(b"RIFF" + b"0" * 40)
            manifest = root / "cases.json"
            manifest.write_text(json.dumps([{"id": "one", "audio": "sample.wav",
                "reference": "你好", "bucket": "安静"}], ensure_ascii=False), encoding="utf-8")
            with patch("apps.api.integrated_server._recognize_audio", return_value=("你号", "qwen_cloud", "http_503")) as transcribe:
                report = run(manifest)
            transcribe.assert_called_once_with(str((root / "sample.wav").resolve()))
            self.assertNotIn("reference", report["cases"][0])
            self.assertNotIn("hypothesis", report["cases"][0])
            self.assertEqual(report["cases"][0]["fallback_reason"], "http_503")
            with patch("apps.api.integrated_server._recognize_audio", return_value=("你好", "qwen_cloud", None)):
                self.assertIn("reference", run(manifest, include_text=True)["cases"][0])
            with patch("apps.api.integrated_server._recognize_audio", side_effect=RuntimeError("private provider payload")):
                failed = run(manifest)["cases"][0]
            self.assertEqual(failed["cer"], 1)
            self.assertEqual(failed["recognition_error"], "RuntimeError")
            self.assertNotIn("private", json.dumps(failed))
            outside = root.parent / "outside.wav"
            manifest.write_text(json.dumps([{"id": "bad", "audio": "../outside.wav", "reference": "你好"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(manifest)
