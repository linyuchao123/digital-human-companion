import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "digital_human_engine" / "train.jioaben" / "inference_eval.py"
)
SPEC = importlib.util.spec_from_file_location("avatar_inference_eval", SCRIPT_PATH)
assert SPEC and SPEC.loader
INFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFERENCE)


class AvatarInferenceEvalTests(unittest.TestCase):
    def test_zero_byte_emotion_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.csv"
            path.touch()

            with self.assertRaisesRegex(ValueError, "不存在或为空"):
                INFERENCE.load_emotion_csv(path, 4)

    def test_prediction_file_can_resume_from_recorded_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "prediction.npy"
            shape = (2, 1, 4, 25)
            values, start, _, _ = INFERENCE.open_prediction_file(
                output_path,
                shape,
                resume=False,
            )
            self.assertEqual(start, 0)
            values[0] = 3.0
            values.flush()
            del values
            _, progress_path = INFERENCE.progress_paths(output_path)
            INFERENCE.save_progress(progress_path, shape, 1)

            resumed, start, _, _ = INFERENCE.open_prediction_file(
                output_path,
                shape,
                resume=True,
            )

            self.assertEqual(start, 1)
            self.assertTrue(np.all(resumed[0] == 3.0))
            del resumed


if __name__ == "__main__":
    unittest.main()
