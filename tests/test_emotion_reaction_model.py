import unittest
from threading import Lock
from unittest.mock import Mock

import numpy as np

from services.avatar.emotion_reaction_model import (
    EmotionReactionModel,
    EmotionReactionModelError,
    EmotionReactionMetadata,
    is_unsupported_mps_error,
    torch,
    validate_emotion_sequence,
)
from services.avatar.model_assets import inspect_face_driver_checkpoint


class EmotionReactionModelTests(unittest.TestCase):
    def test_only_unsupported_mps_operator_error_triggers_device_fallback(self):
        unsupported = NotImplementedError(
            "The operator is not currently implemented for the MPS device"
        )

        self.assertTrue(is_unsupported_mps_error("mps", unsupported))
        self.assertFalse(is_unsupported_mps_error("cpu", unsupported))
        self.assertFalse(is_unsupported_mps_error("mps", RuntimeError("out of memory")))

    @unittest.skipIf(torch is None, "需要 PyTorch")
    def test_predict_retries_once_on_cpu_after_unsupported_mps_operator(self):
        class MovableModel:
            def __init__(self):
                self.target = None

            def to(self, target):
                self.target = str(target)
                return self

            def eval(self):
                return self

        model = EmotionReactionModel.__new__(EmotionReactionModel)
        model.device = torch.device("mps")
        model.model = MovableModel()
        model.stats = {
            "speaker_emotion": {"mean": 0.0, "std": 1.0},
            "listener_emotion": {"mean": 0.0, "std": 1.0},
        }
        model.metadata = EmotionReactionMetadata(
            epoch=111,
            val_loss=0.4,
            parameter_count=1,
            device="mps",
            use_audio=False,
        )
        model._inference_lock = Lock()
        unsupported = NotImplementedError(
            "The operator is not currently implemented for the MPS device"
        )
        model._generate_normalized = Mock(side_effect=[
            unsupported,
            torch.zeros((1, 1, 8, 25), dtype=torch.float32),
        ])

        result = model.predict(np.zeros((8, 25), dtype=np.float32), seed=42)

        self.assertEqual(result.shape, (1, 8, 25))
        self.assertEqual(model._generate_normalized.call_count, 2)
        self.assertEqual(model.metadata.device, "cpu")
        self.assertEqual(model.model.target, "cpu")
        self.assertIn("NotImplementedError", model.metadata.device_fallback_reason)

    def test_emotion_sequence_requires_25_features(self):
        with self.assertRaisesRegex(EmotionReactionModelError, r"\[T, 25\]"):
            validate_emotion_sequence(np.zeros((8, 24), dtype=np.float32))

    def test_emotion_sequence_rejects_non_finite_values(self):
        sequence = np.zeros((8, 25), dtype=np.float32)
        sequence[0, 0] = np.nan

        with self.assertRaisesRegex(EmotionReactionModelError, "NaN"):
            validate_emotion_sequence(sequence)

    @unittest.skipUnless(
        torch is not None and inspect_face_driver_checkpoint().ready,
        "需要 PyTorch 和本地正式权重",
    )
    def test_official_checkpoint_loads_and_predicts(self):
        model = EmotionReactionModel(device="cpu")
        sequence = np.zeros((8, 25), dtype=np.float32)
        sequence[:, 17] = 1.0

        prediction = model.predict(sequence, num_candidates=1, seed=42)

        self.assertEqual(model.metadata.epoch, 111)
        self.assertEqual(model.metadata.parameter_count, 3_926_631)
        self.assertEqual(prediction.shape, (1, 8, 25))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
