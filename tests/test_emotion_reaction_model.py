import unittest

import numpy as np

from services.avatar.emotion_reaction_model import (
    EmotionReactionModel,
    EmotionReactionModelError,
    torch,
    validate_emotion_sequence,
)
from services.avatar.model_assets import inspect_face_driver_checkpoint


class EmotionReactionModelTests(unittest.TestCase):
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
