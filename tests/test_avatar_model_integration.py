import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from apps.api.integrated_server import (
    SessionState,
    _compute_live2d_params,
    _driver_runtime_payload,
    _infer_live2d_params,
)


class _FakeReactionModel:
    def predict(self, sequence, *, num_candidates, seed):
        assert sequence.shape[1] == 25
        assert num_candidates == 1
        assert seed == 42
        prediction = np.zeros((1, sequence.shape[0], 25), dtype=np.float32)
        prediction[0, -1, 18] = 1.0  # Happy
        return prediction


class AvatarModelIntegrationTests(unittest.TestCase):
    def test_runtime_payload_distinguishes_official_model_and_fallback(self):
        class Metadata:
            epoch = 111
            device = "cpu"
            device_fallback_reason = "NotImplementedError: MPS unsupported"

        class LoadedModel:
            metadata = Metadata()

        official = _driver_runtime_payload(LoadedModel())
        fallback = _driver_runtime_payload(None)

        self.assertTrue(official["ready"])
        self.assertEqual(official["mode"], "official")
        self.assertEqual(official["epoch"], 111)
        self.assertTrue(official["device_fallback"])
        self.assertIn("切换 CPU", official["message"])
        self.assertFalse(fallback["ready"])
        self.assertEqual(fallback["mode"], "fallback")

    def test_prediction_uses_official_emotion_mapping(self):
        sequence = np.zeros((8, 25), dtype=np.float32)
        sequence[:, 17] = 1.0

        params = _infer_live2d_params(_FakeReactionModel(), sequence, 0.5)

        self.assertIsNotNone(params)
        self.assertIn("PARAM_MOUTH_FORM", params)
        self.assertGreater(params["PARAM_MOUTH_FORM"], 0)

    def test_prediction_failure_returns_none_for_safe_fallback(self):
        class BrokenModel:
            def predict(self, *args, **kwargs):
                raise RuntimeError("inference failed")

        params = _infer_live2d_params(
            BrokenModel(), np.zeros((8, 25), dtype=np.float32), 0.5
        )

        self.assertIsNone(params)


class AvatarModelCircuitBreakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_disables_official_model_after_unrecoverable_failure(self):
        class BrokenModel:
            def predict(self, *args, **kwargs):
                raise RuntimeError("inference failed")

        state = SessionState("test-session")
        state.model_device = BrokenModel()
        with patch("apps.api.integrated_server.HAS_EMOTION_MAP", True):
            params = await _compute_live2d_params(
                state,
                asyncio.get_running_loop(),
            )

        self.assertIsInstance(params, dict)
        self.assertIsNone(state.model_device)
        self.assertIsNone(state.model_emotion_params)


if __name__ == "__main__":
    unittest.main()
