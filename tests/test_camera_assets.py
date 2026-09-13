import unittest
from fastapi.testclient import TestClient
from apps.api.integrated_server import app


class CameraAssetTests(unittest.TestCase):
    def test_local_assets_and_model_served(self):
        with TestClient(app) as client:
            paths = ['/api/vision/face-landmarker-model', '/static/mediapipe/camera-worker.mjs',
                     '/static/mediapipe/camera-controller.js',
                     '/static/mediapipe/tasks-vision/vision_bundle.mjs',
                     '/static/mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm',
                     '/static/mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm']
            for path in paths:
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertGreater(len(response.content), 100)
                    if path.endswith('.wasm'):
                        self.assertTrue(response.content.startswith(b'\x00asm'))
