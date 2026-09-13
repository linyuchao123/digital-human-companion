import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from apps.api.integrated_server import app


class CameraAssetTests(unittest.TestCase):
    def test_websocket_observation_protocol(self):
        with patch('apps.api.integrated_server.HAS_DRIVER', False), TestClient(app) as client:
            with client.websocket_connect('/ws/main') as ws:
                def vision():
                    for _ in range(100):
                        msg = ws.receive_json()
                        if msg['type'] == 'vision_status':
                            return msg
                    self.fail('缺少视觉状态反馈')
                ws.send_json(dict(type='vision_control',enabled=True,stream_id='test'))
                self.assertEqual(vision()['status'], '观察中')
                ws.send_json(dict(type='vision_features',stream_id='test',seq=0,face_count=0,quality='poor',
                                  features=dict(smile=0,brow_down=0,eye_closed=0,yaw=0,pitch=0,roll=0)))
                self.assertEqual(vision()['status'], '未检测到人脸')
                ws.send_json(dict(type='vision_features',image='base64'))
                self.assertEqual(vision()['code'], 'invalid_fields')
                ws.send_json(dict(type='vision_control',enabled=False,stream_id='test'))
                self.assertEqual(vision()['status'], '已关闭')

    def test_local_assets_and_model_served(self):
        with TestClient(app) as client:
            paths = ['/api/vision/face-landmarker-model', '/static/mediapipe/camera-worker.mjs',
                     '/static/mediapipe/camera-controller.js',
                     '/static/mediapipe/tasks-vision/vision_bundle.mjs',
                     '/static/mediapipe/tasks-vision/vision_bundle.js',
                     '/static/mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm',
                     '/static/mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm']
            for path in paths:
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertGreater(len(response.content), 100)
                    if path.endswith('.wasm'):
                        self.assertTrue(response.content.startswith(b'\x00asm'))
