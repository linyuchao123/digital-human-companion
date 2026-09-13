import unittest
from services.vision.observation import CameraObservation


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.camera = CameraObservation(lambda: self.now)
        self.control = dict(type='vision_control', enabled=True, stream_id='test')
        self.camera.control(self.control)

    def sample(self, sequence, **changes):
        data = dict(type='vision_features', stream_id='test', seq=sequence, face_count=1,
                    quality='good', features=dict(smile=.8, brow_down=0, eye_closed=0, yaw=0, pitch=0, roll=0))
        data.update(changes)
        return data

    def test_stable_expiry_clear(self):
        for seq in range(6):
            self.now = seq * .6
            self.camera.update(self.sample(seq))
        self.assertIn('微笑', self.camera.summary())
        self.now += 6
        self.assertEqual('', self.camera.summary())
        self.camera.control({**self.control, 'enabled': False})
        self.assertIsNone(self.camera.stream)

    def test_invalid_and_budget(self):
        for change in ({'image': 'base64'}, {'stream_id': 'old'}, {'seq': True},
                       {'face_count': 3}, {'features': {'smile': float('nan')}}):
            with self.assertRaises(ValueError):
                self.camera.update(self.sample(0, **change))
        self.camera.update(self.sample(0))
        with self.assertRaises(ValueError):
            self.camera.update(self.sample(0))
        self.camera.update(self.sample(1))
        with self.assertRaisesRegex(ValueError, 'rate_limited'):
            self.camera.update(self.sample(2))

    def test_multiple_faces_and_isolation(self):
        with self.assertRaises(ValueError):
            self.camera.control({**self.control, 'enabled': False, 'stream_id': 'old'})
        self.assertEqual('test', self.camera.stream)
        self.camera.update(self.sample(0, face_count=2))
        self.assertEqual('', self.camera.summary())
        self.assertEqual('', CameraObservation().summary())
        self.now += .6
        self.camera.update(self.sample(1, face_count=0))
        self.assertFalse(self.camera.samples)

    def test_number_validation_and_poor_quality(self):
        for value in (float('nan'), float('inf'), True, '0.5', -1, 2):
            features = self.sample(0)['features']
            features['smile'] = value
            with self.assertRaises(ValueError):
                self.camera.update(self.sample(0, features=features))
        self.camera.update(self.sample(0, quality='poor'))
        self.assertEqual('', self.camera.summary())

if __name__ == '__main__':
    unittest.main()
