"""Ephemeral camera observations. Never images, identity or diagnostic evidence."""
import math
import re
import time
from collections import deque


FIELDS = {'smile', 'brow_down', 'eye_closed', 'yaw', 'pitch', 'roll'}


class CameraObservation:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.clear()

    def clear(self):
        self.stream = None
        self.seq = -1
        self.samples = deque(maxlen=20)
        self.budget = deque()

    def control(self, msg):
        if set(msg) != {'type', 'enabled', 'stream_id'} or type(msg['enabled']) is not bool:
            raise ValueError('invalid_control')
        if not isinstance(msg['stream_id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', msg['stream_id']):
            raise ValueError('invalid_stream')
        self.clear()
        if msg['enabled']:
            self.stream = msg['stream_id']
        return '观察中' if self.stream else '已关闭'

    def update(self, msg):
        if set(msg) != {'type', 'stream_id', 'seq', 'face_count', 'quality', 'features'}:
            raise ValueError('invalid_fields')
        if not self.stream or msg['stream_id'] != self.stream:
            raise ValueError('inactive_stream')
        if type(msg['seq']) is not int or not self.seq < msg['seq'] < 2**53:
            raise ValueError('invalid_sequence')
        if type(msg['face_count']) is not int or not 0 <= msg['face_count'] <= 2:
            raise ValueError('invalid_count')
        if msg['quality'] not in ('good', 'poor'):
            raise ValueError('invalid_quality')
        features = msg['features']
        if not isinstance(features, dict) or set(features) != FIELDS:
            raise ValueError('invalid_features')
        for key, value in features.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('invalid_number')
            if not ((-180 <= value <= 180) if key in ('yaw', 'pitch', 'roll') else (0 <= value <= 1)):
                raise ValueError('invalid_range')
        now = self.clock()
        while self.budget and self.budget[0] <= now - 1:
            self.budget.popleft()
        if len(self.budget) >= 2:
            raise ValueError('rate_limited')
        self.budget.append(now)
        self.seq = msg['seq']
        if msg['face_count'] != 1 or msg['quality'] != 'good':
            self.samples.clear()
            return ('未检测到人脸' if msg['face_count'] == 0 else
                    '多人入镜' if msg['face_count'] > 1 else '画面不清晰')
        if self.samples and now - self.samples[-1][0] > 5:
            self.samples.clear()
        self.samples.append((now, dict(features)))
        return '观察中'

    def summary(self):
        now = self.clock()
        if not self.stream or not self.samples or now - self.samples[-1][0] > 5:
            return ''
        recent = [(t, f) for t, f in self.samples if t >= now - 4]
        if len(recent) < 4 or recent[-1][0] - recent[0][0] < 1.5:
            return ''
        def sustained(key, predicate):
            return sum(predicate(f[key]) for _, f in recent) / len(recent) >= .8
        observations = []
        for key, predicate, label in (
            ('smile', lambda x: x > .55, '嘴角持续上扬，可能在微笑'),
            ('brow_down', lambda x: x > .55, '眉部持续收紧'),
            ('eye_closed', lambda x: x > .7, '近期有持续闭眼现象'),
            ('yaw', lambda x: abs(x) > 25, '头部持续朝向侧面'),
        ):
            if sustained(key, predicate):
                observations.append(label)
        return '；'.join(observations) or '单人面部在画面中，没有明显持续表情变化'
