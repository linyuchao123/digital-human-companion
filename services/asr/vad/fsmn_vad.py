from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VadSegment:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class FsmnVadConfig:
    model_name: str = "fsmn-vad"
    sample_rate_hz: int = 16000


class FsmnVad:
    def __init__(self, config: Optional[FsmnVadConfig] = None):
        self.config = config or FsmnVadConfig()
        from funasr import AutoModel

        self._model = AutoModel(model=self.config.model_name)

    def detect_segments(
        self,
        pcm16: np.ndarray,
        sample_rate_hz: int = 16000,
    ) -> List[VadSegment]:
        if sample_rate_hz != self.config.sample_rate_hz:
            raise ValueError(f"VAD expects {self.config.sample_rate_hz}Hz, got {sample_rate_hz}Hz")

        # 支持int16或float32输入，统一转换为float32
        if pcm16.dtype == np.int16:
            audio_float = pcm16.astype(np.float32) / 32768.0
        elif pcm16.dtype == np.float32:
            audio_float = pcm16
        else:
            raise ValueError("pcm16 must be int16 or float32")

        res = self._model.generate(input=audio_float, sampling_rate=sample_rate_hz)
        segments: List[VadSegment] = []
        for item in res:
            if isinstance(item, dict) and "value" in item:
                v = item["value"]
                if isinstance(v, list):
                    for seg in v:
                        if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                            segments.append(VadSegment(start_ms=int(seg[0]), end_ms=int(seg[1])))
        segments.sort(key=lambda s: (s.start_ms, s.end_ms))
        return segments

