from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ParaformerConfig:
    model_name: str = "paraformer-zh"
    sample_rate_hz: int = 16000


class ParaformerZh:
    def __init__(self, config: Optional[ParaformerConfig] = None):
        self.config = config or ParaformerConfig()
        from funasr import AutoModel

        self._model = AutoModel(model=self.config.model_name)

    def transcribe(
        self,
        pcm16: np.ndarray,
        sample_rate_hz: int = 16000,
        hotwords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if sample_rate_hz != self.config.sample_rate_hz:
            raise ValueError(f"ASR expects {self.config.sample_rate_hz}Hz, got {sample_rate_hz}Hz")
        
        # 支持int16或float32输入，统一转换为float32
        if pcm16.dtype == np.int16:
            audio_float = pcm16.astype(np.float32) / 32768.0
        elif pcm16.dtype == np.float32:
            audio_float = pcm16
        else:
            raise ValueError("pcm16 must be int16 or float32")

        kwargs: Dict[str, Any] = {"sampling_rate": sample_rate_hz}
        if hotwords:
            kwargs["hotword"] = " ".join(hotwords)

        res = self._model.generate(input=audio_float, **kwargs)

        text = ""
        confidence = None
        words: List[Dict[str, Any]] = []
        if res and isinstance(res, list):
            item = res[0]
            if isinstance(item, dict):
                if "text" in item:
                    text = str(item["text"])
                if "confidence" in item:
                    try:
                        confidence = float(item["confidence"])
                    except Exception:
                        confidence = None
                if "timestamp" in item and isinstance(item["timestamp"], list):
                    for w in item["timestamp"]:
                        if isinstance(w, (list, tuple)) and len(w) >= 3:
                            words.append(
                                {
                                    "w": str(w[0]),
                                    "start_ms": int(float(w[1]) * 10),
                                    "end_ms": int(float(w[2]) * 10),
                                }
                            )

        return {"text": text, "confidence": confidence, "words": words}

