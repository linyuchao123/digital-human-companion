from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from packages.common.constants import (
    DATASET_AU_NAMES,
    DATASET_EXPRESSION_8_NAMES,
    DATASET_VA_NAMES,
)
from packages.common.protocols import PerceptionToLLM


TORCH_AVAILABLE = False
torch = None
AutoTokenizer = None
AutoModel = None

try:
    import torch as _torch
    from transformers import AutoTokenizer as _AutoTokenizer
    from transformers import AutoModel as _AutoModel
    TORCH_AVAILABLE = True
    torch = _torch
    AutoTokenizer = _AutoTokenizer
    AutoModel = _AutoModel
except Exception:
    pass


@dataclass(frozen=True)
class FusionConfig:
    window_ms: int = 1000
    hop_ms: int = 200
    text_embedding_model: str = "bert-base-chinese"
    fused_embedding_dim: int = 256
    smoothing_alpha: float = 0.7


class FusionEngine:
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self._text_tokenizer = None
        self._text_model = None
        
        if TORCH_AVAILABLE:
            try:
                # 尝试加载BERT模型，设置本地优先
                import os
                os.environ['HF_HUB_OFFLINE'] = '1'  # 优先使用本地缓存
                
                self._text_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.text_embedding_model,
                    local_files_only=True  # 只使用本地缓存
                )
                self._text_model = AutoModel.from_pretrained(
                    self.config.text_embedding_model,
                    local_files_only=True
                )
                print(f"[FusionEngine] BERT模型加载成功")
            except Exception as e:
                print(f"[FusionEngine] BERT模型加载失败，使用离线模式: {e}")
                self._text_tokenizer = None
                self._text_model = None
        self._smoothing_alpha = self.config.smoothing_alpha
        self._previous_va = {"valence": 0.0, "arousal": 0.0}
        self._previous_emotion_probs = {e: 0.0 for e in DATASET_EXPRESSION_8_NAMES}

    def fuse(self, perception: PerceptionToLLM) -> Dict[str, any]:
        text_features = self._extract_text_features(perception.asr.text)
        audio_features = self._extract_audio_features(perception)
        vision_features = self._extract_vision_features(perception)
        aligned_features = self._align_features(text_features, audio_features, vision_features, perception)
        fused_embedding, fusion_details = self._fuse_features(aligned_features)
        emotion_probs, va = self._estimate_emotion_and_va(fusion_details)
        psych_state = self._estimate_psych_state(va, emotion_probs, perception)

        output = {
            "emotion": {
                "primary": max(emotion_probs, key=emotion_probs.get),
                "valence": va["valence"],
                "arousal": va["arousal"],
                "confidence": 0.8,
                "signals": {
                    "voice": {
                        "enabled": bool(audio_features),
                        "x_features": audio_features.get("aggregated", {}) if audio_features else {}
                    },
                    "text": {
                        "enabled": bool(text_features),
                        "x_features": {}
                    },
                    "vision": {
                        "enabled": bool(vision_features),
                        "x_features": {}
                    }
                }
            },
            "x_ext": {
                "fusion": {
                    "version": "fusion_v1",
                    "window_ms": self.config.window_ms,
                    "hop_ms": self.config.hop_ms,
                    "modalities": {
                        "text": bool(text_features),
                        "voice": bool(audio_features),
                        "vision": bool(vision_features)
                    },
                    "fused_embedding": {
                        "dim": self.config.fused_embedding_dim,
                        "values": fused_embedding.tolist()
                    },
                    "emotion_probs": emotion_probs,
                    "va": {
                        "valence": va["valence"],
                        "arousal": va["arousal"],
                        "confidence": 0.8
                    },
                    "psych_state": psych_state,
                    "alignment": {
                        "time_base": "epoch_ms",
                        "vad": {
                            "speech_start_ms": perception.turn.vad.speech_start_ms if perception.turn.vad else None,
                            "speech_end_ms": perception.turn.vad.speech_end_ms if perception.turn.vad else None
                        },
                        "vision_frames_used": len(vision_features.get("au_15", [])) if vision_features else 0,
                        "audio_frames_used": audio_features.get("energy", []).__len__() if audio_features else 0
                    }
                }
            }
        }

        return output

    def _extract_text_features(self, text: str) -> Optional[Dict[str, any]]:
        if not text:
            return None

        if TORCH_AVAILABLE and self._text_tokenizer and self._text_model:
            inputs = self._text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self._text_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return {
                "embedding": embedding.tolist(),
                "text": text
            }
        else:
            embedding = np.random.rand(768).astype(np.float32)
            return {
                "embedding": embedding.tolist(),
                "text": text
            }

    def _extract_audio_features(self, perception: PerceptionToLLM) -> Optional[Dict[str, any]]:
        if not perception.emotion.signals.voice.enabled:
            return None

        return perception.emotion.signals.voice.x_features

    def _extract_vision_features(self, perception: PerceptionToLLM) -> Optional[Dict[str, any]]:
        if not perception.vision or not perception.vision.enabled:
            return None

        # VisionAU是Pydantic模型，使用属性访问
        au_dict = {}
        for au in perception.vision.features.face.au_15:
            if hasattr(au, 'name') and hasattr(au, 'intensity'):
                au_dict[au.name] = au.intensity
            elif isinstance(au, dict):
                au_dict[au["name"]] = au["intensity"]
        
        features = {
            "au_15": au_dict,
            "va": perception.vision.summary.va.dict() if perception.vision.summary.va else None,
            "expression_8": perception.vision.summary.expression_8.probs if perception.vision.summary.expression_8 else None
        }

        return features

    def _align_features(self, text_features: Optional[Dict[str, any]],
                       audio_features: Optional[Dict[str, any]],
                       vision_features: Optional[Dict[str, any]],
                       perception: PerceptionToLLM) -> Dict[str, any]:
        aligned = {
            "text": text_features,
            "audio": audio_features,
            "vision": vision_features
        }

        return aligned

    def _fuse_features(self, aligned_features: Dict[str, any]) -> Tuple[np.ndarray, Dict[str, any]]:
        features = []

        if aligned_features.get("text"):
            text_embedding = np.array(aligned_features["text"]["embedding"])
            features.append(text_embedding)

        if aligned_features.get("audio"):
            audio_aggregated = aligned_features["audio"].get("aggregated", {})
            mfcc_mean = np.array(audio_aggregated.get("mfcc_20_mean", [0.0]*20))
            mfcc_std = np.array(audio_aggregated.get("mfcc_20_std", [0.0]*20))
            pitch_mean = audio_aggregated.get("pitch_mean", 0.0)
            pitch_std = audio_aggregated.get("pitch_std", 0.0)
            energy_mean = audio_aggregated.get("energy_mean", 0.0)
            energy_std = audio_aggregated.get("energy_std", 0.0)
            audio_features = np.concatenate([mfcc_mean, mfcc_std, [pitch_mean, pitch_std, energy_mean, energy_std]])
            features.append(audio_features)

        if aligned_features.get("vision"):
            vision = aligned_features["vision"]
            au_features = np.array([vision["au_15"].get(au, 0.0) for au in DATASET_AU_NAMES])
            features.append(au_features)
            if vision.get("va"):
                va_features = np.array([vision["va"]["valence"], vision["va"]["arousal"]])
                features.append(va_features)
            if vision.get("expression_8"):
                expr_features = np.array([vision["expression_8"].get(expr, 0.0) for expr in DATASET_EXPRESSION_8_NAMES])
                features.append(expr_features)

        if not features:
            fused = np.zeros(self.config.fused_embedding_dim)
        else:
            concatenated = np.concatenate(features)
            if len(concatenated) > self.config.fused_embedding_dim:
                fused = concatenated[:self.config.fused_embedding_dim]
            else:
                fused = np.pad(concatenated, (0, self.config.fused_embedding_dim - len(concatenated)), 'constant')

        return fused, aligned_features

    def _estimate_emotion_and_va(self, fusion_details: Dict[str, any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        emotion_probs = {e: 0.0 for e in DATASET_EXPRESSION_8_NAMES}
        va = {"valence": 0.0, "arousal": 0.0}

        if fusion_details.get("vision"):
            vision = fusion_details["vision"]
            if vision.get("expression_8"):
                emotion_probs = vision["expression_8"]
            if vision.get("va"):
                va = {"valence": vision["va"]["valence"], "arousal": vision["va"]["arousal"]}

        emotion_probs = self._smooth_emotion_probs(emotion_probs)
        va = self._smooth_va(va)

        return emotion_probs, va

    def _estimate_psych_state(self, va: Dict[str, float], emotion_probs: Dict[str, float], perception: PerceptionToLLM) -> Dict[str, any]:
        valence = va["valence"]
        arousal = va["arousal"]

        phq9_score = max(0, min(27, (1 - valence) * 15 + arousal * 5))
        gad7_score = max(0, min(21, (1 - valence) * 10 + arousal * 8))

        risk_level = "low"
        if phq9_score >= 15 or gad7_score >= 14:
            risk_level = "high"
        elif phq9_score >= 10 or gad7_score >= 8:
            risk_level = "medium"

        return {
            "phq9_score_est": int(phq9_score),
            "gad7_score_est": int(gad7_score),
            "risk_level": risk_level,
            "confidence": 0.55
        }

    def _smooth_emotion_probs(self, current: Dict[str, float]) -> Dict[str, float]:
        smoothed = {}
        for e in DATASET_EXPRESSION_8_NAMES:
            smoothed[e] = self._smoothing_alpha * self._previous_emotion_probs[e] + (1 - self._smoothing_alpha) * current.get(e, 0.0)
        self._previous_emotion_probs = smoothed
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}
        return smoothed

    def _smooth_va(self, current: Dict[str, float]) -> Dict[str, float]:
        smoothed = {}
        for k in ["valence", "arousal"]:
            smoothed[k] = self._smoothing_alpha * self._previous_va[k] + (1 - self._smoothing_alpha) * current.get(k, 0.0)
        self._previous_va = smoothed
        return smoothed
