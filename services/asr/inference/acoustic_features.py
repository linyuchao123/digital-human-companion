from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa


@dataclass(frozen=True)
class AcousticFeaturesConfig:
    sample_rate_hz: int = 16000
    n_mfcc: int = 20
    frame_length_ms: int = 20
    frame_shift_ms: int = 10


class AcousticFeaturesExtractor:
    def __init__(self, config: Optional[AcousticFeaturesConfig] = None):
        self.config = config or AcousticFeaturesConfig()
        self.frame_length = int(self.config.sample_rate_hz * self.config.frame_length_ms / 1000)
        self.frame_shift = int(self.config.sample_rate_hz * self.config.frame_shift_ms / 1000)

    def extract(self, pcm16: np.ndarray, sample_rate_hz: int = 16000) -> Dict[str, any]:
        if sample_rate_hz != self.config.sample_rate_hz:
            raise ValueError(f"Acoustic features extractor expects {self.config.sample_rate_hz}Hz, got {sample_rate_hz}Hz")
        if pcm16.dtype != np.int16:
            raise ValueError("pcm16 must be int16")

        # Convert to float32 and normalize
        audio = pcm16.astype(np.float32) / 32768.0

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate_hz,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.frame_shift
        )

        # Extract pitch
        pitch, _ = librosa.piptrack(
            y=audio,
            sr=sample_rate_hz,
            n_fft=self.frame_length,
            hop_length=self.frame_shift
        )
        pitch = np.max(pitch, axis=0)
        pitch[pitch == 0] = np.nan

        # Extract energy
        energy = np.array([
            np.sum(np.abs(audio[i:i+self.frame_length])**2)
            for i in range(0, len(audio) - self.frame_length, self.frame_shift)
        ])
        if len(energy) < len(mfcc[0]):
            energy = np.pad(energy, (0, len(mfcc[0]) - len(energy)), 'constant')

        # Aggregate features
        aggregated = self._aggregate_features(mfcc, pitch, energy)

        return {
            "mfcc": mfcc.T.tolist(),
            "pitch": pitch.tolist(),
            "energy": energy.tolist(),
            "aggregated": aggregated
        }

    def _aggregate_features(self, mfcc: np.ndarray, pitch: np.ndarray, energy: np.ndarray) -> Dict[str, any]:
        # MFCC aggregation
        mfcc_mean = np.nanmean(mfcc, axis=1).tolist()
        mfcc_std = np.nanstd(mfcc, axis=1).tolist()

        # Pitch aggregation
        pitch_mean = np.nanmean(pitch) if not np.all(np.isnan(pitch)) else 0.0
        pitch_std = np.nanstd(pitch) if not np.all(np.isnan(pitch)) else 0.0

        # Energy aggregation
        energy_mean = np.mean(energy) if len(energy) > 0 else 0.0
        energy_std = np.std(energy) if len(energy) > 0 else 0.0

        return {
            "mfcc_20_mean": mfcc_mean,
            "mfcc_20_std": mfcc_std,
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "energy_mean": float(energy_mean),
            "energy_std": float(energy_std),
            "sample_rate_hz": self.config.sample_rate_hz,
            "feature_window_ms": self.config.frame_length_ms
        }
