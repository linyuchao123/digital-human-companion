#!/usr/bin/env python3
"""
听觉模块整合管道
整合VAD、ASR和声学特征提取
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from services.asr.vad.fsmn_vad import FsmnVad, FsmnVadConfig, VadSegment
from services.asr.inference.paraformer_zh import ParaformerZh, ParaformerConfig
from services.asr.inference.acoustic_features import (
    AcousticFeaturesExtractor,
    AcousticFeaturesConfig,
)


@dataclass(frozen=True)
class AudioPipelineConfig:
    """听觉管道配置"""
    sample_rate_hz: int = 16000
    enable_vad: bool = True
    enable_asr: bool = True
    enable_acoustic_features: bool = True
    hotwords: Optional[List[str]] = None


@dataclass(frozen=True)
class AudioSegmentResult:
    """音频段处理结果"""
    segment: VadSegment
    text: str
    confidence: Optional[float]
    words: List[Dict[str, Any]]
    acoustic_features: Dict[str, Any]


class AudioPipeline:
    """
    听觉处理管道
    
    整合VAD + ASR + 声学特征提取
    输出符合PerceptionToLLM协议的格式
    """
    
    def __init__(self, config: Optional[AudioPipelineConfig] = None):
        self.config = config or AudioPipelineConfig()
        
        # 初始化VAD
        if self.config.enable_vad:
            self._vad = FsmnVad(FsmnVadConfig(sample_rate_hz=self.config.sample_rate_hz))
        else:
            self._vad = None
        
        # 初始化ASR
        if self.config.enable_asr:
            self._asr = ParaformerZh(ParaformerConfig(sample_rate_hz=self.config.sample_rate_hz))
        else:
            self._asr = None
        
        # 初始化声学特征提取
        if self.config.enable_acoustic_features:
            self._feature_extractor = AcousticFeaturesExtractor(
                AcousticFeaturesConfig(sample_rate_hz=self.config.sample_rate_hz)
            )
        else:
            self._feature_extractor = None
    
    def process_audio(
        self,
        pcm16: np.ndarray,
        sample_rate_hz: int = 16000,
    ) -> Dict[str, Any]:
        """
        处理音频数据
        
        Args:
            pcm16: 16-bit PCM音频数据
            sample_rate_hz: 采样率
            
        Returns:
            符合PerceptionToLLM协议的输出
        """
        if sample_rate_hz != self.config.sample_rate_hz:
            raise ValueError(f"期望采样率 {self.config.sample_rate_hz}Hz, 得到 {sample_rate_hz}Hz")
        
        if pcm16.dtype != np.int16:
            raise ValueError("音频数据必须是int16格式")
        
        # VAD检测语音段
        if self._vad:
            segments = self._vad.detect_segments(pcm16, sample_rate_hz)
        else:
            # 不使用VAD，将整个音频作为一个段
            duration_ms = len(pcm16) * 1000 // sample_rate_hz
            segments = [VadSegment(start_ms=0, end_ms=duration_ms)]
        
        # 处理每个语音段
        segment_results: List[AudioSegmentResult] = []
        
        for seg in segments:
            # 提取音频段
            start_sample = int(seg.start_ms * sample_rate_hz / 1000)
            end_sample = int(seg.end_ms * sample_rate_hz / 1000)
            segment_audio = pcm16[start_sample:end_sample]
            
            if len(segment_audio) == 0:
                continue
            
            # ASR识别
            if self._asr:
                asr_result = self._asr.transcribe(
                    segment_audio,
                    sample_rate_hz,
                    hotwords=self.config.hotwords
                )
            else:
                asr_result = {"text": "", "confidence": None, "words": []}
            
            # 提取声学特征
            if self._feature_extractor:
                features = self._feature_extractor.extract(segment_audio, sample_rate_hz)
                aggregated = features.get("aggregated", {})
            else:
                aggregated = {}
            
            segment_results.append(AudioSegmentResult(
                segment=seg,
                text=asr_result["text"],
                confidence=asr_result["confidence"],
                words=asr_result["words"],
                acoustic_features=aggregated
            ))
        
        # 构建输出
        return self._build_output(segment_results, pcm16, sample_rate_hz)
    
    def _build_output(
        self,
        segment_results: List[AudioSegmentResult],
        pcm16: np.ndarray,
        sample_rate_hz: int
    ) -> Dict[str, Any]:
        """构建符合协议的输出"""
        
        # 合并所有段的文本
        full_text = " ".join([r.text for r in segment_results if r.text])
        
        # 取第一个段的置信度（如果有）
        confidence = None
        for r in segment_results:
            if r.confidence is not None:
                confidence = r.confidence
                break
        
        # 合并所有词的时间戳
        all_words = []
        for r in segment_results:
            for w in r.words:
                all_words.append(w)
        
        # 构建VAD信息（使用第一个段或整个音频）
        if segment_results:
            first_seg = segment_results[0].segment
            last_seg = segment_results[-1].segment
            vad_info = {
                "speech_start_ms": first_seg.start_ms,
                "speech_end_ms": last_seg.end_ms
            }
        else:
            duration_ms = len(pcm16) * 1000 // sample_rate_hz
            vad_info = {
                "speech_start_ms": 0,
                "speech_end_ms": duration_ms
            }
        
        # 构建声学特征（使用第一个段的特征）
        acoustic_features = {}
        for r in segment_results:
            if r.acoustic_features:
                acoustic_features = r.acoustic_features
                break
        
        # 构建符合PerceptionToLLM协议的输出
        output = {
            "turn": {
                "utterance_id": f"utt_{int(np.random.randint(0, 1000000))}",
                "input_mode": "voice",
                "barge_in": False,
                "vad": vad_info
            },
            "asr": {
                "text": full_text,
                "language": "zh-CN",
                "confidence": confidence,
                "words": all_words,
                "wer_hint": {
                    "domain": "psychology",
                    "hotwords": self.config.hotwords or []
                }
            },
            "emotion": {
                "signals": {
                    "voice": {
                        "enabled": self.config.enable_acoustic_features,
                        "x_features": acoustic_features
                    }
                }
            },
            "segments": [
                {
                    "start_ms": r.segment.start_ms,
                    "end_ms": r.segment.end_ms,
                    "text": r.text,
                    "confidence": r.confidence
                }
                for r in segment_results
            ]
        }
        
        return output
    
    def process_file(self, audio_path: str) -> Dict[str, Any]:
        """
        处理音频文件
        
        Args:
            audio_path: 音频文件路径（支持wav格式）
            
        Returns:
            处理结果
        """
        # 读取wav文件
        with wave.open(audio_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # 读取音频数据
            raw_data = wf.readframes(n_frames)
            pcm16 = np.frombuffer(raw_data, dtype=np.int16)
            
            # 如果是立体声，转换为单声道
            if n_channels == 2:
                pcm16 = pcm16.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            # 如果采样率不匹配，需要重采样（简化处理，实际应该使用librosa.resample）
            if sample_rate != self.config.sample_rate_hz:
                # 简单的重采样
                ratio = self.config.sample_rate_hz / sample_rate
                new_length = int(len(pcm16) * ratio)
                indices = np.linspace(0, len(pcm16) - 1, new_length)
                pcm16 = np.interp(indices, np.arange(len(pcm16)), pcm16).astype(np.int16)
        
        return self.process_audio(pcm16, self.config.sample_rate_hz)


def load_audio_file(audio_path: str, target_sample_rate: int = 16000) -> np.ndarray:
    """
    加载音频文件并转换为16kHz单声道PCM16
    
    Args:
        audio_path: 音频文件路径
        target_sample_rate: 目标采样率
        
    Returns:
        PCM16音频数据
    """
    import librosa
    
    # 使用librosa加载音频
    audio, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
    
    # 转换为int16
    audio = np.clip(audio * 32767, -32768, 32767)
    pcm16 = audio.astype(np.int16)
    
    return pcm16
