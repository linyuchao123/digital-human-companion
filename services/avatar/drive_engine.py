#!/usr/bin/env python3
"""
数字人面部行为驱动引擎 (DriveEngine)
接收 LLMToDriver 协议输入，驱动2个数字人形象输出逐帧面部参数

数据流:
  LLMToDriver (render.avatar / render.voice) + TTS音频
       ↓
  FaceReactionModel (CNN+LSTM+MLP)
       ↓
  58D face_params + 25D emotion_25 @ 25fps
       ↓
  VRMMapper / Live2DMapper
       ↓
  avatar_drive_stream 帧序列 → 渲染引擎
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

from packages.common.protocols import LLMToDriver
from services.avatar.face_drive.inference.model import FaceReactionModel, ModelConfig
from services.avatar.face_drive.mapping.avatar_mapper import VRMMapper, Live2DMapper, get_mapper
from services.avatar.face_drive.training.dataset import load_audio_mel


@dataclass
class DriveConfig:
    """驱动引擎配置"""
    model_path: Optional[str] = None        # 训练好的模型权重路径
    frame_rate_fps: int = 25                # 输出帧率
    input_sample_rate: int = 16000          # 音频采样率
    smoothing_alpha: float = 0.7            # IIR平滑系数
    device: str = "auto"                    # 推理设备

    # 2个数字人形象配置
    avatar_1: Dict[str, str] = None  # {"id": "avatar_xiao_an", "type": "vrm"}
    avatar_2: Dict[str, str] = None  # {"id": "avatar_xiao_ming", "type": "live2d"}

    def __post_init__(self):
        if self.avatar_1 is None:
            self.avatar_1 = {"id": "avatar_xiao_an", "type": "vrm"}
        if self.avatar_2 is None:
            self.avatar_2 = {"id": "avatar_xiao_ming", "type": "live2d"}

        if self.device == "auto":
            self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"


# Idle动画参数（无输入时的自然状态）
IDLE_BLINK_PERIOD_S = 4.0    # 眨眼周期（秒）
IDLE_BREATH_PERIOD_S = 3.5   # 呼吸周期（秒）
IDLE_HEAD_PERIOD_S = 8.0     # 微小头动周期（秒）


class DriveEngine:
    """
    数字人面部行为驱动引擎

    支持:
    - 2个数字人形象（VRM + Live2D）
    - 音频驱动口型同步
    - 情绪条件驱动表情
    - IIR平滑防抖
    - Idle自然动画（眨眼/呼吸/头动）
    """

    def __init__(self, config: Optional[DriveConfig] = None):
        self.config = config or DriveConfig()

        # 初始化模型
        self._model = self._load_model()

        # 初始化映射器（2个数字人）
        self._mappers = {
            self.config.avatar_1["id"]: get_mapper(self.config.avatar_1["type"]),
            self.config.avatar_2["id"]: get_mapper(self.config.avatar_2["type"]),
        }

        # 平滑状态
        self._prev_params = np.zeros(58, dtype=np.float32)
        self._prev_emo = np.zeros(25, dtype=np.float32)
        self._prev_emo[17] = 1.0  # 默认Neutral

        # Idle状态
        self._t_start = time.time()

        print(f"[DriveEngine] 初始化完成，设备={self.config.device}")
        print(f"[DriveEngine] 数字人1: {self.config.avatar_1}")
        print(f"[DriveEngine] 数字人2: {self.config.avatar_2}")

    def _load_model(self) -> FaceReactionModel:
        """加载面部行为驱动模型"""
        cfg = ModelConfig()
        model = FaceReactionModel(cfg)

        if self.config.model_path and Path(self.config.model_path).exists() and TORCH_AVAILABLE:
            try:
                ckpt = torch.load(self.config.model_path, map_location=self.config.device)
                state = ckpt.get("model", ckpt)
                model.load_state_dict(state)
                print(f"[DriveEngine] 已加载模型: {self.config.model_path}")
            except Exception as e:
                print(f"[DriveEngine] 模型加载失败: {e}，使用随机初始化")
        else:
            print("[DriveEngine] 模型文件不存在，使用随机初始化（训练后替换）")

        # 始终将模型移到目标设备
        if TORCH_AVAILABLE and hasattr(model, 'parameters'):
            model = model.to(self.config.device)
            model.eval()

        return model

    def drive(
        self,
        llm_output: LLMToDriver,
        audio_chunk: Optional[np.ndarray] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        驱动数字人面部行为

        Args:
            llm_output: LLMToDriver协议对象（包含情绪意图和渲染指令）
            audio_chunk: PCM音频数据 (T,) float32 @16kHz，与TTS输出同步
            audio_path: 或者提供音频文件路径

        Returns:
            avatar_drive_stream 格式的帧序列字典
        """
        start_t = time.time()

        # 1. 提取情绪条件
        emotion_intent, emotion_intensity, emotion_cond = self._extract_emotion_condition(llm_output)

        # 2. 提取音频特征
        mel = self._get_mel(audio_chunk, audio_path)

        # 3. 模型推理
        face_params_seq, emotion_25_seq = self._infer(mel, emotion_cond)

        # 4. IIR平滑
        face_params_seq = self._smooth_sequence(face_params_seq)

        # 5. 生成帧序列
        frames = self._build_frames(face_params_seq, emotion_25_seq, emotion_intent, emotion_intensity)

        # 6. 生成两个数字人的驱动参数
        avatar_outputs = {}
        for avatar_id, mapper in self._mappers.items():
            avatar_frames = []
            for i, frame in enumerate(frames):
                fp = np.array(frame["face_params_58"], dtype=np.float32)
                ep = np.array(frame["emotion_25"], dtype=np.float32)
                mapped = mapper.map(fp, ep, emotion_intent, emotion_intensity)
                avatar_frames.append({**frame, "avatar_params": mapped})
            avatar_outputs[avatar_id] = avatar_frames

        elapsed_ms = int((time.time() - start_t) * 1000)

        return {
            "protocol": "avatar_drive_stream",
            "version": "1.0",
            "trace_id": llm_output.trace_id,
            "session_id": llm_output.session_id,
            "turn_id": llm_output.turn_id,
            "frame_rate_fps": self.config.frame_rate_fps,
            "frames": frames,                    # 原始帧序列
            "avatar_outputs": avatar_outputs,    # 各数字人映射后帧序列
            "x_ext": {
                "elapsed_ms": elapsed_ms,
                "emotion_intent": emotion_intent,
                "emotion_intensity": emotion_intensity,
                "n_frames": len(frames),
            }
        }

    def drive_idle(self, duration_s: float = 1.0) -> Dict[str, Any]:
        """
        生成 Idle 待机动画（眨眼/呼吸/微小头动）
        当没有对话输入时持续播放
        """
        n_frames = int(duration_s * self.config.frame_rate_fps)
        frames = []
        t_now = time.time() - self._t_start

        for i in range(n_frames):
            t = t_now + i / self.config.frame_rate_fps
            face_params = np.zeros(58, dtype=np.float32)

            # 眨眼（周期性）
            blink_phase = (t % IDLE_BLINK_PERIOD_S) / IDLE_BLINK_PERIOD_S
            if 0.9 < blink_phase < 1.0:
                blink_val = float(np.sin((blink_phase - 0.9) / 0.1 * np.pi))
                face_params[0] = blink_val   # eyeBlinkLeft
                face_params[1] = blink_val   # eyeBlinkRight

            # 呼吸（胸部轻微起伏，通过头部Y轴微动模拟）
            breath_phase = t / IDLE_BREATH_PERIOD_S * 2 * np.pi
            breath_val = float(np.sin(breath_phase) * 0.02)
            if len(face_params) > 56:
                face_params[56] = breath_val  # headY

            # 微小头动
            head_phase = t / IDLE_HEAD_PERIOD_S * 2 * np.pi
            if len(face_params) > 53:
                face_params[53] = float(np.sin(head_phase) * 0.05)  # headYaw

            # Neutral情绪
            emo = np.zeros(25, dtype=np.float32)
            emo[17] = 1.0  # Neutral

            frames.append({
                "t_ms": int(i * 1000 / self.config.frame_rate_fps),
                "face_params_58": face_params.tolist(),
                "emotion_25": emo.tolist(),
                "aux": {
                    "au_15": {f"AU{j}": 0.0 for j in [1,2,4,6,7,9,10,12,14,15,17,23,24,25,26]},
                    "va": {"valence": 0.0, "arousal": 0.0},
                    "expression_8": {"Neutral": 1.0},
                }
            })

        # 映射到两个数字人
        avatar_outputs = {}
        for avatar_id, mapper in self._mappers.items():
            avatar_frames = []
            for frame in frames:
                fp = np.array(frame["face_params_58"])
                ep = np.array(frame["emotion_25"])
                mapped = mapper.map(fp, ep)
                avatar_frames.append({**frame, "avatar_params": mapped})
            avatar_outputs[avatar_id] = avatar_frames

        return {
            "protocol": "avatar_drive_stream",
            "version": "1.0",
            "mode": "idle",
            "frame_rate_fps": self.config.frame_rate_fps,
            "frames": frames,
            "avatar_outputs": avatar_outputs,
        }

    def _extract_emotion_condition(
        self,
        llm_output: LLMToDriver,
    ):
        """从LLMToDriver提取情绪条件"""
        emotion_intent = None
        emotion_intensity = 0.5

        if llm_output.render:
            if llm_output.render.voice:
                emotion_intent = llm_output.render.voice.emotion
            if llm_output.render.avatar and llm_output.render.avatar.expression:
                expr = llm_output.render.avatar.expression
                if hasattr(expr, "name"):
                    emotion_intent = emotion_intent or expr.name
                if hasattr(expr, "intensity"):
                    emotion_intensity = expr.intensity or 0.5

        # 构建10维情绪条件向量 (2 VA + 8 EXP)
        EMOTION_TO_IDX = {
            "Neutral": 0, "Happy": 1, "happy": 1,
            "Sad": 2, "sad": 2, "gentle": 2,
            "Surprise": 3, "Fear": 4,
            "Disgust": 5, "Anger": 6, "concerned": 6,
            "Contempt": 7,
        }
        exp_vec = np.zeros(8, dtype=np.float32)
        if emotion_intent and emotion_intent in EMOTION_TO_IDX:
            idx = EMOTION_TO_IDX[emotion_intent]
            exp_vec[idx] = emotion_intensity
        else:
            exp_vec[0] = 1.0  # Neutral

        va_vec = np.array([
            0.3 if emotion_intent == "happy" else (-0.3 if emotion_intent in ("sad", "gentle") else 0.0),
            0.2 if emotion_intent == "surprise" else 0.0,
        ], dtype=np.float32)

        emotion_cond = np.concatenate([va_vec, exp_vec])  # (10,)

        return emotion_intent, emotion_intensity, emotion_cond

    def _get_mel(
        self,
        audio_chunk: Optional[np.ndarray],
        audio_path: Optional[str],
        default_frames: int = 75,
    ) -> np.ndarray:
        """提取log-mel特征"""
        if audio_chunk is not None and len(audio_chunk) > 0:
            try:
                import librosa
                import io, soundfile as sf
                mel = load_audio_mel.__wrapped__(audio_chunk) if hasattr(load_audio_mel, '__wrapped__') else None
                if mel is None:
                    # 直接从PCM提取
                    mel = librosa.feature.melspectrogram(
                        y=audio_chunk.astype(np.float32),
                        sr=self.config.input_sample_rate,
                        n_mels=80, hop_length=640,
                    )
                    mel = np.log(mel + 1e-8).T.astype(np.float32)
                return mel
            except Exception:
                pass

        if audio_path and Path(audio_path).exists():
            try:
                return load_audio_mel(audio_path)
            except Exception:
                pass

        # 降级：静音帧
        return np.zeros((default_frames, 80), dtype=np.float32)

    def _infer(
        self,
        mel: np.ndarray,
        emotion_cond: np.ndarray,
    ):
        """模型推理"""
        if TORCH_AVAILABLE and hasattr(self._model, 'parameters'):
            mel_t = torch.from_numpy(mel).unsqueeze(0).to(self.config.device)
            cond_t = torch.from_numpy(emotion_cond).unsqueeze(0).to(self.config.device)
            with torch.no_grad():
                outputs = self._model(mel_t, cond_t)
            face_seq = outputs["face_58"].squeeze(0).cpu().numpy()
            emo_seq = outputs["emotion_25"].squeeze(0).cpu().numpy()
        else:
            outputs = self._model(mel[np.newaxis], emotion_cond[np.newaxis])
            face_seq = outputs["face_58"][0]
            emo_seq = outputs["emotion_25"][0]

        return face_seq, emo_seq

    def _smooth_sequence(self, params: np.ndarray) -> np.ndarray:
        """IIR平滑滤波防抖"""
        alpha = self.config.smoothing_alpha
        smoothed = np.zeros_like(params)
        prev = self._prev_params

        for i in range(len(params)):
            smoothed[i] = alpha * prev + (1 - alpha) * params[i]
            prev = smoothed[i]

        self._prev_params = smoothed[-1]
        return smoothed

    def _build_frames(
        self,
        face_seq: np.ndarray,
        emo_seq: np.ndarray,
        emotion_intent: Optional[str],
        emotion_intensity: float,
    ) -> List[Dict]:
        """构建帧序列"""
        AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU9", "AU10",
                    "AU12", "AU14", "AU15", "AU17", "AU23", "AU24", "AU25", "AU26"]
        EXP_NAMES = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

        frames = []
        for i in range(len(face_seq)):
            au_15 = {AU_NAMES[j]: float(emo_seq[i, j]) for j in range(15)}
            va = {"valence": float(emo_seq[i, 15]), "arousal": float(emo_seq[i, 16])}
            expression_8 = {EXP_NAMES[j]: float(emo_seq[i, 17 + j]) for j in range(8)}

            frame = {
                "t_ms": int(i * 1000 / self.config.frame_rate_fps),
                "face_params_58": face_seq[i].tolist(),
                "emotion_25": emo_seq[i].tolist(),
                "aux": {
                    "au_15": au_15,
                    "va": va,
                    "expression_8": expression_8,
                }
            }
            frames.append(frame)

        return frames
