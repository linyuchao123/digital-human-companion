#!/usr/bin/env python3
"""
数字人面部行为驱动模型 - 数据集与预处理
支持 NoXI + RECOLA 训练集
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TORCH_AVAILABLE = False
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    class Dataset:  # type: ignore[no-redef]
        pass

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# 25维特征顺序（官方固定，不可修改）
FEATURE_ORDER = [
    "AU1", "AU2", "AU4", "AU6", "AU7", "AU9", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU23", "AU24", "AU25", "AU26",  # 15 AU
    "valence", "arousal",                                              # 2 VA
    "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"  # 8 EXP
]

# 音频预处理配置
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "n_mels": 80,
    "hop_length": 640,    # 10ms帧移 @16kHz → 25fps对齐
    "win_length": 1600,   # 100ms窗
    "n_fft": 2048,
    "f_min": 50,
    "f_max": 7600,
}


def load_audio_mel(wav_path: str, cfg: dict = AUDIO_CONFIG) -> np.ndarray:
    """
    加载WAV并提取log-mel spectrogram
    返回: (T_mel, 80) float32
    """
    if not LIBROSA_AVAILABLE:
        # 降级: 返回随机特征（测试用）
        return np.random.rand(100, 80).astype(np.float32)

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if sr != cfg["sample_rate"]:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg["sample_rate"])

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        n_fft=cfg["n_fft"],
        fmin=cfg["f_min"],
        fmax=cfg["f_max"],
    )
    log_mel = np.log(mel + 1e-8).T  # (T_mel, 80)
    return log_mel.astype(np.float32)


def load_face_params(npy_path: str) -> np.ndarray:
    """
    加载3D面部参数
    输入: shape=(T, 1, 58) 或 (T, 58)
    返回: (T, 58) float32
    """
    data = np.load(npy_path)
    if data.ndim == 3:
        data = data[:, 0, :]  # (T, 1, 58) → (T, 58)
    return data.astype(np.float32)


def load_emotion_csv(csv_path: str) -> np.ndarray:
    """
    加载情绪标签CSV
    返回: (T, 25) float32，顺序为官方FEATURE_ORDER
    """
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return np.zeros((0, 25), dtype=np.float32)

    data = []
    for row in rows:
        frame = []
        for feat in FEATURE_ORDER:
            v = row.get(feat, "0")
            try:
                frame.append(float(v))
            except (ValueError, TypeError):
                frame.append(0.0)
        data.append(frame)

    return np.array(data, dtype=np.float32)


def align_sequences(
    mel: np.ndarray,           # (T_mel, 80)
    face: np.ndarray,          # (T_face, 58)
    emotion: np.ndarray,       # (T_emo, 25)
    fps: int = 25,
    audio_fps: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将mel帧对齐到25fps面部参数帧
    mel是~100fps(hop=640@16k)，需要降采样到25fps
    """
    # mel帧率: 16000 / 640 = 25 fps，已经对齐！
    # 取最短长度
    T = min(len(mel), len(face), len(emotion))
    T = min(T, 750)  # 验证集固定750帧上限

    mel_aligned = mel[:T]
    face_aligned = face[:T]
    emo_aligned = emotion[:T]

    return mel_aligned, face_aligned, emo_aligned


def split_into_chunks(
    mel: np.ndarray,
    face: np.ndarray,
    emotion: np.ndarray,
    chunk_frames: int = 75,   # 3秒@25fps
    overlap: int = 12,        # 0.5秒重叠
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """将长序列切分为训练片段"""
    chunks = []
    T = len(mel)
    step = chunk_frames - overlap

    for start in range(0, T - chunk_frames + 1, step):
        end = start + chunk_frames
        chunks.append((
            mel[start:end],
            face[start:end],
            emotion[start:end],
        ))
    return chunks


def scan_dataset(data_root: str) -> List[Dict]:
    """
    扫描训练集，返回所有对齐的(audio, face, emotion)三元组路径
    结构: data_root/Audio_files/NoXI/{session}/{role}/{idx}.wav
          data_root/3D_FV_files/NoXI/{session}/{role}/{idx}.npy
          data_root/Emotion/NoXI/{session}/P1或P2/{idx}.csv
    """
    data_root = Path(data_root)
    samples = []

    # 角色映射: Expert_video→P1, Novice_video→P2
    ROLE_MAP = {"Expert_video": "P1", "Novice_video": "P2"}

    for dataset in ["NoXI", "RECOLA"]:
        audio_base = data_root / "Audio_files" / dataset
        fv_base = data_root / "3D_FV_files" / dataset
        emo_base = data_root / "Emotion" / dataset

        if not audio_base.exists():
            continue

        for session_dir in sorted(audio_base.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name

            for role_dir in sorted(session_dir.iterdir()):
                if not role_dir.is_dir():
                    continue
                role = role_dir.name

                # 对应情绪文件夹
                if dataset == "NoXI":
                    emo_role = ROLE_MAP.get(role, role)
                else:
                    emo_role = role  # RECOLA用原始名

                for wav_file in sorted(role_dir.glob("*.wav")):
                    idx = wav_file.stem
                    fv_path = fv_base / session / role / f"{idx}.npy"
                    emo_path = emo_base / session / emo_role / f"{idx}.csv"

                    if fv_path.exists() and emo_path.exists():
                        samples.append({
                            "audio": str(wav_file),
                            "face": str(fv_path),
                            "emotion": str(emo_path),
                            "session": session,
                            "role": role,
                            "idx": idx,
                            "dataset": dataset,
                        })

    print(f"[Dataset] 扫描完成，找到 {len(samples)} 个有效三元组")
    return samples


class FaceReactionDataset(Dataset):
    """
    面部反应生成数据集

    输入: 说话人音频 log-mel (T, 80)
    输出: 监听者面部参数 (T, 58) + 情绪特征 (T, 25)
    """

    def __init__(
        self,
        samples: List[Dict],
        chunk_frames: int = 75,
        augment: bool = True,
        cache_mel: bool = False,
    ):
        self.chunk_frames = chunk_frames
        self.augment = augment
        self.cache_mel = cache_mel
        self._mel_cache: Dict[str, np.ndarray] = {}

        # 预处理：切分所有样本
        self.chunks = []
        for sample in samples:
            try:
                mel = self._get_mel(sample["audio"])
                face = load_face_params(sample["face"])
                emo = load_emotion_csv(sample["emotion"])
                mel, face, emo = align_sequences(mel, face, emo)
                for c_mel, c_face, c_emo in split_into_chunks(mel, face, emo, chunk_frames):
                    self.chunks.append((c_mel, c_face, c_emo))
            except Exception as e:
                print(f"[Dataset] 跳过样本 {sample['audio']}: {e}")

        print(f"[Dataset] 共 {len(self.chunks)} 个训练片段（chunk_frames={chunk_frames}）")

    def _get_mel(self, wav_path: str) -> np.ndarray:
        if self.cache_mel and wav_path in self._mel_cache:
            return self._mel_cache[wav_path]
        mel = load_audio_mel(wav_path)
        if self.cache_mel:
            self._mel_cache[wav_path] = mel
        return mel

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        mel, face, emo = self.chunks[idx]

        if self.augment:
            mel, face, emo = self._augment(mel, face, emo)

        if TORCH_AVAILABLE:
            return (
                torch.from_numpy(mel),    # (T, 80)
                torch.from_numpy(face),   # (T, 58)
                torch.from_numpy(emo),    # (T, 25)
            )
        return mel, face, emo

    def _augment(
        self,
        mel: np.ndarray,
        face: np.ndarray,
        emo: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据增强"""
        # 音量扰动（±6dB）
        if np.random.rand() < 0.3:
            gain_db = np.random.uniform(-6, 6)
            mel = mel + gain_db / 20.0

        # SpecAugment：频域mask
        if np.random.rand() < 0.2:
            f_mask = np.random.randint(0, 10)
            f_start = np.random.randint(0, 80 - f_mask)
            mel[:, f_start:f_start + f_mask] = mel.mean()

        # SpecAugment：时域mask
        if np.random.rand() < 0.2:
            t_mask = min(np.random.randint(0, 10), mel.shape[0] - 1)
            t_start = np.random.randint(0, mel.shape[0] - t_mask)
            mel[t_start:t_start + t_mask, :] = mel.mean()

        return mel, face, emo
