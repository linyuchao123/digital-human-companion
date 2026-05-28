from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# RECOLA 角色映射: 原始ID → P1/P2
RECOLA_ROLE_MAP = {
    "P25": "P1", "P26": "P2",
    "P41": "P1", "P42": "P2",
    "P45": "P1", "P46": "P2",
}


# ============================================================================
# 原有工具函数 (保留兼容)
# ============================================================================

def collect_samples(root_dir: str | Path) -> List[Dict[str, str]]:
    root = Path(root_dir)
    audio_root = root / "Audio_files"
    emotion_root = root / "emotion"
    target_root = root / "3D_FV_files"
    samples: List[Dict[str, str]] = []
    for session_dir in sorted(audio_root.iterdir()):
        if not session_dir.is_dir():
            continue
        expert_dir = session_dir / "Expert_video"
        if not expert_dir.is_dir():
            continue
        for audio_file in sorted(expert_dir.glob("*.npy")):
            if audio_file.name.startswith("._"):
                continue
            stem = audio_file.stem
            emotion_file = emotion_root / session_dir.name / f"{stem}.csv"
            target_file = target_root / session_dir.name / f"{stem}.npy"
            if emotion_file.exists() and target_file.exists():
                samples.append({
                    "audio": str(audio_file),
                    "emotion": str(emotion_file),
                    "target": str(target_file),
                    "session": session_dir.name,
                    "segment": stem,
                })
    return samples


def load_sample_arrays(sample: Dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio = np.load(sample["audio"]).astype(np.float32)
    emotion = np.loadtxt(sample["emotion"], delimiter=",", skiprows=1, dtype=np.float32)
    target = np.load(sample["target"]).astype(np.float32)
    if target.ndim == 3 and target.shape[1] == 1:
        target = np.squeeze(target, axis=1)
    seq_len = min(audio.shape[0], emotion.shape[0], target.shape[0])
    return audio[:seq_len], emotion[:seq_len], target[:seq_len]


def _compute_mean_std(sum_vec: np.ndarray, sq_sum_vec: np.ndarray, count: int) -> Dict[str, np.ndarray]:
    if count <= 0:
        raise ValueError("count 必须大于 0")
    mean = sum_vec / float(count)
    var = sq_sum_vec / float(count) - np.square(mean)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def compute_stats(samples: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, np.ndarray]]:
    audio_sum = None
    audio_sq_sum = None
    emotion_sum = None
    emotion_sq_sum = None
    target_sum = None
    target_sq_sum = None
    total_count = 0
    for sample in samples:
        audio, emotion, target = load_sample_arrays(sample)
        if audio_sum is None:
            audio_sum = np.zeros(audio.shape[1], dtype=np.float64)
            audio_sq_sum = np.zeros(audio.shape[1], dtype=np.float64)
            emotion_sum = np.zeros(emotion.shape[1], dtype=np.float64)
            emotion_sq_sum = np.zeros(emotion.shape[1], dtype=np.float64)
            target_sum = np.zeros(target.shape[1], dtype=np.float64)
            target_sq_sum = np.zeros(target.shape[1], dtype=np.float64)
        audio_sum += audio.sum(axis=0)
        audio_sq_sum += np.square(audio, dtype=np.float64).sum(axis=0)
        emotion_sum += emotion.sum(axis=0)
        emotion_sq_sum += np.square(emotion, dtype=np.float64).sum(axis=0)
        target_sum += target.sum(axis=0)
        target_sq_sum += np.square(target, dtype=np.float64).sum(axis=0)
        total_count += audio.shape[0]
    return {
        "audio": _compute_mean_std(audio_sum, audio_sq_sum, total_count),
        "emotion": _compute_mean_std(emotion_sum, emotion_sq_sum, total_count),
        "target": _compute_mean_std(target_sum, target_sq_sum, total_count),
    }


class FaceParamDataset(Dataset):
    def __init__(self, root_dir, samples=None, normalize=True, stats=None):
        self.root_dir = str(root_dir)
        self.samples = list(samples) if samples is not None else collect_samples(self.root_dir)
        self.normalize = normalize
        self.stats = stats if stats is not None else (compute_stats(self.samples) if normalize else None)

    def __len__(self):
        return len(self.samples)

    def _normalize(self, array, key):
        if not self.normalize or self.stats is None:
            return array
        mean = self.stats[key]["mean"]
        std = self.stats[key]["std"]
        return (array - mean) / std

    def __getitem__(self, idx):
        audio, emotion, target = load_sample_arrays(self.samples[idx])
        audio = self._normalize(audio, "audio").astype(np.float32)
        emotion = self._normalize(emotion, "emotion").astype(np.float32)
        target = self._normalize(target, "target").astype(np.float32)
        return {
            "audio": torch.from_numpy(audio),
            "emotion": torch.from_numpy(emotion),
            "target": torch.from_numpy(target),
            "length": torch.tensor(audio.shape[0], dtype=torch.long),
        }


def collate_fn(batch):
    max_len = max(int(item["length"]) for item in batch)
    batch_size = len(batch)
    audio_dim = batch[0]["audio"].shape[1]
    emotion_dim = batch[0]["emotion"].shape[1]
    target_dim = batch[0]["target"].shape[1]
    audios = torch.zeros(batch_size, max_len, audio_dim, dtype=torch.float32)
    emotions = torch.zeros(batch_size, max_len, emotion_dim, dtype=torch.float32)
    targets = torch.zeros(batch_size, max_len, target_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    for i, item in enumerate(batch):
        length = int(item["length"])
        audios[i, :length] = item["audio"]
        emotions[i, :length] = item["emotion"]
        targets[i, :length] = item["target"]
        mask[i, :length] = True
        lengths[i] = length
    return {"audio": audios, "emotion": emotions, "target": targets, "mask": mask, "lengths": lengths}


# ============================================================================
# 新增: EmotionReactionDataset - 情感反应预测数据集
# 输入: speaker emotion (25维)
# 目标: listener emotion (25维)
# ============================================================================

def _resolve_emotion_path(
    data_root: Path, dataset: str, session: str,
    role: str, clip_id: str,
) -> Optional[Path]:
    """解析 emotion CSV 路径, 支持 NoXI 和 RECOLA"""
    if dataset == "NoXI":
        path = data_root / "Emotion" / "NoXI" / session / role / f"{clip_id}.csv"
    elif dataset == "RECOLA":
        path = data_root / "Emotion" / "RECOLA" / session / role / f"{clip_id}.csv"
    else:
        return None
    if path.exists() and not path.name.startswith("._"):
        return path
    return None


def _resolve_audio_feature_path(
    feature_root: Optional[Path], dataset: str, session: str,
    video_type: str, clip_id: str,
) -> Optional[Path]:
    """解析预提取音频特征路径 (768维 .npy)"""
    if feature_root is None:
        return None
    path = feature_root / "Audio_files" / session / video_type / f"{clip_id}.npy"
    if path.exists():
        return path
    return None


def collect_reaction_samples(
    data_root: str | Path,
    feature_root: str | Path | None = None,
    include_recola: bool = True,
) -> List[Dict[str, object]]:
    """
    收集 speaker→listener 情感反应样本对

    返回列表, 每个元素:
      {
        "speaker_emotion": str,   # speaker emotion CSV 路径
        "listener_emotion": str,  # listener emotion CSV 路径 (训练目标)
        "speaker_audio": str|None, # 预提取音频特征路径 (可选)
        "session": str,
        "clip_id": str,
        "direction": str,         # "expert2novice" 或 "novice2expert"
      }
    """
    data_root = Path(data_root)
    feature_root = Path(feature_root) if feature_root else None
    samples = []

    # --- NoXI ---
    noxi_video = data_root / "Video_files" / "NoXI"
    if noxi_video.exists():
        for session_dir in sorted(noxi_video.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name
            expert_dir = session_dir / "Expert_video"
            novice_dir = session_dir / "Novice_video"
            if not expert_dir.exists() or not novice_dir.exists():
                continue

            # 收集 clip IDs (取交集)
            expert_clips = {f.stem for f in expert_dir.glob("*.mp4")}
            novice_clips = {f.stem for f in novice_dir.glob("*.mp4")}
            common_clips = sorted(expert_clips & novice_clips, key=lambda x: int(x) if x.isdigit() else x)

            for clip_id in common_clips:
                p1_emo = _resolve_emotion_path(data_root, "NoXI", session, "P1", clip_id)
                p2_emo = _resolve_emotion_path(data_root, "NoXI", session, "P2", clip_id)
                if p1_emo is None or p2_emo is None:
                    continue

                # Expert → Novice
                expert_audio = _resolve_audio_feature_path(
                    feature_root, "NoXI", session, "Expert_video", clip_id
                )
                samples.append({
                    "speaker_emotion": str(p1_emo),
                    "listener_emotion": str(p2_emo),
                    "speaker_audio": str(expert_audio) if expert_audio else None,
                    "session": session,
                    "clip_id": clip_id,
                    "direction": "expert2novice",
                })

                # Novice → Expert
                novice_audio = _resolve_audio_feature_path(
                    feature_root, "NoXI", session, "Novice_video", clip_id
                )
                samples.append({
                    "speaker_emotion": str(p2_emo),
                    "listener_emotion": str(p1_emo),
                    "speaker_audio": str(novice_audio) if novice_audio else None,
                    "session": session,
                    "clip_id": clip_id,
                    "direction": "novice2expert",
                })

    # --- RECOLA ---
    if include_recola:
        recola_video = data_root / "Video_files" / "RECOLA"
        if recola_video.exists():
            for group_dir in sorted(recola_video.iterdir()):
                if not group_dir.is_dir():
                    continue
                group = group_dir.name
                person_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir()])
                if len(person_dirs) < 2:
                    continue
                p_a, p_b = person_dirs[0], person_dirs[1]
                role_a = RECOLA_ROLE_MAP.get(p_a.name, "P1")
                role_b = RECOLA_ROLE_MAP.get(p_b.name, "P2")
                clips_a = {f.stem for f in p_a.glob("*.mp4")}
                clips_b = {f.stem for f in p_b.glob("*.mp4")}
                common = sorted(clips_a & clips_b, key=lambda x: int(x) if x.isdigit() else x)
                for clip_id in common:
                    ea = _resolve_emotion_path(data_root, "RECOLA", group, role_a, clip_id)
                    eb = _resolve_emotion_path(data_root, "RECOLA", group, role_b, clip_id)
                    if ea is None or eb is None:
                        continue
                    samples.append({
                        "speaker_emotion": str(ea),
                        "listener_emotion": str(eb),
                        "speaker_audio": None,
                        "session": f"RECOLA/{group}",
                        "clip_id": clip_id,
                        "direction": f"{p_a.name}2{p_b.name}",
                    })
                    samples.append({
                        "speaker_emotion": str(eb),
                        "listener_emotion": str(ea),
                        "speaker_audio": None,
                        "session": f"RECOLA/{group}",
                        "clip_id": clip_id,
                        "direction": f"{p_b.name}2{p_a.name}",
                    })

    return samples


def compute_emotion_stats(samples: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, np.ndarray]]:
    """计算 speaker emotion 和 listener emotion 的均值/标准差"""
    sp_sum = np.zeros(25, dtype=np.float64)
    sp_sq = np.zeros(25, dtype=np.float64)
    li_sum = np.zeros(25, dtype=np.float64)
    li_sq = np.zeros(25, dtype=np.float64)
    au_sum = np.zeros(768, dtype=np.float64)
    au_sq = np.zeros(768, dtype=np.float64)
    count = 0
    au_count = 0
    for s in samples:
        sp = np.loadtxt(s["speaker_emotion"], delimiter=",", skiprows=1, dtype=np.float64)
        li = np.loadtxt(s["listener_emotion"], delimiter=",", skiprows=1, dtype=np.float64)
        n = min(sp.shape[0], li.shape[0])
        sp_sum += sp[:n].sum(axis=0)
        sp_sq += np.square(sp[:n]).sum(axis=0)
        li_sum += li[:n].sum(axis=0)
        li_sq += np.square(li[:n]).sum(axis=0)
        count += n
        if s["speaker_audio"] is not None:
            a = np.load(s["speaker_audio"]).astype(np.float64)
            m = min(a.shape[0], n)
            au_sum[:a.shape[1]] += a[:m].sum(axis=0)
            au_sq[:a.shape[1]] += np.square(a[:m]).sum(axis=0)
            au_count += m
    stats = {
        "speaker_emotion": _compute_mean_std(sp_sum, sp_sq, count),
        "listener_emotion": _compute_mean_std(li_sum, li_sq, count),
    }
    if au_count > 0:
        stats["audio"] = _compute_mean_std(au_sum, au_sq, au_count)
    return stats


class EmotionReactionDataset(Dataset):
    """
    情感反应预测数据集
    输入: speaker emotion [T, 25] + (可选) speaker audio [T, 768]
    目标: listener emotion [T, 25]
    
    支持数据增强: 时间拉伸、随机噪声
    """

    def __init__(
        self,
        samples: List[Dict[str, object]],
        normalize: bool = True,
        stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        max_seq_len: int = 750,
        augment: bool = False,
        augment_prob: float = 0.5,
    ):
        self.samples = list(samples)
        self.normalize = normalize
        self.stats = stats
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.samples)

    def _norm(self, arr: np.ndarray, key: str) -> np.ndarray:
        if not self.normalize or self.stats is None or key not in self.stats:
            return arr
        return (arr - self.stats[key]["mean"]) / self.stats[key]["std"]

    def _augment_time_stretch(self, audio: np.ndarray, emotion: np.ndarray, target: np.ndarray, 
                               has_audio: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """时间拉伸: 随机改变序列速度 0.9x - 1.1x"""
        if np.random.rand() > self.augment_prob:
            return audio, emotion, target
        
        stretch_factor = np.random.uniform(0.9, 1.1)
        orig_len = emotion.shape[0]
        new_len = int(orig_len * stretch_factor)
        new_len = max(10, min(new_len, self.max_seq_len))
        
        # 线性插值
        old_indices = np.linspace(0, orig_len - 1, orig_len)
        new_indices = np.linspace(0, orig_len - 1, new_len)
        
        emotion = np.array([np.interp(new_indices, old_indices, emotion[:, i]) 
                           for i in range(emotion.shape[1])]).T
        target = np.array([np.interp(new_indices, old_indices, target[:, i]) 
                          for i in range(target.shape[1])]).T
        
        # audio也拉伸到相同长度
        if has_audio and audio.shape[0] > 0:
            audio_orig_len = audio.shape[0]
            audio_old_indices = np.linspace(0, audio_orig_len - 1, audio_orig_len)
            audio_new_indices = np.linspace(0, audio_orig_len - 1, new_len)
            audio = np.array([np.interp(audio_new_indices, audio_old_indices, audio[:, i]) 
                             for i in range(audio.shape[1])]).T
        elif has_audio:
            # has_audio为True但shape[0]==0，创建零数组
            audio = np.zeros((new_len, 768), dtype=np.float32)
        
        return audio, emotion, target

    def _augment_noise(self, audio: np.ndarray, emotion: np.ndarray, target: np.ndarray,
                       has_audio: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """添加高斯噪声"""
        if np.random.rand() > self.augment_prob:
            return audio, emotion, target
        
        # emotion和target添加小噪声
        emotion_noise = np.random.normal(0, 0.02, emotion.shape).astype(np.float32)
        target_noise = np.random.normal(0, 0.02, target.shape).astype(np.float32)
        emotion = emotion + emotion_noise
        target = target + target_noise
        
        if has_audio and audio.shape[0] > 0:
            audio_noise = np.random.normal(0, 0.01, audio.shape).astype(np.float32)
            audio = audio + audio_noise
        
        return audio, emotion, target

    def _augment_dropout(self, emotion: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """随机时间步 dropout (模拟缺失帧)"""
        if np.random.rand() > self.augment_prob * 0.5:
            return emotion, target
        
        seq_len = emotion.shape[0]
        # 随机mask 5%的帧，用插值填充
        mask = np.random.rand(seq_len) > 0.05
        if mask.sum() < seq_len * 0.5:  # 至少保留50%
            return emotion, target
        
        valid_indices = np.where(mask)[0]
        all_indices = np.arange(seq_len)
        
        for i in range(emotion.shape[1]):
            emotion[:, i] = np.interp(all_indices, valid_indices, emotion[valid_indices, i])
        for i in range(target.shape[1]):
            target[:, i] = np.interp(all_indices, valid_indices, target[valid_indices, i])
        
        return emotion, target

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # 加载 speaker / listener emotion (处理空文件)
        sp_emo = np.loadtxt(s["speaker_emotion"], delimiter=",", skiprows=1, dtype=np.float32)
        li_emo = np.loadtxt(s["listener_emotion"], delimiter=",", skiprows=1, dtype=np.float32)

        # 处理空文件或1D数组
        if sp_emo.ndim < 2 or sp_emo.shape[0] == 0:
            sp_emo = np.zeros((self.max_seq_len, 25), dtype=np.float32)
        if li_emo.ndim < 2 or li_emo.shape[0] == 0:
            li_emo = np.zeros((self.max_seq_len, 25), dtype=np.float32)

        seq_len = min(sp_emo.shape[0], li_emo.shape[0], self.max_seq_len)
        sp_emo = sp_emo[:seq_len]
        li_emo = li_emo[:seq_len]

        # 加载 speaker audio (可选)
        has_audio = s["speaker_audio"] is not None
        if has_audio:
            audio = np.load(s["speaker_audio"]).astype(np.float32)
            seq_len = min(seq_len, audio.shape[0])
            audio = audio[:seq_len]
            sp_emo = sp_emo[:seq_len]
            li_emo = li_emo[:seq_len]
        else:
            audio = np.zeros((seq_len, 768), dtype=np.float32)

        # 数据增强 (仅训练集)
        if self.augment:
            audio, sp_emo, li_emo = self._augment_time_stretch(audio, sp_emo, li_emo, has_audio)
            audio, sp_emo, li_emo = self._augment_noise(audio, sp_emo, li_emo, has_audio)
            sp_emo, li_emo = self._augment_dropout(sp_emo, li_emo)
            # 重新计算seq_len
            seq_len = sp_emo.shape[0]
        
        # 归一化
        sp_emo = self._norm(sp_emo, "speaker_emotion").astype(np.float32)
        li_emo = self._norm(li_emo, "listener_emotion").astype(np.float32)
        if has_audio:
            audio = self._norm(audio, "audio").astype(np.float32)

        return {
            "audio": torch.from_numpy(audio),
            "emotion": torch.from_numpy(sp_emo),
            "target": torch.from_numpy(li_emo),
            "has_audio": torch.tensor(has_audio, dtype=torch.bool),
            "length": torch.tensor(seq_len, dtype=torch.long),
        }


def reaction_collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(int(item["length"]) for item in batch)
    bs = len(batch)
    audios = torch.zeros(bs, max_len, 768, dtype=torch.float32)
    emotions = torch.zeros(bs, max_len, 25, dtype=torch.float32)
    targets = torch.zeros(bs, max_len, 25, dtype=torch.float32)
    mask = torch.zeros(bs, max_len, dtype=torch.bool)
    has_audio = torch.zeros(bs, dtype=torch.bool)
    lengths = torch.zeros(bs, dtype=torch.long)
    for i, item in enumerate(batch):
        L = int(item["length"])
        # 确保tensor长度与L一致（数据增强可能导致变化）
        audio_tensor = item["audio"]
        emotion_tensor = item["emotion"]
        target_tensor = item["target"]
        
        # 截断或填充到L
        if audio_tensor.shape[0] > L:
            audio_tensor = audio_tensor[:L]
        if emotion_tensor.shape[0] > L:
            emotion_tensor = emotion_tensor[:L]
        if target_tensor.shape[0] > L:
            target_tensor = target_tensor[:L]
            
        audios[i, :audio_tensor.shape[0]] = audio_tensor
        emotions[i, :emotion_tensor.shape[0]] = emotion_tensor
        targets[i, :target_tensor.shape[0]] = target_tensor
        mask[i, :L] = True
        has_audio[i] = item["has_audio"]
        lengths[i] = L
    return {
        "audio": audios,
        "emotion": emotions,
        "target": targets,
        "mask": mask,
        "has_audio": has_audio,
        "lengths": lengths,
    }
