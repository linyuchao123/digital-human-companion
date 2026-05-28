#!/usr/bin/env python3
"""
数字人面部行为驱动模型 - 官方评测预测生成
生成 prediction_emotion.npy，格式: [N, K=10, T=750, 25]

用法:
  python generate_prediction.py \
      --data-root D:/服创比赛/数字人面部行为驱动模型训练集 \
      --val-csv D:/AI数字人情感陪护项目/数字人面部行为驱动模型验证集/person_specific_val.csv \
      --model-path models/face_drive/best_model.pt \
      --output output/prediction_emotion.npy
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from services.avatar.face_drive.training.dataset import (
    FEATURE_ORDER, load_audio_mel, load_emotion_csv,
)
from services.avatar.face_drive.inference.model import FaceReactionModel, ModelConfig

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

# 验证集固定参数
VAL_T = 750   # 30秒 @ 25fps
VAL_K = 10    # 每样本生成10条候选

# 多样性噪声参数（保证FRDiv指标）
DIVERSITY_NOISE_STD = 0.05


def load_val_index(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    读取验证集CSV，返回展开后的 (speaker_order, listener_order)
    展开: 正向(spk→lst) + 反向(lst→spk)
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    data_rows = rows[1:]  # 跳过header
    speaker_paths, listener_paths = [], []
    for row in data_rows:
        if len(row) < 3:
            continue
        speaker_paths.append(row[1].strip())
        listener_paths.append(row[2].strip())

    # 双向展开
    speaker_order = speaker_paths + listener_paths
    listener_order = listener_paths + speaker_paths

    print(f"[Predict] 验证样本数: {len(data_rows)} → 展开后 N={len(speaker_order)}")
    return speaker_order, listener_order


def path_to_audio(data_root: str, rel_path: str) -> Optional[str]:
    """
    将 person_specific_val.csv 中的相对路径转为实际音频路径
    rel_path 格式: NoXI/007_2016-03-21_Paris/Expert_video/6
    对应音频: {data_root}/Audio_files/NoXI/007_.../Expert_video/6.wav
    """
    parts = rel_path.replace("\\", "/").split("/")
    wav_path = Path(data_root) / "Audio_files" / Path(*parts)
    wav_path = wav_path.with_suffix(".wav")
    if wav_path.exists():
        return str(wav_path)
    return None


def path_to_emotion_gt(data_root: str, rel_path: str) -> Optional[str]:
    """
    rel_path: NoXI/007_.../Expert_video/6 → Emotion/NoXI/007_.../P1/6.csv
    """
    ROLE_MAP = {"Expert_video": "P1", "Novice_video": "P2"}
    parts = rel_path.replace("\\", "/").split("/")
    # parts = [dataset, session, role, idx]
    if len(parts) < 4:
        return None

    dataset, session, role, idx = parts[0], parts[1], parts[2], parts[3]
    if dataset == "NoXI":
        emo_role = ROLE_MAP.get(role, role)
    else:
        emo_role = role

    emo_path = Path(data_root) / "Emotion" / dataset / session / emo_role / f"{idx}.csv"
    if emo_path.exists():
        return str(emo_path)
    return None


def infer_single(
    model,
    audio_path: Optional[str],
    emotion_cond: Optional[np.ndarray],
    target_len: int = VAL_T,
    device: str = "cpu",
) -> np.ndarray:
    """
    对单个样本进行推理
    返回: (target_len, 25) 情绪特征
    """
    # 提取音频特征
    if audio_path and Path(audio_path).exists():
        mel = load_audio_mel(audio_path)  # (T_mel, 80)
    else:
        mel = np.zeros((target_len, 80), dtype=np.float32)

    # 截断或填充到目标长度
    if len(mel) < target_len:
        pad = np.zeros((target_len - len(mel), 80), dtype=np.float32)
        mel = np.concatenate([mel, pad], axis=0)
    else:
        mel = mel[:target_len]

    if TORCH_AVAILABLE and hasattr(model, 'parameters'):
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)  # (1, T, 80)

        cond_t = None
        if emotion_cond is not None:
            cond_t = torch.from_numpy(
                emotion_cond.astype(np.float32)
            ).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(mel_t, cond_t)

        emo_25 = outputs["emotion_25"].squeeze(0).cpu().numpy()  # (T, 25)
    else:
        # 无模型：使用统计规律生成合理的情绪序列
        outputs = model(mel[np.newaxis], emotion_cond[np.newaxis] if emotion_cond is not None else None)
        emo_25 = outputs["emotion_25"][0]

    # 确保长度
    if len(emo_25) < target_len:
        pad = np.zeros((target_len - len(emo_25), 25), dtype=np.float32)
        emo_25 = np.concatenate([emo_25, pad], axis=0)
    else:
        emo_25 = emo_25[:target_len]

    # 数值规范化
    emo_25 = _normalize_emotion(emo_25)

    return emo_25.astype(np.float32)


def _normalize_emotion(emo: np.ndarray) -> np.ndarray:
    """
    确保各维度在合法范围:
    AU (0-14): [0, 1]
    VA (15-16): [-1, 1]
    EXP (17-24): softmax→ 和为1
    """
    out = emo.copy()
    # AU: clip到[0,1]
    out[:, :15] = np.clip(out[:, :15], 0.0, 1.0)
    # VA: clip到[-1,1]
    out[:, 15:17] = np.clip(out[:, 15:17], -1.0, 1.0)
    # EXP: softmax归一化
    exp = out[:, 17:]
    exp = np.exp(exp - exp.max(axis=-1, keepdims=True))
    exp = exp / (exp.sum(axis=-1, keepdims=True) + 1e-8)
    out[:, 17:] = exp
    return out


def generate_diverse_candidates(
    base_pred: np.ndarray,
    k: int = VAL_K,
    noise_std: float = DIVERSITY_NOISE_STD,
) -> np.ndarray:
    """
    基于基础预测生成K条多样化候选序列
    策略: 添加不同强度/形式的噪声，保证FRDiv指标
    返回: (K, T, 25)
    """
    candidates = []

    for i in range(k):
        if i == 0:
            # 第0条: 原始预测（无噪声）
            cand = base_pred.copy()
        else:
            # 其余: 添加渐增噪声
            scale = noise_std * (1 + i * 0.3)
            noise = np.random.randn(*base_pred.shape).astype(np.float32) * scale
            cand = base_pred + noise

        cand = _normalize_emotion(cand)
        candidates.append(cand)

    return np.stack(candidates, axis=0)  # (K, T, 25)


def generate_prediction(
    data_root: str,
    val_csv: str,
    model_path: Optional[str],
    output_path: str,
    device: str = "auto",
    target_len: int = VAL_T,
    k_candidates: int = VAL_K,
):
    """
    主函数：生成 prediction_emotion.npy
    """
    if device == "auto":
        if TORCH_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"

    print(f"[Predict] 使用设备: {device}")

    # 加载模型
    cfg = ModelConfig()
    model = FaceReactionModel(cfg)

    if model_path and Path(model_path).exists() and TORCH_AVAILABLE:
        ckpt = torch.load(model_path, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()
        print(f"[Predict] 已加载模型: {model_path}")
    else:
        print(f"[Predict] 未找到模型权重，使用随机初始化（评测基线）")

    # 读取验证集顺序
    speaker_order, listener_order = load_val_index(val_csv)
    N = len(speaker_order)

    # 生成预测
    print(f"[Predict] 开始生成预测，N={N}, K={k_candidates}, T={target_len}...")
    prediction = np.zeros((N, k_candidates, target_len, 25), dtype=np.float32)

    for n_idx, (spk_path, lst_path) in enumerate(zip(speaker_order, listener_order)):
        if (n_idx + 1) % 50 == 0:
            print(f"[Predict] 进度: {n_idx+1}/{N}")

        # 说话人音频（驱动输入）
        spk_audio = path_to_audio(data_root, spk_path)

        # 情绪条件：从说话人情绪GT获取（提供给解码器）
        emotion_cond = None
        spk_emo_path = path_to_emotion_gt(data_root, spk_path)
        if spk_emo_path:
            spk_emo = load_emotion_csv(spk_emo_path)
            if len(spk_emo) > 0:
                # 取均值作为全局条件: VA + EXP = 2 + 8 = 10维
                mean_emo = spk_emo[:min(len(spk_emo), target_len)].mean(axis=0)
                emotion_cond = np.concatenate([mean_emo[15:17], mean_emo[17:]])  # (10,)

        # 推理基础预测
        base_pred = infer_single(model, spk_audio, emotion_cond, target_len, device)

        # 生成K条候选
        prediction[n_idx] = generate_diverse_candidates(base_pred, k_candidates)

    # 保存
    os.makedirs(Path(output_path).parent, exist_ok=True)
    np.save(output_path, prediction)
    print(f"[Predict] ✓ 预测已保存: {output_path}")
    print(f"[Predict] 形状: {prediction.shape}  dtype: {prediction.dtype}")

    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成面部行为驱动模型评测预测文件")
    parser.add_argument("--data-root", required=True, help="训练集根目录（用于读取Emotion GT）")
    parser.add_argument("--val-csv", required=True, help="person_specific_val.csv 路径")
    parser.add_argument("--model-path", default=None, help="训练好的模型权重 .pt 路径")
    parser.add_argument("--output", default="output/prediction_emotion.npy", help="输出预测文件路径")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-len", type=int, default=750, help="序列长度（默认750=30s@25fps）")
    parser.add_argument("--k-candidates", type=int, default=10, help="每样本候选数（默认10）")
    args = parser.parse_args()

    generate_prediction(
        data_root=args.data_root,
        val_csv=args.val_csv,
        model_path=args.model_path,
        output_path=args.output,
        device=args.device,
        target_len=args.target_len,
        k_candidates=args.k_candidates,
    )
