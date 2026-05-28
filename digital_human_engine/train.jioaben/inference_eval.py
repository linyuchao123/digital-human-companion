"""
评测推理脚本

严格按 person_specific_val.csv 顺序生成:
  prediction_emotion.npy  shape [N=1086, K=10, T=750, 25]

每行样本展开为正向(speaker→listener) + 反向(listener→speaker), 共 N = 543*2 = 1086

使用方法:
  python inference_eval.py \
    --checkpoint /root/autodl-tmp/checkpoints/best_model.pt \
    --val_root /root/autodl-tmp/val \
    --val_csv /root/autodl-tmp/predict/22-【A22】验证集自测包/person_specific_val.csv \
    --output_path /root/autodl-tmp/predict/prediction_emotion.npy \
    --num_candidates 10
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from model import EmotionReactionTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="评测推理")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--val_root", type=str, default="/root/autodl-tmp/val",
                        help="验证集根目录")
    parser.add_argument("--val_csv", type=str,
                        default="/root/autodl-tmp/predict/22-【A22】验证集自测包/person_specific_val.csv",
                        help="person_specific_val.csv 路径")
    parser.add_argument("--output_path", type=str,
                        default="/root/autodl-tmp/predict/prediction_emotion.npy",
                        help="输出 npy 路径")
    parser.add_argument("--num_candidates", type=int, default=10,
                        help="每样本生成 K 条候选")
    parser.add_argument("--target_len", type=int, default=750,
                        help="目标序列长度 T")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="推理 batch size")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def path_to_emotion_csv(val_root: str, video_path: str) -> str:
    """
    将 CSV 中的 video 路径转换为 Emotion CSV 路径

    video_path 格式: NoXI/005_2016-03-18_Paris/Expert_video/2
                     RECOLA/group-1/P25/2

    Emotion 目录结构:
      NoXI: Emotion/NoXI/<session>/P1/<clip_id>.csv  (Expert=P1, Novice=P2)
      RECOLA: Emotion/RECOLA/<group>/<role>/<clip_id>.csv
    """
    parts = video_path.split("/")
    dataset = parts[0]  # NoXI 或 RECOLA

    if dataset == "NoXI":
        session = parts[1]
        video_type = parts[2]  # Expert_video 或 Novice_video
        clip_id = parts[3]
        role = "P1" if video_type == "Expert_video" else "P2"
        return os.path.join(val_root, "Emotion", "NoXI", session, role, f"{clip_id}.csv")
    elif dataset == "RECOLA":
        group = parts[1]
        person_id = parts[2]
        clip_id = parts[3]
        # RECOLA 中 person_id 就是角色目录名
        return os.path.join(val_root, "Emotion", "RECOLA", group, person_id, f"{clip_id}.csv")
    else:
        raise ValueError(f"未知数据集: {dataset}")


def load_emotion_csv(path: str) -> np.ndarray:
    """加载 emotion CSV, 返回 [T, 25], 空文件返回零数组"""
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim < 2 or data.shape[0] == 0:
        return np.zeros((750, 25), dtype=np.float32)
    return data


def pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    """将序列 padding 或裁剪到目标长度"""
    T = arr.shape[0]
    if T >= target_len:
        return arr[:target_len]
    else:
        pad = np.zeros((target_len - T, arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ========== 加载模型 ==========
    print(f"加载模型: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    stats = ckpt.get("stats", None)

    model = EmotionReactionTransformer(
        audio_dim=768,
        emotion_dim=25,
        output_dim=25,
        hidden_dim=config.get("hidden_dim", 256),
        latent_dim=config.get("latent_dim", 64),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 4),
        dropout=0.0,  # 推理关闭 dropout
        use_audio=config.get("use_audio", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  模型加载完成 (epoch {ckpt.get('epoch', '?')})")

    # ========== 读取 val CSV ==========
    print(f"读取样本顺序: {args.val_csv}")
    with open(args.val_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    print(f"  原始行数: {len(rows)}, 展开后 N = {len(rows) * 2}")

    # ========== 构建样本列表 ==========
    # 每行展开为: 正向 (speaker→listener) + 反向 (listener→speaker)
    samples = []
    for row in rows:
        idx, speaker_path, listener_path = row[0], row[1], row[2]
        # 正向: speaker → listener
        samples.append({
            "speaker_path": speaker_path,
            "listener_path": listener_path,
            "direction": "forward",
        })
    for row in rows:
        idx, speaker_path, listener_path = row[0], row[1], row[2]
        # 反向: listener → speaker
        samples.append({
            "speaker_path": listener_path,
            "listener_path": speaker_path,
            "direction": "reverse",
        })

    N = len(samples)
    K = args.num_candidates
    T = args.target_len
    D = 25
    print(f"总样本数 N={N}, K={K}, T={T}, D={D}")

    # ========== 推理 ==========
    predictions = np.zeros((N, K, T, D), dtype=np.float32)

    # 获取归一化统计量
    sp_mean = stats["speaker_emotion"]["mean"] if stats and "speaker_emotion" in stats else np.zeros(25)
    sp_std = stats["speaker_emotion"]["std"] if stats and "speaker_emotion" in stats else np.ones(25)
    li_mean = stats["listener_emotion"]["mean"] if stats and "listener_emotion" in stats else np.zeros(25)
    li_std = stats["listener_emotion"]["std"] if stats and "listener_emotion" in stats else np.ones(25)

    print("开始推理...")
    with torch.no_grad():
        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  样本 {i + 1}/{N} ({sample['direction']})")

            # 加载 speaker emotion
            sp_csv = path_to_emotion_csv(args.val_root, sample["speaker_path"])
            if not os.path.exists(sp_csv):
                print(f"  警告: 找不到 {sp_csv}, 使用零填充")
                predictions[i] = 0.0
                continue

            sp_emo = load_emotion_csv(sp_csv)
            sp_emo = pad_or_trim(sp_emo, T)

            # 归一化
            sp_emo_norm = ((sp_emo - sp_mean) / (sp_std + 1e-8)).astype(np.float32)

            # 构建输入 tensor
            audio_t = torch.zeros(1, T, 768, device=device)  # 无预提取音频
            emotion_t = torch.from_numpy(sp_emo_norm).unsqueeze(0).to(device)  # [1, T, 25]
            mask_t = torch.ones(1, T, dtype=torch.bool, device=device)
            has_audio_t = torch.zeros(1, dtype=torch.bool, device=device)

            # 截断 mask 到实际长度
            actual_len = min(load_emotion_csv(sp_csv).shape[0], T)
            if actual_len < T:
                mask_t[0, actual_len:] = False

            # 生成 K 条候选
            pred_k = model.generate(
                audio_t, emotion_t, mask=mask_t, has_audio=has_audio_t,
                num_candidates=K,
            )  # [1, K, T, 25]

            # 反归一化
            pred_np = pred_k[0].cpu().numpy()  # [K, T, 25]
            pred_np = pred_np * li_std + li_mean

            # 对短序列的 padding 部分复制最后一帧
            if actual_len < T:
                pred_np[:, actual_len:, :] = pred_np[:, actual_len - 1: actual_len, :]

            predictions[i] = pred_np

    # ========== 保存 ==========
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(args.output_path, predictions)
    print(f"\n预测结果已保存: {args.output_path}")
    print(f"  Shape: {predictions.shape}")
    print(f"  数值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  均值: {predictions.mean():.4f}, 标准差: {predictions.std():.4f}")

    # 验证维度
    assert predictions.shape == (N, K, T, D), \
        f"Shape 不匹配! 期望 ({N}, {K}, {T}, {D}), 实际 {predictions.shape}"
    print("维度验证通过!")


if __name__ == "__main__":
    main()
