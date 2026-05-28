#!/usr/bin/env python3
"""
数字人面部行为驱动模型 - 训练脚本
多任务损失: L_face + L_smooth + L_au + L_va + L_expr
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    pass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from services.avatar.face_drive.training.dataset import FaceReactionDataset, scan_dataset
from services.avatar.face_drive.inference.model import FaceReactionModel, ModelConfig


class MultiTaskLoss:
    """
    多任务损失组合
    L_total = L_face + 0.1*L_smooth + 0.5*L_au + 0.3*L_va + 0.2*L_expr
    """

    def __init__(self):
        if TORCH_AVAILABLE:
            self.l1 = nn.L1Loss()
            self.mse = nn.MSELoss()
            self.bce = nn.BCELoss()
            self.kl = nn.KLDivLoss(reduction="batchmean")

    def __call__(self, outputs: dict, face_gt, emo_gt) -> dict:
        if not TORCH_AVAILABLE:
            return {"total": 1.0}

        face_pred = outputs["face_58"]            # (B, T, 58)
        au_pred = outputs["au_15"]                # (B, T, 15)
        va_pred = outputs["va_2"]                 # (B, T, 2)
        exp_pred = outputs["exp_8"]               # (B, T, 8)

        au_gt = emo_gt[:, :, :15]                 # (B, T, 15) AU值 [0,1]
        va_gt = emo_gt[:, :, 15:17]               # (B, T, 2) VA [-1,1]
        exp_gt = emo_gt[:, :, 17:]                # (B, T, 8) 表情概率

        # 主任务：58D面部参数L1
        L_face = self.l1(face_pred, face_gt)

        # 平滑损失：防止抖动
        L_smooth = torch.mean(torch.abs(face_pred[:, 1:, :] - face_pred[:, :-1, :]))

        # 辅助：AU分类BCE（值已经是[0,1]，直接二值化作为标签）
        au_label = (au_gt > 0.5).float()
        au_pred_clamp = torch.clamp(au_pred, 1e-6, 1 - 1e-6)
        L_au = self.bce(au_pred_clamp, au_label)

        # 辅助：VA回归MSE
        va_gt_norm = va_gt  # 已在[-1,1]，Tanh输出也在[-1,1]
        L_va = self.mse(va_pred, va_gt_norm)

        # 辅助：表情分布KL散度
        exp_gt_smooth = exp_gt + 1e-8
        exp_gt_smooth = exp_gt_smooth / exp_gt_smooth.sum(dim=-1, keepdim=True)
        L_expr = self.kl(torch.log(exp_pred + 1e-8), exp_gt_smooth)

        # 加权组合
        L_total = L_face + 0.1 * L_smooth + 0.5 * L_au + 0.3 * L_va + 0.2 * L_expr

        return {
            "total": L_total,
            "face": L_face,
            "smooth": L_smooth,
            "au": L_au,
            "va": L_va,
            "expr": L_expr,
        }


def train(
    data_root: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.1,
    patience: int = 10,
    chunk_frames: int = 75,
    device: str = "auto",
    resume: Optional[str] = None,
):
    if not TORCH_AVAILABLE:
        print("[Train] PyTorch不可用，跳过训练")
        return

    # 设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] 使用设备: {device}")

    # 数据加载
    print(f"[Train] 扫描数据集: {data_root}")
    samples = scan_dataset(data_root)
    if not samples:
        print("[Train] 未找到有效训练样本，退出")
        return

    dataset = FaceReactionDataset(samples, chunk_frames=chunk_frames, augment=True)
    if len(dataset) == 0:
        print("[Train] 数据集为空，退出")
        return

    # 训练/验证划分
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Train] 训练样本: {n_train}, 验证样本: {n_val}")

    # 模型
    cfg = ModelConfig()
    model = FaceReactionModel(cfg).to(device)
    print(f"[Train] 模型参数量: {model.count_parameters():,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MultiTaskLoss()

    # 断点续训
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(output_dir, exist_ok=True)

    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Train] 从 epoch {start_epoch} 继续训练")

    # 训练循环
    for epoch in range(start_epoch, epochs):
        # ---- 训练 ----
        model.train()
        train_losses = []
        t0 = time.time()

        for mel, face_gt, emo_gt in train_loader:
            mel = mel.to(device)
            face_gt = face_gt.to(device)
            emo_gt = emo_gt.to(device)

            # 从情绪GT提取条件向量 (VA + EXP = 2+8 = 10维)
            emotion_cond = torch.cat([emo_gt[:, 0, 15:17], emo_gt[:, 0, 17:]], dim=-1)  # (B, 10)

            optimizer.zero_grad()
            outputs = model(mel, emotion_cond)
            losses = criterion(outputs, face_gt, emo_gt)
            losses["total"].backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(losses["total"].item())

        scheduler.step()
        avg_train = np.mean(train_losses)

        # ---- 验证 ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for mel, face_gt, emo_gt in val_loader:
                mel = mel.to(device)
                face_gt = face_gt.to(device)
                emo_gt = emo_gt.to(device)
                emotion_cond = torch.cat([emo_gt[:, 0, 15:17], emo_gt[:, 0, 17:]], dim=-1)
                outputs = model(mel, emotion_cond)
                losses = criterion(outputs, face_gt, emo_gt)
                val_losses.append(losses["total"].item())

        avg_val = np.mean(val_losses)
        elapsed = time.time() - t0

        print(
            f"[Train] Epoch {epoch+1:3d}/{epochs} | "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存最佳模型
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": cfg.__dict__,
            }, best_path)
            print(f"[Train] ✓ 保存最佳模型 → {best_path} (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] 早停触发（patience={patience}），停止训练")
                break

        # 每10 epoch保存检查点
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_ep{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": cfg.__dict__,
            }, ckpt_path)

    print(f"\n[Train] 训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"[Train] 模型保存于: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练面部行为驱动模型")
    parser.add_argument("--data-root", required=True, help="训练集根目录（含Audio_files/3D_FV_files/Emotion）")
    parser.add_argument("--output-dir", default="models/face_drive", help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--chunk-frames", type=int, default=75, help="训练片段帧数(默认75=3秒@25fps)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None, help="断点续训路径")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        chunk_frames=args.chunk_frames,
        device=args.device,
        resume=args.resume,
    )
