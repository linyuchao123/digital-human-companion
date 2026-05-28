"""
情感反应预测模型训练脚本

输入: speaker audio(768维, 可选) + speaker emotion(25维)
目标: listener emotion(25维)

支持:
- KL退火 (KL Annealing)
- 维度分组加权
- 验证集评估
- 模型检查点保存
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import (
    EmotionReactionDataset,
    collect_reaction_samples,
    compute_emotion_stats,
    reaction_collate_fn,
)
from loss import compute_reaction_loss
from model import EmotionReactionTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="情感反应预测模型训练")
    # 数据路径
    parser.add_argument("--train_root", type=str, default="/root/autodl-tmp/train",
                        help="训练数据根目录")
    parser.add_argument("--val_root", type=str, default="/root/autodl-tmp/val",
                        help="验证数据根目录")
    parser.add_argument("--feature_root", type=str,
                        default="/root/autodl-tmp/data_feature/NoXI_expert",
                        help="预提取音频特征根目录 (可选)")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/checkpoints",
                        help="模型输出目录")

    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_audio", action="store_true", default=True,
                        help="是否使用音频特征")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    # epochs moved to KL section with default 150
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=750)

    # 损失权重
    parser.add_argument("--temporal_weight", type=float, default=0.1)
    parser.add_argument("--au_weight", type=float, default=1.0)
    parser.add_argument("--va_weight", type=float, default=2.0)
    parser.add_argument("--exp_weight", type=float, default=1.5)

    # KL退火 (降低KL权重，提高重建精度)
    parser.add_argument("--kl_weight_max", type=float, default=0.02,
                        help="KL损失最大权重")
    parser.add_argument("--kl_warmup_epochs", type=int, default=30,
                        help="KL退火 warmup 轮数")

    # 早停机制
    parser.add_argument("--early_stop_patience", type=int, default=15,
                        help="早停耐心值，验证损失不改善的轮数")
    parser.add_argument("--epochs", type=int, default=150,
                        help="最大训练轮数")

    # 其他
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=50,
                        help="每N步打印一次日志")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="每N个epoch保存一次检查点")

    return parser.parse_args()


def get_kl_weight(epoch: int, max_weight: float, warmup_epochs: int) -> float:
    """KL退火: 线性增长"""
    if warmup_epochs <= 0:
        return max_weight
    return min(max_weight, max_weight * epoch / warmup_epochs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    total_losses = {
        "total_loss": 0.0, "recon_loss": 0.0, "temporal_loss": 0.0,
        "kl_loss": 0.0, "au_loss": 0.0, "va_loss": 0.0, "exp_loss": 0.0,
    }
    n_batches = 0
    kl_w = get_kl_weight(epoch, args.kl_weight_max, args.kl_warmup_epochs)

    for step, batch in enumerate(loader):
        audio = batch["audio"].to(device)
        emotion = batch["emotion"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        has_audio = batch.get("has_audio", None)
        if has_audio is not None:
            has_audio = has_audio.to(device)

        # 前向
        output = model(audio, emotion, target=target, mask=mask, has_audio=has_audio)

        # 计算损失
        losses = compute_reaction_loss(
            output, target, mask,
            kl_weight=kl_w,
            temporal_weight=args.temporal_weight,
            au_weight=args.au_weight,
            va_weight=args.va_weight,
            exp_weight=args.exp_weight,
        )

        # 反向
        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

        if (step + 1) % args.log_interval == 0:
            avg = {k: v / n_batches for k, v in total_losses.items()}
            print(
                f"  [Epoch {epoch}] Step {step + 1}/{len(loader)} | "
                f"Loss: {avg['total_loss']:.4f} | "
                f"Recon: {avg['recon_loss']:.4f} | "
                f"KL({kl_w:.4f}): {avg['kl_loss']:.4f} | "
                f"AU: {avg['au_loss']:.4f} | VA: {avg['va_loss']:.4f} | EXP: {avg['exp_loss']:.4f}"
            )

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    total_losses = {
        "total_loss": 0.0, "recon_loss": 0.0, "temporal_loss": 0.0,
        "kl_loss": 0.0, "au_loss": 0.0, "va_loss": 0.0, "exp_loss": 0.0,
    }
    n_batches = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        emotion = batch["emotion"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        has_audio = batch.get("has_audio", None)
        if has_audio is not None:
            has_audio = has_audio.to(device)

        output = model(audio, emotion, target=target, mask=mask, has_audio=has_audio)
        losses = compute_reaction_loss(
            output, target, mask,
            kl_weight=args.kl_weight_max,
            temporal_weight=args.temporal_weight,
            au_weight=args.au_weight,
            va_weight=args.va_weight,
            exp_weight=args.exp_weight,
        )

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 数据集 ==========
    print("收集训练样本...")
    feature_root = args.feature_root if args.use_audio and os.path.exists(args.feature_root) else None
    train_samples = collect_reaction_samples(args.train_root, feature_root=feature_root)
    print(f"  训练样本数: {len(train_samples)}")

    val_samples = collect_reaction_samples(args.val_root, feature_root=None)
    print(f"  验证样本数: {len(val_samples)}")

    # 计算归一化统计量
    print("计算归一化统计量...")
    stats = compute_emotion_stats(train_samples)
    np.savez(output_dir / "stats.npz", **{
        f"{k}_{sk}": sv for k, sv in stats.items() for sk, sv in sv.items()
    })
    print("  统计量已保存")

    train_dataset = EmotionReactionDataset(
        train_samples, normalize=True, stats=stats, max_seq_len=args.max_seq_len,
        augment=True, augment_prob=0.5,  # 训练集启用数据增强
    )
    val_dataset = EmotionReactionDataset(
        val_samples, normalize=True, stats=stats, max_seq_len=args.max_seq_len,
        augment=False,  # 验证集不增强
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=reaction_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=reaction_collate_fn,
        pin_memory=True,
    )

    # ========== 模型 ==========
    model = EmotionReactionTransformer(
        audio_dim=768,
        emotion_dim=25,
        output_dim=25,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_audio=args.use_audio,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")

    # ========== 优化器 ==========
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 保存配置
    config = vars(args)
    config["total_params"] = total_params
    config["train_samples"] = len(train_samples)
    config["val_samples"] = len(val_samples)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ========== 训练循环 ==========
    best_val_loss = float("inf")
    best_epoch = 0
    history = []
    patience_counter = 0

    print(f"\n开始训练 ({args.epochs} epochs, 早停耐心={args.early_stop_patience})...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 训练
        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch, args)

        # 验证
        val_losses = validate(model, val_loader, device, args)

        scheduler.step()
        elapsed = time.time() - t0

        kl_w = get_kl_weight(epoch, args.kl_weight_max, args.kl_warmup_epochs)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"LR: {lr:.6f} | KL_w: {kl_w:.4f}\n"
            f"  Train - Total: {train_losses['total_loss']:.4f} | "
            f"Recon: {train_losses['recon_loss']:.4f} | "
            f"KL: {train_losses['kl_loss']:.4f}\n"
            f"  Val   - Total: {val_losses['total_loss']:.4f} | "
            f"Recon: {val_losses['recon_loss']:.4f} | "
            f"AU: {val_losses['au_loss']:.4f} | "
            f"VA: {val_losses['va_loss']:.4f} | "
            f"EXP: {val_losses['exp_loss']:.4f}"
        )

        # 记录历史
        history.append({
            "epoch": epoch,
            "lr": lr,
            "kl_weight": kl_w,
            "train": train_losses,
            "val": val_losses,
        })

        # 保存最佳模型 + 早停检查
        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "stats": stats,
                "config": config,
            }, output_dir / "best_model.pt")
            print(f"  ** 最佳模型已保存 (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"\n  !! 早停触发: 验证损失已连续 {patience_counter} 轮未改善")
                print(f"  !! 最佳模型在第 {best_epoch} 轮 (val_loss: {best_val_loss:.4f})")
                break

        # 定期保存
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_losses["total_loss"],
                "stats": stats,
                "config": config,
            }, output_dir / f"checkpoint_epoch{epoch}.pt")

    # 保存最终模型和历史
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_losses["total_loss"],
        "stats": stats,
        "config": config,
    }, output_dir / "final_model.pt")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n训练完成! 最佳 val_loss: {best_val_loss:.4f}")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
