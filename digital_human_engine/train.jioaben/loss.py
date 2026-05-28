"""
情感反应预测损失函数

25维输出: AU(0-14) + VA(15-16) + EXP(17-24)
按维度分组加权, 包含 CVAE KL散度损失
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


# 25维分组索引
AU_SLICE = slice(0, 15)   # 15个AU
VA_SLICE = slice(15, 17)  # Valence, Arousal
EXP_SLICE = slice(17, 25) # 8个表情概率


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dim_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    带 mask 的 MSE, 支持维度加权

    Args:
        pred: [B, T, D]
        target: [B, T, D]
        mask: [B, T] bool
        dim_weights: [D] 可选的维度权重
    """
    mask_f = mask.unsqueeze(-1).float()  # [B, T, 1]
    diff_sq = (pred - target) ** 2  # [B, T, D]

    if dim_weights is not None:
        diff_sq = diff_sq * dim_weights.to(diff_sq.device)

    loss = (diff_sq * mask_f).sum() / (mask_f.sum() * pred.shape[-1] + 1e-8)
    return loss


def temporal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """时序平滑损失: 相邻帧差分的 MSE"""
    pred_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    temporal_mask = mask[:, 1:] & mask[:, :-1]
    return masked_mse(pred_delta, target_delta, temporal_mask)


def diversity_loss(
    predictions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    多样性正则化: 鼓励 K 条候选序列之间有差异
    仅在训练时使用多个采样时有效

    Args:
        predictions: [B, K, T, D]
        mask: [B, T]
    """
    if predictions.dim() != 4 or predictions.shape[1] < 2:
        return torch.tensor(0.0, device=predictions.device)

    K = predictions.shape[1]
    mask_f = mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, T, 1]

    # 计算所有 K 对之间的平均距离
    total_dist = 0.0
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            diff = (predictions[:, i] - predictions[:, j]) ** 2
            dist = (diff * mask_f[:, 0]).sum() / (mask_f[:, 0].sum() * predictions.shape[-1] + 1e-8)
            total_dist += dist
            count += 1

    # 返回负距离 (最小化此损失 = 最大化多样性)
    return -total_dist / max(count, 1)


def kl_divergence(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    KL(q(z|x,y) || p(z|x)) - CVAE KL散度

    Args:
        posterior_mu/logvar: [B, T, latent_dim] 后验参数
        prior_mu/logvar: [B, T, latent_dim] 先验参数
        mask: [B, T] bool
    """
    kl = 0.5 * (
        prior_logvar - posterior_logvar
        + (torch.exp(posterior_logvar) + (posterior_mu - prior_mu) ** 2) / (torch.exp(prior_logvar) + 1e-8)
        - 1.0
    )  # [B, T, latent_dim]
    mask_f = mask.unsqueeze(-1).float()
    return (kl * mask_f).sum() / (mask_f.sum() + 1e-8)


def get_dimension_weights(
    device: torch.device,
    au_weight: float = 1.0,
    va_weight: float = 2.0,
    exp_weight: float = 1.5,
) -> torch.Tensor:
    """构建25维的维度权重向量"""
    weights = torch.ones(25, device=device)
    weights[AU_SLICE] = au_weight
    weights[VA_SLICE] = va_weight
    weights[EXP_SLICE] = exp_weight
    return weights


def compute_reaction_loss(
    model_output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    kl_weight: float = 0.01,
    temporal_weight: float = 0.1,
    au_weight: float = 1.0,
    va_weight: float = 2.0,
    exp_weight: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """
    情感反应预测总损失

    Args:
        model_output: 模型输出 dict, 包含 pred, posterior_mu/logvar, prior_mu/logvar
        target: [B, T, 25] listener GT emotion
        mask: [B, T] bool
        kl_weight: KL损失权重 (支持退火调节)
        temporal_weight: 时序损失权重

    Returns:
        Dict: total_loss, recon_loss, temporal_loss, kl_loss, 及各维度分组损失
    """
    pred = model_output["pred"]
    dim_weights = get_dimension_weights(pred.device, au_weight, va_weight, exp_weight)

    # 1. 重建损失 (带维度加权)
    l_recon = masked_mse(pred, target, mask, dim_weights)

    # 2. 时序平滑损失
    l_temporal = temporal_loss(pred, target, mask)

    # 3. KL 散度损失
    l_kl = torch.tensor(0.0, device=pred.device)
    if model_output.get("posterior_mu") is not None:
        l_kl = kl_divergence(
            model_output["posterior_mu"],
            model_output["posterior_logvar"],
            model_output["prior_mu"],
            model_output["prior_logvar"],
            mask,
        )

    # 各分组损失 (用于监控)
    l_au = masked_mse(pred[:, :, AU_SLICE], target[:, :, AU_SLICE], mask)
    l_va = masked_mse(pred[:, :, VA_SLICE], target[:, :, VA_SLICE], mask)
    l_exp = masked_mse(pred[:, :, EXP_SLICE], target[:, :, EXP_SLICE], mask)

    # 总损失
    total = l_recon + temporal_weight * l_temporal + kl_weight * l_kl

    return {
        "total_loss": total,
        "recon_loss": l_recon,
        "temporal_loss": l_temporal,
        "kl_loss": l_kl,
        "au_loss": l_au,
        "va_loss": l_va,
        "exp_loss": l_exp,
    }
