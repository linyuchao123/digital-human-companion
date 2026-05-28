"""
EmotionReactionTransformer - 情感反应预测模型

输入: speaker audio (768维, 可选) + speaker emotion (25维)
输出: listener emotion (25维)
支持 CVAE 多样性生成 (K=10)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EmotionGate(nn.Module):
    """情感门控: 用 speaker emotion 调制隐藏特征"""

    def __init__(self, emotion_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        gate = self.fc(emotion)  # [B, T, hidden]
        return hidden * gate


class CVAEHead(nn.Module):
    """
    条件变分自编码器头 - 用于生成多样性候选序列

    训练时: 编码器从 listener GT + condition → z (后验)
    推理时: 从先验 N(0, I) 采样 z
    """

    def __init__(self, condition_dim: int, target_dim: int, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # 后验编码器: condition + target → mu, logvar
        self.posterior_encoder = nn.Sequential(
            nn.Linear(condition_dim + target_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.posterior_mu = nn.Linear(128, latent_dim)
        self.posterior_logvar = nn.Linear(128, latent_dim)

        # 先验编码器: condition → mu, logvar
        self.prior_encoder = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.prior_mu = nn.Linear(128, latent_dim)
        self.prior_logvar = nn.Linear(128, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        condition: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            condition: [B, T, condition_dim] encoder 输出
            target: [B, T, 25] listener GT (训练时)
        Returns:
            z, posterior_mu, posterior_logvar, prior_mu, prior_logvar
        """
        # 先验
        prior_h = self.prior_encoder(condition)
        prior_mu = self.prior_mu(prior_h)
        prior_logvar = self.prior_logvar(prior_h)

        if target is not None:
            # 训练: 后验
            posterior_input = torch.cat([condition, target], dim=-1)
            posterior_h = self.posterior_encoder(posterior_input)
            posterior_mu = self.posterior_mu(posterior_h)
            posterior_logvar = self.posterior_logvar(posterior_h)
            z = self.reparameterize(posterior_mu, posterior_logvar)
            return z, posterior_mu, posterior_logvar, prior_mu, prior_logvar
        else:
            # 推理: 从先验采样
            z = self.reparameterize(prior_mu, prior_logvar)
            return z, None, None, prior_mu, prior_logvar


class EmotionReactionTransformer(nn.Module):
    """
    情感反应预测 Transformer + CVAE

    输入: speaker audio [B, T, 768] + speaker emotion [B, T, 25]
    输出: listener emotion [B, T, 25]
    """

    def __init__(
        self,
        audio_dim: int = 768,
        emotion_dim: int = 25,
        output_dim: int = 25,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_audio: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_audio = use_audio

        # 输入投影
        if use_audio:
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        self.emotion_proj = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 融合层
        if use_audio:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.emotion_gate = EmotionGate(emotion_dim, hidden_dim)

        # 位置编码
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CVAE 头
        self.cvae = CVAEHead(
            condition_dim=hidden_dim,
            target_dim=output_dim,
            latent_dim=latent_dim,
        )

        # 解码器: hidden + z → output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # 时序平滑层
        self.temporal_smooth = nn.Conv1d(
            output_dim, output_dim, kernel_size=5, padding=2, groups=1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(
        self,
        audio: torch.Tensor,
        emotion: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码 speaker 特征
        Returns: [B, T, hidden_dim]
        """
        emo_h = self.emotion_proj(emotion)

        if self.use_audio:
            audio_h = self.audio_proj(audio)

            # 如果某些样本没有音频, 用零填充
            if has_audio is not None:
                audio_mask = has_audio.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
                audio_h = audio_h * audio_mask

            fused = self.fusion(torch.cat([audio_h, emo_h], dim=-1))
            fused = self.emotion_gate(fused, emotion)
        else:
            fused = emo_h

        fused = self.pos_enc(fused)

        # 生成 key_padding_mask (True = 忽略的位置)
        key_padding_mask = ~mask if mask is not None else None

        hidden = self.transformer_encoder(fused, src_key_padding_mask=key_padding_mask)
        return hidden

    def forward(
        self,
        audio: torch.Tensor,
        emotion: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向传播

        Returns:
            pred: [B, T, 25] 预测的 listener emotion
            posterior_mu, posterior_logvar: 后验参数
            prior_mu, prior_logvar: 先验参数
        """
        hidden = self.encode(audio, emotion, mask, has_audio)

        # CVAE
        z, post_mu, post_logvar, prior_mu, prior_logvar = self.cvae(hidden, target)

        # 解码
        dec_input = torch.cat([hidden, z], dim=-1)
        pred = self.decoder(dec_input)

        # 时序平滑
        pred_smooth = self.temporal_smooth(pred.transpose(1, 2)).transpose(1, 2)

        # mask 处理
        if mask is not None:
            pred_smooth = pred_smooth * mask.unsqueeze(-1).float()

        return {
            "pred": pred_smooth,
            "posterior_mu": post_mu,
            "posterior_logvar": post_logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
        }

    @torch.no_grad()
    def generate(
        self,
        audio: torch.Tensor,
        emotion: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
        num_candidates: int = 10,
    ) -> torch.Tensor:
        """
        推理时生成 K 条多样性候选

        Returns: [B, K, T, 25]
        """
        self.eval()
        hidden = self.encode(audio, emotion, mask, has_audio)
        B, T, H = hidden.shape
        results = []

        for _ in range(num_candidates):
            z, _, _, _, _ = self.cvae(hidden, target=None)
            dec_input = torch.cat([hidden, z], dim=-1)
            pred = self.decoder(dec_input)
            pred = self.temporal_smooth(pred.transpose(1, 2)).transpose(1, 2)
            if mask is not None:
                pred = pred * mask.unsqueeze(-1).float()
            results.append(pred)

        return torch.stack(results, dim=1)  # [B, K, T, 25]


# ============================================================================
# 保留旧模型兼容 (用于加载旧权重)
# ============================================================================

class MultiModalFaceFormer(nn.Module):
    """旧模型 (保留兼容), 输出58维3DMM"""

    def __init__(
        self,
        audio_dim=768, emotion_dim=25, hidden_dim=256,
        emotion_hidden_dim=64, output_dim=58, num_heads=4,
        num_layers=4, dropout=0.1, use_cross_attention=False,
        use_emotion_gate=True,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_cross_attention = use_cross_attention
        self.use_emotion_gate = use_emotion_gate

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.emotion_proj = nn.Sequential(
            nn.Linear(emotion_dim, emotion_hidden_dim), nn.ReLU(),
            nn.Linear(emotion_hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        if use_emotion_gate:
            self.emotion_gate = nn.Sequential(
                nn.Linear(emotion_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            )
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.noise_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, audio, emotion, mask=None):
        a = self.audio_proj(audio)
        e = self.emotion_proj(emotion)
        fused = self.fusion(torch.cat([a, e], dim=-1))
        if self.use_emotion_gate:
            gate = self.emotion_gate(emotion)
            fused = fused * gate
        fused = self.pos_enc(fused)
        if self.training:
            fused = fused + self.noise_alpha * torch.randn_like(fused)
        key_padding_mask = ~mask if mask is not None else None
        h = self.transformer_encoder(fused, src_key_padding_mask=key_padding_mask)
        return self.output_head(h)
