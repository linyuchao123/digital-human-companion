#!/usr/bin/env python3
"""
数字人面部行为驱动模型 - 核心网络
架构: Audio CNN Encoder → BiLSTM → Face Decoder (MLP)
多任务输出: 58D面部参数 + 25D情绪特征
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ModelConfig:
    """模型超参数配置"""
    # 输入
    mel_dim: int = 80           # log-mel维度
    emotion_cond_dim: int = 10  # 情绪条件维度 (8EXP + 2VA)
    # 编码器
    enc_channels: Tuple = (256, 512, 256)
    enc_kernel: int = 3
    # 时序模型
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    # 解码器
    dec_hidden: int = 512
    # 输出
    face_dim: int = 58
    emotion_dim: int = 25
    # 训练
    dropout: float = 0.1
    smoothing_alpha: float = 0.7  # IIR平滑系数


if TORCH_AVAILABLE:
    class AudioEncoder(nn.Module):
        """
        轻量CNN音频编码器
        输入: (B, T, 80) → 输出: (B, T/4, 256)
        """

        def __init__(self, cfg: ModelConfig):
            super().__init__()
            in_ch, h1, h2, out_ch = cfg.mel_dim, cfg.enc_channels[0], cfg.enc_channels[1], cfg.enc_channels[2]
            k = cfg.enc_kernel

            self.conv_layers = nn.Sequential(
                # (B, 80, T) → (B, 256, T/2)
                nn.Conv1d(in_ch, h1, kernel_size=k, stride=2, padding=k // 2),
                nn.BatchNorm1d(h1),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                # → (B, 512, T/4)
                nn.Conv1d(h1, h2, kernel_size=k, stride=2, padding=k // 2),
                nn.BatchNorm1d(h2),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                # → (B, 256, T/4)
                nn.Conv1d(h2, out_ch, kernel_size=k, stride=1, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, 80)
            x = x.permute(0, 2, 1)      # (B, 80, T)
            x = self.conv_layers(x)      # (B, 256, T/4)
            x = x.permute(0, 2, 1)      # (B, T/4, 256)
            return x

    class TemporalModel(nn.Module):
        """
        因果LSTM时序建模
        输入: (B, T, 256) → 输出: (B, T, 256)
        """

        def __init__(self, cfg: ModelConfig):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=cfg.enc_channels[-1],
                hidden_size=cfg.lstm_hidden,
                num_layers=cfg.lstm_layers,
                batch_first=True,
                dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(cfg.lstm_hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.norm(out)

    class FaceDecoder(nn.Module):
        """
        MLP面部参数解码器
        输入: 时序特征(256) + 情绪条件(10) → 58D面部参数 + 25D情绪特征
        """

        def __init__(self, cfg: ModelConfig):
            super().__init__()
            in_dim = cfg.lstm_hidden + cfg.emotion_cond_dim

            self.shared = nn.Sequential(
                nn.Linear(in_dim, cfg.dec_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.dec_hidden, 256),
                nn.ReLU(inplace=True),
            )

            # 主输出: 58D面部参数
            self.face_head = nn.Linear(256, cfg.face_dim)

            # 辅助输出: 情绪特征（多任务）
            # AU (0-14): 15维，用Sigmoid限制[0,1]
            self.au_head = nn.Sequential(
                nn.Linear(256, 15),
                nn.Sigmoid()
            )
            # VA (15-16): 2维，用Tanh限制[-1,1]
            self.va_head = nn.Sequential(
                nn.Linear(256, 2),
                nn.Tanh()
            )
            # EXP (17-24): 8维，用Softmax归一化
            self.exp_head = nn.Sequential(
                nn.Linear(256, 8),
                nn.Softmax(dim=-1)
            )

        def forward(
            self,
            feat: torch.Tensor,               # (B, T, lstm_hidden)
            emotion_cond: Optional[torch.Tensor] = None,  # (B, T, 10) or (B, 10)
        ) -> Dict[str, torch.Tensor]:
            B, T, _ = feat.shape

            if emotion_cond is None:
                emotion_cond = torch.zeros(B, T, 10, device=feat.device)
            elif emotion_cond.dim() == 2:
                emotion_cond = emotion_cond.unsqueeze(1).expand(B, T, -1)

            x = torch.cat([feat, emotion_cond], dim=-1)  # (B, T, 266)
            shared = self.shared(x)                       # (B, T, 256)

            face = self.face_head(shared)                 # (B, T, 58)
            au = self.au_head(shared)                     # (B, T, 15)
            va = self.va_head(shared)                     # (B, T, 2)
            exp = self.exp_head(shared)                   # (B, T, 8)

            # 拼装25维情绪特征
            emotion_25 = torch.cat([au, va, exp], dim=-1)  # (B, T, 25)

            return {
                "face_58": face,
                "emotion_25": emotion_25,
                "au_15": au,
                "va_2": va,
                "exp_8": exp,
            }

    class FaceReactionModel(nn.Module):
        """
        完整面部反应生成模型

        输入:
          - audio_mel: (B, T, 80) log-mel spectrogram
          - emotion_cond: (B, 10) 情绪条件向量 [来自LLMToDriver]

        输出:
          - face_58: (B, T_out, 58) 面部参数
          - emotion_25: (B, T_out, 25) 情绪特征
          - au_15, va_2, exp_8: 各子任务输出
        """

        def __init__(self, cfg: Optional[ModelConfig] = None):
            super().__init__()
            self.cfg = cfg or ModelConfig()
            self.encoder = AudioEncoder(self.cfg)
            self.temporal = TemporalModel(self.cfg)
            self.decoder = FaceDecoder(self.cfg)

        def forward(
            self,
            audio_mel: torch.Tensor,
            emotion_cond: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            # 音频编码
            enc = self.encoder(audio_mel)   # (B, T/4, 256)

            # 上采样回目标帧率（25fps已对齐，不需要额外上采样）
            # CNN stride=2x2=4，mel是25fps → 编码后是~6fps → 插值回25fps
            T_target = audio_mel.shape[1]
            if enc.shape[1] != T_target:
                enc = F.interpolate(
                    enc.permute(0, 2, 1),
                    size=T_target,
                    mode="linear",
                    align_corners=False
                ).permute(0, 2, 1)

            # 时序建模
            feat = self.temporal(enc)       # (B, T, 256)

            # 解码
            outputs = self.decoder(feat, emotion_cond)

            return outputs

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:
    class AudioEncoder:  # type: ignore[no-redef]
        def __init__(self, cfg): self.cfg = cfg
        def __call__(self, x): return np.random.rand(x.shape[0], x.shape[1]//4, 256).astype(np.float32)

    class TemporalModel:  # type: ignore[no-redef]
        def __init__(self, cfg): self.cfg = cfg
        def __call__(self, x): return x

    class FaceDecoder:  # type: ignore[no-redef]
        def __init__(self, cfg): self.cfg = cfg

    class FaceReactionModel:  # type: ignore[no-redef]
        def __init__(self, cfg=None):
            self.cfg = cfg or ModelConfig()

        def __call__(self, audio_mel, emotion_cond=None):
            T = audio_mel.shape[1] if hasattr(audio_mel, 'shape') else 75
            B = audio_mel.shape[0] if hasattr(audio_mel, 'shape') else 1
            face = np.random.rand(B, T, 58).astype(np.float32)
            emo = np.random.rand(B, T, 25).astype(np.float32)
            emo[:, :, 17:] = np.abs(emo[:, :, 17:])
            emo[:, :, 17:] /= emo[:, :, 17:].sum(axis=-1, keepdims=True)
            return {"face_58": face, "emotion_25": emo}

        def count_parameters(self): return 0
