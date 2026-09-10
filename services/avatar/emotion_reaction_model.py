"""正式情感反应 Transformer 的加载与推理适配器。"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .model_assets import inspect_face_driver_checkpoint, resolve_checkpoint_path

try:
    import torch
except ImportError:  # pragma: no cover - 由无 ML 依赖的轻量运行环境触发
    torch = None


MODEL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "digital_human_engine"
    / "train.jioaben"
    / "model.py"
)


class EmotionReactionModelError(RuntimeError):
    """正式情感反应模型不可用或输入不合法。"""


@dataclass(frozen=True)
class EmotionReactionMetadata:
    epoch: int | None
    val_loss: float | None
    parameter_count: int
    device: str
    use_audio: bool


def validate_emotion_sequence(sequence: np.ndarray) -> np.ndarray:
    """将输入规范为有限的 float32 `[T, 25]` 情感序列。"""
    value = np.asarray(sequence, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 25:
        raise EmotionReactionModelError(
            f"情感序列必须为 [T, 25]，实际为 {value.shape}"
        )
    if value.shape[0] == 0:
        raise EmotionReactionModelError("情感序列不能为空")
    if not np.isfinite(value).all():
        raise EmotionReactionModelError("情感序列包含 NaN 或无穷值")
    return value


def _select_device(requested: str) -> Any:
    if torch is None:
        raise EmotionReactionModelError(
            "PyTorch 未安装，请使用 Python 3.11/3.12 的 ml 运行环境"
        )
    requested = requested.strip().lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _load_model_class() -> type:
    """从原始训练代码加载模型类，避免带点目录名造成导入冲突。"""
    if not MODEL_SOURCE.is_file():
        raise EmotionReactionModelError(f"模型定义不存在: {MODEL_SOURCE}")
    module_name = "digital_xinyu_emotion_reaction_definition"
    spec = importlib.util.spec_from_file_location(module_name, MODEL_SOURCE)
    if spec is None or spec.loader is None:
        raise EmotionReactionModelError("无法创建正式模型定义的加载器")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.EmotionReactionTransformer


class EmotionReactionModel:
    """用正式 checkpoint 预测陪伴数字人的倾听情绪序列。"""

    def __init__(self, checkpoint_path: Path | None = None, device: str | None = None):
        if torch is None:
            raise EmotionReactionModelError(
                "PyTorch 未安装，请使用 Python 3.11/3.12 的 ml 运行环境"
            )

        self.checkpoint_path = checkpoint_path or resolve_checkpoint_path()
        if checkpoint_path is None:
            asset_status = inspect_face_driver_checkpoint()
            if not asset_status.ready:
                raise EmotionReactionModelError(asset_status.message)
        elif not self.checkpoint_path.is_file():
            raise EmotionReactionModelError(f"模型权重不存在: {self.checkpoint_path}")

        self.device = _select_device(device or os.getenv("DEVICE", "auto"))
        try:
            checkpoint = torch.load(
                str(self.checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
            required = {"model_state_dict", "config", "stats"}
            missing = required - checkpoint.keys()
            if missing:
                raise EmotionReactionModelError(
                    f"正式模型权重缺少字段: {', '.join(sorted(missing))}"
                )

            config = checkpoint["config"]
            model_class = _load_model_class()
            self.model = model_class(
                audio_dim=768,
                emotion_dim=25,
                output_dim=25,
                hidden_dim=config["hidden_dim"],
                latent_dim=config["latent_dim"],
                num_heads=config["num_heads"],
                num_layers=config["num_layers"],
                dropout=0.0,
                use_audio=config.get("use_audio", True),
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.model.eval()
            self.stats = checkpoint["stats"]
            self.metadata = EmotionReactionMetadata(
                epoch=checkpoint.get("epoch"),
                val_loss=checkpoint.get("val_loss"),
                parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
                device=str(self.device),
                use_audio=bool(config.get("use_audio", True)),
            )
            self._inference_lock = threading.Lock()
        except EmotionReactionModelError:
            raise
        except Exception as exc:
            raise EmotionReactionModelError(
                f"正式情感反应模型加载失败: {type(exc).__name__}: {exc}"
            ) from exc

    def predict(
        self,
        speaker_emotion: np.ndarray,
        *,
        num_candidates: int = 1,
        seed: int | None = None,
    ) -> np.ndarray:
        """生成 `[K, T, 25]` 倾听者情绪；当前实时链路使用无音频模式。"""
        if not 1 <= num_candidates <= 10:
            raise EmotionReactionModelError("候选数量必须在 1 到 10 之间")
        sequence = validate_emotion_sequence(speaker_emotion)
        speaker_stats = self.stats["speaker_emotion"]
        listener_stats = self.stats["listener_emotion"]
        normalized = (
            (sequence - speaker_stats["mean"])
            / (speaker_stats["std"] + 1e-8)
        ).astype(np.float32)

        emotion_tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        audio_tensor = torch.zeros(
            1, sequence.shape[0], 768, dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.ones(
            1, sequence.shape[0], dtype=torch.bool, device=self.device
        )
        has_audio_tensor = torch.zeros(1, dtype=torch.bool, device=self.device)

        with self._inference_lock, torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
            prediction = self.model.generate(
                audio_tensor,
                emotion_tensor,
                mask=mask_tensor,
                has_audio=has_audio_tensor,
                num_candidates=num_candidates,
            )

        result = prediction.detach().cpu().numpy()[0]
        result = (
            result * listener_stats["std"] + listener_stats["mean"]
        ).astype(np.float32)
        if not np.isfinite(result).all():
            raise EmotionReactionModelError("正式模型输出包含 NaN 或无穷值")
        return result
