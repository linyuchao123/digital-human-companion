"""生成数字人官方评测预测，支持资产预检、增量写盘和断点续跑。

完整输出形状为 ``[1086, 10, 750, 25]``。每对样本先按正向排列，
再按反向排列，与官方 ``person_specific_val.csv`` 和邻接矩阵保持一致。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from model import EmotionReactionTransformer
from services.avatar.evaluation import (
    emotion_csv_path,
    inspect_evaluation_assets,
    load_evaluation_samples,
)
from services.avatar.model_assets import resolve_checkpoint_path


DEFAULT_EVAL_DIR = PROJECT_ROOT / "数字人面部行为驱动模型验证脚本"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成数字人官方评测预测")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=resolve_checkpoint_path(),
        help="模型检查点路径",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=PROJECT_ROOT / "digital_human_engine" / "val",
        help="验证集根目录",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_EVAL_DIR / "person_specific_val.csv",
        help="官方 person_specific_val.csv 路径",
    )
    parser.add_argument(
        "--neighbor-matrix",
        type=Path,
        default=DEFAULT_EVAL_DIR / "person_specific_masked_neighbour_emotion_val.npy",
        help="官方邻接矩阵路径，仅用于资产预检",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_EVAL_DIR / "prediction_emotion.npy",
        help="输出 npy 路径",
    )
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--target-len", type=int, default=750)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从同名 .partial.npy 和 .progress.json 继续",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="仅生成前 N 条，用于开发冒烟验证，不属于官方评测结果",
    )
    parser.add_argument(
        "--allow-incomplete-assets",
        action="store_true",
        help="仅供冒烟验证：允许验证集其他位置存在无效文件",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境不可用")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("请求使用 MPS，但当前环境不可用")
    return torch.device(requested)


def load_emotion_csv(path: Path, target_len: int) -> tuple[np.ndarray, int]:
    """严格加载 25 维情绪特征，不再用全零数组掩盖损坏文件。"""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"情绪文件不存在或为空: {path}")
    value = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 25 or value.shape[0] == 0:
        raise ValueError(f"情绪文件应为 [T, 25]，实际 {value.shape}: {path}")
    if not np.isfinite(value).all():
        raise ValueError(f"情绪文件包含 NaN 或无穷值: {path}")
    actual_len = min(value.shape[0], target_len)
    if value.shape[0] >= target_len:
        return value[:target_len], actual_len
    padding = np.zeros((target_len - value.shape[0], 25), dtype=np.float32)
    return np.concatenate((value, padding), axis=0), actual_len


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = EmotionReactionTransformer(
        audio_dim=768,
        emotion_dim=25,
        output_dim=25,
        hidden_dim=config.get("hidden_dim", 256),
        latent_dim=config.get("latent_dim", 64),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 4),
        dropout=0.0,
        use_audio=config.get("use_audio", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("stats"), checkpoint.get("epoch", "?")


def normalization_arrays(stats):
    speaker = stats.get("speaker_emotion", {}) if stats else {}
    listener = stats.get("listener_emotion", {}) if stats else {}
    return (
        np.asarray(speaker.get("mean", np.zeros(25)), dtype=np.float32),
        np.asarray(speaker.get("std", np.ones(25)), dtype=np.float32),
        np.asarray(listener.get("mean", np.zeros(25)), dtype=np.float32),
        np.asarray(listener.get("std", np.ones(25)), dtype=np.float32),
    )


def progress_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.with_name(f"{output_path.stem}.partial.npy"),
        output_path.with_name(f"{output_path.stem}.progress.json"),
    )


def open_prediction_file(
    output_path: Path,
    shape: tuple[int, int, int, int],
    *,
    resume: bool,
) -> tuple[np.memmap, int, Path, Path]:
    partial_path, progress_path = progress_paths(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        if not partial_path.is_file() or not progress_path.is_file():
            raise RuntimeError("未找到可恢复的 .partial.npy 和 .progress.json")
        metadata = json.loads(progress_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("shape", ())) != shape:
            raise RuntimeError("断点文件形状与本次参数不一致")
        predictions = np.lib.format.open_memmap(partial_path, mode="r+")
        return predictions, int(metadata["next_index"]), partial_path, progress_path

    predictions = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    progress_path.write_text(
        json.dumps({"shape": shape, "next_index": 0}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return predictions, 0, partial_path, progress_path


def save_progress(progress_path: Path, shape: tuple[int, ...], next_index: int) -> None:
    temporary = progress_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"shape": shape, "next_index": next_index},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, progress_path)


def main() -> int:
    args = parse_args()
    if args.num_candidates <= 0 or args.target_len <= 0:
        raise ValueError("num-candidates 和 target-len 必须大于 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples 必须大于 0")
    if args.allow_incomplete_assets and args.max_samples is None:
        raise ValueError("allow-incomplete-assets 只能与 max-samples 一起使用")

    report = inspect_evaluation_assets(
        val_root=args.val_root,
        index_csv=args.val_csv,
        neighbor_matrix=args.neighbor_matrix,
        target_len=args.target_len,
        require_checkpoint=False,
    )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"模型检查点不存在: {args.checkpoint}")
    if not report.ready and not args.allow_incomplete_assets:
        print(json.dumps(report.to_summary_dict(), ensure_ascii=False, indent=2))
        print(
            "资产预检未通过，已停止生成；请先补齐验证数据。",
            file=sys.stderr,
        )
        return 2
    if not report.ready:
        print("警告：当前为非官方冒烟模式，输出不得用于评测指标。")

    all_samples = load_evaluation_samples(args.val_csv)
    samples = all_samples[: args.max_samples] if args.max_samples else all_samples
    shape = (len(samples), args.num_candidates, args.target_len, 25)
    device = select_device(args.device)
    print(f"加载模型: {args.checkpoint}（设备: {device}）")
    model, stats, epoch = load_model(args.checkpoint, device)
    print(f"模型加载完成（epoch {epoch}），输出形状 {shape}")
    sp_mean, sp_std, li_mean, li_std = normalization_arrays(stats)

    predictions, start_index, partial_path, progress_path = open_prediction_file(
        args.output_path,
        shape,
        resume=args.resume,
    )
    if start_index > len(samples):
        raise RuntimeError("断点位置超出本次样本范围")

    try:
        with torch.no_grad():
            for index in range(start_index, len(samples)):
                sample = samples[index]
                csv_path = emotion_csv_path(args.val_root, sample.speaker_path)
                speaker_emotion, actual_len = load_emotion_csv(csv_path, args.target_len)
                normalized = ((speaker_emotion - sp_mean) / (sp_std + 1e-8)).astype(np.float32)

                emotion = torch.from_numpy(normalized).unsqueeze(0).to(device)
                audio = torch.zeros(1, args.target_len, 768, device=device)
                mask = torch.zeros(1, args.target_len, dtype=torch.bool, device=device)
                mask[:, :actual_len] = True
                has_audio = torch.zeros(1, dtype=torch.bool, device=device)

                torch.manual_seed(args.seed + index)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + index)
                generated = model.generate(
                    audio,
                    emotion,
                    mask=mask,
                    has_audio=has_audio,
                    num_candidates=args.num_candidates,
                )[0].cpu().numpy()
                generated = generated * li_std + li_mean
                if actual_len < args.target_len:
                    generated[:, actual_len:, :] = generated[:, actual_len - 1 : actual_len, :]
                predictions[index] = generated.astype(np.float32)
                predictions.flush()
                save_progress(progress_path, shape, index + 1)
                print(f"样本 {index + 1}/{len(samples)} 完成（{sample.direction}）")
    except Exception:
        predictions.flush()
        print(f"生成中断，可使用 --resume 从 {progress_path} 继续。", file=sys.stderr)
        raise

    minimum = float(predictions.min())
    maximum = float(predictions.max())
    mean = float(predictions.mean())
    std = float(predictions.std())
    predictions.flush()
    del predictions
    os.replace(partial_path, args.output_path)
    progress_path.unlink(missing_ok=True)
    print(f"预测结果已保存: {args.output_path}")
    print(f"数值范围 [{minimum:.4f}, {maximum:.4f}]，均值 {mean:.4f}，标准差 {std:.4f}")
    if args.max_samples:
        print("注意：这是开发冒烟结果，不是完整官方评测文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
