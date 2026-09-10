"""数字人情感反应模型的官方评测资产预检。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .model_assets import inspect_face_driver_checkpoint


RECOLA_ROLE_MAP = {
    "P25": "P1",
    "P26": "P2",
    "P41": "P1",
    "P42": "P2",
    "P45": "P1",
    "P46": "P2",
}


@dataclass(frozen=True)
class EvaluationSample:
    speaker_path: str
    listener_path: str
    direction: str


@dataclass
class EvaluationAssetReport:
    ready: bool = False
    pair_count: int = 0
    expanded_sample_count: int = 0
    valid_emotion_files: int = 0
    neighbor_matrix_shape: tuple[int, ...] | None = None
    checkpoint_ready: bool = False
    missing_files: list[str] = field(default_factory=list)
    invalid_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_summary_dict(self, *, example_limit: int = 10) -> dict[str, object]:
        """返回适合终端展示的摘要，避免缺失资产时刷出数百行路径。"""
        return {
            "ready": self.ready,
            "pair_count": self.pair_count,
            "expanded_sample_count": self.expanded_sample_count,
            "valid_emotion_files": self.valid_emotion_files,
            "missing_file_count": len(self.missing_files),
            "missing_file_examples": self.missing_files[:example_limit],
            "invalid_file_count": len(self.invalid_files),
            "invalid_file_examples": self.invalid_files[:example_limit],
            "neighbor_matrix_shape": self.neighbor_matrix_shape,
            "checkpoint_ready": self.checkpoint_ready,
            "errors": self.errors,
        }


def load_evaluation_samples(index_csv: Path) -> list[EvaluationSample]:
    """按官方要求将每一对样本展开为正向和反向两个样本。"""
    with index_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    if len(rows) <= 1:
        raise ValueError("官方样本索引为空")

    pairs: list[tuple[str, str]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) < 3 or not row[1].strip() or not row[2].strip():
            raise ValueError(f"官方样本索引第 {line_number} 行格式不正确")
        pairs.append((row[1].strip(), row[2].strip()))

    forward = [
        EvaluationSample(speaker, listener, "forward")
        for speaker, listener in pairs
    ]
    reverse = [
        EvaluationSample(listener, speaker, "reverse")
        for speaker, listener in pairs
    ]
    return forward + reverse


def emotion_csv_path(val_root: Path, video_path: str) -> Path:
    """将官方视频相对路径转换为实际 25 维情绪 CSV 路径。"""
    parts = Path(video_path.replace("\\", "/")).parts
    if len(parts) != 4:
        raise ValueError(f"不支持的样本路径: {video_path}")
    dataset, session, role_or_video, clip_id = parts
    if dataset == "NoXI":
        role_map = {"Expert_video": "P1", "Novice_video": "P2"}
        if role_or_video not in role_map:
            raise ValueError(f"未知 NoXI 视频角色: {role_or_video}")
        role = role_map[role_or_video]
    elif dataset == "RECOLA":
        if role_or_video not in RECOLA_ROLE_MAP:
            raise ValueError(f"未知 RECOLA 人员编号: {role_or_video}")
        role = RECOLA_ROLE_MAP[role_or_video]
    else:
        raise ValueError(f"未知评测数据集: {dataset}")
    return val_root / "Emotion" / dataset / session / role / f"{clip_id}.csv"


def _validate_emotion_csv(path: Path, target_len: int) -> str | None:
    if path.stat().st_size == 0:
        return f"{path}: 文件为空"
    try:
        value = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    except Exception as exc:
        return f"{path}: 无法读取（{type(exc).__name__}）"
    if value.ndim != 2 or value.shape[1] != 25:
        return f"{path}: 期望 [T, 25]，实际 {value.shape}"
    if value.shape[0] < target_len:
        return f"{path}: 少于 {target_len} 帧，实际 {value.shape[0]}"
    if not np.isfinite(value[:target_len]).all():
        return f"{path}: 前 {target_len} 帧包含 NaN 或无穷值"
    return None


def inspect_evaluation_assets(
    *,
    val_root: Path,
    index_csv: Path,
    neighbor_matrix: Path,
    target_len: int = 750,
    require_checkpoint: bool = True,
) -> EvaluationAssetReport:
    """完整检查正式评测开始前所需的索引、标签、矩阵和权重。"""
    report = EvaluationAssetReport()
    if not index_csv.is_file():
        report.errors.append(f"官方样本索引不存在: {index_csv}")
        return report
    try:
        samples = load_evaluation_samples(index_csv)
    except (OSError, ValueError) as exc:
        report.errors.append(str(exc))
        return report

    report.pair_count = len(samples) // 2
    report.expanded_sample_count = len(samples)
    unique_paths: dict[str, Path] = {}
    for sample in samples:
        for video_path in (sample.speaker_path, sample.listener_path):
            try:
                unique_paths[video_path] = emotion_csv_path(val_root, video_path)
            except ValueError as exc:
                report.errors.append(str(exc))

    for path in unique_paths.values():
        if not path.is_file():
            report.missing_files.append(str(path))
            continue
        error = _validate_emotion_csv(path, target_len)
        if error:
            report.invalid_files.append(error)
        else:
            report.valid_emotion_files += 1

    if not neighbor_matrix.is_file():
        report.errors.append(f"官方邻接矩阵不存在: {neighbor_matrix}")
    else:
        try:
            matrix = np.load(neighbor_matrix, mmap_mode="r", allow_pickle=False)
            report.neighbor_matrix_shape = tuple(matrix.shape)
            expected_shape = (len(samples), len(samples))
            if matrix.shape != expected_shape:
                report.errors.append(
                    f"邻接矩阵形状应为 {expected_shape}，实际为 {matrix.shape}"
                )
        except Exception as exc:
            report.errors.append(
                f"官方邻接矩阵无法读取: {type(exc).__name__}: {exc}"
            )

    checkpoint_status = inspect_face_driver_checkpoint()
    report.checkpoint_ready = checkpoint_status.ready
    if require_checkpoint and not checkpoint_status.ready:
        report.errors.append(checkpoint_status.message)

    report.ready = not (
        report.errors or report.missing_files or report.invalid_files
    ) and (report.checkpoint_ready or not require_checkpoint)
    return report
