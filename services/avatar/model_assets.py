"""数字人正式模型资产的定位与完整性检查。"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "digital_human_engine" / "checkpoints_v2" / "best_model.pt"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "84fa5cabb83155d3f600fac214aadaeb79068606e329f3d8cf4640c21f475e90"
)
OFFICIAL_CHECKPOINT_SIZE = 49_289_786


@dataclass(frozen=True)
class ModelAssetStatus:
    """单个模型文件的可用性与完整性状态。"""

    filename: str
    exists: bool
    ready: bool
    integrity: str
    size_bytes: int | None = None
    sha256: str | None = None
    expected_sha256: str | None = None
    message: str = ""

    def to_public_dict(self) -> dict[str, object]:
        """返回适合状态接口展示、且不暴露本机绝对路径的数据。"""
        return asdict(self)


def resolve_checkpoint_path() -> Path:
    """读取可覆盖的权重路径，并将相对路径解析到项目根目录。"""
    configured = os.getenv("FACE_DRIVER_CHECKPOINT", "").strip()
    if not configured:
        return DEFAULT_CHECKPOINT_PATH
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def expected_checkpoint_sha256() -> str | None:
    """返回期望摘要；显式设置为空或 off 时仅检查文件是否存在。"""
    configured = os.getenv(
        "FACE_DRIVER_CHECKPOINT_SHA256", OFFICIAL_CHECKPOINT_SHA256
    ).strip()
    return None if configured.lower() in {"", "off", "none"} else configured.lower()


@lru_cache(maxsize=8)
def _sha256_for_file(path: str, size: int, mtime_ns: int) -> str:
    """按文件元数据缓存摘要，避免状态接口反复读取大权重。"""
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> ModelAssetStatus:
    """检查权重是否存在，并在提供预期值时校验大小和 SHA-256。"""
    if not path.is_file():
        return ModelAssetStatus(
            filename=path.name,
            exists=False,
            ready=False,
            integrity="missing",
            expected_sha256=expected_sha256,
            message="正式模型权重不存在",
        )

    try:
        stat = path.stat()
        if expected_size is not None and stat.st_size != expected_size:
            return ModelAssetStatus(
                filename=path.name,
                exists=True,
                ready=False,
                integrity="size_mismatch",
                size_bytes=stat.st_size,
                expected_sha256=expected_sha256,
                message="模型权重大小与正式版本不一致",
            )

        if expected_sha256 is None:
            return ModelAssetStatus(
                filename=path.name,
                exists=True,
                ready=True,
                integrity="unchecked",
                size_bytes=stat.st_size,
                message="模型权重存在，但未配置摘要校验",
            )

        actual_sha256 = _sha256_for_file(str(path), stat.st_size, stat.st_mtime_ns)
        verified = actual_sha256 == expected_sha256.lower()
        return ModelAssetStatus(
            filename=path.name,
            exists=True,
            ready=verified,
            integrity="verified" if verified else "checksum_mismatch",
            size_bytes=stat.st_size,
            sha256=actual_sha256,
            expected_sha256=expected_sha256.lower(),
            message="正式模型权重校验通过" if verified else "模型权重摘要校验失败",
        )
    except OSError as exc:
        return ModelAssetStatus(
            filename=path.name,
            exists=True,
            ready=False,
            integrity="read_error",
            expected_sha256=expected_sha256,
            message=f"模型权重无法读取: {type(exc).__name__}",
        )


def inspect_face_driver_checkpoint() -> ModelAssetStatus:
    """检查项目约定的 best_model.pt 正式权重。"""
    path = resolve_checkpoint_path()
    expected_sha256 = expected_checkpoint_sha256()
    expected_size = (
        OFFICIAL_CHECKPOINT_SIZE
        if expected_sha256 == OFFICIAL_CHECKPOINT_SHA256
        else None
    )
    return inspect_checkpoint(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
