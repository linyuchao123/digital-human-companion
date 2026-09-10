#!/usr/bin/env python3
"""下载经过项目评测的本地 RAG 嵌入模型，不把权重提交到 Git。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "BAAI/bge-small-zh-v1.5"
DEFAULT_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "embedding" / "bge-small-zh-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载数字心屿 RAG 本地嵌入模型")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model_info = HfApi().model_info(args.repository, revision=args.revision)
    resolved_revision = model_info.sha
    snapshot_download(
        repo_id=args.repository,
        revision=resolved_revision,
        local_dir=destination,
        ignore_patterns=("*.bin",),
    )
    metadata = {
        "repository": args.repository,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "model_source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"path": str(destination), **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
