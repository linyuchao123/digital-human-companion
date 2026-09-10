#!/usr/bin/env python3
"""检查官方数字人评测资产是否完整。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.avatar.evaluation import inspect_evaluation_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查数字人官方评测资产")
    parser.add_argument(
        "--val-root",
        type=Path,
        default=PROJECT_ROOT / "digital_human_engine" / "val",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=PROJECT_ROOT / "数字人面部行为驱动模型验证脚本" / "person_specific_val.csv",
    )
    parser.add_argument(
        "--neighbor-matrix",
        type=Path,
        default=(
            PROJECT_ROOT
            / "数字人面部行为驱动模型验证脚本"
            / "person_specific_masked_neighbour_emotion_val.npy"
        ),
    )
    parser.add_argument("--target-len", type=int, default=750)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出全部缺失和无效文件；默认只显示前 10 个示例",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_evaluation_assets(
        val_root=args.val_root,
        index_csv=args.index_csv,
        neighbor_matrix=args.neighbor_matrix,
        target_len=args.target_len,
    )
    payload = report.to_dict() if args.verbose else report.to_summary_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
