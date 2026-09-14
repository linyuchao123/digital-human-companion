#!/usr/bin/env python3
"""检查数字人评测预测文件是否满足官方提交格式。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.avatar.evaluation import inspect_prediction_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查数字人官方预测文件")
    parser.add_argument(
        "--prediction",
        type=Path,
        default=(
            PROJECT_ROOT
            / "数字人面部行为驱动模型验证脚本"
            / "prediction_emotion.npy"
        ),
    )
    parser.add_argument("--samples", type=int, default=1086)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--frames", type=int, default=750)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_prediction_file(
        args.prediction,
        expected_shape=(args.samples, args.candidates, args.frames, 25),
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
