#!/usr/bin/env python3
"""Run the production ASR route against a private, human-labelled WAV manifest."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.asr.evaluation import error_counts, summarize


def load_cases(manifest: Path):
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取合法的 JSON 清单") from exc
    if not isinstance(data, list) or not data or len(data) > 500:
        raise ValueError("清单必须是包含 1 到 500 条样本的数组")
    base, cases, seen = manifest.resolve().parent, [], set()
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict) or set(item) - {"id", "audio", "reference", "bucket"}:
            raise ValueError(f"第 {index} 条包含未知或错误字段")
        case_id, audio, reference = item.get("id"), item.get("audio"), item.get("reference")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"第 {index} 条 id 无效或重复")
        if not isinstance(audio, str) or not audio.lower().endswith(".wav"):
            raise ValueError(f"{case_id}: 仅支持 WAV")
        if not isinstance(reference, str) or not reference.strip() or len(reference) > 2000:
            raise ValueError(f"{case_id}: 参考文本须为 1 到 2000 字")
        candidate = base / audio
        path = candidate.resolve()
        if base not in path.parents or candidate.is_symlink():
            raise ValueError(f"{case_id}: 音频必须位于清单目录内且不能是符号链接")
        if not path.is_file():
            raise ValueError(f"{case_id}: 音频不存在")
        if not 44 <= path.stat().st_size <= 3 * 1024 * 1024:
            raise ValueError(f"{case_id}: WAV 大小须为 44 字节到 3 MiB")
        seen.add(case_id)
        cases.append((case_id, path, reference, item.get("bucket") or "未分类"))
    return cases


def run(manifest: Path, include_text=False):
    from apps.api.integrated_server import _recognize_audio
    results = []
    for case_id, audio, reference, bucket in load_cases(manifest):
        recognition_error = None
        try:
            hypothesis, provider, fallback = _recognize_audio(str(audio))
        except Exception as exc:
            hypothesis, fallback = "", None
            provider = "qwen_cloud" if os.environ.get("ASR_PROVIDER", "funasr") == "qwen" else "funasr_paraformer"
            raw_error = str(exc)
            recognition_error = raw_error if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", raw_error) else type(exc).__name__
        fallback_code = fallback if isinstance(fallback, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", fallback) else None
        metric = {**error_counts(reference, hypothesis), "id": case_id, "bucket": bucket,
                  "provider": provider, "fallback": bool(fallback),
                  "fallback_reason": fallback_code or ("provider_failed" if fallback else None),
                  "recognition_error": recognition_error}
        if include_text:
            metric.update(reference=reference, hypothesis=hypothesis)
        results.append(metric)
    return {"summary": summarize(results), "cases": results}


def main():
    parser = argparse.ArgumentParser(description="用人工标注录音评测当前生产 ASR 路径")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-cer", type=float, default=.10, help="字符错误率门槛，默认 0.10")
    parser.add_argument("--no-gate", action="store_true", help="只生成报告，不因门槛失败退出")
    parser.add_argument("--include-text", action="store_true", help="在输出中包含原文（可能含隐私）")
    parser.add_argument("--output", type=Path, help="JSON 报告路径；默认仅输出到终端")
    args = parser.parse_args()
    if not 0 <= args.max_cer <= 1:
        parser.error("--max-cer 必须在 0 到 1 之间")
    try:
        report = run(args.manifest, args.include_text)
    except Exception as exc:
        print(f"ASR 评测失败：{exc}", file=sys.stderr)
        return 1
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    cer = report["summary"]["cer"]
    if not args.no_gate and cer > args.max_cer:
        print(f"未达门槛：CER {cer:.2%} > {args.max_cer:.2%}", file=sys.stderr)
        return 2
    print(f"评测完成：{report['summary']['cases']} 条，CER {cer:.2%}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
