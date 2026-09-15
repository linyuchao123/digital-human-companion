"""ASR transcript evaluation without retaining audio or provider credentials."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict


def tokenize(text: str) -> list[str]:
    """Chinese characters are individual tokens; latin text is split into words."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", normalized)


def error_counts(reference: str, hypothesis: str) -> dict[str, int | float]:
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    if not ref:
        raise ValueError("reference_empty")
    dp = [[(j, 0, 0, j) for j in range(len(hyp) + 1)]]
    for i in range(1, len(ref) + 1):
        row = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                row.append(dp[i - 1][j - 1])
                continue
            sub, delete, insert = dp[i - 1][j - 1], dp[i - 1][j], row[j - 1]
            options = ((sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                       (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                       (insert[0] + 1, insert[1], insert[2], insert[3] + 1))
            row.append(min(options, key=lambda item: item[0]))
        dp.append(row)
    distance, substitutions, deletions, insertions = dp[-1][-1]
    return {"reference_tokens": len(ref), "hypothesis_tokens": len(hyp),
            "substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "errors": distance, "cer": distance / len(ref)}


def summarize(results: list[dict]) -> dict:
    if not results:
        raise ValueError("results_empty")
    def totals(items):
        reference = sum(item["reference_tokens"] for item in items)
        errors = sum(item["errors"] for item in items)
        return {"cases": len(items), "reference_tokens": reference, "errors": errors,
                "cer": errors / reference if reference else None,
                "sentence_error_rate": sum(item["errors"] > 0 for item in items) / len(items)}
    buckets = defaultdict(list)
    for result in results:
        buckets[result.get("bucket") or "未分类"].append(result)
    return {**totals(results), "buckets": {name: totals(items) for name, items in sorted(buckets.items())}}
