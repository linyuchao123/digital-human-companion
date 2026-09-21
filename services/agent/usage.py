"""Per-request model usage collection without leaking credentials or prompts."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelUsage:
    provider: str
    model: str
    request_kind: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    estimated: bool = False
    status: str = "success"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_collector: ContextVar[list[ModelUsage] | None] = ContextVar(
    "model_usage_collector", default=None
)


def begin_usage_collection() -> Token:
    return _collector.set([])


def finish_usage_collection(token: Token) -> list[dict[str, Any]]:
    rows = list(_collector.get() or [])
    _collector.reset(token)
    return [row.to_dict() for row in rows]


def estimate_tokens(parts: Iterable[str]) -> int:
    """Conservative local estimate used only when a provider omits usage."""
    text = "\n".join(part for part in parts if part)
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    non_cjk = re.sub(r"[\u3400-\u9fff\s]", "", text)
    return max(1, cjk + math.ceil(len(non_cjk) / 4))


def record_usage(
    *,
    provider: str,
    model: str,
    request_kind: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    estimated: bool = False,
    status: str = "success",
) -> None:
    collector = _collector.get()
    if collector is None:
        return
    collector.append(
        ModelUsage(
            provider=provider,
            model=model,
            request_kind=request_kind,
            input_tokens=max(0, int(input_tokens or 0)),
            output_tokens=max(0, int(output_tokens or 0)),
            cached_input_tokens=max(0, int(cached_input_tokens or 0)),
            estimated=bool(estimated),
            status=status if status in {"success", "failed"} else "failed",
        )
    )


def usage_from_payload(payload: Any) -> tuple[int, int, int] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens", 0)) if isinstance(details, dict) else 0
    try:
        return max(0, int(prompt or 0)), max(0, int(completion or 0)), max(0, int(cached or 0))
    except (TypeError, ValueError):
        return None
