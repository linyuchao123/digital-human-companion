from __future__ import annotations

import secrets
import time
from dataclasses import dataclass


def now_epoch_ms() -> int:
    return int(time.time() * 1000)


def new_trace_id() -> str:
    return secrets.token_hex(16)


@dataclass(frozen=True)
class TurnRef:
    trace_id: str
    session_id: str
    turn_id: int

