"""Bounded recovery for existing read-only tools; never retries mutations."""
import asyncio
import math
from time import monotonic

import httpx


async def execute_read(operation, *, timeout, attempts=2):
    """Share one deadline across at most two transient-network attempts.

    Cancellation, unavailable/configuration errors and invalid data propagate.
    The caller owns arguments: no model-provided URLs or executable commands.
    """
    if not isinstance(attempts, int) or not 1 <= attempts <= 2 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('invalid execution budget')
    deadline = monotonic() + timeout
    for index in range(attempts):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('read budget exhausted')
        try:
            return await asyncio.wait_for(operation(), timeout=remaining)
        except (httpx.TransportError, TimeoutError):
            if index + 1 == attempts or deadline - monotonic() <= 0.15:
                raise
            await asyncio.sleep(0.1)
