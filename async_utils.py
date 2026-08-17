"""Helpers for awaiting external services with a clear timeout."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar


RESULT = TypeVar("RESULT")


class ServiceTimeoutError(TimeoutError):
    """Raised when an external service does not respond in time."""


async def run_async(
    awaitable: Awaitable[RESULT],
    *,
    timeout_seconds: float,
    operation: str,
) -> RESULT:
    """Await a service call without letting it wait indefinitely."""
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise ServiceTimeoutError(
            f"{operation} không phản hồi trong {timeout_seconds:g} giây. "
            "Hãy kiểm tra kết nối mạng và thử lại."
        ) from exc
