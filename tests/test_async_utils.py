import asyncio

import pytest

from async_utils import ServiceTimeoutError, run_async


def test_async_call_completes():
    async def scenario():
        return await run_async(
            asyncio.sleep(0, result="ok"),
            timeout_seconds=1,
            operation="Dịch vụ thử nghiệm",
        )

    assert asyncio.run(scenario()) == "ok"


def test_async_call_has_a_clear_timeout():
    async def scenario():
        await run_async(
            asyncio.sleep(0.05),
            timeout_seconds=0.01,
            operation="Dịch vụ thử nghiệm",
        )

    with pytest.raises(ServiceTimeoutError, match="không phản hồi"):
        asyncio.run(scenario())
