import asyncio
from typing import Optional, Sequence

import aiohttp


async def _async_get_request(
    session: aiohttp.ClientSession, url: str
) -> Optional[bytes]:
    try:
        resp = await session.get(url)
        return await resp.read()
    except Exception:
        return None


async def async_batch_get_request(urls: Sequence[str], timeout: float = 1.0):
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        return await asyncio.gather(*[_async_get_request(session, url) for url in urls])
