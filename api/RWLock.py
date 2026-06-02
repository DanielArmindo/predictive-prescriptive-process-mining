import asyncio


# Reader–Writer Lock (RWLock)
class RWLock:
    def __init__(self):
        self._readers = 0
        self._readers_lock = asyncio.Lock()
        self._resource_lock = asyncio.Lock()

    async def acquire_read(self):
        # The first reader blocks the resource
        async with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                await self._resource_lock.acquire()

    async def release_read(self):
        async with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                self._resource_lock.release()

    async def acquire_write(self):
        await self._resource_lock.acquire()

    def release_write(self):
        self._resource_lock.release()
