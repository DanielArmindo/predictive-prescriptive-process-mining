import asyncio
import os
from sklearn.metrics.pairwise import cosine_similarity


class OllamaContextParser:
    def __init__(self, model="all-minilm:l6-v2", timeout: int = 30):
        self.model = model
        self.timeout = timeout
        self.failed_logs: list[tuple[tuple[str, str, str], Exception]] = []
        self.cache_lock = asyncio.Lock()

        self.urls = [url.strip() for url in os.getenv("LLM_URL", "").split(",") if url.strip()]
        if not self.urls:
            raise ValueError("A variável de ambiente LLM_URL não está definida ou está vazia.")

        self.max_concurrent_per_instance = 1


    async def parse_context(self, history: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], float]:
        self.failed_logs.clear()
        queue = asyncio.Queue()

        # Puts all tasks in the queue
        for content in history:
            queue.put_nowait(content)

        results_dict: dict[tuple[str, str, str], float] = {}
        results_lock = asyncio.Lock()

        async def worker(base_url: str):
            sem = asyncio.Semaphore(self.max_concurrent_per_instance)
            while True:
                try:
                    item1, item2, item3 = await queue.get()
                    key = (item1, item2, item3)
                except asyncio.CancelledError:
                    break

                try:
                    # Outgoing call
                    async with sem:
                        emb1 = await self._call_ollama(base_url, item1)
                        emb2 = await self._call_ollama(base_url, item2)
                        prob = cosine_similarity(emb1, emb2)[0][0] * 100

                    async with results_lock:
                        results_dict[key] = prob

                    queue.task_done()

                except Exception as e:
                    async with results_lock:
                        results_dict[key] = float(0)
                    self.failed_logs.append((key, e))
                    queue.task_done()

                except asyncio.CancelledError:
                    queue.task_done()
                    raise

        # Create workers
        workers = [asyncio.create_task(worker(url)) for url in self.urls]

        try:
            await queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        return results_dict


    async def _call_ollama(self, base_url: str, content: str) -> list:
        payload = {
            "model": self.model,
            "prompt": content,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }

        try:
            async with asyncio.timeout(self.timeout):
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{base_url}/api/embeddings",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        ssl=False
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

            raw_response = data['embedding']

            if not raw_response:
                raise RuntimeError("Resposta vazia do Ollama")

            return [raw_response]

        except asyncio.TimeoutError as te:
            raise TimeoutError(f"Timeout após {self.timeout}s ao chamar {base_url}") from te
        except Exception as e:
            raise RuntimeError(f"Erro na chamada ao Ollama ({base_url}): {e}") from e
