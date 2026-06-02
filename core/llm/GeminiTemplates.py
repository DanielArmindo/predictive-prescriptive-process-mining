from typing import Optional
from google import genai
import asyncio
from .prompts.gemini_templates import prompt
import json
from pathlib import Path


class GeminiTemplates:
    def __init__(self, model="gemini-3-flash-preview", timeout: int = 240):
        self.model = model
        self.timeout = timeout
        self.failed_logs: list[tuple[str, dict, Exception]] = []

        self.client = genai.Client()
        self.max_concurrent_per_instance = 3

    async def parse_logs(self, logs: dict[str, list[str]]) -> dict:
        self.failed_logs.clear()
        queue = asyncio.Queue()

        # Puts all tasks in the queue
        for activity, content_list in logs.items():
            queue.put_nowait((activity, content_list))

        results_dict: dict[str, Optional[dict]] = {}
        results_lock = asyncio.Lock()

        sem = asyncio.Semaphore(self.max_concurrent_per_instance)

        async def worker():
            while True:
                try:
                    key, activity_content = await queue.get()
                except asyncio.CancelledError:
                    break

                try:
                    # Outgoing call
                    async with sem:
                        template = await self._call_gemini(activity_content)

                    async with results_lock:
                        results_dict[key] = template

                    queue.task_done()

                except Exception as e:
                    async with results_lock:
                        results_dict[key] = None
                    self.failed_logs.append((key, activity_content, e))
                    queue.task_done()

                except asyncio.CancelledError:
                    queue.task_done()
                    raise

        # Create workers
        num_workers = self.max_concurrent_per_instance
        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

        try:
            await queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        return results_dict

    def get_failed_logs(self) -> list[tuple[str, dict, Exception]]:
        return list(self.failed_logs)

    async def _call_gemini(self, content: list[str]) -> dict:
        content_formated = ""
        for item in content:
            content_formated += f"{item}\n\n---\n\n"

        final_content = prompt.format(texts=content_formated)

        try:
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model, 
                    contents=final_content,
                    config={
                        "response_mime_type": "application/json",
                    }
                ),
                timeout=self.timeout
            )

            if not response or response.text is None:
                raise ValueError("JSON data is invalid")

            try:
                result_dict = json.loads(response.text)
                return result_dict
            except json.JSONDecodeError as je:
                raise ValueError(f"Erro ao fazer parse do JSON retornado: {je}")

        except asyncio.TimeoutError as te:
            raise TimeoutError(f"Timeout após {self.timeout}s ao chamar O GEMINI") from te
        except Exception as e:
            raise RuntimeError(f"Erro na chamada ao GEMINI: {e}") from e


    def dump_final_content(self, content: list[str], file_path: str, act: str) -> None:
        """
        Gera o mesmo final_content usado no _call_gemini e escreve num ficheiro.
        Debug.
        """
        content_formated = ""
        for item in content:
            content_formated += f"{item}\n\n---\n\n"

        final_content = prompt.format(texts=content_formated)

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as f:
            f.write(f"Atividade: {act}\n")
            f.write(final_content)
