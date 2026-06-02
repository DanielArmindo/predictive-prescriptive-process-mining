import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import fuzz
from .prompts.ollama_extractor_system import prompt as prompt_system


class OllamaContentExtractor:
    def __init__(self, model="llama3.1:8b", timeout: int = 120, similarity_threshold: float = 90):
        self.model = model
        self.timeout = timeout
        self.similarity_threshold = similarity_threshold
        self.failed_logs: List[Tuple[str, Exception]] = []
        self.template_cache: Dict[str, Any] = {}
        self.cache_lock = asyncio.Lock()

        self.urls = [url.strip() for url in os.getenv("LLM_URL", "").split(",") if url.strip()]
        if not self.urls:
            raise ValueError("A variável de ambiente LLM_URL não está definida ou está vazia.")

        self.max_concurrent_per_instance = 1


    async def parse_logs(self, history: list[str], template_metadata: dict) -> dict[str, Optional[dict]]:
        self.failed_logs.clear()
        self.template_cache.clear()
        queue = asyncio.Queue()

        # Puts all tasks in the queue
        for content in history:
            queue.put_nowait((content))

        # System Prompt 
        system_prompt = self._get_system_prompt(template_metadata)

        results_dict: Dict[str, Optional[dict]] = {}
        results_lock = asyncio.Lock()

        async def worker(base_url: str):
            sem = asyncio.Semaphore(self.max_concurrent_per_instance)
            while True:
                try:
                    # Attempting to retrieve an item from the queue — can be canceled!
                    key = await queue.get()
                except asyncio.CancelledError:
                    # If the worker is canceled before picking up an item, Don't call task_done()
                    break

                try:
                    # Cache validation
                    async with self.cache_lock:
                        if key in self.template_cache:
                            result = self.template_cache[key]
                            async with results_lock:
                                results_dict[key] = result
                            queue.task_done()
                            continue

                        found = False
                        for cached_key, template in self.template_cache.items():
                            if fuzz.ratio(cached_key, key) >= self.similarity_threshold:
                                self.template_cache[key] = template
                                result = template
                                async with results_lock:
                                    results_dict[key] = result
                                found = True
                                break
                        if found:
                            queue.task_done()
                            continue

                    # Outgoing call
                    async with sem:
                        response = await self._call_ollama(base_url, key, system_prompt)

                    async with self.cache_lock:
                        if key not in self.template_cache:
                            self.template_cache[key] = response

                    async with results_lock:
                        results_dict[key] = response

                    queue.task_done()

                except Exception as e:
                    async with results_lock:
                        results_dict[key] = None
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


    def get_failed_logs(self) -> List[Tuple[str, Exception]]:
        return list(self.failed_logs)


    def get_cache_template(self) -> Dict[str, Any]:
        return self.template_cache


    async def _call_ollama(self, base_url: str, log_line: str, system_prompt: str) -> Dict[Any, Any]:
        formatted_prompt = """Analisa o seguinte texto:\n\n{content}\n\n---\n\nExtrai os valores e retorne em JSON."""
        formatted_prompt = formatted_prompt.format(content=log_line)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }

        try:
            async with asyncio.timeout(self.timeout):
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{base_url}/api/chat",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        ssl=False
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

            raw_response = data['message']["content"]
            if not raw_response:
                raise RuntimeError("Resposta vazia do Ollama")

            try:
                data = json.loads(raw_response)
                # data["original"] = key_log
                return data
            except json.JSONDecodeError as je:
                print(f"[JSON ERROR] Resposta inválida: {raw_response[:200]}...")
                raise ValueError(f"Erro ao decodificar JSON: {je}") from je

        except asyncio.TimeoutError as te:
            raise TimeoutError(f"Timeout após {self.timeout}s ao chamar {base_url}") from te
        except Exception as e:
            raise RuntimeError(f"Erro na chamada ao Ollama ({base_url}): {e}") from e


    def _get_system_prompt(self, metadata: dict) -> str:
        # Extract information DIRECTLY from the metadata
        template_text = metadata.get("template_text", "")
        variable_blocks = metadata.get("variable_blocks", [])
        
        # Build statements by parameter (using ONLY what is in the metadata)
        param_instructions = []
        for block in variable_blocks:
            pos = block["position"]
            examples = block.get("examples", [])
            inferred_type = block.get("inferred_type", "unknown")
            
            # Format examples directly from the metadata
            examples_str = ", ".join(f'"{e}"' for e in examples[:5])  # Max 5 exemplos
            
            instruction = f"""Parameter {pos} ({inferred_type}):\n- Sample values: {examples_str}\n"""
            param_instructions.append(instruction)
        
        # Set up the full system prompt
        system_prompt = prompt_system.format(template=template_text, parameters_examples=''.join(param_instructions))

        return system_prompt
