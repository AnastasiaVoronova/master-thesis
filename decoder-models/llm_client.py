import logging
from typing import Optional, Dict, Any
import httpx
import json
from config.config import settings
from datetime import datetime


info_logger = logging.getLogger("llm_info")
info_logger.setLevel(logging.INFO)
info_formatter = logging.Formatter('%(message)s')

info_console = logging.StreamHandler()
info_console.setFormatter(info_formatter)
info_logger.addHandler(info_console)

info_file = logging.FileHandler("llm_client.log", encoding="utf-8")
info_file.setFormatter(info_formatter)
info_logger.addHandler(info_file)


error_logger = logging.getLogger("llm_error")
error_logger.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

error_console = logging.StreamHandler()
error_console.setFormatter(error_formatter)
error_logger.addHandler(error_console)

error_file = logging.FileHandler("llm_client_error.log", encoding="utf-8")
error_file.setFormatter(error_formatter)
error_logger.addHandler(error_file)


class LLMClient:

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0,
                model: str = "deepseek-chat"):
        self.base_url = (base_url or settings.model_api_url).rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.model = model

    async def chat(
        self,
        text: str,
        system_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "prefix": True,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if metadata:
            payload["metadata"] = metadata

        try:
            r = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {settings.model_api_key}"}
            )
            r.raise_for_status()
            data = r.json()

            usage = data.get("usage", {})
            cache_hit = usage.get("prompt_cache_hit_tokens")
            cache_miss = usage.get("prompt_cache_miss_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            log_entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "cache_hit": cache_hit,
                "cache_miss": cache_miss,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            info_logger.info(json.dumps(log_entry, ensure_ascii=False))

            reply = data.get("choices", [{}])[0].get("message", {}).get("content")
            if not reply:
                error_logger.error(f"Пустой ответ от LLM: {data}")
                return "⚠️ Ошибка: пустой ответ от LLM"

            return reply

        except httpx.RequestError as e:
            error_logger.error(f"Request failed: {e}")
            return "⚠️ Ошибка соединения с LLM. Попробуй позже."
        except httpx.HTTPStatusError as e:
            error_logger.error(f"HTTP error: {e.response.status_code} {e.response.text}")
            return "⚠️ LLM вернул ошибку. Попробуй позже."



llm_client = LLMClient()
