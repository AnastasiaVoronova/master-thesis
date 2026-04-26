import logging
from typing import Optional, Dict, Any
import json
from openai import AsyncOpenAI, APIConnectionError, APIStatusError
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

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0,
                model: str = "deepseek-chat", thinking=None):
        self.client = AsyncOpenAI(
            api_key=settings.model_api_key,
            base_url=(base_url or settings.model_api_url).rstrip("/"),
            timeout=timeout,
        )
        self.model = model
        self.thinking = thinking  # None | True | False

    async def chat(
        self,
        text: str,
        system_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:

        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if self.thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if self.thinking else "disabled"}}

        if metadata:
            kwargs["metadata"] = metadata

        try:
            response = await self.client.chat.completions.create(**kwargs)

            usage = response.usage.model_dump() if response.usage else {}
            log_entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "usage": usage,
            }
            info_logger.info(json.dumps(log_entry, ensure_ascii=False))

            reply = response.choices[0].message.content
            if not reply:
                error_logger.error(f"Пустой ответ от LLM: {response}")
                return "None", "⚠️ Ошибка: пустой ответ от LLM"

            return reply, response

        except APIConnectionError as e:
            error_logger.error(f"Request failed: {e}")
            return "None", "⚠️ Ошибка соединения с LLM. Попробуй позже."
        except APIStatusError as e:
            error_logger.error(f"HTTP error: {e.status_code} {e.response.text}")
            return "None", "⚠️ LLM вернул ошибку. Попробуй позже."



llm_client = LLMClient()
