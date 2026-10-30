"""Models behind an OpenAI-compatible chat API, e.g. a vLLM server running Qwen3.5.

Spec format: ``openai:<model-name>``, e.g. ``openai:Qwen/Qwen3.5-4B``. The endpoint comes from
``QDU_OPENAI_BASE_URL`` (default ``http://localhost:8000/v1``) and the key from
``QDU_OPENAI_API_KEY`` (default ``EMPTY``, which vLLM accepts).

Decoding is greedy and short, because answers are short extractive spans. Qwen3.5 reasons
("thinks") by default, which would distort answers, latency and token cost, so thinking is
disabled on every request through ``chat_template_kwargs``.
"""

import base64
import io
import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from qdu.models.base import Generation

if TYPE_CHECKING:
    from PIL.Image import Image

DEFAULT_BASE_URL = "http://localhost:8000/v1"


def image_to_data_url(image: "Image") -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_messages(prompt: str, images: Sequence["Image"]) -> list[dict[str, Any]]:
    if not images:
        return [{"role": "user", "content": prompt}]
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}} for image in images
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class OpenAICompatModel:
    supports_images = True

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 64,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        disable_thinking: bool = True,
        client: Any = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url or os.environ.get("QDU_OPENAI_BASE_URL", DEFAULT_BASE_URL),
                api_key=api_key or os.environ.get("QDU_OPENAI_API_KEY", "EMPTY"),
                timeout=timeout_s,
                max_retries=max_retries,
            )
        self._client = client
        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking

    def generate(self, prompt: str, images: Sequence["Image"] = ()) -> Generation:
        extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}} if self.disable_thinking else {}
        )
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=build_messages(prompt, images),
            temperature=0.0,
            max_tokens=self.max_tokens,
            extra_body=extra_body,
        )
        latency = time.perf_counter() - start
        usage = response.usage
        return Generation(
            text=response.choices[0].message.content or "",
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_s=latency,
        )
