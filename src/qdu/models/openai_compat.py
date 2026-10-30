"""Models behind an OpenAI-compatible chat API, e.g. a vLLM server running Qwen3.5.

Spec format: ``openai:<model-name>``, e.g. ``openai:Qwen/Qwen3.5-4B``. The endpoint comes from
``QDU_OPENAI_BASE_URL`` (default ``http://localhost:8000/v1``) and the key from
``QDU_OPENAI_API_KEY`` (default ``EMPTY``, which vLLM accepts).

Decoding is greedy and short, because answers are short extractive spans. Qwen3.5 reasons
("thinks") by default, which would distort answers, latency and token cost, so thinking is
disabled on every request through ``chat_template_kwargs``.

Page images are downscaled (aspect ratio kept) to at most ``max_image_pixels`` before sending
(override with ``QDU_MAX_IMAGE_PIXELS``). The default is Qwen3.5's own processor limit
(``size.longest_edge`` = 16,777,216 pixels in its ``preprocessor_config.json``, about 16k image
tokens), so we impose no stricter cap than the model itself: a 4 MP cap in the pilot
downscaled a quarter of DocVQA and InfographicVQA pages and favoured the hybrid path, whose
OCR still saw the full page. The server needs ``--max-model-len`` above ~17k tokens for the
largest pages. The cap applies equally to the vision and hybrid paths; OCR always runs on the
full-resolution page.

The server's launch flags are not visible through the API, so the runbook passes them in
``QDU_SERVER_ARGS`` and they are recorded with every run.
"""

import base64
import io
import json
import math
import os
import time
import urllib.request
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from qdu.models.base import Generation

if TYPE_CHECKING:
    from PIL.Image import Image

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MAX_IMAGE_PIXELS = 16_777_216  # Qwen3.5 preprocessor size.longest_edge
_BICUBIC = 3  # PIL.Image.Resampling.BICUBIC


def limit_pixels(image: "Image", max_pixels: int | None) -> "Image":
    """Downscale so that width x height <= max_pixels, keeping the aspect ratio."""
    width, height = image.size
    if not max_pixels or width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, resample=_BICUBIC)


def image_to_data_url(image: "Image", max_pixels: int | None = None) -> str:
    buffer = io.BytesIO()
    limit_pixels(image.convert("RGB"), max_pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_messages(
    prompt: str, images: Sequence["Image"], max_pixels: int | None = None
) -> list[dict[str, Any]]:
    if not images:
        return [{"role": "user", "content": prompt}]
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_to_data_url(image, max_pixels)}}
        for image in images
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _fetch_json(url: str) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception:  # server info is best effort; never block a run on it
        return None


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
        max_image_pixels: int | None = None,
        client: Any = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("QDU_OPENAI_BASE_URL", DEFAULT_BASE_URL)
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.base_url,
                api_key=api_key or os.environ.get("QDU_OPENAI_API_KEY", "EMPTY"),
                timeout=timeout_s,
                max_retries=max_retries,
            )
        self._client = client
        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self.max_image_pixels = max_image_pixels or int(
            os.environ.get("QDU_MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS)
        )

    def describe(self) -> dict[str, Any]:
        """Request settings plus, when reachable, the server's version and model list."""
        root = self.base_url.rstrip("/").removesuffix("/v1")
        return {
            "provider": "openai_compat",
            "base_url": self.base_url,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "thinking_disabled": self.disable_thinking,
            "max_image_pixels": self.max_image_pixels,
            "server_args": os.environ.get("QDU_SERVER_ARGS"),
            "server_version": _fetch_json(f"{root}/version"),
            "server_models": _fetch_json(f"{root}/v1/models"),
        }

    def generate(self, prompt: str, images: Sequence["Image"] = ()) -> Generation:
        extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}} if self.disable_thinking else {}
        )
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=build_messages(prompt, images, self.max_image_pixels),
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
