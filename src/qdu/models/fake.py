"""Deterministic stand-in model for tests and dry runs. Never use its numbers in results."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qdu.models.base import Generation

if TYPE_CHECKING:
    from PIL.Image import Image

Responder = Callable[[str, int], str]
"""Receives the prompt and the number of images, returns the model output."""


def _empty(prompt: str, n_images: int) -> str:
    return ""


@dataclass
class FakeModel:
    name: str = "fake"
    supports_images: bool = True
    responder: Responder = _empty
    calls: list[tuple[str, int]] = field(default_factory=list)

    def generate(self, prompt: str, images: Sequence["Image"] = ()) -> Generation:
        if images and not self.supports_images:
            raise ValueError(f"{self.name} does not accept images")
        self.calls.append((prompt, len(images)))
        text = self.responder(prompt, len(images))
        return Generation(
            text=text,
            model=self.name,
            input_tokens=len(prompt.split()) + 1000 * len(images),
            output_tokens=len(text.split()),
            latency_s=0.0,
        )
