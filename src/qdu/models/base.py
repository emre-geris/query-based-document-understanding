from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True)
class Generation:
    """One model call. Token counts come from the provider; None when it does not report them."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float = 0.0


class LanguageModel(Protocol):
    """A text-only LLM or a vision-language model behind a single interface."""

    name: str
    supports_images: bool

    def generate(self, prompt: str, images: Sequence["Image"] = ()) -> Generation: ...
