from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True)
class OcrResult:
    text: str
    engine: str
    latency_s: float
    """Time the engine took when the text was first extracted (also when served from cache)."""
    cached: bool = False


class OcrEngine(Protocol):
    name: str

    def extract(self, image: "Image", key: str) -> OcrResult:
        """Return the page text. ``key`` identifies the page for caching (e.g. a doc id)."""
        ...
