import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image

# Characters Windows does not allow in file names ("/" separates cache sub-directories).
_UNSAFE = re.compile(r'[<>:"\\|?*\x00-\x1f]')


def cache_file(cache_dir: Path, key: str) -> Path:
    """Cache file for a page key such as ``docvqa/4662``, valid on every OS.

    Some InfographicVQA ids are image URLs (``...infographic.jpg?format=1500w``). Unsafe
    characters are replaced and a short hash of the original key keeps names unique; safe keys
    map to ``<key>.json`` unchanged, so existing caches stay valid.
    """
    safe = _UNSAFE.sub("_", key)
    if safe != key:
        safe += "-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return cache_dir / f"{safe}.json"


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
