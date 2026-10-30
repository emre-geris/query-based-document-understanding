"""Tesseract OCR with an on-disk cache.

PSM 3 (automatic layout analysis) is the default: in an early comparison of PSM 3/4/6/11/12
on DocVQA it kept the most gold answers in the OCR text.

Results are cached per (psm, page key) as JSON because OCR
dominates the runtime of text and hybrid pipelines and the same page is reused across runs.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from qdu.ocr.base import OcrResult, cache_file

if TYPE_CHECKING:
    from PIL.Image import Image

WINDOWS_DEFAULT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def find_tesseract() -> str | None:
    """Return the Tesseract executable, checking PATH and the default Windows location."""
    found = shutil.which("tesseract")
    if found:
        return found
    if os.path.exists(WINDOWS_DEFAULT_PATH):
        return WINDOWS_DEFAULT_PATH
    return None


class TesseractOcr:
    def __init__(self, psm: int = 3, cache_dir: Path | None = None, lang: str = "eng") -> None:
        import pytesseract

        executable = find_tesseract()
        if executable is None:
            raise RuntimeError(
                "Tesseract not found. Install it, e.g. on Windows:"
                " winget install UB-Mannheim.TesseractOCR"
            )
        pytesseract.pytesseract.tesseract_cmd = executable
        self._pytesseract = pytesseract
        self.psm = psm
        self.lang = lang
        self.version = str(pytesseract.get_tesseract_version())
        self.name = f"tesseract-{self.version}-psm{psm}"
        self.cache_dir = cache_dir / f"psm{psm}" if cache_dir else None

    def _cache_path(self, key: str) -> Path | None:
        return cache_file(self.cache_dir, key) if self.cache_dir else None

    def extract(self, image: "Image", key: str) -> OcrResult:
        path = self._cache_path(key)
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return OcrResult(data["text"], data["engine"], data["latency_s"], cached=True)

        start = time.perf_counter()
        text = self._pytesseract.image_to_string(image, lang=self.lang, config=f"--psm {self.psm}")
        result = OcrResult(text, self.name, time.perf_counter() - start)

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"text": text, "engine": self.name, "latency_s": result.latency_s}
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return result
