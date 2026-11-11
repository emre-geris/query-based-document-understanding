"""Tesseract OCR with an on-disk cache.

PSM 3 (automatic layout analysis) is the default: in an early comparison of PSM 3/4/6/11/12
on DocVQA it kept the most gold answers in the OCR text.

Results are cached per (psm, page key) as JSON because OCR
dominates the runtime of text and hybrid pipelines and the same page is reused across runs.

One ``image_to_data`` call gives the words with their confidence and box, grouped here into
lines like PP-OCRv5's (`OcrResult.lines`: text, mean word confidence 0-1, box), so the OCR
signals work on either engine. The page text is built from those lines, one per row and a
blank row between paragraphs, as ``image_to_string`` lays it out (Nov 2026; caches written
before hold the ``image_to_string`` text and no lines).
"""

import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from qdu.ocr.base import OcrLine, OcrResult, cache_file, read_cache, write_cache
from qdu.provenance import cpu_info

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


def lines_from_data(data: dict[str, list]) -> tuple[str, tuple[OcrLine, ...]]:
    """Page text and lines from ``image_to_data`` output (one entry per word, with block,
    paragraph and line numbers, confidence 0-100 and box)."""
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        if str(word).strip() and float(data["conf"][i]) >= 0:
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append(i)
    lines, rows, last_par = [], [], None
    for (block, par, _), idx in groups.items():
        words = " ".join(str(data["text"][i]).strip() for i in idx)
        x0 = min(data["left"][i] for i in idx)
        y0 = min(data["top"][i] for i in idx)
        x1 = max(data["left"][i] + data["width"][i] for i in idx)
        y1 = max(data["top"][i] + data["height"][i] for i in idx)
        score = sum(float(data["conf"][i]) for i in idx) / len(idx) / 100
        lines.append(OcrLine(words, round(score, 4), (x0, y0, x1, y1)))
        if last_par is not None and (block, par) != last_par:
            rows.append("")
        rows.append(words)
        last_par = (block, par)
    return "\n".join(rows), tuple(lines)


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
        self.hardware = cpu_info()

    def _cache_path(self, key: str) -> Path | None:
        return cache_file(self.cache_dir, key) if self.cache_dir else None

    def extract(self, image: "Image", key: str) -> OcrResult:
        path = self._cache_path(key)
        if (cached := read_cache(path, image.size)) is not None:
            return cached

        start = time.perf_counter()
        data = self._pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=f"--psm {self.psm}",
            output_type=self._pytesseract.Output.DICT,
        )
        latency = time.perf_counter() - start
        text, lines = lines_from_data(data)
        result = OcrResult(
            text, self.name, latency, hardware=self.hardware, lines=lines, page_size=image.size
        )
        write_cache(path, result)
        return result
