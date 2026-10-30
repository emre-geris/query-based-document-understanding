"""OCR engines used by the text and hybrid pipelines."""

from pathlib import Path

from qdu.ocr.base import OcrEngine, OcrResult
from qdu.ocr.tesseract import TesseractOcr, find_tesseract

OCR_SPECS = ("tesseract", "paddle-mobile", "paddle-server")


def create_ocr(spec: str, cache_dir: Path | None = None, psm: int = 3) -> OcrEngine:
    """Build an OCR engine from a spec: ``tesseract``, ``paddle-mobile`` or ``paddle-server``."""
    if spec == "tesseract":
        return TesseractOcr(psm=psm, cache_dir=cache_dir)
    if spec.startswith("paddle-"):
        from qdu.ocr.paddle import PaddleOcr

        return PaddleOcr(spec.removeprefix("paddle-"), cache_dir=cache_dir)
    raise ValueError(f"unknown OCR spec {spec!r}; choose from {', '.join(OCR_SPECS)}")


__all__ = ["OCR_SPECS", "OcrEngine", "OcrResult", "TesseractOcr", "create_ocr", "find_tesseract"]
