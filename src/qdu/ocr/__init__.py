"""OCR engines used by the text and hybrid pipelines."""

from qdu.ocr.base import OcrEngine, OcrResult
from qdu.ocr.tesseract import TesseractOcr, find_tesseract

__all__ = ["OcrEngine", "OcrResult", "TesseractOcr", "find_tesseract"]
