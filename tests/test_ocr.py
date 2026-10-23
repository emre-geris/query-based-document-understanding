from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from qdu.ocr import TesseractOcr, find_tesseract

requires_tesseract = pytest.mark.skipif(find_tesseract() is None, reason="tesseract not installed")


def render(text: str) -> Image.Image:
    image = Image.new("RGB", (900, 160), "white")
    ImageDraw.Draw(image).text((20, 40), text, fill="black", font=ImageFont.load_default(size=56))
    return image


@requires_tesseract
def test_reads_rendered_text() -> None:
    result = TesseractOcr(psm=6).extract(render("INVOICE 1975"), key="page")
    assert "INVOICE" in result.text
    assert "1975" in result.text
    assert not result.cached
    assert result.engine.startswith("tesseract-")


@requires_tesseract
def test_second_call_is_served_from_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ocr = TesseractOcr(psm=6, cache_dir=tmp_path)
    first = ocr.extract(render("CACHE ME"), key="doc1")

    def fail(*args: object, **kwargs: object) -> str:
        raise AssertionError("tesseract should not run for a cached page")

    monkeypatch.setattr(ocr._pytesseract, "image_to_string", fail)
    second = ocr.extract(render("CACHE ME"), key="doc1")
    assert second.cached
    assert second.text == first.text
    assert second.latency_s == first.latency_s
    assert (tmp_path / "psm6" / "doc1.json").exists()
