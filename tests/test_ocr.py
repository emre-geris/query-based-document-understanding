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


def test_cache_file_is_valid_on_windows(tmp_path) -> None:
    from qdu.ocr.base import cache_file

    assert cache_file(tmp_path, "docvqa/4662") == tmp_path / "docvqa/4662.json"
    a = cache_file(tmp_path, "infographicvqa/cities.jpg?format=1500w")
    b = cache_file(tmp_path, "infographicvqa/cities.jpg*format=1500w")
    assert a.parent == tmp_path / "infographicvqa"
    assert "?" not in a.name and a != b
    a.parent.mkdir(parents=True)
    a.write_text("{}", encoding="utf-8")  # fails on Windows if the name is invalid


def test_warm_cache_runs_each_page_once() -> None:
    from qdu.data.records import QARecord
    from qdu.ocr.base import OcrResult
    from qdu.ocr.warm import warm_cache

    class CountingOcr:
        name = "counting"

        def __init__(self) -> None:
            self.seen: list[str] = []

        def extract(self, image, key: str) -> OcrResult:
            cached = key in self.seen
            self.seen.append(key)
            return OcrResult("text", self.name, 0.5, cached=cached, hardware="cpu x 1")

    def rec(qid: str, doc: str) -> QARecord:
        return QARecord(question_id=qid, dataset="docvqa", doc_id=doc, question="q", answers=["a"])

    engine = CountingOcr()
    records = [
        rec("docvqa/1", "docvqa/d1"),
        rec("docvqa/2", "docvqa/d1"),
        rec("docvqa/3", "docvqa/d2"),
    ]
    stats = warm_cache(records, lambda r: Image.new("RGB", (4, 4)), engine)
    assert engine.seen == ["docvqa/d1", "docvqa/d2"]
    assert (stats.pages, stats.extracted, stats.cached, stats.seconds) == (2, 2, 0, 1.0)


def test_cpu_info_names_the_core_count() -> None:
    import os

    from qdu.provenance import cpu_info

    assert cpu_info().endswith(f"x {os.cpu_count()}")


def test_cache_keeps_lines_boxes_and_page_size(tmp_path) -> None:
    from qdu.ocr.base import OcrLine, OcrResult, read_cache, write_cache

    lines = (OcrLine("TOTAL 42", 0.9654, (16, 182, 245, 239)),)
    fresh = OcrResult("TOTAL 42", "e", 1.5, hardware="cpu x 2", lines=lines, page_size=(700, 300))
    write_cache(tmp_path / "p.json", fresh)
    cached = read_cache(tmp_path / "p.json")
    assert cached is not None and cached.cached
    assert (cached.lines, cached.page_size, cached.hardware) == (lines, (700, 300), "cpu x 2")
    write_cache(tmp_path / "t.json", OcrResult("x", "tesseract", 0.1))
    assert read_cache(tmp_path / "t.json").lines is None  # engines without boxes


def test_cache_entry_for_a_different_page_size_is_ignored(tmp_path) -> None:
    from qdu.ocr.base import OcrResult, read_cache, write_cache

    write_cache(tmp_path / "p.json", OcrResult("old", "e", 0.1, page_size=(224, 224)))
    assert read_cache(tmp_path / "p.json", (224, 224)) is not None
    assert read_cache(tmp_path / "p.json", (1653, 2339)) is None  # the page image changed
    write_cache(tmp_path / "legacy.json", OcrResult("old", "e", 0.1))
    assert read_cache(tmp_path / "legacy.json", (1653, 2339)) is not None  # size unknown
