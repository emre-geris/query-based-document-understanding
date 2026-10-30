from pathlib import Path

import pytest
from PIL import Image

from qdu.data import QARecord
from qdu.evaluation.em_f1 import contains_answer
from qdu.ocr import OcrResult, create_ocr
from qdu.ocr.compare import compare_engines, summarize_engines, write_comparison

RECORDS = [
    QARecord("docvqa/1", "docvqa/d1", "Total?", ("$1,250",), dataset="docvqa"),
    QARecord("chartqa/2", "chartqa/c2", "Peak?", ("42",), dataset="chartqa"),
]


class FixedOcr:
    def __init__(self, name: str, text: str, latency: float) -> None:
        self.name, self.text, self.latency = name, text, latency

    def extract(self, image: Image.Image, key: str) -> OcrResult:
        return OcrResult(self.text, self.name, self.latency)


def test_contains_answer_uses_normalized_contiguous_tokens() -> None:
    assert contains_answer("TOTAL 1250 USD", ["$1,250"])
    assert contains_answer("Received by John Smith.", ["john smith"])
    assert not contains_answer("Smith John", ["john smith"])
    assert not contains_answer("anything", [""])


def test_compare_and_summarize(tmp_path: Path) -> None:
    engines = {
        "good": FixedOcr("good", "total 1250 and peak 42", 2.0),
        "weak": FixedOcr("weak", "tota1 l25O", 0.5),
    }
    results = compare_engines(RECORDS, lambda r: Image.new("RGB", (4, 4)), engines)
    assert len(results) == 4
    summary = {(r.dataset, r.engine): r for r in summarize_engines(results)}
    assert summary[("ALL", "good")].coverage == 1.0
    assert summary[("ALL", "weak")].coverage == 0.0
    assert summary[("ALL", "good")].mean_latency_s == pytest.approx(2.0)
    assert summary[("docvqa", "weak")].n == 1

    write_comparison(tmp_path, results, list(summary.values()), {"x": 1})
    assert (tmp_path / "pages.csv").read_text().count("\n") == 5
    assert (tmp_path / "summary.csv").exists()


def test_unknown_ocr_spec() -> None:
    with pytest.raises(ValueError, match="unknown OCR spec"):
        create_ocr("easyocr")
