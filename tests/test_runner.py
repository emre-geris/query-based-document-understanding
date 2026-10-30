from pathlib import Path

from PIL import Image

from qdu.data import QARecord
from qdu.evaluation.oracle import summarize
from qdu.models import FakeModel
from qdu.ocr import OcrResult
from qdu.pipelines import (
    HybridPipeline,
    PipelineInput,
    PipelineOutput,
    TextPipeline,
    VisionPipeline,
)
from qdu.runner import read_predictions, run_pipelines, scored_predictions

RECORDS = [
    QARecord("q1", "d1", "Total?", ("12.50",), ("table/list",), "table/list"),
    QARecord("q2", "d2", "Who signed?", ("John Smith",), ("handwritten",), "handwritten"),
]


class StaticOcr:
    name = "static"

    def extract(self, image: Image.Image, key: str) -> OcrResult:
        return OcrResult("TOTAL 12.50", self.name, 0.1)


class Broken:
    name = "vision"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, inp: PipelineInput) -> PipelineOutput:
        self.calls += 1
        raise TimeoutError("provider timed out")


def blank(record: QARecord) -> Image.Image:
    return Image.new("RGB", (8, 8))


def pipelines():
    # text answers from OCR, vision/hybrid answer "John Smith" -> each wins one question
    text_model = FakeModel(responder=lambda prompt, n: "12.50" if "TOTAL 12.50" in prompt else "")
    vlm = FakeModel(responder=lambda prompt, n: "John Smith")
    return [
        TextPipeline(text_model, StaticOcr()),
        VisionPipeline(vlm),
        HybridPipeline(vlm, StaticOcr()),
    ]


def test_end_to_end_run_and_analysis(tmp_path: Path) -> None:
    out = tmp_path / "predictions.jsonl"
    stats = run_pipelines(RECORDS, blank, pipelines(), out)
    assert (stats.completed, stats.failed, stats.skipped) == (6, 0, 0)

    rows = list(read_predictions(out))
    assert {r["pipeline"] for r in rows} == {"text", "vision", "hybrid"}
    q1_text = next(r for r in rows if r["question_id"] == "q1" and r["pipeline"] == "text")
    assert q1_text["anls"] == 1.0
    assert q1_text["ocr_latency_s"] == 0.1

    summary = summarize(scored_predictions(out), ("text", "vision", "hybrid"))
    assert summary.overall.oracle_score == 1.0
    assert summary.labels == {"q1": "text", "q2": "vision"}


def test_rerun_skips_completed_pairs(tmp_path: Path) -> None:
    out = tmp_path / "predictions.jsonl"
    run_pipelines(RECORDS, blank, pipelines(), out)
    stats = run_pipelines(RECORDS, blank, pipelines(), out)
    assert (stats.completed, stats.skipped) == (0, 6)


def test_failures_are_recorded_and_retried(tmp_path: Path) -> None:
    out = tmp_path / "predictions.jsonl"
    broken = Broken()
    first = run_pipelines(RECORDS[:1], blank, [broken], out)
    assert first.failed == 1
    row = next(read_predictions(out))
    assert row["error"] == "TimeoutError: provider timed out"
    assert row["anls"] is None
    assert scored_predictions(out) == []

    run_pipelines(RECORDS[:1], blank, [broken], out)
    assert broken.calls == 2  # failed pair is retried


def test_call_cost_measures() -> None:
    import pytest

    from qdu.runner import call_cost

    row = {"model_latency_s": 1.5, "ocr_latency_s": 0.5, "input_tokens": 100, "output_tokens": 4}
    assert call_cost(row, "latency") == 2.0
    assert call_cost({**row, "ocr_latency_s": None}, "latency") == 1.5
    assert call_cost(row, "tokens") == 104.0
    assert call_cost({**row, "input_tokens": None}, "tokens") is None
    with pytest.raises(ValueError, match="unknown cost"):
        call_cost(row, "dollars")
