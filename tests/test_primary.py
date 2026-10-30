import json
from pathlib import Path

import pytest

from qdu.evaluation.primary import primary_metric, primary_score
from qdu.runner import scored_predictions


@pytest.mark.parametrize(
    ("dataset", "metric"),
    [
        ("docvqa", "anls"),
        ("infographicvqa", "anls"),
        ("chartqa", "relaxed_accuracy"),
        ("tatdqa", "f1"),
        ("unknown", "anls"),
    ],
)
def test_primary_metric(dataset: str, metric: str) -> None:
    assert primary_metric(dataset) == metric


def test_chartqa_uses_relaxed_accuracy() -> None:
    # 2.0 is within 5% of 1.98: correct under relaxed accuracy, 0 under ANLS.
    assert primary_score("chartqa", "2.0", ["1.98"]) == 1.0
    assert primary_score("docvqa", "2.0", ["1.98"]) == 0.0


def test_scored_predictions_use_the_primary_metric(tmp_path: Path) -> None:
    row = {
        "question_id": "chartqa/test-1",
        "stratum": "chartqa",
        "pipeline": "vision",
        "prediction": "2.0",
        "answers": ["1.98"],
        "anls": 0.0,
        "model_latency_s": 0.5,
        "ocr_latency_s": None,
        "error": None,
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    [scored] = scored_predictions(path)
    assert scored.score == 1.0
    assert scored.cost == 0.5
