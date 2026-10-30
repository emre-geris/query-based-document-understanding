import pytest

from qdu.evaluation.metrics import metric_table
from qdu.evaluation.relaxed import parse_number, relaxed_match


@pytest.mark.parametrize(
    ("text", "number"),
    [("1,234.5", 1234.5), ("45%", 45.0), ("$3.2", 3.2), ("-7", -7.0), ("Africa", None), ("", None)],
)
def test_parse_number(text: str, number: float | None) -> None:
    assert parse_number(text) == number


@pytest.mark.parametrize(
    ("prediction", "answers", "expected"),
    [
        ("104", ["100"], True),  # within 5%
        ("106", ["100"], False),
        ("45%", ["45"], True),
        ("0", ["0"], True),
        ("Africa", ["africa"], True),
        ("Asia", ["Africa"], False),
        ("12", ["twelve", "12.3"], True),
    ],
)
def test_relaxed_match(prediction: str, answers: list[str], expected: bool) -> None:
    assert relaxed_match(prediction, answers) is expected


def row(qid: str, pipeline: str, prediction: str, anls: float, model_s: float, ocr_s=None):
    return {
        "question_id": qid,
        "pipeline": pipeline,
        "prediction": prediction,
        "answers": ["100"],
        "anls": anls,
        "model_latency_s": model_s,
        "ocr_latency_s": ocr_s,
        "input_tokens": 10,
        "output_tokens": 1,
        "error": None,
    }


def test_metric_table_reports_cost_per_pipeline_and_oracle() -> None:
    rows = [
        row("chartqa/q1", "text", "103", 0.0, 1.0, 0.5),
        row("chartqa/q1", "vision", "100", 1.0, 3.0),
        row("chartqa/q2", "text", "100", 1.0, 1.0, 0.5),
        row("chartqa/q2", "vision", "100", 1.0, 3.0),
    ]
    labels = {"chartqa/q1": "vision", "chartqa/q2": "text"}
    table = {(m.dataset, m.pipeline): m for m in metric_table(rows, ("text", "vision"), labels)}

    text = table[("chartqa", "text")]
    assert (text.n, text.anls, text.relaxed_accuracy) == (2, 0.5, 1.0)
    assert (text.model_s, text.ocr_s, text.total_s) == (1.0, 0.5, 1.5)
    oracle = table[("chartqa", "oracle")]
    assert oracle.anls == 1.0
    assert oracle.total_s == pytest.approx((3.0 + 1.5) / 2)
    assert table[("ALL", "vision")].n == 2
