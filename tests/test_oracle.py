import pytest

from qdu.evaluation.oracle import ScoredPrediction, oracle_label, summarize

ORDER = ("text", "vision", "hybrid")


def rows(table: dict[str, tuple[str, float, float, float]]) -> list[ScoredPrediction]:
    out = []
    for qid, (stratum, *scores) in table.items():
        for pipeline, score in zip(ORDER, scores, strict=True):
            out.append(ScoredPrediction(qid, stratum, pipeline, score))
    return out


TABLE = {
    # qid: (stratum, text, vision, hybrid)
    "q1": ("form", 1.0, 1.0, 1.0),  # all correct -> cheapest (text)
    "q2": ("form", 0.0, 1.0, 0.8),  # only vision fully right
    "q3": ("figure", 0.0, 0.0, 0.9),  # only hybrid
    "q4": ("figure", 0.0, 0.0, 0.0),  # nobody -> cheapest
}


def test_oracle_label_prefers_cheapest_among_ties() -> None:
    assert oracle_label({"text": 1.0, "vision": 1.0, "hybrid": 1.0}, ORDER, 0.0) == "text"
    assert oracle_label({"text": 0.9, "vision": 1.0, "hybrid": 1.0}, ORDER, 0.0) == "vision"
    assert oracle_label({"text": 0.95, "vision": 1.0, "hybrid": 1.0}, ORDER, 0.1) == "text"


def test_overall_summary() -> None:
    s = summarize(rows(TABLE), ORDER).overall
    assert s.n_questions == 4
    assert s.mean_score == pytest.approx({"text": 0.25, "vision": 0.5, "hybrid": 0.675})
    assert s.oracle_score == pytest.approx((1.0 + 1.0 + 0.9 + 0.0) / 4)
    assert s.best_single == "hybrid"
    assert s.headroom == pytest.approx(0.725 - 0.675)
    assert s.label_counts == {"text": 2, "vision": 1, "hybrid": 1}


def test_by_stratum() -> None:
    summary = summarize(rows(TABLE), ORDER)
    assert set(summary.by_stratum) == {"figure", "form"}
    assert summary.by_stratum["form"].best_single == "vision"
    assert summary.by_stratum["figure"].oracle_score == pytest.approx(0.45)
    assert summary.labels["q3"] == "hybrid"


def test_incomplete_questions_are_excluded_and_counted() -> None:
    data = [*rows(TABLE), ScoredPrediction("q5", "form", "text", 1.0)]
    summary = summarize(data, ORDER)
    assert summary.overall.n_questions == 4
    assert summary.incomplete_questions == 1


def test_no_complete_question_raises() -> None:
    with pytest.raises(ValueError, match="no question"):
        summarize([ScoredPrediction("q1", "form", "text", 1.0)], ORDER)


def test_measured_cost_overrides_the_fixed_order() -> None:
    scores = {"text": 1.0, "vision": 1.0, "hybrid": 1.0}
    costs = {"text": 1.8, "vision": 1.2, "hybrid": 2.4}  # OCR made text slower than vision
    assert oracle_label(scores, ORDER, 0.0, costs) == "vision"
    assert oracle_label(scores, ORDER, 0.0, {"text": None, "vision": 1.2, "hybrid": 2.4}) == "text"


def test_summarize_uses_costs_from_rows() -> None:
    rows = [
        ScoredPrediction("q1", "form", "text", 1.0, cost=2.0),
        ScoredPrediction("q1", "form", "vision", 1.0, cost=1.0),
        ScoredPrediction("q1", "form", "hybrid", 1.0, cost=3.0),
    ]
    assert summarize(rows, ORDER).labels == {"q1": "vision"}
