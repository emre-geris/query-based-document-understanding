import pytest

from qdu.evaluation.bootstrap import paired_differences


def test_estimate_is_the_mean_difference_and_ci_contains_it() -> None:
    values = {
        "base": {"q1": 0.0, "q2": 1.0, "q3": 0.0, "q4": 1.0},
        "better": {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0},
    }
    document = {"q1": "d1", "q2": "d1", "q3": "d2", "q4": "d3"}
    d = paired_differences(values, document, "base", resamples=200)["better"]
    assert d.estimate == pytest.approx(0.5)
    assert d.low <= d.estimate <= d.high
    assert (d.n_questions, d.n_documents) == (4, 3)


def test_identical_methods_have_a_zero_width_interval() -> None:
    v = {"q1": 0.3, "q2": 0.9}
    d = paired_differences({"a": v, "b": dict(v)}, {"q1": "d1", "q2": "d2"}, "a")["b"]
    assert (d.estimate, d.low, d.high) == (0.0, 0.0, 0.0)
    assert not d.excludes_zero


def test_resampling_documents_keeps_their_questions_together() -> None:
    # Two documents: one where "b" wins all 9 questions, one where it loses 1. Resampling
    # documents gives only 3 distinct means; resampling questions would give many more.
    values = {"a": {}, "b": {}}
    document = {}
    for i in range(9):
        values["a"][f"x{i}"], values["b"][f"x{i}"], document[f"x{i}"] = 0.0, 1.0, "dx"
    values["a"]["y"], values["b"]["y"], document["y"] = 1.0, 0.0, "dy"
    d = paired_differences(values, document, "a", resamples=500)["b"]
    assert d.low == pytest.approx(-1.0)
    assert d.high == pytest.approx(1.0)


def test_same_seed_same_interval_and_order_independent() -> None:
    values = {
        "a": {f"q{i}": i % 3 / 2 for i in range(30)},
        "b": {f"q{i}": i % 2 for i in range(30)},
    }
    document = {f"q{i}": f"d{i // 2}" for i in range(30)}
    first = paired_differences(values, document, "a", seed=7)["b"]
    shuffled = {m: dict(reversed(list(v.items()))) for m, v in values.items()}
    assert paired_differences(shuffled, document, "a", seed=7)["b"] == first


def test_methods_must_cover_the_same_questions() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        paired_differences({"a": {"q1": 1.0}, "b": {"q2": 1.0}}, {"q1": "d", "q2": "d"}, "a")
