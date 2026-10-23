import pytest

from qdu.evaluation import anls_score, levenshtein, mean_anls


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("same", "same", 0),
    ],
)
def test_levenshtein(a: str, b: str, distance: int) -> None:
    assert levenshtein(a, b) == distance
    assert levenshtein(b, a) == distance


def test_exact_match_ignores_case_and_whitespace() -> None:
    assert anls_score("  John   SMITH ", ["john smith"]) == 1.0


def test_partial_match_below_threshold_is_scored() -> None:
    # 1 edit over 10 characters -> NL = 0.1 -> score 0.9
    assert anls_score("0.28 milli", ["0.28 mills"]) == pytest.approx(0.9)


def test_distance_at_or_above_threshold_scores_zero() -> None:
    assert anls_score("abcd", ["wxyz"]) == 0.0
    # NL exactly 0.5 is not below the threshold
    assert anls_score("ab", ["ax"]) == 0.0


def test_best_gold_answer_wins() -> None:
    assert anls_score("1975", ["nineteen seventy five", "1975"]) == 1.0


def test_empty_prediction_against_non_empty_gold() -> None:
    assert anls_score("", ["answer"]) == 0.0


def test_mean_anls() -> None:
    assert mean_anls(["a", "b"], [["a"], ["zzz"]]) == pytest.approx(0.5)


def test_mean_anls_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="2 predictions but 1"):
        mean_anls(["a", "b"], [["a"]])
