import pytest

from qdu.evaluation.em_f1 import exact_match, normalize_tokens, token_f1


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("645,791", ["645791"]),
        ("645791.0", ["645791"]),
        ("12.345", ["12.35"]),  # rounded to two decimals, as in the official evaluation
        ("The Oxaydo product.", ["oxaydo", "product"]),
        ("45%", ["45"]),
    ],
)
def test_normalize_tokens(text: str, tokens: list[str]) -> None:
    assert normalize_tokens(text) == tokens


def test_exact_match_numbers_and_text() -> None:
    assert exact_match("645,791", ["645791"])
    assert exact_match("the AMER region", ["AMER region"])
    assert not exact_match("645,792", ["645,791"])


def test_token_f1_partial_and_best_gold() -> None:
    assert token_f1("total assets 2018", ["total assets"]) == pytest.approx(0.8)
    assert token_f1("x", ["nothing", "x"]) == 1.0
    assert token_f1("unrelated", ["answer"]) == 0.0
    assert token_f1("", [""]) == 1.0
