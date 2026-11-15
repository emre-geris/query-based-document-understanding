from qdu.evaluation.primary import primary_score
from qdu.evaluation.relaxed import is_year, relaxed_match


def test_official_relaxed_accuracy_gives_years_numeric_tolerance():
    # the known quirk of the official metric: a wrong year within 5% counts as correct
    assert relaxed_match("2019", ["2020"])
    assert relaxed_match("1925", ["2020"])


def test_exact_years_requires_the_same_year():
    assert not relaxed_match("2019", ["2020"], exact_years=True)
    assert relaxed_match("2020", ["2020"], exact_years=True)
    assert relaxed_match("2020.0", ["2020"], exact_years=True)
    # other numbers keep their tolerance
    assert relaxed_match("101", ["100"], exact_years=True)
    assert relaxed_match("Blue", ["blue"], exact_years=True)
    assert not relaxed_match("2019", ["2020", "2021"], exact_years=True)
    assert relaxed_match("2021", ["2020", "2021"], exact_years=True)


def test_is_year():
    assert is_year("2020") and is_year(" 1999 ")
    assert not is_year("1850") and not is_year("20.5") and not is_year("12345")


def test_primary_score_option_only_affects_chartqa():
    assert primary_score("chartqa", "2019", ["2020"]) == 1.0
    assert primary_score("chartqa", "2019", ["2020"], chartqa_exact_years=True) == 0.0
    assert primary_score("docvqa", "2019", ["2020"], chartqa_exact_years=True) == primary_score(
        "docvqa", "2019", ["2020"]
    )
