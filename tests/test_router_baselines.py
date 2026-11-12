import pytest

from qdu.ocr.base import OcrResult
from qdu.router.baselines import (
    EcoDocRules,
    GridPoint,
    QuestionOnly,
    select_point,
    train_ecodoc,
    train_question_only,
)
from qdu.router.labels import LabelRow

OCR_S = 0.5
MODEL_S = {"text": 0.2, "vision": 2.0, "hybrid": 2.2}

TEXT_QUESTIONS = ("what is the invoice number", "what is the date of the letter")
IMAGE_QUESTIONS = (
    "what colour is the largest bar in the chart",
    "which line in the graph is highest",
)


def row(i: int, split: str, image: bool) -> LabelRow:
    """Text answers the text questions; only vision and hybrid answer the image questions."""
    question = (IMAGE_QUESTIONS if image else TEXT_QUESTIONS)[i % 2]
    score = {"text": 0.0 if image else 1.0, "vision": 1.0, "hybrid": 1.0}
    cost_c = {p: MODEL_S[p] + (0.0 if p == "vision" else OCR_S) for p in MODEL_S}
    return LabelRow(
        question_id=f"docvqa/{i}",
        doc_id=f"doc/{i}",
        dataset="docvqa",
        split=split,
        stratum="docvqa/x",
        question=question,
        questions_on_page=1,
        ocr_s=OCR_S,
        score=score,
        model_s=dict(MODEL_S),
        cost_c=cost_c,
        cost_b=dict(cost_c),
        tokens={"text": 500.0, "vision": 3000.0, "hybrid": 3500.0},
        oracle_latency="vision" if image else "text",
        oracle_tokens="vision" if image else "text",
    )


@pytest.fixture
def rows() -> tuple[list[LabelRow], list[LabelRow]]:
    all_rows = [row(i, "train" if i < 80 else "val", image=(i // 2) % 2 == 1) for i in range(120)]
    return [r for r in all_rows if r.split == "train"], [r for r in all_rows if r.split == "val"]


def ocr_for(rows: list[LabelRow]) -> dict[str, OcrResult]:
    """Text pages contain the question's terms; image pages have unrelated OCR text."""
    pages = {}
    for r in rows:
        text = r.question if r.score["text"] else "annual report 2021 summary"
        pages[r.doc_id] = OcrResult(text=text, engine="fake", latency_s=OCR_S)
    return pages


def test_select_point_budget_and_fallback():
    shares = {"text": 0.5, "vision": 0.5, "hybrid": 0.0}
    grid = [GridPoint((0.1,), 0.9, 2.0, shares), GridPoint((0.5,), 0.8, 1.0, shares)]
    assert select_point(grid, 1.5) == (grid[1], True)
    assert select_point(grid, 0.5) == (grid[0], False)
    with pytest.raises(ValueError):
        select_point([], 1.0)


def test_question_only_learns_image_questions(rows):
    train, val = rows
    router = QuestionOnly().fit(train)
    prob, seconds = router.probabilities(val)
    for r in val:
        assert (prob[r.question_id] > 0.5) == (r.score["text"] == 0.0)
        assert seconds[r.question_id] >= 0.0
    policy = router.decide(val, (0.5,))
    assert not policy.uses_ocr  # it decides from the question alone
    assert {policy.choice[r.question_id] for r in val} == {"text", "vision"}


def test_question_only_needs_both_labels(rows):
    train, _ = rows
    only_text = [r for r in train if r.score["text"] == 1.0]
    with pytest.raises(ValueError, match="both labels"):
        QuestionOnly().fit(only_text)


def test_train_question_only_routes_perfectly_within_budget(rows):
    train, val = rows
    baseline = train_question_only(train, val)
    assert baseline.met_budget
    best = next(g for g in baseline.grid if g.params == baseline.params)
    assert best.score == pytest.approx(1.0)
    assert best.shares["text"] == pytest.approx(0.5)
    assert baseline.policy(val).choice == {
        r.question_id: "text" if r.score["text"] else "vision" for r in val
    }


def test_ecodoc_rules_pay_ocr_only_after_rule_one(rows):
    train, val = rows
    router = EcoDocRules(ocr_for(train + val)).fit(train)
    gap, _ = router.intent(val)
    for r in val:
        assert (gap[r.question_id] > 0) == (r.score["text"] == 0.0)
    policy = router.decide(val, (0.0, 0.5))
    for r in val:
        image = r.score["text"] == 0.0
        assert policy.choice[r.question_id] == ("vision" if image else "text")
        assert policy.uses_ocr[r.question_id] is (not image)


def test_ecodoc_relevance_sends_unmatched_pages_to_vision(rows):
    train, val = rows
    router = EcoDocRules(ocr_for(train + val)).fit(train)
    policy = router.decide(val, (1.0, 0.5))  # rule 1 off: everything is checked against OCR
    for r in val:
        assert policy.uses_ocr[r.question_id]
        assert policy.choice[r.question_id] == ("text" if r.score["text"] else "vision")


def test_train_ecodoc_needs_ocr_for_every_page(rows):
    train, val = rows
    pages = ocr_for(train)
    with pytest.raises(ValueError, match="no OCR"):
        train_ecodoc(train, val, pages)
    baseline = train_ecodoc(train, val, ocr_for(train + val))
    assert baseline.met_budget
    assert len(baseline.grid) == 30
