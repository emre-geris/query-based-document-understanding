import random
from pathlib import Path

import pytest

from qdu.router import Policy, read_policy
from qdu.router.labels import LabelRow
from qdu.router.train import (
    CurvePoint,
    QualityModel,
    decision_costs,
    select_lambda,
    train_router,
    write_choices,
    write_curve,
)

OCR_S = 0.5
MODEL_S = {"text": 0.2, "vision": 2.0, "hybrid": 2.2}
TOKENS = {"text": 500.0, "vision": 3000.0, "hybrid": 3500.0}


def row(qid: str, split: str, easy: bool, question: str) -> LabelRow:
    """Text answers the easy questions (signal s_easy = 1); vision and hybrid answer all."""
    score = {"text": 1.0 if easy else 0.0, "vision": 1.0, "hybrid": 1.0}
    cost_c = {p: MODEL_S[p] + (0.0 if p == "vision" else OCR_S) for p in MODEL_S}
    return LabelRow(
        question_id=qid,
        doc_id=f"doc/{qid}",
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
        tokens=dict(TOKENS),
        oracle_latency="text" if easy else "vision",
        oracle_tokens="text" if easy else "vision",
    )


@pytest.fixture
def data() -> tuple[list[LabelRow], list[LabelRow], dict[str, dict[str, float]]]:
    rng = random.Random(0)
    rows, features = [], {}
    for i in range(240):
        easy = i % 2 == 0
        split = "train" if i < 160 else "val"
        question = "what is the total amount" if easy else "which color is the largest bar"
        r = row(f"docvqa/{i}", split, easy, question)
        rows.append(r)
        features[r.question_id] = {"s_easy": float(easy), "s_noise": rng.random()}
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "val"]
    return train, val, features


def router_cost(rows: list[LabelRow]) -> dict[str, dict]:
    ids = [r.question_id for r in rows]
    return {
        "uses_ocr": dict.fromkeys(ids, True),
        "page_s": dict.fromkeys(ids, 0.0),
        "question_s": dict.fromkeys(ids, 0.001),
    }


@pytest.mark.parametrize("kind", ["ridge", "lgbm"])
def test_quality_model_learns_which_questions_text_answers(data, kind):
    train, val, features = data
    model = QualityModel.fit(train, features, kind=kind, tfidf=True)
    predicted = model.predict(val, features)
    assert set(predicted) == {r.question_id for r in val}
    for r in val:
        q = predicted[r.question_id]
        assert all(0.0 <= v <= 1.0 for v in q.values())
        if r.score["text"] == 1.0:
            assert q["text"] > 0.7
        else:
            assert q["text"] < 0.3


def test_quality_model_is_deterministic(data):
    train, val, features = data
    a = QualityModel.fit(train, features, kind="lgbm", seed=3).predict(val, features)
    b = QualityModel.fit(train, features, kind="lgbm", seed=3).predict(val, features)
    assert a == b


def test_quality_model_rejects_unknown_kind(data):
    train, _, features = data
    with pytest.raises(ValueError, match="unknown model kind"):
        QualityModel.fit(train, features, kind="forest")


def test_missing_features_are_allowed(data):
    train, val, features = data
    sparse_features = {
        q: {k: v for k, v in f.items() if k != "s_noise"} for q, f in features.items()
    }
    sparse_features[train[0].question_id]["s_noise"] = 0.5  # known to the model, absent elsewhere
    for kind in ("ridge", "lgbm"):
        model = QualityModel.fit(train, sparse_features, kind=kind, tfidf=False)
        assert model.feature_names == ["s_easy", "s_noise"]
        assert len(model.predict(val, sparse_features)) == len(val)


def test_decision_costs(data):
    train, _, _ = data
    assert decision_costs(train, ocr_paid=True) == pytest.approx(MODEL_S)
    unpaid = decision_costs(train, ocr_paid=False)
    assert unpaid == pytest.approx({"text": 0.7, "vision": 2.0, "hybrid": 2.7})
    assert decision_costs(train, ocr_paid=True, scenario="A") == pytest.approx(TOKENS)


def test_select_lambda_prefers_most_accurate_within_budget():
    shares = {"text": 0.0, "vision": 1.0, "hybrid": 0.0}
    curve = [
        CurvePoint(0.0, 0.90, 2.0, shares),
        CurvePoint(0.1, 0.88, 1.4, shares),
        CurvePoint(0.2, 0.88, 1.2, shares),
        CurvePoint(0.5, 0.70, 0.9, shares),
    ]
    best, met = select_lambda(curve, budget=1.5)
    assert (best.lam, met) == (0.2, True)  # equal score: the cheaper point
    best, met = select_lambda(curve, budget=0.5)
    assert (best.lam, met) == (0.0, False)  # nothing fits: most accurate, flagged


def test_train_router_routes_easy_questions_to_text(data, tmp_path: Path):
    train, val, features = data
    rc = router_cost([*train, *val])
    trained = train_router(train, val, features, rc, uses_ocr=True, kind="lgbm", tfidf=False)
    vision = Policy("v", {r.question_id: "vision" for r in val})  # no signals, no OCR
    assert trained.budget == pytest.approx(sum(vision.cost(r, "C") for r in val) / len(val))
    assert trained.met_budget
    chosen = next(c for c in trained.curve if c.lam == trained.lam)
    assert chosen.score == pytest.approx(1.0)
    assert chosen.shares["text"] == pytest.approx(0.5)

    write_curve(trained.curve, tmp_path / "curve.csv")
    assert (tmp_path / "curve.csv").read_text(encoding="utf-8").startswith("lambda,score,cost")


def test_write_choices_round_trips(data, tmp_path: Path):
    _, val, _ = data
    choice = {r.question_id: ("text" if r.score["text"] else "vision") for r in val}
    policy = Policy(
        "r",
        choice,
        uses_ocr=dict.fromkeys(choice, True),
        page_s=dict.fromkeys(choice, 0.05),
        question_s=dict.fromkeys(choice, 0.001),
    )
    path = tmp_path / "my-router.csv"
    write_choices(policy, val, path)
    back = read_policy(path)
    assert back.name == "my-router"
    assert back.choice == choice
    assert back.uses_ocr == dict.fromkeys(choice, True)
    assert back.page_s == pytest.approx(dict.fromkeys(choice, 0.05))
    assert back.question_s == pytest.approx(dict.fromkeys(choice, 0.001))
