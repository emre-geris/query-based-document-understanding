import math
from pathlib import Path

import pytest

from qdu.ocr.base import OcrResult
from qdu.router import Policy, read_policy
from qdu.router.labels import LabelRow
from qdu.router.learned_cascade import answer_features, train_learned_cascade
from qdu.router.train import write_choices

OCR_S = 0.5
MODEL_S = {"text": 0.2, "vision": 2.0, "hybrid": 2.2}


def row(i: int, split: str, text_right: bool) -> LabelRow:
    cost_c = {p: MODEL_S[p] + (0.0 if p == "vision" else OCR_S) for p in MODEL_S}
    return LabelRow(
        question_id=f"docvqa/{i}",
        doc_id=f"doc/{i}",
        dataset="docvqa",
        split=split,
        stratum="docvqa/x",
        question="what is the total amount",
        questions_on_page=1,
        ocr_s=OCR_S,
        score={"text": 1.0 if text_right else 0.0, "vision": 1.0, "hybrid": 1.0},
        model_s=dict(MODEL_S),
        cost_c=cost_c,
        cost_b=dict(cost_c),
        tokens={"text": 500.0, "vision": 3000.0, "hybrid": 3500.0},
        oracle_latency="text" if text_right else "vision",
        oracle_tokens="text" if text_right else "vision",
    )


def test_answer_features():
    rows = [row(0, "train", True), row(1, "train", False), row(2, "train", False)]
    answers = {"docvqa/0": "$1,200.50", "docvqa/1": "Not mentioned in the document"}
    ocr = {
        "doc/0": OcrResult("Total: $ 1,200.50 due", "fake", OCR_S),
        "doc/1": OcrResult("Invoice", "fake", OCR_S),
        "doc/2": OcrResult("", "fake", OCR_S),
    }
    confidence = {"sequence": {"docvqa/0": 0.9}, "mean_token": {}, "min_token": {}}
    features, seconds = answer_features(rows, answers, confidence, ocr)
    a, b, c = (features[f"docvqa/{i}"] for i in range(3))
    assert (a["ans_in_ocr"], a["ans_number"], a["ans_missing"]) == (1.0, 1.0, 0.0)
    assert a["ans_conf_sequence"] == 0.9 and math.isnan(a["ans_conf_mean_token"])
    assert (b["ans_missing"], b["ans_in_ocr"], b["ans_words"]) == (1.0, 0.0, 5.0)
    assert (c["ans_empty"], c["ans_in_ocr"], c["ans_chars"]) == (1.0, 0.0, 0.0)
    assert set(seconds) == {"docvqa/0", "docvqa/1", "docvqa/2"}


@pytest.fixture
def data():
    rows = [row(i, "train" if i < 240 else "val", text_right=i % 2 == 0) for i in range(320)]
    features = {
        r.question_id: {
            "ans_in_ocr": float(r.score["text"] == 1.0),
            "ans_conf_sequence": 0.9 if r.score["text"] else 0.2,
            "s1_n_words": 5.0,
        }
        for r in rows
    }
    seconds = {r.question_id: 0.001 for r in rows}
    layout_s = {r.question_id: 0.1 for r in rows}
    return rows, features, seconds, layout_s


def test_learned_cascade_escalates_doubtful_answers_and_pays_twice(data):
    rows, features, seconds, layout_s = data
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "val"]
    cascade = train_learned_cascade(
        train, val, features, seconds, layout_s, signals=("s1",), signals_need_layout=False
    )
    assert cascade.met_budget
    policy = cascade.policy(val, features, seconds, layout_s)
    for r in val:
        q = r.question_id
        assert policy.uses_ocr[q] and policy.page_s[q] == 0.0
        if r.score["text"]:
            assert policy.choice[q] == "text" and q not in policy.called_before
            assert policy.cost(r, "C") == pytest.approx(0.2 + OCR_S + 0.001)
        else:
            assert policy.choice[q] == "vision" and policy.called_before[q] == ("text",)
            assert policy.cost(r, "C") == pytest.approx(0.2 + 2.0 + OCR_S + 0.001)


def test_choices_round_trip_with_called_before(data, tmp_path: Path):
    rows, *_ = data
    val = [r for r in rows if r.split == "val"][:4]
    choice = {r.question_id: "vision" for r in val}
    before = {val[0].question_id: ("text",), val[1].question_id: ("text", "hybrid")}
    write_choices(Policy("c", choice, called_before=before), val, tmp_path / "c.csv")
    assert (
        (tmp_path / "c.csv").read_text(encoding="utf-8").splitlines()[0].endswith("called_before")
    )
    back = read_policy(tmp_path / "c.csv")
    assert back.called_before == before
    assert back.cost(val[1], "C") == pytest.approx(0.2 + 2.2 + 2.0 + OCR_S)

    write_choices(Policy("r", choice), val, tmp_path / "r.csv")  # no cascade: no extra column
    header = (tmp_path / "r.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "question_id,pipeline,uses_ocr,page_s,question_s"
    assert read_policy(tmp_path / "r.csv").called_before == {}


def test_read_policy_rejects_unknown_called_before(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("question_id,pipeline,called_before\ndocvqa/1,vision,ocr\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown called_before"):
        read_policy(path)
