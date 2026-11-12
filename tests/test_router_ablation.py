import math
from pathlib import Path

import pytest

from qdu.router.ablation import (
    AblationResult,
    Config,
    auc_comparisons,
    config_cost,
    format_ablation,
    format_comparisons,
    run_ablation,
    with_dataset,
    write_ablation,
)
from qdu.router.labels import LabelRow

OCR_S = 0.5
MODEL_S = {"text": 0.2, "vision": 2.0, "hybrid": 2.2}
NEEDS_OCR = {"s1": False, "s2": True, "s3": False, "s4": True}
NEEDS_LAYOUT = {"s1": False, "s2": False, "s3": True, "s4": True}


def row(i: int) -> LabelRow:
    """Even questions: text answers (and the OCR signal s2 says so); odd: only vision does."""
    text_ok = i % 2 == 0
    cost_c = {p: MODEL_S[p] + (0.0 if p == "vision" else OCR_S) for p in MODEL_S}
    return LabelRow(
        question_id=f"q/{i}",
        doc_id=f"doc/{i}",
        dataset="docvqa" if i % 3 else "chartqa",
        split="train" if i < 240 else "val",
        stratum="x",
        question="what is the value",
        questions_on_page=1,
        ocr_s=OCR_S,
        score={"text": 1.0 if text_ok else 0.0, "vision": 1.0, "hybrid": 1.0 if i % 5 else 0.0},
        model_s=dict(MODEL_S),
        cost_c=cost_c,
        cost_b=dict(cost_c),
        tokens={"text": 500.0, "vision": 3000.0, "hybrid": 3500.0},
        oracle_latency="text" if text_ok else "vision",
        oracle_tokens="text" if text_ok else "vision",
    )


@pytest.fixture
def data():
    rows = [row(i) for i in range(320)]
    features = {
        r.question_id: {"s1_len": 4.0, "s2_conf": float(r.score["text"] == 1.0)} for r in rows
    }
    seconds = {r.question_id: {"s1": 0.001, "s2": 0.002, "s3": 0.0, "s4": 0.0} for r in rows}
    layout_s = {r.question_id: 0.1 for r in rows}
    return rows, features, seconds, layout_s


def test_with_dataset_adds_one_hot(data):
    rows, features, _, _ = data
    out = with_dataset(rows, features)
    assert out["q/0"]["dataset_chartqa"] == 1.0 and out["q/0"]["dataset_docvqa"] == 0.0
    assert "dataset_chartqa" not in features["q/0"]  # input untouched


def test_config_cost_follows_the_groups(data):
    _, _, seconds, layout_s = data
    ids = ["q/1"]
    question_only = config_cost(
        Config("a", ("s1",)), ids, seconds, layout_s, NEEDS_OCR, NEEDS_LAYOUT
    )
    assert question_only == {"uses_ocr": {"q/1": False}, "page_s": {"q/1": 0.0},
                             "question_s": {"q/1": 0.001}}  # fmt: skip
    ocr = config_cost(Config("b", ("s1", "s2", "s3", "dataset")), ids, seconds, layout_s,
                      NEEDS_OCR, NEEDS_LAYOUT)  # fmt: skip
    assert ocr["uses_ocr"] == {"q/1": True}
    assert ocr["page_s"] == {"q/1": 0.1}
    assert ocr["question_s"] == {"q/1": pytest.approx(0.003)}


def test_run_ablation_finds_the_informative_signal(data, tmp_path: Path):
    rows, features, seconds, layout_s = data
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "val"]
    configs = (Config("S1", ("s1",), tfidf=False), Config("S1+S2", ("s1", "s2"), tfidf=False))
    results = run_ablation(
        train, val, features, seconds, layout_s, NEEDS_OCR, NEEDS_LAYOUT, configs=configs
    )
    s1, s12 = results
    assert s1.auc["text_ok"] == pytest.approx(0.5)  # a constant feature knows nothing
    assert s12.auc["text_ok"] == pytest.approx(1.0)
    assert not s1.uses_ocr and s12.uses_ocr
    assert not math.isnan(s12.auc_within_text_ok)

    assert "S1+S2" in format_ablation(results)
    write_ablation(results, tmp_path / "ablation.csv")
    lines = (tmp_path / "ablation.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("configuration,groups,tfidf,auc_text_ok")
    assert len(lines) == 3


def test_auc_comparisons_paired_bootstrap(data):
    rows, *_ = data
    val = [r for r in rows if r.split == "val"]
    perfect = {r.question_id: float(r.score["text"] >= r.score["vision"]) for r in val}
    flat = dict.fromkeys(perfect, 0.5)
    results = [
        AblationResult(Config("good", ()), {}, 0.0, 0.0, 0.0, 0.0, False, perfect),
        AblationResult(Config("flat", ()), {}, 0.0, 0.0, 0.0, 0.0, False, flat),
    ]
    pairs = (("good", "flat", "good over flat"), ("good", "good", "same"), ("x", "flat", "skip"))
    better, same = auc_comparisons(results, val, pairs, resamples=200)
    assert better.estimate == pytest.approx(0.5)
    assert better.excludes_zero and better.low > 0.4
    assert (same.estimate, same.low, same.high) == (0.0, 0.0, 0.0)
    assert not same.excludes_zero
    assert "good over flat" in format_comparisons([better, same])
