import pytest

from qdu.router.labels import LabelRow
from qdu.router.two_stage import lgbm_params
from qdu.router.variants import (
    StageModel,
    format_variants,
    run_variants,
    tune,
)

OCR_S = 0.5
MODEL_S = {"text": 0.2, "vision": 2.0, "hybrid": 2.2}
NEEDS_LAYOUT = {"s1": False, "s2": False, "s3": True, "s4": True}
SMALL_GRID = ({"num_leaves": 3, "min_child_samples": 5}, {"num_leaves": 7, "min_child_samples": 5})


def row(i: int, split: str, kind: str) -> LabelRow:
    """chart: vision only (seen in the question); hidden: vision only (seen in s2); plain: text."""
    text_ok = kind == "plain"
    cost_c = {p: MODEL_S[p] + (0.0 if p == "vision" else OCR_S) for p in MODEL_S}
    return LabelRow(
        question_id=f"docvqa/{i}",
        doc_id=f"doc/{i // 2}",
        dataset="docvqa",
        split=split,
        stratum="x",
        question="which bar in the chart is highest" if kind == "chart" else "what is the code",
        questions_on_page=2,
        ocr_s=OCR_S,
        score={"text": 1.0 if text_ok else 0.0, "vision": 1.0, "hybrid": 1.0},
        model_s=dict(MODEL_S),
        cost_c=cost_c,
        cost_b=dict(cost_c),
        tokens={"text": 500.0, "vision": 3000.0, "hybrid": 3500.0},
        oracle_latency="text" if text_ok else "vision",
        oracle_tokens="text" if text_ok else "vision",
    )


@pytest.fixture
def data():
    kinds = ("chart", "hidden", "plain", "plain")
    rows = [row(i, "train" if i < 240 else "val", kinds[i % 4]) for i in range(320)]
    features, seconds = {}, {}
    for i, r in enumerate(rows):
        kind = kinds[i % 4]
        features[r.question_id] = {
            "s1_cue_visual": float(kind == "chart"),
            "s3_figure_share": float(kind == "chart"),
            "s2_low_conf_share": float(kind == "hidden"),
        }
        seconds[r.question_id] = {"s1": 0.001, "s2": 0.002, "s3": 0.003, "s4": 0.0}
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "val"]
    return train, val, features, seconds, {r.question_id: 0.1 for r in rows}


def test_lgbm_params_overrides():
    assert lgbm_params(3)["random_state"] == 3
    assert lgbm_params(0, num_leaves=7)["num_leaves"] == 7


@pytest.mark.parametrize("objective", ["classify", "weighted", "regress"])
def test_stage_model_objectives(data, objective):
    train, val, features, _, _ = data
    model = StageModel(("s1_", "s2_", "s3_"), objective=objective).fit(train, features)
    p = model.predict(val, features)
    needs_image = [r.question_id for r in val if r.score["text"] == 0.0]
    text = [r.question_id for r in val if r.score["text"] == 1.0]
    assert min(p[q] for q in needs_image) > max(p[q] for q in text)


def test_stage_model_rejects_unknown_objective(data):
    train, _, features, _, _ = data
    with pytest.raises(ValueError, match="unknown objective"):
        StageModel(("s1_",), objective="ranking").fit(train, features)


def test_tune_picks_from_the_grid(data):
    train, _, features, _, _ = data
    params, auc = tune(train, features, ("s1_", "s2_", "s3_"), grid=SMALL_GRID)
    assert params in [dict(g) for g in SMALL_GRID]
    assert auc == pytest.approx(1.0)


def test_run_variants(data):
    train, val, features, seconds, layout_s = data
    results = run_variants(
        train,
        val,
        features,
        seconds,
        layout_s,
        stage2=("s1", "s2", "s3"),
        needs_layout=NEEDS_LAYOUT,
        tune_grid=SMALL_GRID,
    )
    assert [r.variant for r in results] == ["default", "weighted", "regress", "tuned"]
    for r in results:
        assert r.met_budget
        assert r.auc_text_ok == pytest.approx(1.0)
        assert r.score == pytest.approx(1.0)
    assert set(results[-1].params) == {"stage1", "stage2"}
    assert "regress" in format_variants(results)
    with pytest.raises(ValueError, match="unknown variant"):
        run_variants(train, val, features, seconds, layout_s, needs_layout=NEEDS_LAYOUT,
                     variants=("boosted",))  # fmt: skip
