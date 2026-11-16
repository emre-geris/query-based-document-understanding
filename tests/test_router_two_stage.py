from dataclasses import replace

import pytest

from qdu.cli import main
from qdu.router.baselines import Budget, parse_budget, path_budget
from qdu.router.labels import LabelRow
from qdu.router.lodo import leave_one_out
from qdu.router.two_stage import NeedsImage, load_two_stage, save_two_stage, train_two_stage

OCR_S = 0.5
LAYOUT_S = 0.1
MODEL_S = {"text": 0.2, "vision": 2.0, "hybrid": 2.2}
NEEDS_LAYOUT = {"s1": False, "s2": False, "s3": True, "s4": True}


def row(i: int, split: str, kind: str) -> LabelRow:
    """kind: "chart" (vision only, visible from the question), "hidden" (vision only, visible
    only from the OCR signal s2), "plain" (text answers) or "dense" (only hybrid answers,
    visible only from the OCR signal s2_dense)."""
    question = {
        "chart": "which bar in the chart is highest",
        "hidden": "what is the reference code",
        "plain": "what is the reference code",
        "dense": "what is the reference code",
    }[kind]
    text_ok = kind == "plain"
    score = {
        "text": 1.0 if text_ok else 0.0,
        "vision": 0.0 if kind == "dense" else 1.0,
        "hybrid": 1.0,
    }
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
        oracle_latency="text" if text_ok else ("hybrid" if kind == "dense" else "vision"),
        oracle_tokens="text" if text_ok else ("hybrid" if kind == "dense" else "vision"),
    )


def make_data(kinds):
    rows = [row(i, "train" if i < 240 else "val", kinds[i % len(kinds)]) for i in range(320)]
    features, seconds = {}, {}
    for r in rows:
        kind = kinds[int(r.question_id.split("/")[1]) % len(kinds)]
        features[r.question_id] = {
            "s1_cue_visual": float(kind == "chart"),
            "s3_figure_share": float(kind == "chart"),
            "s2_low_conf_share": float(kind in ("hidden", "dense")),
            "s2_dense": float(kind == "dense"),
        }
        seconds[r.question_id] = {"s1": 0.001, "s2": 0.002, "s3": 0.003}
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "val"]
    layout_s = {r.question_id: LAYOUT_S for r in rows}
    return train, val, features, seconds, layout_s


@pytest.fixture
def data():
    return make_data(("chart", "hidden", "plain", "plain"))


@pytest.fixture
def dense_data():
    return make_data(("chart", "hidden", "plain", "dense"))


def test_needs_image_uses_only_its_prefixes(data):
    train, val, features, _, _ = data
    stage1 = NeedsImage(("s1_", "s3_")).fit(train, features)
    assert stage1.feature_names == ["s1_cue_visual", "s3_figure_share"]
    p = stage1.predict(val, features)
    charts = [r.question_id for r in val if "chart" in r.question]
    assert min(p[q] for q in charts) > max(v for q, v in p.items() if q not in charts)


def test_two_stage_routes_and_charges_ocr_only_in_stage_two(data):
    train, val, features, seconds, layout_s = data
    router = train_two_stage(
        train,
        val,
        features,
        seconds,
        layout_s,
        stage1=("s1", "s3"),
        stage2=("s1", "s2", "s3"),
        needs_layout=NEEDS_LAYOUT,
        paths=("text", "vision"),
    )
    assert router.hybrid is None and len(router.params) == 2
    assert router.met_budget
    best = next(g for g in router.grid if g.params == router.params)
    assert best.score == pytest.approx(1.0)

    policy = router.policy(val, features, seconds, layout_s)
    for r in val:
        q = r.question_id
        if "chart" in r.question:  # stage 1: vision, no OCR, stage-1 time and layout only
            assert (policy.choice[q], policy.uses_ocr[q]) == ("vision", False)
            assert policy.question_s[q] == pytest.approx(0.004)
        else:  # stage 2: OCR paid, stage-2 extractor time added
            assert policy.uses_ocr[q]
            assert policy.question_s[q] == pytest.approx(0.006)
            assert policy.choice[q] == ("text" if r.score["text"] else "vision")
        assert policy.page_s[q] == pytest.approx(LAYOUT_S)


def test_stage_one_without_layout_charges_no_layout_there(data):
    train, val, features, seconds, layout_s = data
    router = train_two_stage(
        train,
        val,
        features,
        seconds,
        layout_s,
        stage1=("s1",),
        stage2=("s1", "s2", "s3"),
        needs_layout=NEEDS_LAYOUT,
        paths=("text", "vision"),
    )
    policy = router.policy(val, features, seconds, layout_s, params=(0.5, 0.5))
    for r in val:
        q = r.question_id
        expected = 0.0 if not policy.uses_ocr[q] else LAYOUT_S
        assert policy.page_s[q] == pytest.approx(expected)


def test_cli_rejects_ocr_signals_in_stage_one(capsys):
    assert main(["router", "two-stage", "--stage1", "s1,s2"]) == 2
    assert "stage 1 must not need OCR" in capsys.readouterr().err


def test_needs_image_hybrid_target(dense_data):
    train, val, features, _, _ = dense_data
    head = NeedsImage(("s1_", "s2_", "s3_"), target="hybrid").fit(train, features)
    p = head.predict(val, features)
    dense = [r.question_id for r in val if r.score["vision"] == 0.0 and r.score["hybrid"] == 1.0]
    assert min(p[q] for q in dense) > max(v for q, v in p.items() if q not in dense)
    with pytest.raises(ValueError, match="unknown target"):
        NeedsImage(("s1_",), target="nope").fit(train, features)


def test_three_way_router_sends_hybrid_gains_to_hybrid(dense_data):
    train, val, features, seconds, layout_s = dense_data
    budget = 2.0 + OCR_S + LAYOUT_S + 0.01  # hybrid's extra model time needs headroom
    router = train_two_stage(
        train,
        val,
        features,
        seconds,
        layout_s,
        stage1=("s1", "s3"),
        stage2=("s1", "s2", "s3"),
        needs_layout=NEEDS_LAYOUT,
        budget=budget,
    )
    assert router.hybrid is not None and len(router.params) == 3
    best = next(g for g in router.grid if g.params == router.params)
    assert best.score == pytest.approx(1.0)
    assert best.shares["hybrid"] == pytest.approx(0.25)

    policy = router.policy(val, features, seconds, layout_s)
    for r in val:
        q = r.question_id
        if r.score["vision"] == 0.0:  # dense: hybrid, decided in stage 2 after OCR
            assert policy.choice[q] == "hybrid" and policy.uses_ocr[q]
            assert policy.question_s[q] == pytest.approx(0.006)  # no extra time for the head
            # C cost: hybrid model time + OCR once + layout + signal time
            assert policy.cost(r, "C") == pytest.approx(2.2 + OCR_S + LAYOUT_S + 0.006)
        elif "chart" in r.question:
            assert (policy.choice[q], policy.uses_ocr[q]) == ("vision", False)


def test_t3_off_is_the_two_way_router(dense_data):
    train, val, features, seconds, layout_s = dense_data
    kwargs = dict(stage1=("s1", "s3"), stage2=("s1", "s2", "s3"), needs_layout=NEEDS_LAYOUT)
    three = train_two_stage(train, val, features, seconds, layout_s, **kwargs)
    two = train_two_stage(
        train, val, features, seconds, layout_s, paths=("text", "vision"), **kwargs
    )
    for a, b in [(0.3, 0.5), (0.5, 0.2), (1.01, 0.5)]:
        p3 = three.policy(val, features, seconds, layout_s, params=(a, b, 1.01))
        p2 = two.policy(val, features, seconds, layout_s, params=(a, b))
        assert dict(p3.choice) == dict(p2.choice) and "hybrid" not in p3.choice.values()
    # every two-way setting is in the three-way grid with the same score and cost
    by_params = {g.params[:2]: g for g in three.grid if g.params[2] == 1.01}
    for g in two.grid:
        assert by_params[g.params].score == pytest.approx(g.score)
        assert by_params[g.params].cost == pytest.approx(g.cost)


def test_paths_must_include_text_and_vision(data):
    train, val, features, seconds, layout_s = data
    with pytest.raises(ValueError, match="paths must include"):
        train_two_stage(
            train, val, features, seconds, layout_s, needs_layout=NEEDS_LAYOUT, paths=("vision",)
        )


def test_cli_rejects_bad_paths(capsys):
    assert main(["router", "two-stage", "--paths", "text,hybrid"]) == 2
    assert "--paths must include text and vision" in capsys.readouterr().err


def test_three_way_needs_hybrid_gains_in_train(data):
    train, val, features, seconds, layout_s = data  # hybrid never beats vision here
    with pytest.raises(ValueError, match="hybrid > vision"):
        train_two_stage(train, val, features, seconds, layout_s, needs_layout=NEEDS_LAYOUT)


def test_cli_rejects_bad_budget(capsys):
    assert main(["router", "two-stage", "--budget", "cheap"]) == 2
    assert "--budget must be" in capsys.readouterr().err


def test_saved_router_routes_like_the_trained_one(dense_data, tmp_path):
    train, val, features, seconds, layout_s = dense_data
    router = train_two_stage(
        train,
        val,
        features,
        seconds,
        layout_s,
        stage1=("s1", "s3"),
        stage2=("s1", "s2", "s3"),
        needs_layout=NEEDS_LAYOUT,
        budget=10.0,
    )
    path = save_two_stage(router, tmp_path / "r", {"router": "r", "t1": router.params[0]})
    loaded, meta = load_two_stage(tmp_path / "r")  # the folder works as well as the file
    assert path.name == "model.joblib" and meta == {"router": "r", "t1": router.params[0]}
    assert loaded.params == router.params
    assert loaded.stage1_extractors == ("s1", "s3")
    before = router.policy(val, features, seconds, layout_s)
    after = loaded.policy(val, features, seconds, layout_s)
    assert dict(after.choice) == dict(before.choice)
    assert dict(after.page_s) == dict(before.page_s)
    for a, b in zip(
        router.probabilities(val, features), loaded.probabilities(val, features), strict=True
    ):
        assert a == pytest.approx(b)


def test_load_rejects_other_files(tmp_path):
    import joblib

    joblib.dump({"something": 1}, tmp_path / "model.joblib")
    with pytest.raises(ValueError, match="not a two-stage router"):
        load_two_stage(tmp_path)


def test_parse_budget():
    _, val, *_ = make_data(("chart", "hidden", "plain", "plain"))
    hybrid = path_budget(val, "hybrid")
    assert parse_budget("hybrid").on(val) == pytest.approx(hybrid)
    quarter_off = parse_budget("hybrid:0.75")
    assert quarter_off.name == "always-hybrid x 0.75"
    assert quarter_off.on(val) == pytest.approx(0.75 * hybrid)
    assert parse_budget("vision").on(val) == pytest.approx(path_budget(val, "vision"))
    assert parse_budget("1.5") == Budget("fixed", seconds=1.5)
    assert parse_budget("none").on(val) == float("inf")
    for bad in ("cheap", "hybrid:x", "hybrid:0", "vision:-1"):
        with pytest.raises(ValueError):
            parse_budget(bad)


def two_datasets():
    """The same synthetic data twice, as docvqa and chartqa (ids kept apart)."""
    train, val, features, seconds, layout_s = make_data(("chart", "hidden", "plain", "dense"))
    other = {}

    def copy(r):
        qid = r.question_id.replace("docvqa/", "chartqa/")
        other[qid] = r.question_id
        return replace(r, question_id=qid, doc_id=f"c{r.doc_id}", dataset="chartqa")

    train2, val2 = [copy(r) for r in train], [copy(r) for r in val]
    for new, old in other.items():
        features[new], seconds[new], layout_s[new] = features[old], seconds[old], layout_s[old]
    return [*train, *train2], [*val, *val2], features, seconds, layout_s


def test_each_held_out_dataset_is_never_seen_in_fitting_or_tuning():
    train, val, features, seconds, layout_s = two_datasets()
    held = leave_one_out(
        train,
        val,
        [*train, *val],
        features,
        seconds,
        layout_s,
        budget=parse_budget("hybrid"),
        stage1=("s1", "s3"),
        stage2=("s1", "s2", "s3"),
        needs_layout=NEEDS_LAYOUT,
    )
    assert [h.dataset for h in held] == ["chartqa", "docvqa"]
    for h in held:
        assert {r.dataset for r in h.rows} == {h.dataset}
        assert set(h.policy.choice) == {r.question_id for r in h.rows}
        others = [r for r in val if r.dataset != h.dataset]
        assert h.budget_s == pytest.approx(path_budget(others, "hybrid"))
        # the other dataset is an exact copy, so the held-out router routes it perfectly
        assert all(r.score[h.policy.choice[r.question_id]] == 1.0 for r in h.rows)


def test_leaving_out_the_only_dataset_fails():
    train, val, features, seconds, layout_s = make_data(("chart", "plain"))
    with pytest.raises(ValueError, match="leaves no train"):
        leave_one_out(
            train,
            val,
            val,
            features,
            seconds,
            layout_s,
            budget=parse_budget("hybrid"),
            stage1=("s1", "s3"),
            stage2=("s1", "s2", "s3"),
            needs_layout=NEEDS_LAYOUT,
        )


def test_cli_lodo_rejects_bad_budget(capsys):
    assert main(["router", "lodo", "--budget", "hybrid:x"]) == 2
    assert "--budget must be" in capsys.readouterr().err


def with_vision_low(rows):
    """vision-low: as good as vision except on chart questions (which need full detail)."""
    out = []
    for r in rows:
        low = 0.0 if "chart" in r.question else r.score["vision"]
        out.append(
            replace(
                r,
                score={**r.score, "vision-low": low},
                model_s={**r.model_s, "vision-low": 0.5},
                cost_c={**r.cost_c, "vision-low": 0.5},
                cost_b={**r.cost_b, "vision-low": 0.5},
                tokens={**r.tokens, "vision-low": 800.0},
            )
        )
    return out


def test_detail_head_sends_the_low_resolution_page_unless_detail_is_needed():
    train, val, features, seconds, layout_s = make_data(("chart", "hidden", "plain", "plain"))
    train, val = with_vision_low(train), with_vision_low(val)
    router = train_two_stage(
        train,
        val,
        features,
        seconds,
        layout_s,
        stage1=("s1", "s3"),
        stage2=("s1", "s2", "s3"),
        needs_layout=NEEDS_LAYOUT,
        paths=("text", "vision", "vision-low"),
    )
    assert router.detail is not None and len(router.params) == 4
    policy = router.policy(val, features, seconds, layout_s)
    for r in val:
        chosen = policy.choice[r.question_id]
        assert r.score[chosen] == 1.0, r.question_id  # every question answered
        if "chart" in r.question:  # needs full detail
            assert chosen == "vision"
        else:  # the low-resolution page is as good and cheaper than text (which pays OCR)
            assert chosen == "vision-low"
    # with t4 = 0 the router never chooses vision-low: the three-path router
    plain = router.policy(val, features, seconds, layout_s, params=(*router.params[:3], 0.0))
    assert "vision-low" not in set(plain.choice.values())
    assert router.choose(0.9, params=(0.5, 0.5, 1.01, 0.5), p4=0.1) == ("vision-low", 1)
    assert router.choose(0.9, params=(0.5, 0.5, 1.01, 0.5), p4=0.9) == ("vision", 1)
