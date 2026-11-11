import json
from pathlib import Path

import pytest

from qdu.cli import main
from qdu.data import QARecord, write_manifest
from qdu.ocr.base import OcrLine, OcrResult, cache_file, write_cache
from qdu.router import (
    Policy,
    QuestionShape,
    build_features,
    build_labels,
    choose,
    cost_quality_curve,
    evaluate,
    load_ocr,
    mean_costs,
    pack_ocr,
    read_labels,
    read_policy,
    route,
    router_inputs,
    write_labels,
)


# question: (doc, split, gold, {path: (prediction, model s, input tokens)}); OCR 0.5 s per page
def paths(text: tuple, vision: tuple, hybrid: tuple) -> dict[str, tuple]:
    return {"text": text, "vision": vision, "hybrid": hybrid}


QUESTIONS = {
    "docvqa/1": ("docvqa/10", "train", "abc",
                 paths(("abc", 0.2, 500), ("abc", 1.0, 3000), ("abc", 1.2, 3500))),
    "docvqa/2": ("docvqa/10", "train", "xyz",
                 paths(("no", 0.2, 500), ("xyz", 1.0, 3000), ("xyz", 1.2, 3500))),
    "docvqa/3": ("docvqa/11", "val", "q",
                 paths(("no", 0.3, 600), ("no", 1.1, 3100), ("q", 1.3, 3600))),
    "chartqa/4": ("chartqa/12", "val", "7",
                  paths(("7", 0.1, 200), ("7", 0.3, 500), ("8", 0.5, 700))),
}  # fmt: skip
OCR_S = 0.5


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records, rows = [], []
    for qid, (doc, split, gold, by_path) in QUESTIONS.items():
        dataset = qid.split("/")[0]
        records.append(
            QARecord(qid, doc, f"What is {qid}?", (gold,), (), f"{dataset}/x", dataset, split)
        )
        for path, (prediction, model_s, tokens) in by_path.items():
            base = {"question_id": qid, "doc_id": doc, "stratum": f"{dataset}/x"}
            rows.append(
                {
                    **base,
                    "pipeline": path,
                    "question": f"What is {qid}?",
                    "answers": [gold],
                    "prediction": prediction,
                    "anls": 0.0,
                    "input_tokens": tokens,
                    "output_tokens": 2,
                    "model_latency_s": model_s,
                    "ocr_latency_s": None if path == "vision" else OCR_S,
                    "error": None,
                }
            )
    rows.insert(0, {**rows[0], "error": "ConnectionError", "prediction": None})  # retried later
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    manifest = tmp_path / "manifest.csv"
    write_manifest(records, manifest)
    return run_dir, manifest


def test_labels_scores_costs_and_oracle(run) -> None:
    rows = {r.question_id: r for r in build_labels(*run)}
    r = rows["docvqa/2"]
    assert (r.split, r.questions_on_page, r.ocr_s) == ("train", 2, OCR_S)
    assert r.score == {"text": 0.0, "vision": 1.0, "hybrid": 1.0}
    assert r.cost_c == pytest.approx({"text": 0.7, "vision": 1.0, "hybrid": 1.7})
    assert r.cost_b == pytest.approx({"text": 0.45, "vision": 1.0, "hybrid": 1.45})
    assert r.tokens == {"text": 502, "vision": 3002, "hybrid": 3502}
    assert (r.oracle_latency, r.oracle_tokens) == ("vision", "vision")
    assert rows["docvqa/1"].oracle_latency == "text"  # all right: cheapest
    assert rows["docvqa/3"].oracle_latency == "hybrid"
    # ChartQA uses relaxed accuracy; under C text (0.1 + 0.5 OCR) costs more than vision (0.3)
    assert rows["chartqa/4"].oracle_latency == "vision"
    assert rows["chartqa/4"].oracle_tokens == "text"


def test_labels_round_trip_and_filters(run, tmp_path: Path) -> None:
    rows = build_labels(*run)
    path = tmp_path / "labels.csv"
    write_labels(rows, path)
    back = read_labels(path)
    assert [r.question_id for r in back] == [r.question_id for r in rows]
    assert back[0].cost_b == pytest.approx(rows[0].cost_b)
    assert [r.question_id for r in read_labels(path, splits=["val"], datasets=["docvqa"])] == [
        "docvqa/3"
    ]


def test_baselines_and_policy_costs(run) -> None:
    rows = build_labels(*run)
    results = {(m.group, m.method): m for m in evaluate(rows, resamples=50)}
    vision = results[("ALL", "always-vision")]
    assert vision.score == pytest.approx(0.75)
    assert vision.cost_c == pytest.approx((1.0 + 1.0 + 1.1 + 0.3) / 4)
    assert vision.score_vs_baseline is None
    oracle = results[("ALL", "oracle")]
    assert oracle.score == 1.0
    assert oracle.score_vs_baseline.estimate == pytest.approx(0.25)
    assert results[("docvqa", "always-text")].n == 3


def test_router_pays_for_its_own_ocr(run) -> None:
    rows = {r.question_id: r for r in build_labels(*run)}
    r = rows["docvqa/2"]
    plain = Policy("p", {r.question_id: "vision"})
    with_ocr = Policy("p", {r.question_id: "vision"}, uses_ocr={r.question_id: True})
    q = r.question_id
    with_layout = Policy("p", {q: "text"}, page_s={q: 0.2}, question_s={q: 0.01})
    assert plain.cost(r, "C") == pytest.approx(1.0)
    assert with_ocr.cost(r, "C") == pytest.approx(1.5)
    assert with_ocr.cost(r, "B") == pytest.approx(1.25)  # OCR shared by the page's 2 questions
    assert with_ocr.cost(r, "A") == 3002
    assert with_layout.cost(r, "C") == pytest.approx(0.2 + 0.5 + 0.2 + 0.01)
    assert with_layout.cost(r, "B") == pytest.approx(0.2 + 0.35 + 0.01)


def test_policy_must_cover_every_question(run) -> None:
    rows = build_labels(*run)
    with pytest.raises(ValueError, match="no choice"):
        evaluate(rows, [Policy("partial", {"docvqa/1": "text"})], resamples=10)
    with pytest.raises(ValueError, match="reserved"):
        evaluate(rows, [Policy("oracle", {r.question_id: "text" for r in rows})], resamples=10)


def test_decision_rule() -> None:
    quality = {"text": 0.8, "vision": 0.9, "hybrid": 0.9}
    cost = {"text": 0.5, "vision": 1.5, "hybrid": 2.3}
    assert choose(quality, cost, 0.0) == "vision"  # tie on quality: cheaper
    assert choose(quality, cost, 0.2) == "text"
    assert choose({"text": 0.5, "vision": 0.5}, {"text": 1.0, "vision": 1.0}, 0.0) == "vision"


def test_route_and_curve(run) -> None:
    rows = build_labels(*run)
    train = [r for r in rows if r.split == "train"]
    cost = mean_costs(train)
    assert cost["text"] == pytest.approx(0.7)
    predicted = {r.question_id: r.score for r in rows}  # a perfect quality model
    policy = route(predicted, cost, 0.0)
    assert sum(r.score[policy.choice[r.question_id]] for r in rows) == len(rows)
    curve = cost_quality_curve(rows, predicted, cost, [0.0, 10.0])
    assert curve[0][1] == 1.0
    assert curve[1][2] < curve[0][2]  # a high λ buys cheaper paths


def test_read_policy_csv(tmp_path: Path) -> None:
    path = tmp_path / "mine.csv"
    path.write_text("question_id,pipeline,uses_ocr,question_s\nq1,text,1,0.01\nq2,vision,0,\n")
    p = read_policy(path)
    assert (p.name, p.choice) == ("mine", {"q1": "text", "q2": "vision"})
    assert p.uses_ocr == {"q1": True, "q2": False}
    assert p.question_s == {"q1": 0.01}
    path.write_text("question_id,pipeline\nq1,ocr\n")
    with pytest.raises(ValueError, match="unknown pipeline"):
        read_policy(path)


def _cache(tmp_path: Path) -> Path:
    """An engine folder as a Drive zip delivers it: one name mangled, one decoy with the wrong
    time under the right name and the real page renamed with (1)."""
    engine = tmp_path / "ocr" / "rapid-mobile-gpu"
    lines = (OcrLine("abc", 0.9, (0, 0, 10, 10)), OcrLine("xyz", 0.4, (0, 20, 10, 30)))
    write_cache(cache_file(engine, "docvqa/10"), OcrResult("abc\nxyz", "e", OCR_S, lines=lines))
    write_cache(engine / "docvqa" / "11.json", OcrResult("decoy", "e", 9.9))
    write_cache(engine / "docvqa" / "11(1).json", OcrResult("q", "e", OCR_S))
    write_cache(engine / "chartqa" / "12.json", OcrResult("7", "e", OCR_S))
    return engine


def test_pack_ocr_matches_pages_by_their_run_time(run, tmp_path: Path) -> None:
    rows = build_labels(*run)
    out = tmp_path / "ocr.jsonl.gz"
    assert pack_ocr(rows, _cache(tmp_path), out) == 3
    ocr = load_ocr(out)
    assert ocr["docvqa/11"].text == "q"  # not the decoy
    assert ocr["docvqa/10"].lines[1] == OcrLine("xyz", 0.4, (0, 20, 10, 30))
    (tmp_path / "ocr" / "rapid-mobile-gpu" / "chartqa" / "12.json").unlink()
    with pytest.raises(ValueError, match="1 pages"):
        pack_ocr(rows, tmp_path / "ocr" / "rapid-mobile-gpu", out)


def test_features_get_ocr_and_time_extractors(run, tmp_path: Path) -> None:
    rows = build_labels(*run)
    out = tmp_path / "ocr.jsonl.gz"
    pack_ocr(rows, _cache(tmp_path), out)

    class OcrWords:
        name, needs_ocr = "ocr", True

        def __call__(self, item):
            low = sum(line.score < 0.5 for line in item.ocr.lines)
            return {"words": len(item.ocr.text.split()), "low_conf": low}

    table = build_features(router_inputs(rows, load_ocr(out)), [QuestionShape(), OcrWords()])
    assert table.needs_ocr
    assert table.features["docvqa/1"]["ocr_words"] == 2
    assert table.features["docvqa/1"]["ocr_low_conf"] == 1
    assert table.features["docvqa/1"]["q_wh_what"] == 1.0
    assert set(table.seconds["docvqa/1"]) == {"q", "ocr"}
    csv_out = tmp_path / "features.csv"
    table.write(csv_out)
    assert csv_out.read_text().splitlines()[0].startswith("question_id,ocr_low_conf,ocr_words")
    with pytest.raises(ValueError, match="needs OCR"):
        build_features(router_inputs(rows), [OcrWords()])
    with pytest.raises(ValueError, match="no OCR for 1 pages"):
        router_inputs(rows, {k: v for k, v in load_ocr(out).items() if k != "chartqa/12"})


def test_cli_labels_and_evaluate(run, tmp_path: Path, capsys) -> None:
    run_dir, manifest = run
    labels = tmp_path / "labels.csv"
    assert (
        main(["router", "labels", str(run_dir), "--manifest", str(manifest), "--out", str(labels)])
        == 0
    )
    choices = tmp_path / "text_router.csv"
    choices.write_text("question_id,pipeline\ndocvqa/3,text\nchartqa/4,text\n")
    args = ["router", "evaluate", "--labels", str(labels), "--choices", str(choices)]
    assert main([*args, "--resamples", "20", "--out", str(tmp_path / "eval.csv")]) == 0
    out = capsys.readouterr().out
    assert "Split val: 2 questions" in out
    assert "text_router" in out
    assert (tmp_path / "eval.csv").exists()
