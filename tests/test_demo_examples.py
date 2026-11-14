import json

import pytest
from PIL import Image
from test_router_live import SIZE, FakeLayout, FakeOcr, page_layout, page_ocr, trained  # noqa: F401

from qdu.cli import main
from qdu.data import QARecord
from qdu.data.records import write_manifest
from qdu.demo import (
    DemoService,
    SharedOcr,
    StoredLayout,
    StoredOcr,
    demo_fake_model,
    describe,
    load_examples,
    load_recorded,
    showcase,
)
from qdu.router import EXTRACTORS
from qdu.router.labels import LabelRow, write_labels
from qdu.router.live import LiveRouter

SCORES = {
    "text_ok": (1.0, 1.0, 1.0),
    "vision": (0.0, 1.0, 1.0),
    "hybrid_only": (0.0, 0.0, 1.0),
    "other": (0.5, 0.5, 0.5),
}
# page -> (split, kinds of its questions)
PAGES = {
    "docvqa/a": ("val", ("vision", "hybrid_only", "text_ok", "other")),
    "docvqa/b": ("val", ("text_ok", "vision", "hybrid_only")),  # fewer questions: first
    "docvqa/c": ("val", ("text_ok", "vision")),  # no hybrid-only question
    "docvqa/t": ("test", ("text_ok", "vision", "hybrid_only")),  # test split: never offered
}


def label(qid, doc, split, kind):
    t, v, h = SCORES[kind]
    return LabelRow(
        question_id=qid,
        doc_id=doc,
        dataset="docvqa",
        split=split,
        stratum="docvqa/x",
        question=f"question {qid}",
        questions_on_page=1,
        ocr_s=0.5,
        score={"text": t, "vision": v, "hybrid": h},
        model_s={"text": 0.2, "vision": 1.4, "hybrid": 1.6},
        cost_c={"text": 0.7, "vision": 1.4, "hybrid": 2.1},
        cost_b={"text": 0.7, "vision": 1.4, "hybrid": 2.1},
        tokens={"text": 500.0, "vision": 3000.0, "hybrid": 3500.0},
        oracle_latency="text",
        oracle_tokens="text",
    )


@pytest.fixture
def run(tmp_path):
    rows, records, preds = [], [], []
    for doc, (split, kinds) in PAGES.items():
        for i, kind in enumerate(kinds):
            qid = f"{doc}-{i}"
            rows.append(label(qid, doc, split, kind))
            records.append(QARecord(qid, doc, f"question {qid}", (f"gold {qid}",), split=split))
            for path in ("text", "vision", "hybrid"):
                right = rows[-1].score[path] >= 1
                preds.append(
                    {
                        "question_id": qid,
                        "pipeline": path,
                        "prediction": f"gold {qid}" if right else "wrong",
                        "model": "Qwen/Qwen3.5-4B",
                        "input_tokens": 100,
                        "output_tokens": 3,
                        "model_latency_s": 1.0,
                        "ocr_latency_s": 0.8 if path != "vision" else None,
                        "error": None,
                    }
                )
    # a failed retry that must be ignored, and a later successful row that wins
    preds.append({**preds[0], "prediction": "", "error": "timeout"})
    write_labels(rows, tmp_path / "labels.csv")
    write_manifest(records, tmp_path / "full.csv")
    with (tmp_path / "predictions.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in preds)
    return tmp_path


def test_examples_are_val_only_and_classified(run):
    examples = load_examples(run, run / "full.csv")
    assert {e.doc_id for e in examples} == {"docvqa/a", "docvqa/b", "docvqa/c"}
    kinds = {e.question_id: e.kind for e in examples}
    assert kinds["docvqa/a-1"] == "hybrid_only" and kinds["docvqa/a-3"] == "other"
    assert examples[0].record.answers == ("gold docvqa/a-0",)


def test_showcase_picks_pages_covering_all_three_paths(run):
    pages = showcase(load_examples(run, run / "full.csv"), per_dataset=2)
    assert [p[0].doc_id for p in pages] == ["docvqa/b", "docvqa/a"]
    assert [e.kind for e in pages[0]] == ["text_ok", "vision", "hybrid_only"]
    assert len(showcase(load_examples(run, run / "full.csv"))) == 1


def test_recorded_answers_skip_failed_rows(run):
    rec = load_recorded(["docvqa/a-0", "docvqa/b-2"], run)
    assert set(rec) == {"docvqa/a-0", "docvqa/b-2"}
    out = rec["docvqa/a-0"]["text"]
    assert out.answer == "wrong" and out.model_latency_s == 1.0 and out.ocr_latency_s == 0.8
    assert rec["docvqa/b-2"]["hybrid"].answer == "gold docvqa/b-2"


def test_stored_ocr_and_layout_build_the_live_fallback_once():
    built = []

    def factory():
        built.append(1)
        return FakeOcr({"new": page_ocr("plain")})

    ocr = StoredOcr({"old": page_ocr("dense")}, factory)
    img = Image.new("RGB", SIZE, "white")
    assert ocr.extract(img, "old").engine == "fake" and built == []
    ocr.extract(img, "new")
    ocr.extract(img, "new")
    assert built == [1]
    with pytest.raises(KeyError, match="no stored OCR"):
        StoredOcr({}).extract(img, "x")
    layout = StoredLayout({"old": page_layout("old", "chart")}, lambda: FakeLayout({}))
    assert layout.detect(img, "old").regions[0].label == "chart"
    with pytest.raises(KeyError, match="no stored layout"):
        StoredLayout({}).detect(img, "x")


def test_service_scores_and_uses_recorded_answers(trained, run):  # noqa: F811
    router, val, ocr, layout, _, kinds = trained
    shared = SharedOcr(FakeOcr(ocr))
    live = LiveRouter(
        router, {k: EXTRACTORS[k]() for k in ("s1", "s2", "s3")}, shared, FakeLayout(layout)
    )
    service = DemoService(live, demo_fake_model(), shared)
    chart = next(r for r in val if kinds[r.doc_id] == "chart")
    recorded = load_recorded(["docvqa/b-1"], run)["docvqa/b-1"]  # vision right, text wrong
    a = service.ask(
        chart.question, Image.new("RGB", SIZE, "white"), chart.doc_id, compare=True,
        answers=("gold docvqa/b-1",), dataset="docvqa", recorded=recorded,
    )  # fmt: skip
    assert a.path == "vision" and a.run.recorded and a.metric == "anls"
    assert a.score == 1.0 and a.compare["text"].score == 0.0
    assert a.run.model_s == 1.0 and a.compare["text"].ocr_s == pytest.approx(0.8)
    text = describe(a)
    assert "recorded in the labelled run" in text and "Gold: gold docvqa/b-1; anls = 1.00" in text


def test_forced_text_with_recorded_answer_still_charges_the_page_ocr(trained, run):  # noqa: F811
    router, val, ocr, layout, _, _ = trained
    shared = SharedOcr(FakeOcr(ocr))
    live = LiveRouter(
        router, {k: EXTRACTORS[k]() for k in ("s1", "s2", "s3")}, shared, FakeLayout(layout)
    )
    service = DemoService(live, demo_fake_model(), shared)
    r = val[0]
    recorded = load_recorded(["docvqa/b-0"], run)["docvqa/b-0"]
    a = service.ask(
        r.question, Image.new("RGB", SIZE, "white"), r.doc_id, "text", recorded=recorded
    )
    assert a.run.recorded and a.ocr_s == pytest.approx(0.5)  # the page's OCR, read once
    assert shared.engine.calls == [r.doc_id]


def test_cli_demo_ask_needs_image_or_example(tmp_path, capsys):
    assert main(["demo", "ask"]) == 2
    assert "either --image" in capsys.readouterr().err
    assert main(["demo", "ask", "--image", "p.png", "--example", "docvqa/1"]) == 2
    assert main(["demo", "ask", "--image", "p.png"]) == 2
    assert "needs at least one --question" in capsys.readouterr().err


def test_cli_demo_examples_lists_pages(run, capsys):
    assert main(["demo", "examples", "--run", str(run), "--manifest", str(run / "full.csv")]) == 0
    out = capsys.readouterr().out
    assert "docvqa/b" in out and "hybrid_only" in out and "docvqa/t" not in out
