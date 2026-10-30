import json
import re
from pathlib import Path

import pytest
from PIL import Image

from qdu import benchmark
from qdu.data import read_manifest
from qdu.models import FakeModel
from qdu.ocr import OcrResult
from qdu.runner import scored_predictions


class PrintedCodeOcr:
    """Pretends to OCR the fake pages: the page for doc 500+i contains CODEi."""

    name = "printed-code"

    def extract(self, image: Image.Image, key: str) -> OcrResult:
        return OcrResult(f"CODE{int(key.split('/')[1]) - 500}", self.name, 0.1)


def reads_code_from_ocr(prompt: str, n_images: int) -> str:
    match = re.search(r"CODE\d+", prompt.split("Question:")[0])
    return match.group(0) if match else "unknown"


def guesses_every_third(prompt: str, n_images: int) -> str:
    i = int(re.search(r"code (\d+)", prompt).group(1))
    return f"CODE{i}" if i % 3 == 0 else "no idea"


def test_build_sample_can_take_every_question(tmp_path: Path, fake_docvqa) -> None:
    manifest = tmp_path / "all.csv"
    info = benchmark.build_sample(
        manifest, datasets=["docvqa"], per_dataset=None, split=None, loaded={"docvqa": fake_docvqa}
    )
    records = read_manifest(manifest)
    assert len(records) == len(fake_docvqa) == info["questions"]
    assert info["one_per_document"] is False
    assert all(r.stratum.startswith("docvqa/") for r in records)


def test_build_sample_writes_manifest_and_info(tmp_path: Path, fake_docvqa) -> None:
    manifest = tmp_path / "sample.csv"
    info = benchmark.build_sample(
        manifest,
        datasets=["docvqa"],
        per_dataset=12,
        split=None,
        min_per_stratum=2,
        loaded={"docvqa": fake_docvqa},
    )
    records = read_manifest(manifest)
    assert len(records) == 12
    assert all(r.question_id.startswith("docvqa/") and r.split for r in records)
    source = info["sources"]["docvqa"]
    assert source["sample_strata"] == {s: 3 for s in source["population_strata"]}
    assert {r.stratum for r in records} == {f"docvqa/{s}" for s in source["population_strata"]}
    saved = json.loads(benchmark.info_path(manifest).read_text())
    assert saved["manifest_sha256"] == info["manifest_sha256"]


def test_run_and_report(tmp_path: Path, fake_docvqa) -> None:
    manifest = tmp_path / "sample.csv"
    benchmark.build_sample(
        manifest,
        datasets=["docvqa"],
        per_dataset=12,
        split=None,
        min_per_stratum=2,
        loaded={"docvqa": fake_docvqa},
    )
    run_dir = tmp_path / "run"

    def run():
        return benchmark.run(
            run_dir,
            text_model=FakeModel(name="text-fake", responder=reads_code_from_ocr),
            vision_model=FakeModel(name="vlm-fake", responder=guesses_every_third),
            ocr=PrintedCodeOcr(),
            manifest=manifest,
            loaded={"docvqa": fake_docvqa},
        )

    first = run()
    assert (first.completed, first.failed) == (36, 0)
    assert run().skipped == 36  # resumable
    assert json.loads((run_dir / "run_info.json").read_text())["text_model"] == "text-fake"

    result = benchmark.report(run_dir)
    summary, by_dataset = result.by_stratum, result.by_dataset
    assert set(by_dataset.by_stratum) == {"docvqa"}
    rows = {(m.dataset, m.pipeline): m for m in result.metrics}
    assert rows[("ALL", "text")].anls == 1.0
    assert rows[("ALL", "text")].ocr_s == pytest.approx(
        0.1
    )  # PrintedCodeOcr reports 0.1 s per page
    assert rows[("ALL", "vision")].ocr_s == 0.0
    assert rows[("ALL", "oracle")].n == 12
    assert "relaxed" in benchmark.format_metrics(result.metrics)
    assert summary.overall.mean_score["text"] == 1.0
    assert summary.overall.oracle_score == 1.0
    # Ties are broken by measured latency: the fake VLM takes 0 s, the text path pays 0.1 s of
    # OCR, so questions the vision path also answers are labelled "vision", the rest "text".
    vision_correct = {
        r.question_id
        for r in scored_predictions(run_dir / "predictions.jsonl")
        if r.pipeline == "vision" and r.score == 1.0
    }
    assert vision_correct
    assert summary.labels == {
        q: "vision" if q in vision_correct else "text" for q in summary.labels
    }
    for name in (
        "summary.json",
        "metrics.csv",
        "by_dataset.csv",
        "by_stratum.csv",
        "oracle_labels.csv",
    ):
        assert (run_dir / name).exists()
    assert "ALL" in benchmark.format_summary(summary)


def test_rescore_rederives_answers_from_raw_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "question_id": "chartqa/1",
            "pipeline": "hybrid",
            "answers": ["140858"],
            "raw_output": "140 858",
            "prediction": "140 858",
            "anls": 0.0,
            "error": None,
        },
        {
            "question_id": "chartqa/1",
            "pipeline": "vision",
            "answers": ["140858"],
            "raw_output": "140858",
            "prediction": "140858",
            "anls": 1.0,
            "error": None,
        },
        {
            "question_id": "chartqa/2",
            "pipeline": "text",
            "answers": ["x"],
            "raw_output": None,
            "prediction": None,
            "anls": None,
            "error": "Timeout",
        },
    ]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    stats = benchmark.rescore(run_dir)
    assert (stats.rows, stats.changed) == (2, 1)
    assert stats.anls_before == 0.5 and stats.anls_after == 1.0
    saved = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert saved[0]["prediction"] == "140858" and saved[0]["anls"] == 1.0
    assert saved[2]["error"] == "Timeout"  # failed rows untouched
    assert (run_dir / "rescore_info.json").exists()
