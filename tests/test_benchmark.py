import json
import re
from pathlib import Path

import pytest
from PIL import Image

from qdu import benchmark
from qdu.data import read_manifest
from qdu.models import FakeModel
from qdu.ocr import OcrResult


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
    assert summary.overall.mean_anls["text"] == 1.0
    assert summary.overall.oracle_anls == 1.0
    assert summary.overall.label_counts == {"text": 12, "vision": 0, "hybrid": 0}
    for name in (
        "summary.json",
        "metrics.csv",
        "by_dataset.csv",
        "by_stratum.csv",
        "oracle_labels.csv",
    ):
        assert (run_dir / name).exists()
    assert "ALL" in benchmark.format_summary(summary)
