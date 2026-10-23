"""DocVQA benchmark: build a fixed sample, run pipelines over it, and report results.

The benchmark is how the system is evaluated (per-pipeline ANLS and cost) and where the
router's training labels come from: for every question, the best pipeline according to the
routing oracle (see :mod:`qdu.evaluation.oracle`).
"""

import csv
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdu.data import label_counts, read_manifest, stratified_sample, write_manifest
from qdu.data.docvqa import DATASET_CONFIG, DATASET_ID, DocVQAImages, iter_records, load_docvqa
from qdu.data.sampling import assign_stratum
from qdu.evaluation.oracle import DEFAULT_COST_ORDER, GroupSummary, OracleSummary, summarize
from qdu.models.base import LanguageModel
from qdu.ocr.base import OcrEngine
from qdu.pipelines import build_pipeline
from qdu.provenance import file_sha256, run_metadata
from qdu.runner import PredictionRow, RunStats, run_pipelines, scored_predictions

if TYPE_CHECKING:
    from datasets import Dataset

DEFAULT_MANIFEST = Path("benchmarks/docvqa_validation_200.csv")
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_OCR_CACHE = Path("data/ocr_cache/docvqa")


def info_path(manifest: Path) -> Path:
    return manifest.with_suffix(".info.json")


def build_sample(
    manifest: Path = DEFAULT_MANIFEST,
    *,
    n: int = 200,
    min_per_stratum: int = 10,
    seed: int = 42,
    split: str = "validation",
    dataset: "Dataset | None" = None,
) -> dict[str, Any]:
    """Write the sample manifest and its ``.info.json``; return the info."""
    records = list(iter_records(dataset if dataset is not None else load_docvqa(split)))
    counts = label_counts(records)
    population = Counter(assign_stratum(r.question_types, counts) for r in records)
    sample = stratified_sample(records, n, min_per_stratum=min_per_stratum, seed=seed)
    write_manifest(sample, manifest)

    info = run_metadata(
        dataset=f"{DATASET_ID}/{DATASET_CONFIG}",
        split=split,
        seed=seed,
        n=n,
        min_per_stratum=min_per_stratum,
        one_per_document=True,
        population_questions=len(records),
        population_documents=len({r.doc_id for r in records}),
        population_label_counts=dict(counts.most_common()),
        population_strata=dict(population.most_common()),
        sample_strata=dict(Counter(r.stratum for r in sample).most_common()),
        manifest_sha256=file_sha256(manifest),
    )
    info_path(manifest).write_text(json.dumps(info, ensure_ascii=False, indent=2), "utf-8")
    return info


def run(
    run_dir: Path,
    *,
    text_model: LanguageModel,
    vision_model: LanguageModel,
    ocr: OcrEngine,
    pipelines: Sequence[str] = DEFAULT_COST_ORDER,
    manifest: Path = DEFAULT_MANIFEST,
    limit: int | None = None,
    split: str = "validation",
    dataset: "Dataset | None" = None,
    on_row: Callable[[PredictionRow], None] | None = None,
) -> RunStats:
    """Run the pipelines on the manifest; resumable, appends to ``predictions.jsonl``."""
    records = read_manifest(manifest)[:limit]
    built = [
        build_pipeline(p, text_model=text_model, vision_model=vision_model, ocr=ocr)
        for p in pipelines
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    info = run_metadata(
        text_model=text_model.name,
        vision_model=vision_model.name,
        pipelines=list(pipelines),
        ocr_engine=ocr.name,
        manifest=manifest.as_posix(),
        manifest_sha256=file_sha256(manifest),
        questions=len(records),
    )
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    images = DocVQAImages(dataset if dataset is not None else load_docvqa(split))
    return run_pipelines(
        records,
        lambda r: images.get(r.question_id),
        built,
        run_dir / "predictions.jsonl",
        on_row=on_row,
    )


def _group_dict(g: GroupSummary) -> dict[str, Any]:
    return {
        "n_questions": g.n_questions,
        "mean_anls": g.mean_anls,
        "oracle_anls": g.oracle_anls,
        "best_single": g.best_single,
        "headroom": g.headroom,
        "oracle_label_counts": g.label_counts,
    }


def report(run_dir: Path, *, tolerance: float = 0.0) -> OracleSummary:
    """Write ``summary.json``, ``by_stratum.csv`` and ``oracle_labels.csv`` for a run."""
    rows = scored_predictions(run_dir / "predictions.jsonl")
    present = {r.pipeline for r in rows}
    pipelines = tuple(p for p in DEFAULT_COST_ORDER if p in present)
    summary = summarize(rows, pipelines, tolerance=tolerance)

    result = {
        "pipelines": list(summary.pipelines),
        "tolerance": tolerance,
        "incomplete_questions": summary.incomplete_questions,
        "overall": _group_dict(summary.overall),
        "by_stratum": {s: _group_dict(g) for s, g in summary.by_stratum.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (run_dir / "by_stratum.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stratum", "n", *pipelines, "oracle", "best_single", "headroom"])
        for name, g in [("ALL", summary.overall), *summary.by_stratum.items()]:
            w.writerow(
                [
                    name,
                    g.n_questions,
                    *(f"{g.mean_anls[p]:.4f}" for p in pipelines),
                    f"{g.oracle_anls:.4f}",
                    g.best_single,
                    f"{g.headroom:.4f}",
                ]
            )

    with (run_dir / "oracle_labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "oracle_pipeline"])
        w.writerows(sorted(summary.labels.items()))
    return summary


def format_summary(summary: OracleSummary) -> str:
    pipelines = summary.pipelines
    lines = [
        f"  {'stratum':<18}{'n':>5}"
        + "".join(f"{p:>9}" for p in pipelines)
        + f"{'oracle':>9}   labels ({'/'.join(pipelines)})"
    ]
    for name, g in [("ALL", summary.overall), *summary.by_stratum.items()]:
        scores = "".join(f"{g.mean_anls[p]:>9.3f}" for p in pipelines)
        labels = "/".join(str(g.label_counts[p]) for p in pipelines)
        lines.append(f"  {name:<18}{g.n_questions:>5}{scores}{g.oracle_anls:>9.3f}   {labels}")
    if summary.incomplete_questions:
        lines.append(f"\n  {summary.incomplete_questions} questions skipped (missing a result)")
    return "\n".join(lines)
