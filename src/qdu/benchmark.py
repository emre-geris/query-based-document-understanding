"""Benchmark: build a fixed question sample, run pipelines over it, and report results.

The benchmark is how the system is evaluated (per-pipeline ANLS and cost) and where the
router's training labels come from: for every question, the best pipeline according to the
routing oracle (see :mod:`qdu.evaluation.oracle`).
"""

import csv
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdu.data import (
    ImageStore,
    get_source,
    iter_source_records,
    label_counts,
    load_source,
    read_manifest,
    stratified_sample,
    write_manifest,
)
from qdu.data.sampling import assign_stratum
from qdu.evaluation.metrics import MetricRow, metric_table
from qdu.evaluation.oracle import (
    DEFAULT_COST_ORDER,
    GroupSummary,
    OracleSummary,
    ScoredPrediction,
    summarize,
)
from qdu.models.base import LanguageModel
from qdu.ocr.base import OcrEngine
from qdu.pipelines import build_pipeline
from qdu.provenance import file_sha256, run_metadata
from qdu.runner import (
    PredictionRow,
    RunStats,
    read_predictions,
    run_pipelines,
    scored_predictions,
)

if TYPE_CHECKING:
    from datasets import Dataset

PILOT_DATASETS = ("docvqa", "infographicvqa", "chartqa")
DEFAULT_MANIFEST = Path("benchmarks/pilot_300.csv")
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_OCR_CACHE = Path("data/ocr_cache")


def info_path(manifest: Path) -> Path:
    return manifest.with_suffix(".info.json")


def build_sample(
    manifest: Path = DEFAULT_MANIFEST,
    *,
    datasets: Sequence[str] = PILOT_DATASETS,
    per_dataset: int = 100,
    split: str | None = "train",
    min_per_stratum: int = 5,
    seed: int = 42,
    loaded: Mapping[str, "Dataset"] | None = None,
) -> dict[str, Any]:
    """Sample ``per_dataset`` questions from each dataset (one per document) into a manifest.

    ``split`` restricts sampling to one router split. The pilot draws from ``train`` so its
    labels can be reused for router training without touching validation or test documents.
    Strata are ``<dataset>/<rarest question-type label>``.
    """
    sample = []
    per_source: dict[str, Any] = {}
    for name in datasets:
        dataset = loaded[name] if loaded and name in loaded else load_source(name)
        records = [r for r in iter_source_records(name, dataset) if split in (None, r.split)]
        counts = label_counts(records)
        population = Counter(assign_stratum(r.question_types, counts) for r in records)
        chosen = stratified_sample(records, per_dataset, min_per_stratum=min_per_stratum, seed=seed)
        sample += [replace(r, stratum=f"{name}/{r.stratum}") for r in chosen]

        spec = get_source(name)
        per_source[name] = {
            "hf": f"{spec.hf_id}/{spec.hf_config or 'default'}:{spec.hf_split}",
            "license": spec.license,
            "population_questions": len(records),
            "population_documents": len({r.doc_id for r in records}),
            "population_strata": dict(population.most_common()),
            "sample_strata": dict(Counter(r.stratum for r in chosen).most_common()),
        }

    write_manifest(sample, manifest)
    info = run_metadata(
        datasets=list(datasets),
        router_split=split,
        per_dataset=per_dataset,
        min_per_stratum=min_per_stratum,
        seed=seed,
        one_per_document=True,
        questions=len(sample),
        sources=per_source,
        manifest_sha256=file_sha256(manifest),
    )
    info_path(manifest).write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
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
    loaded: Mapping[str, "Dataset"] | None = None,
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

    images = ImageStore(loaded)
    return run_pipelines(records, images.get, built, run_dir / "predictions.jsonl", on_row=on_row)


def _group_dict(g: GroupSummary) -> dict[str, Any]:
    return {
        "n_questions": g.n_questions,
        "mean_anls": g.mean_anls,
        "oracle_anls": g.oracle_anls,
        "best_single": g.best_single,
        "headroom": g.headroom,
        "oracle_label_counts": g.label_counts,
    }


def _write_groups(path: Path, summary: OracleSummary, label: str) -> None:
    pipelines = summary.pipelines
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([label, "n", *pipelines, "oracle", "best_single", "headroom"])
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


def _dataset_of(question_id: str) -> str:
    return question_id.split("/", 1)[0] if "/" in question_id else "unknown"


@dataclass
class Report:
    by_stratum: OracleSummary
    by_dataset: OracleSummary
    metrics: list[MetricRow]


def report(run_dir: Path, *, tolerance: float = 0.0) -> Report:
    """Write summary.json, metrics.csv, by_dataset.csv, by_stratum.csv and oracle_labels.csv."""
    rows = scored_predictions(run_dir / "predictions.jsonl")
    present = {r.pipeline for r in rows}
    pipelines = tuple(p for p in DEFAULT_COST_ORDER if p in present)
    by_stratum = summarize(rows, pipelines, tolerance=tolerance)
    by_dataset = summarize(
        [
            ScoredPrediction(r.question_id, _dataset_of(r.question_id), r.pipeline, r.anls)
            for r in rows
        ],
        pipelines,
        tolerance=tolerance,
    )

    result = {
        "pipelines": list(by_stratum.pipelines),
        "tolerance": tolerance,
        "incomplete_questions": by_stratum.incomplete_questions,
        "overall": _group_dict(by_stratum.overall),
        "by_dataset": {s: _group_dict(g) for s, g in by_dataset.by_stratum.items()},
        "by_stratum": {s: _group_dict(g) for s, g in by_stratum.by_stratum.items()},
    }
    metrics = metric_table(
        read_predictions(run_dir / "predictions.jsonl"), pipelines, by_stratum.labels
    )
    result["metrics"] = [asdict(m) for m in metrics]
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(MetricRow.__dataclass_fields__))
        w.writeheader()
        w.writerows(asdict(m) for m in metrics)
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_groups(run_dir / "by_dataset.csv", by_dataset, "dataset")
    _write_groups(run_dir / "by_stratum.csv", by_stratum, "stratum")

    with (run_dir / "oracle_labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "oracle_pipeline"])
        w.writerows(sorted(by_stratum.labels.items()))
    return Report(by_stratum, by_dataset, metrics)


def format_summary(summary: OracleSummary, label: str = "stratum") -> str:
    pipelines = summary.pipelines
    width = max([len(label), 5, *(len(s) for s in summary.by_stratum)]) + 2
    lines = [
        f"  {label:<{width}}{'n':>5}"
        + "".join(f"{p:>9}" for p in pipelines)
        + f"{'oracle':>9}   labels ({'/'.join(pipelines)})"
    ]
    for name, g in [("ALL", summary.overall), *summary.by_stratum.items()]:
        scores = "".join(f"{g.mean_anls[p]:>9.3f}" for p in pipelines)
        labels = "/".join(str(g.label_counts[p]) for p in pipelines)
        lines.append(f"  {name:<{width}}{g.n_questions:>5}{scores}{g.oracle_anls:>9.3f}   {labels}")
    if summary.incomplete_questions:
        lines.append(f"\n  {summary.incomplete_questions} questions skipped (missing a result)")
    return "\n".join(lines)


def format_metrics(metrics: list[MetricRow]) -> str:
    """Quality and cost per dataset and pipeline; relaxed accuracy matters for ChartQA."""
    lines = [
        f"  {'dataset':<16}{'pipeline':<9}{'n':>5}{'ANLS':>8}{'relaxed':>9}"
        f"{'model s':>9}{'OCR s':>8}{'total s':>9}{'in tok':>9}{'out tok':>9}"
    ]
    for m in metrics:
        tokens_in = f"{m.input_tokens:>9.0f}" if m.input_tokens is not None else f"{'-':>9}"
        tokens_out = f"{m.output_tokens:>9.1f}" if m.output_tokens is not None else f"{'-':>9}"
        lines.append(
            f"  {m.dataset:<16}{m.pipeline:<9}{m.n:>5}{m.anls:>8.3f}{m.relaxed_accuracy:>9.3f}"
            f"{m.model_s:>9.2f}{m.ocr_s:>8.2f}{m.total_s:>9.2f}{tokens_in}{tokens_out}"
        )
    return "\n".join(lines)
