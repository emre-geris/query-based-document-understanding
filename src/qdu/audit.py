"""Audit a run: are its scores and costs consistent and traceable to the raw rows?

The report (``qdu benchmark report``) aggregates; the audit re-derives the same numbers from
``predictions.jsonl`` by a separate, simpler path and checks invariants that the cost and
score accounting relies on, so a reader can trust (or question) every figure:

- every (question, pipeline) has one successful row; errors and retries are counted
- scores: the stored ANLS matches a fresh computation; the primary metric is recomputed
- costs: vision rows carry no OCR; text and hybrid rows of one page report the same OCR time,
  engine and CPU; model latency and token counts are present and positive
- per dataset and pipeline, the primary score and the C / B costs recomputed here match
  ``metrics.csv`` and ``summary.json`` when they exist
- a few rows per dataset are printed (question, gold, prediction, score) for eyeballing
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from qdu.evaluation.anls import anls_score
from qdu.evaluation.primary import primary_metric, primary_score
from qdu.runner import read_predictions


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def audit(run_dir: Path, *, examples: int = 3) -> tuple[list[str], list[str]]:
    """Return (report lines, problems). An empty problem list means every check passed."""
    lines: list[str] = []
    problems: list[str] = []
    rows = list(read_predictions(run_dir / "predictions.jsonl"))
    errors = [r for r in rows if r.get("error")]
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if not r.get("error"):
            latest[(r["question_id"], r["pipeline"])] = r
    retried = len(rows) - len(errors) - len(latest)
    lines.append(
        f"rows {len(rows)}: {len(latest)} successful (question, pipeline) pairs, "
        f"{len(errors)} failed calls, {retried} duplicate successes"
    )
    if not latest:
        problems.append(f"no successful predictions in {run_dir} (wrong path or empty run?)")
    if retried:
        problems.append(f"{retried} (question, pipeline) pairs were answered more than once")
    for r in errors[:5]:
        lines.append(f"  error {r['question_id']} {r['pipeline']}: {r['error']}")

    pipelines = sorted({p for _, p in latest})
    questions = sorted({q for q, _ in latest})
    complete = [q for q in questions if all((q, p) in latest for p in pipelines)]
    lines.append(f"questions {len(questions)}, complete for {pipelines}: {len(complete)}")

    # Scores
    stale = [
        key
        for key, r in latest.items()
        if abs(anls_score(r["prediction"], r["answers"]) - r["anls"]) > 1e-9
    ]
    if stale:
        problems.append(f"{len(stale)} rows whose stored ANLS differs from a fresh computation")

    # Costs
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_issues: dict[str, list[str]] = defaultdict(list)
    for (_, pipeline), r in latest.items():
        where = f"{r['question_id']} {pipeline}"
        if r["model_latency_s"] is None or r["model_latency_s"] <= 0:
            row_issues["model latency missing or not positive"].append(where)
        if not r.get("input_tokens") or r.get("output_tokens") is None:
            row_issues["token counts missing"].append(where)
        if pipeline == "vision":
            if r.get("ocr_latency_s") or r.get("ocr_engine"):
                row_issues["vision row carries OCR cost"].append(where)
        else:
            if r.get("ocr_latency_s") is None:
                row_issues["OCR time missing"].append(where)
            if not r.get("ocr_hardware"):
                row_issues["OCR CPU not recorded (runs before Nov 2026 lack it)"].append(where)
            by_page[r["doc_id"]].append(r)
    for issue, where in row_issues.items():
        problems.append(f"{issue}: {len(where)} rows, e.g. {', '.join(where[:3])}")
    mixed = [
        doc
        for doc, page_rows in by_page.items()
        if len({(r["ocr_latency_s"], r["ocr_engine"], r.get("ocr_hardware")) for r in page_rows})
        > 1
    ]
    if mixed:
        problems.append(f"{len(mixed)} pages with different OCR time/engine/CPU across rows")
    engines = sorted({r["ocr_engine"] for rs in by_page.values() for r in rs})
    cpus = sorted({str(r.get("ocr_hardware")) for rs in by_page.values() for r in rs})
    lines.append(f"OCR engines {engines}; OCR measured on {cpus}")

    # Per dataset and pipeline, from the complete questions only (as the report does)
    doc_questions: dict[str, set[str]] = defaultdict(set)
    for q in complete:
        doc_questions[latest[(q, pipelines[0])]["doc_id"]].add(q)
    table: dict[tuple[str, str], dict[str, float]] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for q in complete:
        for p in pipelines:
            groups[(q.split("/", 1)[0], p)].append(latest[(q, p)])
    lines.append("")
    lines.append(
        f"  {'dataset':<16}{'pipeline':<9}{'n':>4}{'metric':>9}{'score':>8}"
        f"{'model s':>9}{'OCR s':>8}{'C s':>8}{'B s':>8}{'in tok':>8}{'out tok':>8}"
    )
    for (dataset, p), rs in sorted(groups.items()):
        model = [r["model_latency_s"] for r in rs]
        ocr = [r.get("ocr_latency_s") or 0.0 for r in rs]
        shared = [o / len(doc_questions[r["doc_id"]]) for r, o in zip(rs, ocr, strict=True)]
        row = {
            "score": _mean([primary_score(dataset, r["prediction"], r["answers"]) for r in rs]),
            "model_s": _mean(model),
            "ocr_s": _mean(ocr),
            "total_s": _mean([m + o for m, o in zip(model, ocr, strict=True)]),
            "total_shared_s": _mean([m + o for m, o in zip(model, shared, strict=True)]),
        }
        table[(dataset, p)] = row
        metric = primary_metric(dataset).split("_")[0]
        lines.append(
            f"  {dataset:<16}{p:<9}{len(rs):>4}{metric:>9}{row['score']:>8.3f}"
            f"{row['model_s']:>9.2f}{row['ocr_s']:>8.2f}{row['total_s']:>8.2f}"
            f"{row['total_shared_s']:>8.2f}{_mean([r['input_tokens'] for r in rs]):>8.0f}"
            f"{_mean([r['output_tokens'] for r in rs]):>8.1f}"
        )

    # Cross-check against the written report
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as f:
            for m in csv.DictReader(f):
                ours = table.get((m["dataset"], m["pipeline"]))
                if ours is None:
                    continue
                for field in ("model_s", "ocr_s", "total_s", "total_shared_s"):
                    if field in m and abs(float(m[field]) - ours[field]) > 1e-6:
                        problems.append(
                            f"metrics.csv {m['dataset']}/{m['pipeline']} {field} "
                            f"{float(m[field]):.4f} != recomputed {ours[field]:.4f}"
                        )
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for dataset, g in summary.get("by_dataset", {}).items():
            for p, value in g["mean_score"].items():
                ours = table.get((dataset, p))
                if ours is not None and abs(value - ours["score"]) > 1e-6:
                    problems.append(
                        f"summary.json {dataset}/{p} score {value:.4f} != recomputed "
                        f"{ours['score']:.4f}"
                    )
    else:
        lines.append("(no report yet: run `qdu benchmark report` to cross-check it)")

    # Examples
    lines.append("")
    for dataset in sorted({q.split("/", 1)[0] for q in complete}):
        for q in [q for q in complete if q.startswith(dataset + "/")][:examples]:
            first = latest[(q, pipelines[0])]
            lines.append(f"{q}  Q: {first['question'][:90]}")
            lines.append(f"    gold: {first['answers'][:3]}")
            for p in pipelines:
                r = latest[(q, p)]
                score = primary_score(dataset, r["prediction"], r["answers"])
                lines.append(
                    f"    {p:<7} {score:5.2f}  {r['model_latency_s']:5.2f} s"
                    f"  {r['input_tokens']:>6} tok  {r['prediction']!r:.60}"
                )
    return lines, problems
