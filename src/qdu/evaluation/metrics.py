"""Per-dataset, per-pipeline quality and cost table for a run.

Cost of answering one question with a pipeline, in the three scenarios of the project:

- **C** (``total_s``): model latency + OCR time, every question paying its page's OCR. OCR time
  is the original extraction time, also when the page was served from cache.
- **B** (``total_shared_s``): model latency + OCR time divided by the number of questions on
  that page, as when OCR runs once per document at ingestion and is shared by its questions.
  With one question per document (the pilot) B equals C.
- **A**: tokens (input + output); OCR is free.

The ``oracle`` row uses, for every question, the pipeline the oracle picks, so its cost is
what a perfect router would pay before counting the router's own signal computation.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from qdu.evaluation.em_f1 import exact_match, token_f1
from qdu.evaluation.relaxed import relaxed_match


@dataclass(frozen=True)
class MetricRow:
    dataset: str
    pipeline: str
    n: int
    anls: float
    relaxed_accuracy: float
    exact_match: float
    f1: float
    model_s: float
    ocr_s: float
    total_s: float
    total_shared_s: float
    input_tokens: float | None
    output_tokens: float | None


def dataset_of(question_id: str) -> str:
    return question_id.split("/", 1)[0] if "/" in question_id else "unknown"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if len(present) == len(values) and present else None


def _row(
    dataset: str, pipeline: str, rows: list[dict[str, Any]], doc_questions: Mapping[str, int]
) -> MetricRow:
    model = [r.get("model_latency_s") or 0.0 for r in rows]
    ocr = [r.get("ocr_latency_s") or 0.0 for r in rows]
    shared = [o / doc_questions[r["doc_id"]] for r, o in zip(rows, ocr, strict=True)]
    return MetricRow(
        dataset=dataset,
        pipeline=pipeline,
        n=len(rows),
        anls=_mean([r["anls"] for r in rows]),
        relaxed_accuracy=_mean(
            [float(relaxed_match(r["prediction"] or "", r["answers"])) for r in rows]
        ),
        exact_match=_mean([float(exact_match(r["prediction"] or "", r["answers"])) for r in rows]),
        f1=_mean([token_f1(r["prediction"] or "", r["answers"]) for r in rows]),
        model_s=_mean(model),
        ocr_s=_mean(ocr),
        total_s=_mean([m + o for m, o in zip(model, ocr, strict=True)]),
        total_shared_s=_mean([m + o for m, o in zip(model, shared, strict=True)]),
        input_tokens=_mean_or_none([r.get("input_tokens") for r in rows]),
        output_tokens=_mean_or_none([r.get("output_tokens") for r in rows]),
    )


def metric_table(
    rows: Iterable[dict[str, Any]],
    pipelines: tuple[str, ...],
    oracle_labels: Mapping[str, str],
) -> list[MetricRow]:
    """Rows for every (dataset, pipeline) plus ``oracle``, and the same over ``ALL`` datasets.

    Only questions that have a result for every pipeline (the oracle's complete cases) count.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if r.get("error") is None and r["question_id"] in oracle_labels:
            latest[(r["question_id"], r["pipeline"])] = r

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (qid, pipeline), r in latest.items():
        for dataset in (dataset_of(qid), "ALL"):
            groups[(dataset, pipeline)].append(r)
            if oracle_labels[qid] == pipeline:
                groups[(dataset, "oracle")].append(r)

    questions_by_doc: dict[str, set[str]] = defaultdict(set)
    for (qid, _), r in latest.items():
        questions_by_doc[r["doc_id"]].add(qid)
    doc_questions = {doc: len(qids) for doc, qids in questions_by_doc.items()}

    datasets = [*sorted({d for d, _ in groups} - {"ALL"}), "ALL"]
    return [
        _row(dataset, pipeline, groups[(dataset, pipeline)], doc_questions)
        for dataset in datasets
        for pipeline in (*pipelines, "oracle")
        if groups.get((dataset, pipeline))
    ]
