"""Run pipelines over a sample and append scored predictions to a JSONL file.

Runs are resumable: (question, pipeline) pairs that already have a successful row are
skipped, and failed calls are recorded with their error and retried on the next run.
"""

import json
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdu.data.records import QARecord
from qdu.evaluation.anls import anls_score
from qdu.evaluation.oracle import ScoredPrediction
from qdu.evaluation.primary import primary_score
from qdu.pipelines.base import Pipeline, PipelineInput

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True)
class PredictionRow:
    question_id: str
    doc_id: str
    stratum: str
    pipeline: str
    question: str
    answers: list[str]
    prediction: str | None
    anls: float | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    model_latency_s: float | None
    ocr_engine: str | None
    ocr_latency_s: float | None
    ocr_chars: int | None
    raw_output: str | None
    error: str | None = None


@dataclass
class RunStats:
    completed: int = 0
    skipped: int = 0
    failed: int = 0


def read_predictions(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


COST_MEASURES = ("latency", "tokens")


def call_cost(row: dict[str, Any], measure: str = "latency") -> float | None:
    """Cost of one call: seconds (model + OCR) or tokens (input + output; OCR is free here).

    Latency is what a self-hosted deployment pays and depends on the hardware; tokens are a
    hardware-independent proxy for API pricing, under which OCR costs nothing.
    """
    if measure == "latency":
        if row.get("model_latency_s") is None:
            return None
        return row["model_latency_s"] + (row.get("ocr_latency_s") or 0.0)
    if measure == "tokens":
        if row.get("input_tokens") is None or row.get("output_tokens") is None:
            return None
        return float(row["input_tokens"] + row["output_tokens"])
    raise ValueError(f"unknown cost measure {measure!r}; choose from {', '.join(COST_MEASURES)}")


def scored_predictions(path: Path, cost: str = "latency") -> list[ScoredPrediction]:
    """Latest successful row per (question, pipeline), ready for the oracle analysis.

    Each row is scored with its dataset's primary metric (ANLS, or relaxed accuracy for
    ChartQA), taken from the question id prefix, so oracle labels follow the convention of
    each benchmark.
    """
    latest: dict[tuple[str, str], ScoredPrediction] = {}
    for row in read_predictions(path):
        if row["error"] is None:
            key = (row["question_id"], row["pipeline"])
            latest[key] = ScoredPrediction(
                row["question_id"],
                row["stratum"],
                row["pipeline"],
                primary_score(
                    row["question_id"].split("/", 1)[0], row["prediction"], row["answers"]
                ),
                call_cost(row, cost),
            )
    return list(latest.values())


def run_pipelines(
    records: Sequence[QARecord],
    get_image: Callable[[QARecord], "Image"],
    pipelines: Sequence[Pipeline],
    out_path: Path,
    *,
    on_row: Callable[[PredictionRow], None] | None = None,
) -> RunStats:
    done = {(r["question_id"], r["pipeline"]) for r in read_predictions(out_path) if not r["error"]}
    stats = RunStats()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as out:
        for record in records:
            todo = [p for p in pipelines if (record.question_id, p.name) not in done]
            stats.skipped += len(pipelines) - len(todo)
            if not todo:
                continue
            image = get_image(record)
            inp = PipelineInput(record.question, image, page_key=record.doc_id)
            for pipeline in todo:
                row = _run_one(pipeline, inp, record)
                out.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                out.flush()
                if row.error is None:
                    stats.completed += 1
                else:
                    stats.failed += 1
                if on_row:
                    on_row(row)
    return stats


def _run_one(pipeline: Pipeline, inp: PipelineInput, record: QARecord) -> PredictionRow:
    base = {
        "question_id": record.question_id,
        "doc_id": record.doc_id,
        "stratum": record.stratum,
        "pipeline": pipeline.name,
        "question": record.question,
        "answers": list(record.answers),
    }
    try:
        out = pipeline.run(inp)
    except Exception as exc:  # recorded and retried on the next run
        last = traceback.format_exception_only(exc)[-1].strip()
        return PredictionRow(
            **base,
            prediction=None,
            anls=None,
            model=None,
            input_tokens=None,
            output_tokens=None,
            model_latency_s=None,
            ocr_engine=None,
            ocr_latency_s=None,
            ocr_chars=None,
            raw_output=None,
            error=last,
        )
    return PredictionRow(
        **base,
        prediction=out.answer,
        anls=anls_score(out.answer, record.answers),
        model=out.model,
        input_tokens=out.input_tokens,
        output_tokens=out.output_tokens,
        model_latency_s=out.model_latency_s,
        ocr_engine=out.ocr_engine,
        ocr_latency_s=out.ocr_latency_s,
        ocr_chars=out.ocr_chars,
        raw_output=out.raw_output,
    )
