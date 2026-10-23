"""Dataset-independent record type and manifest I/O.

A manifest is a CSV listing exactly which questions a benchmark uses. Committing it makes
the sample reproducible without re-running the sampler or storing any images.

Ids are globally unique across datasets: ``question_id`` and ``doc_id`` are prefixed with
the dataset name (``docvqa/49153``), so questions from several datasets can share one
manifest, one predictions file and one OCR cache.
"""

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FIELDS = (
    "dataset",
    "split",
    "question_id",
    "doc_id",
    "question",
    "answers",
    "question_types",
    "stratum",
)


@dataclass(frozen=True)
class QARecord:
    """One question about one document page. Images are loaded separately on demand."""

    question_id: str
    doc_id: str
    question: str
    answers: tuple[str, ...]
    question_types: tuple[str, ...] = ()
    stratum: str = ""
    dataset: str = ""
    split: str = ""
    """Router split (train / val / test), assigned per document; see qdu.data.splits."""


def write_manifest(records: Iterable[QARecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "dataset": r.dataset,
                    "split": r.split,
                    "question_id": r.question_id,
                    "doc_id": r.doc_id,
                    "question": r.question,
                    "answers": json.dumps(list(r.answers), ensure_ascii=False),
                    "question_types": json.dumps(list(r.question_types), ensure_ascii=False),
                    "stratum": r.stratum,
                }
            )


def read_manifest(path: Path) -> list[QARecord]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            QARecord(
                question_id=row["question_id"],
                doc_id=row["doc_id"],
                question=row["question"],
                answers=tuple(json.loads(row["answers"])),
                question_types=tuple(json.loads(row["question_types"])),
                stratum=row["stratum"],
                dataset=row.get("dataset", ""),
                split=row.get("split", ""),
            )
            for row in csv.DictReader(f)
        ]
