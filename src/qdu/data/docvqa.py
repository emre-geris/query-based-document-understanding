"""DocVQA access through the Hugging Face ``lmms-lab/DocVQA`` mirror.

The first call downloads the split into the Hugging Face cache (outside the repo); later calls
are local and memory-mapped. Only the validation split has public answers.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from qdu.data.records import QARecord

if TYPE_CHECKING:
    from datasets import Dataset
    from PIL.Image import Image

DATASET_ID = "lmms-lab/DocVQA"
DATASET_CONFIG = "DocVQA"
METADATA_COLUMNS = ["questionId", "docId", "question", "answers", "question_types"]


def load_docvqa(split: str = "validation") -> "Dataset":
    from datasets import load_dataset

    return load_dataset(DATASET_ID, DATASET_CONFIG, split=split)


def record_from_row(row: dict[str, Any]) -> QARecord:
    return QARecord(
        question_id=str(row["questionId"]),
        doc_id=str(row["docId"]),
        question=row["question"],
        answers=tuple(row.get("answers") or ()),
        question_types=tuple(row.get("question_types") or ()),
    )


def iter_records(dataset: "Dataset") -> Iterator[QARecord]:
    """Yield metadata for every row without decoding page images."""
    for row in dataset.select_columns(METADATA_COLUMNS):
        yield record_from_row(row)


class DocVQAImages:
    """Look up page images by question id."""

    def __init__(self, dataset: "Dataset") -> None:
        self._dataset = dataset
        ids = dataset.select_columns(["questionId"])["questionId"]
        self._row_by_question = {str(qid): i for i, qid in enumerate(ids)}

    def __contains__(self, question_id: str) -> bool:
        return question_id in self._row_by_question

    def get(self, question_id: str) -> "Image":
        return self._dataset[self._row_by_question[question_id]]["image"]
