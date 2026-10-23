"""Registry of the question-answering datasets the project uses.

Every source maps its Hugging Face rows onto :class:`QARecord` with globally unique,
dataset-prefixed ids. We use each dataset's **labelled evaluation split** (not its training
split): Qwen2.5-VL was trained on the official training sets, so their questions would
overstate every pipeline. Router train / val / test splits are then cut from these questions
by document (see :mod:`qdu.data.splits`).

The first ``load_source`` call downloads the split into the Hugging Face cache (outside the
repo); later calls are local and memory-mapped.
"""

import hashlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from qdu.data.records import QARecord
from qdu.data.splits import assign_split

if TYPE_CHECKING:
    from datasets import Dataset
    from PIL.Image import Image

RowAdapter = Callable[[int, dict[str, Any]], QARecord]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    hf_id: str
    hf_config: str | None
    hf_split: str
    role: str
    license: str
    metadata_columns: tuple[str, ...]
    adapter: RowAdapter
    image_column: str = "image"
    needs_image_bytes: bool = False
    """True when the adapter derives the document id from the raw image bytes."""


def _docvqa(index: int, row: dict[str, Any]) -> QARecord:
    return QARecord(
        question_id=f"docvqa/{row['questionId']}",
        doc_id=f"docvqa/{row['docId']}",
        question=row["question"],
        answers=tuple(row.get("answers") or ()),
        question_types=tuple(row.get("question_types") or ()),
        dataset="docvqa",
    )


def _infographicvqa(index: int, row: dict[str, Any]) -> QARecord:
    url = row.get("image_url") or ""
    doc = url.rstrip("/").rsplit("/", 1)[-1] if url else f"row{index}"
    labels = [*(row.get("answer_type") or ()), *(row.get("operation/reasoning") or ())]
    return QARecord(
        question_id=f"infographicvqa/{row['questionId']}",
        doc_id=f"infographicvqa/{doc}",
        question=row["question"],
        answers=tuple(row.get("answers") or ()),
        question_types=tuple(label.strip().lower() for label in labels if label.strip()),
        dataset="infographicvqa",
    )


def _image_digest(image: dict[str, Any]) -> str:
    data = image.get("bytes")
    if data is None:
        raise ValueError("ChartQA rows must be read with image decoding disabled")
    return hashlib.sha1(data).hexdigest()[:16]


def _chartqa(index: int, row: dict[str, Any]) -> QARecord:
    origin = "human" if row.get("human_or_machine") in (0, "human") else "augmented"
    return QARecord(
        question_id=f"chartqa/test-{index:05d}",
        doc_id=f"chartqa/{_image_digest(row['image'])}",
        question=row["query"],
        answers=tuple(row.get("label") or ()),
        question_types=(origin,),
        dataset="chartqa",
    )


SOURCES: dict[str, SourceSpec] = {
    "docvqa": SourceSpec(
        name="docvqa",
        hf_id="lmms-lab/DocVQA",
        hf_config="DocVQA",
        hf_split="validation",
        role="router train/val/test",
        license="RRC DocVQA terms (research use)",
        metadata_columns=("questionId", "docId", "question", "answers", "question_types"),
        adapter=_docvqa,
    ),
    "infographicvqa": SourceSpec(
        name="infographicvqa",
        hf_id="lmms-lab/DocVQA",
        hf_config="InfographicVQA",
        hf_split="validation",
        role="router train/val/test",
        license="RRC InfographicVQA terms (research use)",
        metadata_columns=(
            "questionId",
            "question",
            "answers",
            "answer_type",
            "operation/reasoning",
            "image_url",
        ),
        adapter=_infographicvqa,
    ),
    "chartqa": SourceSpec(
        name="chartqa",
        hf_id="HuggingFaceM4/ChartQA",
        hf_config=None,
        hf_split="test",
        role="router train/val/test",
        license="GPL-3.0 (ChartQA repository)",
        metadata_columns=("query", "label", "human_or_machine", "image"),
        adapter=_chartqa,
        needs_image_bytes=True,
    ),
}

DatasetLoader = Callable[[SourceSpec], "Dataset"]


def _hf_loader(spec: SourceSpec) -> "Dataset":
    from datasets import load_dataset

    return load_dataset(spec.hf_id, spec.hf_config, split=spec.hf_split)


def get_source(name: str) -> SourceSpec:
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(SOURCES)}") from None


def load_source(name: str, loader: DatasetLoader | None = None) -> "Dataset":
    return (loader or _hf_loader)(get_source(name))


def iter_source_records(name: str, dataset: "Dataset") -> Iterator[QARecord]:
    """Yield every question with its router split, without decoding page images."""
    spec = get_source(name)
    view = dataset
    if spec.needs_image_bytes:
        from datasets import Image as ImageFeature

        view = view.cast_column(spec.image_column, ImageFeature(decode=False))
    view = view.select_columns(list(spec.metadata_columns))
    for index, row in enumerate(view):
        record = spec.adapter(index, row)
        yield replace(record, split=assign_split(record.dataset, record.doc_id))


class ImageStore:
    """Look up page images by question id across several datasets."""

    def __init__(
        self,
        datasets: Mapping[str, "Dataset"] | None = None,
        loader: DatasetLoader | None = None,
    ) -> None:
        self._datasets = dict(datasets or {})
        self._loader = loader
        self._rows: dict[str, dict[str, int]] = {}

    def _dataset(self, name: str) -> "Dataset":
        if name not in self._datasets:
            self._datasets[name] = load_source(name, self._loader)
        return self._datasets[name]

    def get(self, record: QARecord) -> "Image":
        name = record.dataset or record.question_id.split("/", 1)[0]
        dataset = self._dataset(name)
        if name not in self._rows:
            self._rows[name] = {
                r.question_id: i for i, r in enumerate(iter_source_records(name, dataset))
            }
        row = self._rows[name][record.question_id]
        return dataset[row][get_source(name).image_column]
