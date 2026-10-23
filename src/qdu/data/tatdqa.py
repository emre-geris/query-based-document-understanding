"""TAT-DQA (test) as a page-level QA dataset.

The official release ships a gold JSON and a zip with, per document, the PDF, page PNGs
(``<uid>_<page>.png``) and a JSON of text blocks per page. Every question is mapped to the
single page that holds its evidence (the page of its ``block_mapping`` blocks), so it fits the
same one-page pipelines as the other datasets. Questions whose evidence spans two pages are
skipped; :func:`build_tatdqa` reports how many.

TAT-DQA is used only as unseen test data. Its official metric (exact match / F1 with
numeric scale) differs from ANLS; numeric gold answers are rounded to two decimals as in the
official evaluation.
"""

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

HF_REPO = "next-tat/TAT-DQA"
GOLD_FILE = "tatdqa_dataset_test_gold.json"
DOCS_FILE = "tatdqa_docs_test.zip"


def format_answer(answer: Any) -> list[str]:
    """Gold answer(s) as strings; a multi-span answer becomes one comma-joined string."""
    if isinstance(answer, list):
        return [", ".join(str(a).strip() for a in answer)]
    if isinstance(answer, float):
        rounded = round(answer, 2)
        return [str(int(rounded)) if rounded.is_integer() else str(rounded)]
    return [str(answer).strip()]


def _evidence_page(question: dict[str, Any], block_pages: dict[str, int]) -> int | None:
    pages = {
        block_pages[block]
        for mapping in question.get("block_mapping") or ()
        for block in mapping
        if block in block_pages
    }
    return pages.pop() if len(pages) == 1 else None


def iter_rows(gold_path: Path, docs_zip: Path, stats: dict[str, int]) -> Iterator[dict[str, Any]]:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    with zipfile.ZipFile(docs_zip) as z:
        for entry in gold:
            uid = entry["doc"]["uid"]
            layout = json.loads(z.read(f"test/{uid}.json"))
            block_pages = {
                block["uuid"]: number
                for number, page in enumerate(layout["pages"], start=1)
                for block in page["blocks"]
            }
            for q in entry["questions"]:
                page = 1 if entry["doc"]["page"] == 1 else _evidence_page(q, block_pages)
                if page is None:
                    stats["skipped_multi_page"] = stats.get("skipped_multi_page", 0) + 1
                    continue
                stats["questions"] = stats.get("questions", 0) + 1
                yield {
                    "uid": q["uid"],
                    "doc_uid": uid,
                    "page": page,
                    "question": q["question"],
                    "answers": format_answer(q["answer"]),
                    "answer_type": q.get("answer_type") or "",
                    "scale": q.get("scale") or "",
                    "image": {"bytes": z.read(f"test/{uid}_{page}.png"), "path": None},
                }


def build_tatdqa(gold_path: Path, docs_zip: Path) -> tuple["Dataset", dict[str, int]]:
    """Build the page-level dataset in memory and return it with its counts."""
    from datasets import Dataset, Features, Image, Sequence, Value

    features = Features(
        {
            "uid": Value("string"),
            "doc_uid": Value("string"),
            "page": Value("int64"),
            "question": Value("string"),
            "answers": Sequence(Value("string")),
            "answer_type": Value("string"),
            "scale": Value("string"),
            "image": Image(),
        }
    )
    stats: dict[str, int] = {}
    rows = list(iter_rows(gold_path, docs_zip, stats))
    return Dataset.from_list(rows, features=features), stats


def load_tatdqa() -> "Dataset":
    """Download the official files into the HF cache (first call) and build the dataset."""
    from huggingface_hub import hf_hub_download

    gold = hf_hub_download(HF_REPO, GOLD_FILE, repo_type="dataset")
    docs = hf_hub_download(HF_REPO, DOCS_FILE, repo_type="dataset")
    dataset, _ = build_tatdqa(Path(gold), Path(docs))
    return dataset
