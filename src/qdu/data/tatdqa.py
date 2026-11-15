"""TAT-DQA (test) as a page-level QA dataset.

The official release ships a gold JSON and a zip with, per document, the PDF, page PNGs
(``<uid>_<page>.png``) and a JSON of text blocks per page. The PNGs are 224 x 224 thumbnails,
unreadable for OCR or a VLM, so pages are rendered from the PDF at ``RENDER_DPI`` instead
(an A4 page becomes 1653 x 2339 pixels, comparable to DocVQA scans). Every question is mapped to the
single page that holds its evidence (the page of its ``block_mapping`` blocks), so it fits the
same one-page pipelines as the other datasets. Questions whose evidence spans two pages are
skipped; :func:`build_tatdqa` reports how many.

Each TAT-DQA document is a few pages cut from a company's annual report, and one report can
give several documents (277 documents from 126 reports), so router splits group the documents
by report (:func:`report_of_documents`, ``benchmarks/tatdqa_reports.csv``) to keep a report's
tables out of two splits. Its official metric (exact match / F1 with numeric scale) differs from
ANLS; numeric gold answers are rounded to two decimals as in the official evaluation.
"""

import csv
import io
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
RENDER_DPI = 200


def render_page(pdf: bytes, page: int, dpi: int = RENDER_DPI) -> bytes:
    """PNG bytes of 1-based ``page`` of a PDF, rendered at ``dpi``."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf)
    try:
        image = document[page - 1].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        document.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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
            pages: dict[int, bytes] = {}  # rendered once per page, shared by its questions
            for q in entry["questions"]:
                page = 1 if entry["doc"]["page"] == 1 else _evidence_page(q, block_pages)
                if page is None:
                    stats["skipped_multi_page"] = stats.get("skipped_multi_page", 0) + 1
                    continue
                if page not in pages:
                    pages[page] = render_page(z.read(f"test/{uid}.pdf"), page)
                stats["questions"] = stats.get("questions", 0) + 1
                yield {
                    "uid": q["uid"],
                    "doc_uid": uid,
                    "page": page,
                    "question": q["question"],
                    "answers": format_answer(q["answer"]),
                    "answer_type": q.get("answer_type") or "",
                    "scale": q.get("scale") or "",
                    "image": {"bytes": pages[page], "path": None},
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


def report_of_documents(gold_path: Path) -> dict[str, str]:
    """Document uid -> the annual report (PDF file name) it was cut from."""
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    return {entry["doc"]["uid"]: entry["doc"]["source"] for entry in gold}


def write_reports(reports: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["doc_uid", "report"])
        for uid in sorted(reports):
            w.writerow([uid, reports[uid]])


def read_reports(path: Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {r["doc_uid"]: r["report"] for r in csv.DictReader(f)}


def gold_file() -> Path:
    """The official test gold JSON, downloaded into the HF cache on first use."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, GOLD_FILE, repo_type="dataset"))


CACHE_VERSION = 1
"""Bump when :func:`iter_rows` changes what a row holds, so a stale built copy is not reused."""


def built_cache_dir(docs_zip: Path) -> Path:
    """Where the built dataset is kept: next to the HF cache, keyed by the downloaded snapshot
    (its folder name is the dataset commit), the render DPI and :data:`CACHE_VERSION`."""
    from huggingface_hub.constants import HF_HOME

    snapshot = Path(docs_zip).parent.name
    return Path(HF_HOME) / "qdu" / f"tatdqa-test-{snapshot}-dpi{RENDER_DPI}-v{CACHE_VERSION}"


def load_tatdqa() -> "Dataset":
    """Download the official files into the HF cache (first call) and build the dataset.

    Rendering every evidence page from its PDF takes ~20 s, so the built dataset is saved
    once (:func:`built_cache_dir`) and later calls load it from disk."""
    import shutil

    from datasets import load_from_disk
    from huggingface_hub import hf_hub_download

    gold = hf_hub_download(HF_REPO, GOLD_FILE, repo_type="dataset")
    docs = hf_hub_download(HF_REPO, DOCS_FILE, repo_type="dataset")
    cache = built_cache_dir(Path(docs))
    if (cache / "dataset_info.json").exists():
        return load_from_disk(str(cache))
    dataset, _ = build_tatdqa(Path(gold), Path(docs))
    partial = cache.with_name(cache.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    dataset.save_to_disk(str(partial))
    partial.rename(cache)  # complete copies only: an interrupted save is never loaded
    return load_from_disk(str(cache))
