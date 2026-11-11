"""The run's OCR in one file, keyed by page, for signals that read OCR (S2, S4).

The OCR cache is one JSON file per page, written on the Colab machine (Linux). Copied to
Windows it breaks: two InfographicVQA pages differ only in letter case
(``Social-Media-Infographic.jpg`` and ``social-media-infographic.jpg``) and map to one file,
and a Google Drive zip renames files (``%`` and ``;`` become ``_``, a case clash gets ``(1)``).
`pack_ocr` therefore collects the pages of a label table into one gzipped JSONL keyed by
``doc_id``, taking each page from the candidate file whose recorded OCR time equals the time
in the label table, so a page can never be paired with another page's text.
"""

import gzip
import json
from collections.abc import Iterable
from pathlib import Path

from qdu.gz import open_gzip_text
from qdu.ocr.base import OcrLine, OcrResult, cache_file, read_cache
from qdu.router.labels import LabelRow

# The full run's OCR: PP-OCRv5 mobile through RapidOCR on an L4.
DEFAULT_OCR = Path("results/full-qwen3.5-4b/ocr.jsonl.gz")
_TOLERANCE_S = 1e-5


def _candidates(path: Path) -> list[Path]:
    """The cache file and the names a Drive zip gives it."""
    drive = path.with_name(path.name.replace("%", "_").replace(";", "_"))
    out = [path, drive]
    for p in (path, drive):
        out += [p.with_name(f"{p.stem}({i}){p.suffix}") for i in range(1, 4)]
    return out


def pack_ocr(rows: Iterable[LabelRow], cache_dir: Path, out: Path) -> int:
    """Write the OCR of every page in ``rows`` from ``cache_dir`` (an engine folder such as
    ``ocr_cache/rapid-mobile-gpu``) to ``out``; returns the number of pages."""
    pages = {r.doc_id: r.ocr_s for r in rows}
    found: dict[str, OcrResult] = {}
    missing = []
    listed: dict[Path, set[str]] = {}
    for doc_id, seconds in sorted(pages.items()):
        for path in _candidates(cache_file(cache_dir, doc_id)):
            # Exact name match: on Windows exists() ignores case, so check the listed names.
            if path.parent not in listed:
                parent = path.parent
                listed[parent] = {p.name for p in parent.iterdir()} if parent.is_dir() else set()
            if path.name not in listed[path.parent]:
                continue
            result = read_cache(path)
            if result is not None and abs(result.latency_s - seconds) <= _TOLERANCE_S:
                found[doc_id] = result
                break
        else:
            missing.append(doc_id)
    if missing:
        raise ValueError(
            f"{len(missing)} pages have no cache file with the run's OCR time, e.g. {missing[:3]}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open_gzip_text(out) as f:
        for doc_id, r in found.items():
            record = {
                "doc_id": doc_id,
                "engine": r.engine,
                "latency_s": r.latency_s,
                "hardware": r.hardware,
                "page_size": list(r.page_size) if r.page_size else None,
                "text": r.text,
                "lines": [[ln.text, ln.score, list(ln.box)] for ln in r.lines or ()],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(found)


def load_ocr(path: Path = DEFAULT_OCR) -> dict[str, OcrResult]:
    """Page id -> OCR result (text, lines with confidence and box, page size, time)."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["doc_id"]] = OcrResult(
                d["text"],
                d["engine"],
                d["latency_s"],
                cached=True,
                hardware=d["hardware"],
                lines=tuple(OcrLine(t, s, tuple(b)) for t, s, b in d["lines"]),
                page_size=tuple(d["page_size"]) if d["page_size"] else None,
            )
    return out
