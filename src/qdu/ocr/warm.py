"""Fill the OCR cache for every page of a manifest before the GPU run.

OCR runs on the CPU, so doing it in a separate pass (on a cheap CPU runtime) keeps the paid GPU
busy with model calls only. The text and hybrid pipelines then read each page's text and its
original extraction time from the cache, so the OCR cost they report is unchanged.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qdu.data.records import QARecord
from qdu.ocr.base import OcrEngine, OcrResult

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class WarmStats:
    pages: int = 0
    extracted: int = 0
    cached: int = 0
    seconds: float = 0.0
    """Extraction time of the pages run in this pass."""


def warm_cache(
    records: Sequence[QARecord],
    get_image: Callable[[QARecord], "Image"],
    engine: OcrEngine,
    on_page: Callable[[QARecord, OcrResult], None] | None = None,
) -> WarmStats:
    """OCR each distinct page (``doc_id``) once; pages already cached are skipped."""
    stats = WarmStats()
    seen: set[str] = set()
    for record in records:
        if record.doc_id in seen:
            continue
        seen.add(record.doc_id)
        result = engine.extract(get_image(record), record.doc_id)
        stats.pages += 1
        if result.cached:
            stats.cached += 1
        else:
            stats.extracted += 1
            stats.seconds += result.latency_s
        if on_page:
            on_page(record, result)
    return stats
