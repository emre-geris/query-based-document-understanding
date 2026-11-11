"""Router signals: feature extractors and the feature table keyed by question id.

An extractor turns one question (with lazy access to its page image and cached OCR) into named
numbers. Each declares whether it needs OCR, so the evaluation can charge OCR to a router that
uses it (scenario C), and its time per question is measured, since the decision must stay
under 50 ms. Loading the image or the cached OCR is not timed: OCR time is charged separately
from the label table, and the image is decoded anyway by the path that answers.

S1-S4 are to be written as extractors like `QuestionShape`, an example S1-style extractor.
"""

import csv
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from qdu.ocr.base import OcrResult, cache_file, read_cache
from qdu.router.labels import LabelRow

if TYPE_CHECKING:
    from PIL.Image import Image

# The OCR of the full labelling run (PP-OCRv5 mobile, RapidOCR on an L4).
DEFAULT_OCR_ENGINE_DIR = "rapid-mobile-gpu"


@dataclass
class RouterInput:
    """What a router may look at for one question: never the gold answer or the scores."""

    question_id: str
    doc_id: str
    question: str
    _ocr: Callable[[], OcrResult | None] = field(repr=False)
    _image: Callable[[], "Image"] | None = field(default=None, repr=False)

    @cached_property
    def ocr(self) -> OcrResult | None:
        """Cached OCR of the page (text, lines with confidences and boxes, page size)."""
        return self._ocr()

    @cached_property
    def image(self) -> "Image":
        if self._image is None:
            raise RuntimeError("no image loader given (pass images= to router_inputs)")
        return self._image()


class FeatureExtractor(Protocol):
    name: str
    """Prefix of the extractor's feature names, e.g. ``s1``."""
    needs_ocr: bool

    def __call__(self, item: RouterInput) -> Mapping[str, float]: ...


def router_inputs(
    rows: Iterable[LabelRow],
    ocr_cache: Path | None,
    *,
    engine_dir: str = DEFAULT_OCR_ENGINE_DIR,
    images: Callable[[LabelRow], "Image"] | None = None,
) -> list[RouterInput]:
    """Inputs for ``rows``; OCR is read from ``<ocr_cache>/<engine_dir>`` when first used."""

    def ocr_loader(doc_id: str) -> Callable[[], OcrResult | None]:
        if ocr_cache is None:
            return lambda: None
        return lambda: read_cache(cache_file(ocr_cache / engine_dir, doc_id))

    return [
        RouterInput(
            r.question_id,
            r.doc_id,
            r.question,
            ocr_loader(r.doc_id),
            (lambda r=r: images(r)) if images else None,
        )
        for r in rows
    ]


@dataclass
class FeatureTable:
    features: dict[str, dict[str, float]]
    """question id -> feature name -> value."""
    seconds: dict[str, dict[str, float]]
    """question id -> extractor name -> time it took."""
    needs_ocr: bool

    def write(self, path: Path) -> None:
        names = sorted({k for f in self.features.values() for k in f})
        extractors = sorted({k for s in self.seconds.values() for k in s})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["question_id", *names, *(f"time_s_{e}" for e in extractors)])
            for qid in sorted(self.features):
                w.writerow(
                    [
                        qid,
                        *(self.features[qid].get(n, "") for n in names),
                        *(f"{self.seconds[qid][e]:.6f}" for e in extractors),
                    ]
                )


def build_features(
    items: Iterable[RouterInput], extractors: Sequence[FeatureExtractor]
) -> FeatureTable:
    """Run every extractor on every item; feature names get the extractor's prefix."""
    features: dict[str, dict[str, float]] = {}
    seconds: dict[str, dict[str, float]] = {}
    for item in items:
        if any(e.needs_ocr for e in extractors):
            _ = item.ocr  # load before the clock starts
        row: dict[str, float] = {}
        times: dict[str, float] = {}
        for e in extractors:
            start = time.perf_counter()
            values = e(item)
            times[e.name] = time.perf_counter() - start
            row.update({f"{e.name}_{k}": float(v) for k, v in values.items()})
        features[item.question_id] = row
        seconds[item.question_id] = times
    return FeatureTable(features, seconds, any(e.needs_ocr for e in extractors))


_WH = ("what", "which", "who", "when", "where", "how many", "how much", "how", "why")
_NUMERIC = re.compile(r"\b(how many|how much|total|sum|difference|average|percent|%|ratio)\b")
_VISUAL = re.compile(r"\b(colou?r|chart|graph|bar|line|figure|image|logo|icon|shown|left|right)\b")


@dataclass(frozen=True)
class QuestionShape:
    """Example extractor (S1-style, no OCR): length, wh-word and cue words of the question."""

    name: str = "q"
    needs_ocr: bool = False

    def __call__(self, item: RouterInput) -> dict[str, float]:
        text = item.question.lower()
        wh = next((w for w in _WH if text.startswith(w) or f" {w} " in text), "other")
        return {
            "n_words": len(text.split()),
            "has_digit": any(c.isdigit() for c in text),
            "numeric_cue": bool(_NUMERIC.search(text)),
            "visual_cue": bool(_VISUAL.search(text)),
            **{f"wh_{w.replace(' ', '_')}": wh == w for w in (*_WH, "other")},
        }
