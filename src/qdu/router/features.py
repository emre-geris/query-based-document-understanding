"""Router signals: feature extractors and the feature table keyed by question id.

An extractor turns one question (with its page's OCR and lazy access to the page image) into
named numbers. Each declares whether it needs OCR, so the evaluation can charge OCR to a router that
uses it (scenario C), and its time per question is measured, since the decision must stay
under 50 ms. Loading the image is not timed (the path that answers decodes it anyway), nor is
the OCR, whose time is charged separately from the label table.

The project's signals S1-S4 are in `qdu.router.signals`.
"""

import csv
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from qdu.layout import PageLayout
from qdu.ocr.base import OcrResult
from qdu.router.labels import LabelRow

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class RouterInput:
    """What a router may look at for one question: never the gold answer or the scores."""

    question_id: str
    doc_id: str
    question: str
    ocr: OcrResult | None = field(default=None, repr=False)
    """The page's OCR from the run (text, lines with confidences and boxes, page size)."""
    layout: PageLayout | None = field(default=None, repr=False)
    """The page's layout regions (`qdu.layout`)."""
    _image: Callable[[], "Image"] | None = field(default=None, repr=False)

    @cached_property
    def image(self) -> "Image":
        if self._image is None:
            raise RuntimeError("no image loader given (pass images= to router_inputs)")
        return self._image()


class FeatureExtractor(Protocol):
    name: str
    """Prefix of the extractor's feature names, e.g. ``s1``."""
    needs_ocr: bool
    needs_layout: bool

    def __call__(self, item: RouterInput) -> Mapping[str, float]: ...


def router_inputs(
    rows: Iterable[LabelRow],
    ocr: Mapping[str, OcrResult] | None = None,
    layout: Mapping[str, PageLayout] | None = None,
    *,
    images: Callable[[LabelRow], "Image"] | None = None,
) -> list[RouterInput]:
    """Inputs for ``rows``. ``ocr`` (`qdu.router.ocr.load_ocr`) and ``layout``
    (`qdu.layout.load_layout`) map page ids to the page's OCR and regions; a page missing from
    either is an error, so a signal never silently sees an empty page."""
    rows = list(rows)
    for name, pages in (("OCR", ocr), ("layout", layout)):
        if pages is not None:
            missing = sorted({r.doc_id for r in rows} - set(pages))
            if missing:
                raise ValueError(f"no {name} for {len(missing)} pages, e.g. {missing[:3]}")
    return [
        RouterInput(
            r.question_id,
            r.doc_id,
            r.question,
            ocr[r.doc_id] if ocr is not None else None,
            layout[r.doc_id] if layout is not None else None,
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
    needs_layout: bool = False

    def router_cost(
        self, layout: Mapping[str, PageLayout] | None = None, doc_of: Mapping[str, str] = {}
    ) -> dict[str, dict]:
        """What a router built on these features pays per question, as ``uses_ocr``,
        ``page_s`` and ``question_s`` for `qdu.router.Policy` / `route`: OCR if any extractor
        read it, the page's layout time if any used layout, and the extractors' own time."""
        if self.needs_layout and (layout is None or not doc_of):
            raise ValueError("layout signals: pass the layout and each question's doc_id")
        return {
            "uses_ocr": dict.fromkeys(self.features, self.needs_ocr),
            "page_s": {
                q: layout[doc_of[q]].latency_s if self.needs_layout else 0.0  # type: ignore[index]
                for q in self.features
            },
            "question_s": {q: sum(t.values()) for q, t in self.seconds.items()},
        }

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
    needs_ocr = any(e.needs_ocr for e in extractors)
    needs_layout = any(getattr(e, "needs_layout", False) for e in extractors)
    for item in items:
        if item.ocr is None and needs_ocr:
            raise ValueError(f"{item.question_id}: an extractor needs OCR but none was given")
        if item.layout is None and needs_layout:
            raise ValueError(f"{item.question_id}: an extractor needs layout but none was given")
        row: dict[str, float] = {}
        times: dict[str, float] = {}
        for e in extractors:
            start = time.perf_counter()
            values = e(item)
            times[e.name] = time.perf_counter() - start
            row.update({f"{e.name}_{k}": float(v) for k, v in values.items()})
        features[item.question_id] = row
        seconds[item.question_id] = times
    return FeatureTable(features, seconds, needs_ocr, needs_layout)
