"""Layout detection: text, table, figure and other regions of each page, with their time.

PP-DocLayout (PaddleOCR 3.x, ``LayoutDetection``) runs on the full-resolution page, like OCR,
and writes one JSONL row per page: model, time, hardware, page size and the regions as
``[label, score, x0, y0, x1, y1]``. Writing is resumable (pages already in the file are
skipped) and the result is gzipped once complete, so a run's layout ships as one file next to
its OCR (``results/<run>/layout.jsonl.gz``).

The model: PP-DocLayout-M (Nov 2026). On the project laptop's CPU it takes ~0.09 s per page
against ~0.05 s for -S, both far below the ~0.8 s OCR, and -M is the more accurate of the two.
Layout time is part of a router's cost (page-level: shared by the page's questions in scenario
B), measured here on a CPU, since a deployment can run it next to the GPU's OCR and model.
"""

import gzip
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdu.gz import open_gzip_text
from qdu.provenance import cpu_info

if TYPE_CHECKING:
    from PIL.Image import Image

    from qdu.data.records import QARecord

DEFAULT_LAYOUT_MODEL = "PP-DocLayout-M"
DEFAULT_LAYOUT = Path("results/full-qwen3.5-4b/layout.jsonl.gz")


@dataclass(frozen=True)
class Region:
    label: str
    """PP-DocLayout class, e.g. text, paragraph_title, table, image, chart, formula, header."""
    score: float
    box: tuple[float, float, float, float]
    """x0, y0, x1, y1 on the full-resolution page."""

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass(frozen=True)
class PageLayout:
    doc_id: str
    model: str
    latency_s: float
    hardware: str
    page_size: tuple[int, int]
    regions: tuple[Region, ...]


class PaddleLayout:
    """PP-DocLayout through PaddleOCR (``uv sync --extra paddle``), on the CPU."""

    def __init__(self, model: str = DEFAULT_LAYOUT_MODEL) -> None:
        try:
            from paddleocr import LayoutDetection
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError("PaddleOCR is not installed: run `uv sync --extra paddle`") from exc
        import numpy as np

        self.model = model
        self.hardware = cpu_info()
        self._detector = LayoutDetection(model_name=model)
        # One-off initialisation would otherwise land on the first page.
        self._detector.predict(np.full((1600, 1200, 3), 255, dtype=np.uint8))

    def detect(self, image: "Image", doc_id: str) -> PageLayout:
        import numpy as np

        array = np.array(image.convert("RGB"))
        start = time.perf_counter()
        out = self._detector.predict(array)
        latency = time.perf_counter() - start
        regions = tuple(
            Region(
                b["label"],
                round(float(b["score"]), 4),
                tuple(round(float(v), 1) for v in b["coordinate"]),
            )
            for page in out
            for b in page["boxes"]
        )
        return PageLayout(doc_id, self.model, latency, self.hardware, image.size, regions)


def _to_json(p: PageLayout) -> dict[str, Any]:
    return {
        "doc_id": p.doc_id,
        "model": p.model,
        "latency_s": p.latency_s,
        "hardware": p.hardware,
        "page_size": list(p.page_size),
        "regions": [[r.label, r.score, *r.box] for r in p.regions],
    }


def _from_json(d: Mapping[str, Any]) -> PageLayout:
    return PageLayout(
        d["doc_id"],
        d["model"],
        d["latency_s"],
        d["hardware"],
        tuple(d["page_size"]),
        tuple(Region(r[0], r[1], tuple(r[2:6])) for r in d["regions"]),
    )


def _read_lines(path: Path) -> Iterable[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield from (line for line in f if line.strip())


def load_layout(path: Path = DEFAULT_LAYOUT) -> dict[str, PageLayout]:
    """Page id -> layout, from a ``.jsonl`` or ``.jsonl.gz`` file."""
    return {(p := _from_json(json.loads(line))).doc_id: p for line in _read_lines(path)}


def detect_pages(
    records: Sequence["QARecord"],
    get_image: Callable[["QARecord"], "Image"],
    detector: PaddleLayout,
    out: Path,
    on_page: Callable[[PageLayout], None] | None = None,
) -> int:
    """Append the layout of every page of ``records`` not yet in ``out`` (a ``.jsonl``);
    returns the number of pages detected in this call."""
    done = set(load_layout(out)) if out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for record in records:
            if record.doc_id in done:
                continue
            done.add(record.doc_id)
            page = detector.detect(get_image(record), record.doc_id)
            f.write(json.dumps(_to_json(page), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if on_page:
                on_page(page)
    return n


def compress(src: Path, dst: Path) -> int:
    """Write ``src`` (.jsonl) as gzipped ``dst``, pages sorted by id; returns the page count."""
    pages = sorted(load_layout(src).values(), key=lambda p: p.doc_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open_gzip_text(dst) as f:
        for p in pages:
            f.write(json.dumps(_to_json(p), ensure_ascii=False) + "\n")
    return len(pages)
