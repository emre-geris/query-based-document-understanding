"""Compare OCR engines on a manifest: answer coverage and time per page.

Coverage is the share of questions whose gold answer appears in the page's OCR text (see
:func:`qdu.evaluation.em_f1.contains_answer`). It is an upper bound for the text path and
a direct measure of how much evidence each engine keeps; time is the engine's own
extraction time per page (the original time, also when served from cache).
"""

import csv
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qdu.data.records import QARecord
from qdu.evaluation.em_f1 import contains_answer
from qdu.ocr.base import OcrEngine

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True)
class OcrPageResult:
    question_id: str
    dataset: str
    engine: str
    answer_in_text: bool
    chars: int
    latency_s: float


@dataclass(frozen=True)
class OcrSummaryRow:
    dataset: str
    engine: str
    n: int
    coverage: float
    mean_chars: float
    mean_latency_s: float


def compare_engines(
    records: Sequence[QARecord],
    get_image: Callable[[QARecord], "Image"],
    engines: Mapping[str, OcrEngine],
    on_result: Callable[[OcrPageResult], None] | None = None,
) -> list[OcrPageResult]:
    results = []
    for record in records:
        image = get_image(record)
        for spec, engine in engines.items():
            ocr = engine.extract(image, record.doc_id)
            result = OcrPageResult(
                question_id=record.question_id,
                dataset=record.dataset,
                engine=spec,
                answer_in_text=contains_answer(ocr.text, record.answers),
                chars=len(ocr.text),
                latency_s=ocr.latency_s,
            )
            results.append(result)
            if on_result:
                on_result(result)
    return results


def summarize_engines(results: Sequence[OcrPageResult]) -> list[OcrSummaryRow]:
    groups: dict[tuple[str, str], list[OcrPageResult]] = defaultdict(list)
    for r in results:
        groups[(r.dataset, r.engine)].append(r)
        groups[("ALL", r.engine)].append(r)
    rows = []
    for (dataset, engine), items in groups.items():
        n = len(items)
        rows.append(
            OcrSummaryRow(
                dataset=dataset,
                engine=engine,
                n=n,
                coverage=sum(r.answer_in_text for r in items) / n,
                mean_chars=sum(r.chars for r in items) / n,
                mean_latency_s=sum(r.latency_s for r in items) / n,
            )
        )
    order = {"ALL": "~"}  # sort ALL last
    return sorted(rows, key=lambda r: (order.get(r.dataset, r.dataset), r.engine))


def write_comparison(
    out_dir: Path, results: Sequence[OcrPageResult], summary: Sequence[OcrSummaryRow], info: dict
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pages.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=list(OcrPageResult.__dataclass_fields__), lineterminator="\n"
        )
        w.writeheader()
        w.writerows(asdict(r) for r in results)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=list(OcrSummaryRow.__dataclass_fields__), lineterminator="\n"
        )
        w.writeheader()
        w.writerows(asdict(r) for r in summary)
    (out_dir / "run_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def format_comparison(summary: Sequence[OcrSummaryRow]) -> str:
    lines = [f"  {'dataset':<16}{'engine':<16}{'n':>5}{'coverage':>10}{'chars':>9}{'s/page':>9}"]
    for r in summary:
        lines.append(
            f"  {r.dataset:<16}{r.engine:<16}{r.n:>5}{r.coverage:>10.3f}"
            f"{r.mean_chars:>9.0f}{r.mean_latency_s:>9.2f}"
        )
    return "\n".join(lines)
