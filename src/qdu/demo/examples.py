"""Example pages for the demo, with gold answers and the full run's recorded results.

Examples come from the **val** split only: the test split stays untouched until the final
evaluation (CLAUDE.md, Router work). Under split scheme v2 this includes TAT-DQA val pages.

For an example the demo can work without a GPU and still show real numbers:

- the router reads the run's committed OCR and layout (`StoredOcr`, `StoredLayout`), so its
  signals, and hence its choice, are exactly those of the evaluation;
- each path's answer, model time and tokens come from the labelled run on the L4
  (`load_recorded`, from ``predictions.jsonl``), so nothing is invented. Live answers from a
  model replace them once the vLLM server runs (`qdu demo` with ``--model openai:...``).

`showcase` picks pages whose questions need different paths (one text is enough, one needs
the image, one only hybrid answers), so a presentation can walk through all three. They are
chosen to illustrate routing, not to represent its accuracy: the val and test tables report
that.
"""

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qdu.data import QARecord, read_manifest
from qdu.layout import PageLayout
from qdu.ocr.base import OcrEngine, OcrResult
from qdu.pipelines import PipelineOutput
from qdu.router.labels import LabelRow, read_labels

if TYPE_CHECKING:
    from PIL.Image import Image

    from qdu.router.live import LayoutDetector

DEFAULT_RUN = Path("results/full-qwen3.5-4b")
DEFAULT_MANIFEST = Path("benchmarks/full.csv")
KINDS = ("text_ok", "vision", "hybrid_only")


@dataclass(frozen=True)
class Example:
    record: QARecord
    """The question with its gold answers (``answers``) and how to load its page image."""
    label: LabelRow
    """The run's measured score, time and tokens of each path for this question."""

    @property
    def question_id(self) -> str:
        return self.record.question_id

    @property
    def doc_id(self) -> str:
        return self.record.doc_id

    @property
    def dataset(self) -> str:
        return self.label.dataset

    @property
    def kind(self) -> str:
        """What the question needs, by the run's scores: ``text_ok`` (text as good as vision),
        ``vision`` (text fails, vision answers), ``hybrid_only`` (hybrid answers, text and
        vision both fail) or ``other``."""
        s = self.label.score
        if s["text"] >= 1 and s["vision"] >= 1:
            return "text_ok"
        if s["text"] < 0.5 and s["vision"] >= 1:
            return "vision"
        if s["hybrid"] >= 1 and s["vision"] < 0.5 and s["text"] < 0.5:
            return "hybrid_only"
        return "other"


def load_examples(
    run: Path = DEFAULT_RUN,
    manifest: Path = DEFAULT_MANIFEST,
    *,
    datasets: Sequence[str] | None = None,
) -> list[Example]:
    """Every val question of the label table, joined with its manifest record."""
    labels = read_labels(run / "labels.csv", splits=["val"], datasets=datasets)
    records = {r.question_id: r for r in read_manifest(manifest)}
    missing = [r.question_id for r in labels if r.question_id not in records]
    if missing:
        raise ValueError(f"{len(missing)} labelled questions are not in {manifest}: {missing[:3]}")
    return [Example(records[r.question_id], r) for r in labels]


def showcase(examples: Iterable[Example], per_dataset: int = 1) -> list[list[Example]]:
    """Pages whose questions cover all three kinds, ``per_dataset`` per dataset (fewest
    questions first, then page id), each as [text_ok, vision, hybrid_only] questions."""
    pages: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        pages[e.doc_id].append(e)
    chosen: dict[str, list[list[Example]]] = defaultdict(list)
    for doc_id in sorted(pages, key=lambda d: (len(pages[d]), d)):
        qs = sorted(pages[doc_id], key=lambda e: e.question_id)
        by_kind = {k: next((e for e in qs if e.kind == k), None) for k in KINDS}
        dataset = qs[0].dataset
        if all(by_kind.values()) and len(chosen[dataset]) < per_dataset:
            chosen[dataset].append([by_kind[k] for k in KINDS])  # type: ignore[misc]
    return [page for dataset in sorted(chosen) for page in chosen[dataset]]


def load_recorded(
    question_ids: Iterable[str], run: Path = DEFAULT_RUN, extra_runs: Iterable[Path] = ()
) -> dict[str, dict[str, PipelineOutput]]:
    """The labelled run's answer of each path for ``question_ids`` (the last successful row
    per question and path, as the label table uses); ``extra_runs`` add their paths (e.g.
    the vision-low run)."""
    wanted = set(question_ids)
    out: dict[str, dict[str, PipelineOutput]] = defaultdict(dict)
    for path in (run, *extra_runs):
        _read_recorded(path, wanted, out)
    return dict(out)


def _read_recorded(run: Path, wanted: set[str], out: dict[str, dict[str, PipelineOutput]]) -> None:
    with (run / "predictions.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["question_id"] not in wanted or d.get("error"):
                continue
            out[d["question_id"]][d["pipeline"]] = PipelineOutput(
                pipeline=d["pipeline"],
                answer=d["prediction"],
                raw_output=d.get("raw_output") or d["prediction"],
                model=d["model"],
                input_tokens=d.get("input_tokens"),
                output_tokens=d.get("output_tokens"),
                model_latency_s=d["model_latency_s"],
                ocr_engine=d.get("ocr_engine"),
                ocr_latency_s=d.get("ocr_latency_s"),
                ocr_chars=d.get("ocr_chars"),
                ocr_hardware=d.get("ocr_hardware"),
            )


class StoredOcr:
    """The run's OCR for pages it has (time and hardware as measured on the L4); other pages
    go to the engine ``fallback()`` builds on first use (so a laptop needs no OCR install to
    show examples), or are an error without one."""

    def __init__(
        self, pages: Mapping[str, OcrResult], fallback: Callable[[], OcrEngine] | None = None
    ) -> None:
        self.pages = pages
        self._factory = fallback
        self._engine: OcrEngine | None = None
        self.name = next(iter(pages.values())).engine if pages else "stored"

    def extract(self, image: "Image", key: str) -> OcrResult:
        if key in self.pages:
            return self.pages[key]
        if self._factory is None:
            raise KeyError(f"no stored OCR for {key!r} and no live OCR engine")
        self._engine = self._engine or self._factory()
        return self._engine.extract(image, key)


class StoredLayout:
    """The run's page layouts, with a live detector (built on first use) for other pages."""

    def __init__(
        self,
        pages: Mapping[str, PageLayout],
        fallback: "Callable[[], LayoutDetector] | None" = None,
    ) -> None:
        self.pages = pages
        self._factory = fallback
        self._detector: LayoutDetector | None = None

    def detect(self, image: "Image", doc_id: str) -> PageLayout:
        if doc_id in self.pages:
            return self.pages[doc_id]
        if self._factory is None:
            raise KeyError(f"no stored layout for {doc_id!r} and no live layout detector")
        self._detector = self._detector or self._factory()
        return self._detector.detect(image, doc_id)
