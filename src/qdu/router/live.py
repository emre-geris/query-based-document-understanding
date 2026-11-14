"""Route one new question on one page, live: the demo's decision step.

The evaluation routes questions from the label table, whose OCR, layout and embeddings were
computed beforehand. A user of the demo uploads a new page, so the router must compute its
signals itself, and only the ones it needs:

1. Page-level work the signals of stage 1 need (layout for S3, the CLIP page vector for S6).
2. Stage 1 (OCR-free signals). If it decides, the question goes to vision and **no OCR runs**.
3. Otherwise OCR runs, the remaining stage-2 signals are computed, and the stage-2 and hybrid
   heads choose text, vision or hybrid (`TwoStageRouter.choose`, the same rule the
   evaluation uses).

Every step is timed and returned in a `LiveDecision`, which the demo shows as the decision
steps and the time breakdown. Page-level results (layout, OCR, page embedding) are kept per
page, so a second question on the same page reuses them; the decision says which ones it
reused, so the demo can show scenario C (each question pays) and B (shared per page).

S5/S6 normally read precomputed files (`qdu.router.embed`); live routing uses
`LiveQuestionEmbeddingSignals` / `LivePageEmbeddingSignals`, which embed on the spot with the
same models and produce the same feature names.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from qdu.layout import PageLayout
from qdu.ocr.base import OcrEngine, OcrResult
from qdu.router.features import FeatureExtractor, RouterInput, build_features
from qdu.router.two_stage import OFF, TwoStageRouter

if TYPE_CHECKING:
    from PIL.Image import Image

    from qdu.router.embed import Encoder


class LayoutDetector(Protocol):
    def detect(self, image: "Image", doc_id: str) -> PageLayout: ...


@dataclass
class LiveQuestionEmbeddingSignals:
    """S5 computed on the spot (MiniLM), with the feature names of the precomputed S5."""

    encoder: "Encoder"
    name: str = "s5"
    needs_ocr: bool = False
    needs_layout: bool = False

    def __call__(self, item: RouterInput) -> dict[str, float]:
        v = self.encoder.encode([item.question], normalize_embeddings=True)[0]
        return {f"e{i}": float(x) for i, x in enumerate(v)}


@dataclass
class LivePageEmbeddingSignals:
    """S6 computed on the spot (CLIP): the page vector once per page (its time is page-level,
    returned by ``cost``), the question's CLIP text vector per question."""

    encoder: "Encoder"
    name: str = "s6"
    needs_ocr: bool = False
    needs_layout: bool = False
    _pages: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict, repr=False)

    def _page(self, item: RouterInput) -> tuple[np.ndarray, float]:
        if item.doc_id not in self._pages:
            start = time.perf_counter()
            v = self.encoder.encode([item.image.convert("RGB")], normalize_embeddings=True)[0]
            self._pages[item.doc_id] = (
                np.asarray(v, dtype=np.float32),
                time.perf_counter() - start,
            )
        return self._pages[item.doc_id]

    def __call__(self, item: RouterInput) -> dict[str, float]:
        page, _ = self._page(item)
        text = self.encoder.encode([item.question], normalize_embeddings=True)[0]
        return {**{f"e{i}": float(x) for i, x in enumerate(page)}, "qsim": float(text @ page)}

    def cost(self, item: RouterInput) -> tuple[float, float]:
        """No extra per-question time (the text embedding is timed by `build_features`); the
        page's CLIP time as page-level time."""
        return 0.0, self._page(item)[1]


@dataclass(frozen=True)
class LiveDecision:
    """One routed question: the path, why, and what it cost to decide."""

    path: str
    stage: int
    """1: decided without OCR; 2: after OCR."""
    p_image_stage1: float
    p_image_stage2: float | None
    p_hybrid: float | None
    thresholds: tuple[float, ...]
    """(t1, t2, t3); t3 = 1.01 when the router never chooses hybrid."""
    ocr: OcrResult | None
    """The page's OCR if this question needed it (stage 2); the text path uses it too."""
    layout_s: float
    """Layout time of the page (0 if no signal needs layout)."""
    ocr_s: float
    """OCR time of the page (0 if decided in stage 1)."""
    page_signal_s: float
    """Other page-level signal time, e.g. the CLIP page vector (S6)."""
    signal_s: float
    """Per-question signal time (all extractors used)."""
    decide_s: float
    """Time of the heads' predictions."""
    reused: tuple[str, ...] = ()
    """Page-level results taken from an earlier question on the same page: ``layout``,
    ``ocr``. Their times above are still the page's measured times (scenario C)."""

    @property
    def router_s(self) -> float:
        """The router's own time for this question, excluding OCR and layout."""
        return self.page_signal_s + self.signal_s + self.decide_s


@dataclass
class _Page:
    layout: PageLayout | None = None
    ocr: OcrResult | None = None


class LiveRouter:
    """A trained two-stage router (`qdu.router.two_stage.load_two_stage`) that routes new
    questions, computing only the signals each stage needs."""

    def __init__(
        self,
        router: TwoStageRouter,
        extractors: Mapping[str, FeatureExtractor],
        ocr: OcrEngine,
        layout: LayoutDetector | None = None,
    ) -> None:
        names = (*router.stage1_extractors, *router.stage2_extractors)
        missing = sorted({n for n in names if n not in extractors})
        if missing:
            raise ValueError(f"the router uses signals {missing}, but no extractor was given")
        if layout is None and any(extractors[n].needs_layout for n in names):
            raise ValueError("the router's signals need page layout, but no detector was given")
        self.router = router
        self.extractors = extractors
        self.ocr_engine = ocr
        self.layout_detector = layout
        self._pages: dict[str, _Page] = {}
        self._asked = 0

    def _features(
        self, names: tuple[str, ...], item: RouterInput
    ) -> tuple[dict[str, float], float, float]:
        if not names:
            return {}, 0.0, 0.0
        table = build_features([item], [self.extractors[n] for n in names])
        q = item.question_id
        return table.features[q], sum(table.seconds[q].values()), table.page_time(q)

    def decide(
        self,
        question: str,
        image: "Image",
        page_id: str,
        question_id: str | None = None,
    ) -> LiveDecision:
        """Route ``question`` about the page ``image`` (``page_id`` keys the per-page cache)."""
        r = self.router
        self._asked += 1
        qid = question_id or f"{page_id}#q{self._asked}"
        page = self._pages.setdefault(page_id, _Page())
        had_layout, reused = page.layout is not None, []
        stage1 = tuple(r.stage1_extractors)
        stage2_only = tuple(e for e in r.stage2_extractors if e not in stage1)
        layout_s = 0.0

        def get_layout(names: tuple[str, ...]) -> PageLayout | None:
            nonlocal layout_s
            if any(self.extractors[n].needs_layout for n in names):
                if page.layout is None:
                    assert self.layout_detector is not None
                    page.layout = self.layout_detector.detect(image, page_id)
                elif had_layout and "layout" not in reused:
                    reused.append("layout")
                layout_s = page.layout.latency_s
            return page.layout

        item = RouterInput(qid, page_id, question, None, get_layout(stage1), lambda: image)
        features, signal_s, page_signal_s = self._features(stage1, item)
        row = SimpleNamespace(question_id=qid, question=question)
        start = time.perf_counter()
        p1 = r.stage1.predict([row], {qid: features})[qid]
        decide_s = time.perf_counter() - start
        params = tuple(r.params) if len(r.params) == 3 else (*r.params, OFF)
        if p1 >= params[0]:
            path, _ = r.choose(p1, params=params)
            return LiveDecision(
                path, 1, p1, None, None, params, None, layout_s, 0.0, page_signal_s,
                signal_s, decide_s, tuple(reused),
            )  # fmt: skip

        if page.ocr is None:
            page.ocr = self.ocr_engine.extract(image, page_id)
        else:
            reused.append("ocr")
        item = RouterInput(qid, page_id, question, page.ocr, get_layout(stage2_only), lambda: image)
        more, more_s, more_page_s = self._features(stage2_only, item)
        features.update(more)
        start = time.perf_counter()
        p2 = r.stage2.predict([row], {qid: features})[qid]
        p3 = r.hybrid.predict([row], {qid: features})[qid] if r.hybrid is not None else None
        decide_s += time.perf_counter() - start
        path, _ = r.choose(p1, p2, p3, params)
        return LiveDecision(
            path, 2, p1, p2, p3, params, page.ocr, layout_s, page.ocr.latency_s,
            page_signal_s + more_page_s, signal_s + more_s, decide_s, tuple(reused),
        )  # fmt: skip

    def forget(self, page_id: str) -> None:
        """Drop a page's cached layout and OCR (e.g. when the user removes it)."""
        self._pages.pop(page_id, None)


def live_extractors(
    question_encoder: "Encoder | None" = None, clip_encoder: "Encoder | None" = None
) -> dict[str, Any]:
    """S1-S4 as usual, and S5/S6 computed on the spot when their encoders are given."""
    from qdu.router import EXTRACTORS

    out: dict[str, Any] = {k: EXTRACTORS[k]() for k in ("s1", "s2", "s3", "s4")}
    if question_encoder is not None:
        out["s5"] = LiveQuestionEmbeddingSignals(question_encoder)
    if clip_encoder is not None:
        out["s6"] = LivePageEmbeddingSignals(clip_encoder)
    return out
