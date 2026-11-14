"""Answer one question about an uploaded page: route it, run the chosen path, add up the cost.

Costs follow the evaluation's scenarios (`qdu.router.evaluate.Policy.cost`):

- **C** (each question pays): model time of every path run + the page's OCR time if a path or
  the router needed OCR + layout time if the router used layout + the router's own time.
- **B** (page work shared): as C, but OCR, layout and page-level signal time are divided by
  the number of questions asked so far about the page.
- **A**: tokens of the paths run.

OCR runs at most once per page (`SharedOcr`): the router and the text/hybrid paths read the
same result, as in the labelled run, so its time is counted once. With ``compare=True`` all
three paths answer (for showing what routing saves); the chosen path is still the router's.

With gold ``answers`` (an example page, `qdu.demo.examples`) every answer is scored by the
dataset's primary metric. With ``recorded`` outputs (the labelled run's answers for an example)
a path takes the recorded answer, model time and tokens instead of calling the model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qdu.evaluation.primary import primary_metric, primary_score
from qdu.models.base import LanguageModel
from qdu.ocr.base import OcrEngine, OcrResult
from qdu.pipelines import PipelineInput, PipelineOutput, build_pipeline
from qdu.router.live import LiveDecision, LiveRouter

if TYPE_CHECKING:
    from PIL.Image import Image

PATHS = ("text", "vision", "hybrid")
USES_OCR = {"text": True, "vision": False, "hybrid": True}
MODES = ("auto", *PATHS)
"""``auto``: the router chooses; a path name forces that path (the router is not run)."""


class SharedOcr:
    """An OCR engine that reads each page once and hands the same result to every caller."""

    def __init__(self, engine: OcrEngine) -> None:
        self.engine = engine
        self.name = engine.name
        self._pages: dict[str, OcrResult] = {}

    def extract(self, image: "Image", key: str) -> OcrResult:
        if key not in self._pages:
            self._pages[key] = self.engine.extract(image, key)
        return self._pages[key]

    def peek(self, key: str) -> OcrResult | None:
        """The page's OCR if it was read already, without running it."""
        return self._pages.get(key)

    def forget(self, key: str) -> None:
        self._pages.pop(key, None)


@dataclass(frozen=True)
class PathRun:
    """One path's answer and what it cost on its own (as if always that path, scenario C)."""

    path: str
    output: PipelineOutput
    model_s: float
    ocr_s: float
    tokens_in: int
    tokens_out: int
    recorded: bool = False
    """True when the answer comes from the labelled run, not a live model call."""
    score: float | None = None
    """Primary-metric score against the gold answers, when they are known."""

    @property
    def answer(self) -> str:
        return self.output.answer

    @property
    def cost_c_s(self) -> float:
        return self.model_s + self.ocr_s


@dataclass(frozen=True)
class DemoAnswer:
    question: str
    page_id: str
    mode: str
    path: str
    """The path whose answer is shown: the router's choice, or the forced one."""
    run: PathRun
    decision: LiveDecision | None
    """None when a path was forced (the router did not run)."""
    compare: dict[str, PathRun] = field(default_factory=dict)
    """Every path's run, when comparing (includes ``run``)."""
    questions_on_page: int = 1
    """Questions asked about this page so far, this one included (scenario B's divisor)."""
    ocr_s: float = 0.0
    """The page's OCR time if this question needed OCR (router or path), else 0."""
    layout_s: float = 0.0
    page_signal_s: float = 0.0
    router_s: float = 0.0
    """The router's own per-question time: signals and decision (0 when forced)."""
    gold: tuple[str, ...] = ()
    """Gold answers (example pages only)."""
    metric: str | None = None
    """The dataset's primary metric used for ``score`` (anls, relaxed_accuracy or f1)."""

    @property
    def score(self) -> float | None:
        return self.run.score

    @property
    def model_s(self) -> float:
        return self.run.model_s

    @property
    def total_c_s(self) -> float:
        """Scenario C: this question pays the page's OCR and layout in full."""
        return self.ocr_s + self.layout_s + self.page_signal_s + self.router_s + self.model_s

    @property
    def total_b_s(self) -> float:
        """Scenario B: page-level work shared by the page's questions so far."""
        shared = (self.ocr_s + self.layout_s + self.page_signal_s) / self.questions_on_page
        return shared + self.router_s + self.model_s

    @property
    def tokens(self) -> tuple[int, int]:
        return self.run.tokens_in, self.run.tokens_out


def _run(
    pipeline,
    path: str,
    question: str,
    image: "Image",
    page_id: str,
    recorded: PipelineOutput | None = None,
    score=None,
) -> PathRun:
    out = recorded or pipeline.run(PipelineInput(question, image, page_id))
    return PathRun(
        path,
        out,
        out.model_latency_s,
        (out.ocr_latency_s or 0.0) if USES_OCR[path] else 0.0,
        out.input_tokens or 0,
        out.output_tokens or 0,
        recorded=recorded is not None,
        score=score(out.answer) if score else None,
    )


class DemoService:
    """Router + the three paths on one model, with OCR shared between them."""

    def __init__(self, router: LiveRouter, model: LanguageModel, ocr: SharedOcr) -> None:
        if router.ocr_engine is not ocr:
            raise ValueError("give the router the same SharedOcr, so a page is read only once")
        self.router = router
        self.ocr = ocr
        self.pipelines = {
            p: build_pipeline(p, text_model=model, vision_model=model, ocr=ocr) for p in PATHS
        }
        self._asked: dict[str, int] = {}

    def ask(
        self,
        question: str,
        image: "Image",
        page_id: str,
        mode: str = "auto",
        compare: bool = False,
        *,
        answers: Sequence[str] = (),
        dataset: str | None = None,
        recorded: Mapping[str, PipelineOutput] | None = None,
    ) -> DemoAnswer:
        """``answers`` and ``dataset`` score the answers; ``recorded`` (path -> the labelled
        run's output) replaces the model call for the paths it has."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
        n = self._asked[page_id] = self._asked.get(page_id, 0) + 1
        decision = self.router.decide(question, image, page_id) if mode == "auto" else None
        path = decision.path if decision else mode
        metric = primary_metric(dataset or "") if answers else None
        score = (lambda a: primary_score(dataset or "", a, answers)) if answers else None
        to_run = PATHS if compare else (path,)
        if recorded and any(USES_OCR[p] for p in to_run):
            self.ocr.extract(image, page_id)  # recorded paths skip the pipeline, not the page
        runs = {
            p: _run(self.pipelines[p], p, question, image, page_id, (recorded or {}).get(p), score)
            for p in to_run
        }
        needs_ocr = USES_OCR[path] or (decision is not None and decision.stage == 2)
        page_ocr = self.ocr.peek(page_id)
        return DemoAnswer(
            question=question,
            page_id=page_id,
            mode=mode,
            path=path,
            run=runs[path],
            decision=decision,
            compare=runs if compare else {},
            questions_on_page=n,
            ocr_s=page_ocr.latency_s if needs_ocr and page_ocr else 0.0,
            layout_s=decision.layout_s if decision else 0.0,
            page_signal_s=decision.page_signal_s if decision else 0.0,
            router_s=(decision.signal_s + decision.decide_s) if decision else 0.0,
            gold=tuple(answers),
            metric=metric,
        )

    def forget(self, page_id: str) -> None:
        """Drop everything cached about a page (OCR, layout, question count)."""
        self.ocr.forget(page_id)
        self.router.forget(page_id)
        self._asked.pop(page_id, None)


def demo_fake_model() -> LanguageModel:
    """The test model with a readable answer, for building the interface without a GPU. Its
    answers and times mean nothing and are never reported."""
    from qdu.models.fake import FakeModel

    def respond(prompt: str, n_images: int) -> str:
        if not n_images:
            return "[fake model] answer from the text path"
        return (
            "[fake model] answer from the hybrid path"
            if "OCR text:" in prompt
            else "[fake model] answer from the vision path"
        )

    return FakeModel(name="fake", responder=respond)


def describe(answer: DemoAnswer) -> str:
    """A plain-text report of one answer: decision, answer, cost (for ``qdu demo ask``)."""
    d = answer.decision
    lines = [f"Question: {answer.question}"]
    if d is None:
        lines.append(f"Path: {answer.path} (forced; router not run)")
    else:
        t1, t2, t3 = d.thresholds
        lines.append(f"Path: {answer.path} (router, decided in stage {d.stage})")
        lines.append(f"  stage 1: p(image needed) = {d.p_image_stage1:.3f}, threshold t1 = {t1}")
        if d.stage == 2:
            lines.append(
                f"  stage 2: p(image needed) = {d.p_image_stage2:.3f}, threshold t2 = {t2}"
            )
            if d.p_hybrid is not None:
                lines.append(f"  hybrid head: p(hybrid > vision) = {d.p_hybrid:.4f}, t3 = {t3:.4f}")
        if d.reused:
            lines.append(f"  reused from an earlier question on this page: {', '.join(d.reused)}")
    source = " (recorded in the labelled run)" if answer.run.recorded else ""
    lines.append(f"Answer: {answer.run.answer}{source}")
    if answer.gold:
        lines.append(f"Gold: {' | '.join(answer.gold)}; {answer.metric} = {answer.score:.2f}")
    lines.append(
        f"Time (scenario C): {answer.total_c_s:.3f} s = OCR {answer.ocr_s:.3f} + layout "
        f"{answer.layout_s:.3f} + page signals {answer.page_signal_s:.3f} + router "
        f"{answer.router_s:.3f} + model {answer.model_s:.3f}"
    )
    lines.append(
        f"Time (scenario B, {answer.questions_on_page} question(s) on this page): "
        f"{answer.total_b_s:.3f} s"
    )
    lines.append(f"Tokens (scenario A): {answer.tokens[0]} in, {answer.tokens[1]} out")
    for p, run in answer.compare.items():
        mark = "  <- shown" if p == answer.path else ""
        scored = f", {answer.metric} {run.score:.2f}" if run.score is not None else ""
        lines.append(
            f"  always-{p}: {run.answer!r}{scored}, {run.cost_c_s:.3f} s, "
            f"{run.tokens_in + run.tokens_out} tokens{mark}"
        )
    return "\n".join(lines)
