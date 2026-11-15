"""Crop path (Phase 7, direction B): send only the part of the page that holds the evidence.

The evidence-location signal S4 finds the OCR lines that contain the question's terms; as a
routing feature it added nothing, so here the location becomes an action. The crop is the union
of the best-matching lines, grown to the layout regions (PP-DocLayout) that contain them, plus a
margin, and never smaller than :data:`MIN_WIDTH` x :data:`MIN_HEIGHT` of the page (a value often
sits next to its label, outside the line that matched). When no line matches, the whole page is
sent, so the path never loses the evidence it could not locate; the share of the page sent is
recorded per answer (``image_fraction``), so the fallback rate can be counted.

The crop is cut from the full-resolution page and then goes through the same model and pixel cap
as the vision path. Localisation is training-free (OCR + layout); prior cropping work localises
with model attention or trained heads (ViCrop, Q-Zoom, C²-Routing; ``docs/related_work.md``).
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from qdu.layout import PageLayout
from qdu.models.base import LanguageModel
from qdu.ocr.base import OcrEngine, OcrLine
from qdu.pipelines.base import PipelineInput, PipelineOutput
from qdu.pipelines.prompts import clean_answer, crop_prompt, vision_prompt

if TYPE_CHECKING:
    from PIL.Image import Image

Box = tuple[float, float, float, float]

MARGIN = 0.03
"""Margin around the crop, as a share of the page's width and height."""
MIN_WIDTH = 0.5
MIN_HEIGHT = 0.25
"""The crop is at least this share of the page's width and height."""


def _contains(region: Box, point: tuple[float, float]) -> bool:
    x0, y0, x1, y1 = region
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


def _grow(lo: float, hi: float, minimum: float, limit: float) -> tuple[float, float]:
    """Widen [lo, hi] around its centre to at least ``minimum``, kept inside [0, limit]."""
    if hi - lo >= minimum:
        return max(0.0, lo), min(limit, hi)
    centre = (lo + hi) / 2
    lo, hi = centre - minimum / 2, centre + minimum / 2
    if lo < 0:
        lo, hi = 0.0, minimum
    if hi > limit:
        lo, hi = limit - minimum, limit
    return max(0.0, lo), min(limit, hi)


def evidence_lines(question: str, lines: Sequence[OcrLine]) -> list[OcrLine]:
    """The OCR lines sharing the most content terms with the question (none if no line does);
    terms are matched as in the evidence signal S4."""
    from qdu.router.signals import terms  # lazy: qdu.router imports the runner and pipelines

    q = terms(question)
    scored = [(line, len(terms(line.text) & q)) for line in lines]
    best = max((n for _, n in scored), default=0)
    return [line for line, n in scored if n == best] if best else []


def crop_box(
    question: str,
    lines: Sequence[OcrLine],
    page_size: tuple[int, int],
    layout: PageLayout | None = None,
) -> Box | None:
    """The region to send, in page pixels, or None when no OCR line matches the question."""
    hits = evidence_lines(question, lines)
    if not hits:
        return None
    boxes: list[Box] = [tuple(line.box) for line in hits]  # type: ignore[misc]
    for line in hits:
        x0, y0, x1, y1 = line.box
        centre = ((x0 + x1) / 2, (y0 + y1) / 2)
        for region in layout.regions if layout else ():
            if _contains(region.box, centre):
                boxes.append(tuple(region.box))  # type: ignore[arg-type]
    width, height = page_size
    x0 = min(b[0] for b in boxes) - MARGIN * width
    y0 = min(b[1] for b in boxes) - MARGIN * height
    x1 = max(b[2] for b in boxes) + MARGIN * width
    y1 = max(b[3] for b in boxes) + MARGIN * height
    x0, x1 = _grow(x0, x1, MIN_WIDTH * width, width)
    y0, y1 = _grow(y0, y1, MIN_HEIGHT * height, height)
    return (x0, y0, x1, y1)


class CropPipeline:
    """Evidence region of the page image -> vision-language model (OCR + layout to locate it)."""

    name = "crop"

    def __init__(
        self,
        model: LanguageModel,
        ocr: OcrEngine,
        layout_of: Callable[[str], PageLayout | None],
    ) -> None:
        if not model.supports_images:
            raise ValueError(f"crop pipeline needs an image-capable model, got {model.name}")
        self.model = model
        self.ocr = ocr
        self.layout_of = layout_of

    def run(self, inp: PipelineInput) -> PipelineOutput:
        ocr = self.ocr.extract(inp.image, inp.page_key)
        size = ocr.page_size or inp.image.size
        box = crop_box(inp.question, ocr.lines or (), size, self.layout_of(inp.page_key))
        image: Image = inp.image
        fraction = 1.0
        if box is not None:
            # OCR boxes refer to the page OCR saw; scale them if the image differs in size
            sx, sy = inp.image.size[0] / size[0], inp.image.size[1] / size[1]
            pixels = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
            image = inp.image.crop(tuple(round(v) for v in pixels))
            fraction = (image.size[0] * image.size[1]) / (inp.image.size[0] * inp.image.size[1])
        prompt = crop_prompt(inp.question) if box is not None else vision_prompt(inp.question)
        gen = self.model.generate(prompt, images=[image])
        return PipelineOutput(
            pipeline="crop",
            answer=clean_answer(gen.text),
            raw_output=gen.text,
            model=gen.model,
            input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens,
            model_latency_s=gen.latency_s,
            token_logprobs=gen.token_logprobs,
            ocr_engine=ocr.engine,
            ocr_latency_s=ocr.latency_s,
            ocr_chars=len(ocr.text),
            ocr_hardware=ocr.hardware,
            image_fraction=fraction,
        )
