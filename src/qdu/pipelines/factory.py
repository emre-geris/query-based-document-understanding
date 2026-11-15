from collections.abc import Callable

from qdu.layout import PageLayout
from qdu.models.base import LanguageModel
from qdu.ocr.base import OcrEngine
from qdu.pipelines.base import Pipeline, PipelineName
from qdu.pipelines.crop import CropPipeline
from qdu.pipelines.hybrid import HybridPipeline
from qdu.pipelines.text import TextPipeline
from qdu.pipelines.vision import VisionPipeline

PIPELINE_NAMES: tuple[PipelineName, ...] = ("text", "vision", "hybrid")
"""The three paths of the labelled run (and the default of ``qdu benchmark run``)."""
EXTRA_PIPELINES: tuple[PipelineName, ...] = ("vision-low", "crop")
"""Phase 7 candidate paths, run separately and merged into the label table if they pass."""
ALL_PIPELINES: tuple[PipelineName, ...] = (*PIPELINE_NAMES, *EXTRA_PIPELINES)
VISION_LOW_FRACTION = 0.25
"""Share of the page's pixels the ``vision-low`` path sends."""


def build_pipeline(
    name: str,
    *,
    text_model: LanguageModel | None = None,
    vision_model: LanguageModel | None = None,
    ocr: OcrEngine | None = None,
    layout_of: Callable[[str], PageLayout | None] | None = None,
) -> Pipeline:
    """Build one pipeline, checking that the components it needs were provided."""
    if name == "text":
        if text_model is None or ocr is None:
            raise ValueError("text pipeline needs a text model and an OCR engine")
        return TextPipeline(text_model, ocr)
    if name == "vision":
        if vision_model is None:
            raise ValueError("vision pipeline needs a vision model")
        return VisionPipeline(vision_model)
    if name == "vision-low":
        if vision_model is None:
            raise ValueError("vision-low pipeline needs a vision model")
        return VisionPipeline(vision_model, name="vision-low", pixel_fraction=VISION_LOW_FRACTION)
    if name == "crop":
        if vision_model is None or ocr is None or layout_of is None:
            raise ValueError("crop pipeline needs a vision model, an OCR engine and page layouts")
        return CropPipeline(vision_model, ocr, layout_of)
    if name == "hybrid":
        if vision_model is None or ocr is None:
            raise ValueError("hybrid pipeline needs a vision model and an OCR engine")
        return HybridPipeline(vision_model, ocr)
    raise ValueError(f"unknown pipeline {name!r}; choose from {', '.join(ALL_PIPELINES)}")
