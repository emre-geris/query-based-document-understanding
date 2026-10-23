from qdu.models.base import LanguageModel
from qdu.ocr.base import OcrEngine
from qdu.pipelines.base import Pipeline, PipelineName
from qdu.pipelines.hybrid import HybridPipeline
from qdu.pipelines.text import TextPipeline
from qdu.pipelines.vision import VisionPipeline

PIPELINE_NAMES: tuple[PipelineName, ...] = ("text", "vision", "hybrid")


def build_pipeline(
    name: str,
    *,
    text_model: LanguageModel | None = None,
    vision_model: LanguageModel | None = None,
    ocr: OcrEngine | None = None,
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
    if name == "hybrid":
        if vision_model is None or ocr is None:
            raise ValueError("hybrid pipeline needs a vision model and an OCR engine")
        return HybridPipeline(vision_model, ocr)
    raise ValueError(f"unknown pipeline {name!r}; choose from {', '.join(PIPELINE_NAMES)}")
