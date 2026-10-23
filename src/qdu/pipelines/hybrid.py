from qdu.models.base import LanguageModel
from qdu.ocr.base import OcrEngine
from qdu.pipelines.base import PipelineInput, PipelineOutput
from qdu.pipelines.prompts import clean_answer, hybrid_prompt


class HybridPipeline:
    """Page image + OCR text -> vision-language model."""

    name = "hybrid"

    def __init__(self, model: LanguageModel, ocr: OcrEngine) -> None:
        if not model.supports_images:
            raise ValueError(f"hybrid pipeline needs an image-capable model, got {model.name}")
        self.model = model
        self.ocr = ocr

    def run(self, inp: PipelineInput) -> PipelineOutput:
        ocr = self.ocr.extract(inp.image, inp.page_key)
        gen = self.model.generate(hybrid_prompt(inp.question, ocr.text), images=[inp.image])
        return PipelineOutput(
            pipeline="hybrid",
            answer=clean_answer(gen.text),
            raw_output=gen.text,
            model=gen.model,
            input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens,
            model_latency_s=gen.latency_s,
            ocr_engine=ocr.engine,
            ocr_latency_s=ocr.latency_s,
            ocr_chars=len(ocr.text),
        )
