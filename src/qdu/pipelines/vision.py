from qdu.models.base import LanguageModel
from qdu.pipelines.base import PipelineInput, PipelineOutput
from qdu.pipelines.prompts import clean_answer, vision_prompt


class VisionPipeline:
    """Page image -> vision-language model. No OCR; most robust to layout, costliest input."""

    name = "vision"

    def __init__(self, model: LanguageModel) -> None:
        if not model.supports_images:
            raise ValueError(f"vision pipeline needs an image-capable model, got {model.name}")
        self.model = model

    def run(self, inp: PipelineInput) -> PipelineOutput:
        gen = self.model.generate(vision_prompt(inp.question), images=[inp.image])
        return PipelineOutput(
            pipeline="vision",
            answer=clean_answer(gen.text),
            raw_output=gen.text,
            model=gen.model,
            input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens,
            model_latency_s=gen.latency_s,
        )
