import math
from typing import TYPE_CHECKING

from qdu.models.base import LanguageModel
from qdu.pipelines.base import PipelineInput, PipelineOutput
from qdu.pipelines.prompts import clean_answer, vision_prompt

if TYPE_CHECKING:
    from PIL.Image import Image


def scale_pixels(image: "Image", fraction: float) -> "Image":
    """``image`` downscaled to ``fraction`` of its pixels (aspect ratio kept)."""
    if fraction >= 1.0:
        return image
    scale = math.sqrt(fraction)
    size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
    return image.resize(size, resample=3)  # bicubic, like the model's own pixel cap


class VisionPipeline:
    """Page image -> vision-language model. No OCR; most robust to layout, costliest input.

    ``pixel_fraction`` < 1 sends the page downscaled to that share of its pixels (the
    ``vision-low`` path, Phase 7 direction A). Resizing happens before the model call, so like
    the image encoding it is not part of the measured model latency."""

    def __init__(
        self, model: LanguageModel, *, name: str = "vision", pixel_fraction: float = 1.0
    ) -> None:
        if not model.supports_images:
            raise ValueError(f"vision pipeline needs an image-capable model, got {model.name}")
        if not 0 < pixel_fraction <= 1:
            raise ValueError(f"pixel_fraction must be in (0, 1], got {pixel_fraction}")
        self.model = model
        self.name = name
        self.pixel_fraction = pixel_fraction

    def run(self, inp: PipelineInput) -> PipelineOutput:
        image = scale_pixels(inp.image, self.pixel_fraction)
        gen = self.model.generate(vision_prompt(inp.question), images=[image])
        return PipelineOutput(
            pipeline=self.name,  # type: ignore[arg-type]
            answer=clean_answer(gen.text),
            raw_output=gen.text,
            model=gen.model,
            input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens,
            model_latency_s=gen.latency_s,
            token_logprobs=gen.token_logprobs,
            image_fraction=self.pixel_fraction if self.pixel_fraction < 1 else None,
        )
