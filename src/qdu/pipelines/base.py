from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image

PipelineName = Literal["text", "vision", "hybrid", "vision-low", "crop"]


@dataclass(frozen=True)
class PipelineInput:
    question: str
    image: "Image"
    page_key: str
    """Stable page identifier, used as the OCR cache key."""


@dataclass(frozen=True)
class PipelineOutput:
    pipeline: PipelineName
    answer: str
    raw_output: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    model_latency_s: float
    ocr_engine: str | None = None
    ocr_latency_s: float | None = None
    ocr_chars: int | None = None
    ocr_hardware: str | None = None
    token_logprobs: tuple[float, ...] | None = None
    image_fraction: float | None = None
    """Share of the page's pixels sent to the model, when a path sends less than the page
    (vision-low, crop; 1.0 when the crop falls back to the whole page)."""


class Pipeline(Protocol):
    name: PipelineName

    def run(self, inp: PipelineInput) -> PipelineOutput: ...
