import pytest
from PIL import Image

from qdu.models import FakeModel, create_model
from qdu.ocr import OcrResult
from qdu.pipelines import (
    HybridPipeline,
    PipelineInput,
    TextPipeline,
    VisionPipeline,
    build_pipeline,
)
from qdu.pipelines.prompts import clean_answer


class StaticOcr:
    name = "static"

    def __init__(self, text: str) -> None:
        self.text = text
        self.keys: list[str] = []

    def extract(self, image: Image.Image, key: str) -> OcrResult:
        self.keys.append(key)
        return OcrResult(self.text, self.name, latency_s=0.25)


@pytest.fixture
def inp() -> PipelineInput:
    return PipelineInput("What is the total?", Image.new("RGB", (10, 10)), page_key="doc7")


def answer_with(text: str) -> FakeModel:
    return FakeModel(responder=lambda prompt, n_images: text)


def test_text_pipeline_sends_ocr_text_and_no_image(inp: PipelineInput) -> None:
    model, ocr = answer_with("Answer: $12.50."), StaticOcr("TOTAL $12.50")
    out = TextPipeline(model, ocr).run(inp)
    prompt, n_images = model.calls[0]
    assert "TOTAL $12.50" in prompt
    assert "What is the total?" in prompt
    assert n_images == 0
    assert ocr.keys == ["doc7"]
    assert out.pipeline == "text"
    assert out.answer == "$12.50"
    assert out.ocr_latency_s == 0.25
    assert out.ocr_chars == len("TOTAL $12.50")


def test_vision_pipeline_sends_image_and_no_ocr(inp: PipelineInput) -> None:
    model = answer_with("12.50")
    out = VisionPipeline(model).run(inp)
    prompt, n_images = model.calls[0]
    assert n_images == 1
    assert "OCR" not in prompt
    assert out.pipeline == "vision"
    assert out.ocr_latency_s is None


def test_hybrid_pipeline_sends_both(inp: PipelineInput) -> None:
    model = answer_with("12.50")
    out = HybridPipeline(model, StaticOcr("TOTAL $12.50")).run(inp)
    prompt, n_images = model.calls[0]
    assert n_images == 1
    assert "TOTAL $12.50" in prompt
    assert out.pipeline == "hybrid"


@pytest.mark.parametrize("cls", [VisionPipeline, HybridPipeline])
def test_image_pipelines_reject_text_only_models(cls: type) -> None:
    text_only = FakeModel(supports_images=False)
    with pytest.raises(ValueError, match="image-capable"):
        cls(text_only) if cls is VisionPipeline else cls(text_only, StaticOcr(""))


def test_all_pipelines_share_the_answer_instruction(inp: PipelineInput) -> None:
    models = [FakeModel(), FakeModel(), FakeModel()]
    TextPipeline(models[0], StaticOcr("x")).run(inp)
    VisionPipeline(models[1]).run(inp)
    HybridPipeline(models[2], StaticOcr("x")).run(inp)
    assert all(m.calls[0][0].endswith("Do not explain.") for m in models)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Answer: John Smith.", "John Smith"),
        ('"1975"', "1975"),
        ("\n\n  0.28\nbecause the table says so", "0.28"),
        ("Final answer - ITC Limited", "ITC Limited"),
        ("12.", "12"),
        ("140 858", "140858"),
        ("1 413.68", "1413.68"),
        ("1\u202f234\u202f567", "1234567"),
        ("2019 2020", "2019 2020"),  # not a thousands group
        ("Top 3 brands", "Top 3 brands"),
        ("1. courage\n2. clout\n3. astuteness", "courage, clout, astuteness"),
        ("2. Hong Kong\n1. Singapore", "Singapore, Hong Kong"),  # numbered items in order
        ("- red\n- blue", "red, blue"),
        ("1. FC Köln", "1. FC Köln"),  # a single line is not a list
        ("Answer: 12\nbecause the table says so", "12"),  # not every line is an item
        ("", ""),
    ],
)
def test_clean_answer(raw: str, expected: str) -> None:
    assert clean_answer(raw) == expected


def test_registry() -> None:
    assert create_model("fake").name == "fake"
    with pytest.raises(ValueError, match="unknown model spec"):
        create_model("gpt-x")


def test_factory_builds_each_pipeline_and_checks_components() -> None:
    model, ocr = FakeModel(), StaticOcr("x")
    assert build_pipeline("text", text_model=model, ocr=ocr).name == "text"
    assert build_pipeline("vision", vision_model=model).name == "vision"
    assert build_pipeline("hybrid", vision_model=model, ocr=ocr).name == "hybrid"
    with pytest.raises(ValueError, match="needs a text model"):
        build_pipeline("text", vision_model=model)
    with pytest.raises(ValueError, match="unknown pipeline"):
        build_pipeline("audio")
