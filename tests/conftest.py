import io

import pytest
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

TYPES = ["table/list", "form", "handwritten", "figure/diagram"]


def render_page(text: str) -> PILImage.Image:
    image = PILImage.new("RGB", (800, 200), "white")
    ImageDraw.Draw(image).text((20, 60), text, fill="black", font=ImageFont.load_default(size=40))
    return image


def png_bytes(image: PILImage.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def fake_docvqa() -> Dataset:
    """48 questions on 48 pages in the lmms-lab/DocVQA schema; page i prints CODEi."""
    n = 48
    features = Features(
        {
            "questionId": Value("string"),
            "docId": Value("int64"),
            "question": Value("string"),
            "answers": Sequence(Value("string")),
            "question_types": Sequence(Value("string")),
            "image": Image(),
        }
    )
    return Dataset.from_dict(
        {
            "questionId": [str(1000 + i) for i in range(n)],
            "docId": [500 + i for i in range(n)],
            "question": [f"What is code {i}?" for i in range(n)],
            "answers": [[f"CODE{i}"] for i in range(n)],
            "question_types": [[TYPES[i % 4]] for i in range(n)],
            "image": [
                {"bytes": png_bytes(render_page(f"CODE{i}")), "path": None} for i in range(n)
            ],
        },
        features=features,
    )


@pytest.fixture
def fake_chartqa() -> Dataset:
    """6 questions on 3 charts (two questions per chart) in the HuggingFaceM4/ChartQA schema."""
    charts = [png_bytes(render_page(f"CHART{c}")) for c in range(3)]
    features = Features(
        {
            "image": Image(),
            "query": Value("string"),
            "label": Sequence(Value("string")),
            "human_or_machine": Value("int64"),
        }
    )
    return Dataset.from_dict(
        {
            "image": [{"bytes": charts[i // 2], "path": None} for i in range(6)],
            "query": [f"Value {i}?" for i in range(6)],
            "label": [[str(i)] for i in range(6)],
            "human_or_machine": [i % 2 for i in range(6)],
        },
        features=features,
    )
