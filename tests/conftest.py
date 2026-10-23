from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

TYPES = ["table/list", "form", "handwritten", "figure/diagram"]


def render_page(text: str) -> Image.Image:
    image = Image.new("RGB", (800, 200), "white")
    ImageDraw.Draw(image).text((20, 60), text, fill="black", font=ImageFont.load_default(size=40))
    return image


class FakeDocVQA:
    """Minimal stand-in for the Hugging Face dataset API used by qdu.data.docvqa."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def select_columns(self, columns: list[str]) -> "FakeDocVQA":
        return FakeDocVQA([{c: r[c] for c in columns} for r in self.rows])

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self.rows]
        return self.rows[key]


@pytest.fixture
def fake_docvqa() -> FakeDocVQA:
    """24 questions on 24 pages; the answer to question i is printed on page i."""
    return FakeDocVQA(
        [
            {
                "questionId": 1000 + i,
                "docId": 500 + i,
                "question": f"What is code {i}?",
                "answers": [f"CODE{i}"],
                "question_types": [TYPES[i % 4]],
                "image": render_page(f"CODE{i}"),
            }
            for i in range(24)
        ]
    )
