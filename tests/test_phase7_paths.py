"""The Phase 7 candidate paths: vision-low (direction A) and crop (direction B)."""

import json
from dataclasses import dataclass, field

import pytest
from PIL import Image

from qdu.layout import PageLayout, Region
from qdu.models.base import Generation
from qdu.ocr.base import OcrLine, OcrResult
from qdu.pipelines import ALL_PIPELINES, PIPELINE_NAMES, PipelineInput, build_pipeline
from qdu.pipelines.crop import MIN_HEIGHT, MIN_WIDTH, crop_box, evidence_lines
from qdu.pipelines.factory import VISION_LOW_FRACTION
from qdu.pipelines.prompts import ANSWER_INSTRUCTION, crop_prompt, vision_prompt
from qdu.pipelines.vision import scale_pixels

PAGE = (1000, 2000)
GOLD = {"docvqa/1": "a", "docvqa/2": "b"}


@dataclass
class SeeingModel:
    """Records the prompt and image size of every call."""

    name: str = "seeing"
    supports_images: bool = True
    seen: list[tuple[str, tuple[int, int]]] = field(default_factory=list)

    def generate(self, prompt, images=()):
        self.seen.append((prompt, images[0].size if images else (0, 0)))
        return Generation("Acme", self.name, 10, 1, 0.5)


@dataclass
class OneOcr:
    result: OcrResult
    name: str = "stored"

    def extract(self, image, key):
        return self.result


LINES = (
    OcrLine("Annual report 2019", 0.99, (100, 50, 600, 90)),
    OcrLine("Total revenue 4,512", 0.98, (100, 1500, 500, 1540)),
    OcrLine("Contact: Acme Corp", 0.97, (100, 1900, 400, 1940)),
)


def test_core_paths_stay_the_default_and_candidates_are_extra():
    assert PIPELINE_NAMES == ("text", "vision", "hybrid")
    assert ALL_PIPELINES == ("text", "vision", "hybrid", "vision-low", "crop")


def test_scale_pixels_keeps_aspect_and_quarter_of_the_pixels():
    image = Image.new("RGB", PAGE, "white")
    small = scale_pixels(image, 0.25)
    assert small.size == (500, 1000)
    assert scale_pixels(image, 1.0) is image


def test_vision_low_sends_a_quarter_of_the_pixels_with_the_vision_prompt():
    model = SeeingModel()
    pipeline = build_pipeline("vision-low", vision_model=model)
    out = pipeline.run(PipelineInput("what?", Image.new("RGB", PAGE, "white"), "p"))
    prompt, size = model.seen[0]
    assert prompt == vision_prompt("what?")
    assert size[0] * size[1] == pytest.approx(VISION_LOW_FRACTION * PAGE[0] * PAGE[1], rel=0.01)
    assert out.pipeline == "vision-low" and out.image_fraction == VISION_LOW_FRACTION
    plain = build_pipeline("vision", vision_model=SeeingModel())
    assert plain.run(PipelineInput("q", Image.new("RGB", PAGE), "p")).image_fraction is None


def test_evidence_lines_are_the_best_matching_ones():
    assert [line.text for line in evidence_lines("What was the total revenue?", LINES)] == [
        "Total revenue 4,512"
    ]
    assert evidence_lines("Who signed it?", LINES) == []


def test_crop_box_grows_to_the_layout_region_and_a_minimum_size():
    region = Region("table", 0.9, (80, 1400, 900, 1700))
    layout = PageLayout("p", "m", 0.1, "cpu", PAGE, (region,))
    x0, y0, x1, y1 = crop_box("total revenue", LINES, PAGE, layout)
    assert x0 <= 80 and x1 >= 900 and y0 <= 1400 and y1 >= 1700  # the enclosing table
    assert y1 - y0 >= MIN_HEIGHT * PAGE[1] - 1e-6 and x1 - x0 >= MIN_WIDTH * PAGE[0] - 1e-6
    assert x0 >= 0 and y0 >= 0 and x1 <= PAGE[0] and y1 <= PAGE[1]
    bare = crop_box("total revenue", LINES, PAGE)  # no layout: the line plus margin, min size
    assert bare[1] < 1500 < 1540 < bare[3] and bare[3] - bare[1] == pytest.approx(500)
    assert crop_box("who signed it", LINES, PAGE) is None


def test_crop_pipeline_sends_the_crop_or_the_whole_page():
    ocr = OneOcr(OcrResult("...", "stored", 0.8, lines=LINES, page_size=PAGE, hardware="L4"))
    model = SeeingModel()
    pipeline = build_pipeline("crop", vision_model=model, ocr=ocr, layout_of=lambda key: None)
    page = Image.new("RGB", PAGE, "white")
    out = pipeline.run(PipelineInput("What was the total revenue?", page, "p"))
    prompt, size = model.seen[0]
    assert prompt == crop_prompt("What was the total revenue?") and ANSWER_INSTRUCTION in prompt
    assert size[0] < PAGE[0] or size[1] < PAGE[1]
    assert out.pipeline == "crop" and 0 < out.image_fraction < 1
    assert out.ocr_latency_s == 0.8 and out.ocr_hardware == "L4"  # the crop pays the page's OCR
    fallback = pipeline.run(PipelineInput("Who signed it?", page, "p"))
    assert model.seen[1] == (vision_prompt("Who signed it?"), PAGE)
    assert fallback.image_fraction == 1.0


def test_crop_needs_ocr_and_layout():
    with pytest.raises(ValueError, match="crop pipeline needs"):
        build_pipeline("crop", vision_model=SeeingModel())
    with pytest.raises(ValueError, match="unknown pipeline"):
        build_pipeline("zoom", vision_model=SeeingModel())


def test_labels_merge_an_extra_run(tmp_path):
    from qdu.data import QARecord, write_manifest
    from qdu.router import build_labels, read_labels, write_labels
    from qdu.router.evaluate import evaluate

    manifest = tmp_path / "m.csv"
    records = [
        QARecord("docvqa/1", "docvqa/d1", "q1", ("a",), (), "docvqa", split="val"),
        QARecord("docvqa/2", "docvqa/d2", "q2", ("b",), (), "docvqa", split="val"),
    ]
    write_manifest(records, manifest)

    def row(qid, path, answer, latency, tokens):
        return {
            "question_id": qid, "doc_id": qid.replace("/", "/d"), "stratum": "docvqa/x",
            "pipeline": path, "question": "q", "answers": [GOLD[qid]],
            "prediction": answer, "anls": None, "model": "m", "input_tokens": tokens,
            "output_tokens": 1, "model_latency_s": latency,
            "ocr_engine": "o" if path in ("text", "hybrid") else None,
            "ocr_latency_s": 0.5 if path in ("text", "hybrid") else None,
            "ocr_chars": 10, "raw_output": answer, "error": None,
        }  # fmt: skip

    main, extra = tmp_path / "main", tmp_path / "extra"
    main.mkdir()
    extra.mkdir()
    lines = [
        row(q, p, "a" if q.endswith("1") else "b", 1.0, 100)
        for q in ("docvqa/1", "docvqa/2")
        for p in ("text", "vision", "hybrid")
    ]
    (main / "predictions.jsonl").write_text("\n".join(map(json.dumps, lines)) + "\n")
    low = [row("docvqa/1", "vision-low", "a", 0.4, 30), row("docvqa/2", "vision-low", "x", 0.4, 30)]
    (extra / "predictions.jsonl").write_text("\n".join(map(json.dumps, low)) + "\n")

    paths = ("text", "vision", "hybrid", "vision-low")
    rows = build_labels(main, manifest, extra_runs=[extra], paths=paths)
    assert [r.score["vision-low"] for r in rows] == [1.0, 0.0]
    assert rows[0].oracle_latency == "vision-low"  # as good and cheapest
    out = tmp_path / "labels.csv"
    write_labels(rows, out, paths)
    back = read_labels(out)
    assert set(back[0].score) == set(paths)
    methods = {m.method for m in evaluate(back, resamples=10) if m.group == "ALL"}
    assert {"always-vision-low", "always-hybrid", "oracle"} <= methods
    assert len(build_labels(main, manifest)) == 2  # without the extra run: the three paths only
