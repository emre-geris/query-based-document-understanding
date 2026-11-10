import json
from pathlib import Path

from qdu import benchmark
from qdu.audit import audit


def row(qid: str, pipeline: str, prediction: str, model_s: float, ocr_s: float | None, **extra):
    ocr = pipeline != "vision"
    return {
        "question_id": qid,
        "doc_id": "docvqa/page",
        "stratum": "docvqa/form",
        "pipeline": pipeline,
        "question": "What is the total?",
        "answers": ["42"],
        "prediction": prediction,
        "anls": 1.0 if prediction == "42" else 0.0,
        "model": "m",
        "input_tokens": 100,
        "output_tokens": 2,
        "model_latency_s": model_s,
        "ocr_engine": "paddle" if ocr else None,
        "ocr_latency_s": ocr_s if ocr else None,
        "ocr_chars": 10 if ocr else None,
        "raw_output": prediction,
        "error": None,
        "ocr_hardware": "cpu x 2" if ocr else None,
        **extra,
    }


def write(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir()
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def consistent_rows() -> list[dict]:
    rows = []
    for q in ("docvqa/1", "docvqa/2"):  # two questions on one page
        rows += [
            row(q, "text", "42", 0.2, 2.0),
            row(q, "vision", "41", 1.0, None),
            row(q, "hybrid", "42", 1.2, 2.0),
        ]
    return rows


def test_consistent_run_passes_and_matches_the_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write(run_dir, consistent_rows())
    benchmark.report(run_dir)
    lines, problems = audit(run_dir)
    assert problems == []
    text = "\n".join(lines)
    assert "complete for ['hybrid', 'text', 'vision']: 2" in text
    assert "gold: ['42']" in text


def test_inconsistent_costs_and_scores_are_reported(tmp_path: Path) -> None:
    rows = consistent_rows()
    rows[0]["anls"] = 0.5  # stale score
    rows[3]["ocr_latency_s"] = 3.0  # same page, different OCR time
    rows[4]["ocr_latency_s"] = 0.7  # vision row charged for OCR
    rows.append({**rows[1], "error": "timeout", "prediction": None})
    run_dir = tmp_path / "run"
    write(run_dir, rows)
    lines, problems = audit(run_dir)
    joined = "\n".join(problems)
    assert "stored ANLS differs" in joined
    assert "pages with different OCR time" in joined
    assert "vision row carries OCR cost" in joined
    assert "1 failed calls" in lines[0]
