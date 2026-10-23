from collections import Counter
from pathlib import Path

import pytest

from qdu.data import (
    QARecord,
    assign_stratum,
    label_counts,
    read_manifest,
    stratified_sample,
    write_manifest,
)
from qdu.data.docvqa import record_from_row


def make_records() -> list[QARecord]:
    """60 docs x 2 questions: mostly common types, a few rare ones."""
    records = []
    for doc in range(60):
        for q in range(2):
            qid = f"{doc:03d}{q}"
            if doc < 4:
                types = ("figure/diagram",)
            elif doc < 10:
                types = ("table/list", "handwritten")
            elif doc % 2:
                types = ("table/list",)
            else:
                types = ("form",)
            records.append(QARecord(qid, f"d{doc}", f"q{qid}?", ("a",), types))
    return records


def test_rarest_label_defines_stratum() -> None:
    counts = Counter({"table/list": 50, "handwritten": 6})
    assert assign_stratum(("table/list", "handwritten"), counts) == "handwritten"
    assert assign_stratum((), counts) == "unknown"


def test_sample_size_one_question_per_document_and_deterministic() -> None:
    records = make_records()
    a = stratified_sample(records, 30, min_per_stratum=4, seed=1)
    b = stratified_sample(list(reversed(records)), 30, min_per_stratum=4, seed=1)
    assert len(a) == 30
    assert len({r.doc_id for r in a}) == 30
    assert a == b  # input order does not matter


def test_rare_strata_reach_the_minimum() -> None:
    sample = stratified_sample(make_records(), 20, min_per_stratum=4, seed=0)
    per_stratum = Counter(r.stratum for r in sample)
    assert per_stratum["figure/diagram"] == 4
    assert per_stratum["handwritten"] >= 4
    assert sum(per_stratum.values()) == 20


def test_different_seeds_give_different_samples() -> None:
    records = make_records()
    a = {r.question_id for r in stratified_sample(records, 20, seed=0)}
    b = {r.question_id for r in stratified_sample(records, 20, seed=1)}
    assert a != b


def test_too_large_n_is_rejected() -> None:
    with pytest.raises(ValueError, match="only 60 eligible"):
        stratified_sample(make_records(), 61)


def test_label_counts() -> None:
    counts = label_counts(make_records())
    assert counts["figure/diagram"] == 8


def test_manifest_round_trip(tmp_path: Path) -> None:
    records = stratified_sample(make_records(), 10, seed=3)
    path = tmp_path / "manifest.csv"
    write_manifest(records, path)
    assert read_manifest(path) == records


def test_record_from_docvqa_row() -> None:
    row = {
        "questionId": 49153,
        "docId": 14465,
        "question": "What is the value?",
        "answers": ["0.28"],
        "question_types": ["figure/diagram"],
    }
    record = record_from_row(row)
    assert record.question_id == "49153"
    assert record.doc_id == "14465"
    assert record.answers == ("0.28",)
