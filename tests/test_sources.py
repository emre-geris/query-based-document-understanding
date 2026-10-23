from collections import Counter

import pytest
from datasets import Dataset

from qdu.data import ImageStore, assign_split, get_source, iter_source_records
from qdu.data.sources import _docvqa, _infographicvqa


def test_split_is_deterministic_and_document_level() -> None:
    assert assign_split("docvqa", "docvqa/1") == assign_split("docvqa", "docvqa/1")
    counts = Counter(assign_split("docvqa", f"docvqa/{i}") for i in range(10_000))
    assert counts["train"] / 10_000 == pytest.approx(0.6, abs=0.02)
    assert counts["val"] / 10_000 == pytest.approx(0.2, abs=0.02)
    assert counts["test"] / 10_000 == pytest.approx(0.2, abs=0.02)


def test_split_rejects_bad_ratios() -> None:
    with pytest.raises(ValueError, match="summing to 1"):
        assign_split("docvqa", "x", ratios=(0.5, 0.5, 0.5))


def test_docvqa_adapter_prefixes_ids() -> None:
    record = _docvqa(
        0,
        {
            "questionId": "49153",
            "docId": 14465,
            "question": "What is the value?",
            "answers": ["0.28"],
            "question_types": ["figure/diagram"],
        },
    )
    assert (record.question_id, record.doc_id, record.dataset) == (
        "docvqa/49153",
        "docvqa/14465",
        "docvqa",
    )


def test_infographicvqa_adapter_uses_image_url_as_document() -> None:
    record = _infographicvqa(
        3,
        {
            "questionId": "65718",
            "question": "How many?",
            "answers": ["3"],
            "answer_type": ["single span"],
            "operation/reasoning": ["Counting"],
            "image_url": "https://example.org/img/38933.jpeg",
        },
    )
    assert record.doc_id == "infographicvqa/38933.jpeg"
    assert record.question_types == ("single span", "counting")


def test_chartqa_records_share_a_document_per_chart(fake_chartqa: Dataset) -> None:
    records = list(iter_source_records("chartqa", fake_chartqa))
    assert [r.question_id for r in records[:2]] == ["chartqa/test-00000", "chartqa/test-00001"]
    assert len({r.doc_id for r in records}) == 3
    assert records[0].doc_id == records[1].doc_id
    assert records[0].split == records[1].split  # same chart -> same split
    assert {r.question_types for r in records} == {("human",), ("augmented",)}


def test_image_store_resolves_questions_across_datasets(fake_docvqa, fake_chartqa) -> None:
    store = ImageStore({"docvqa": fake_docvqa, "chartqa": fake_chartqa})
    docvqa_record = next(iter_source_records("docvqa", fake_docvqa))
    chart_record = list(iter_source_records("chartqa", fake_chartqa))[4]
    assert store.get(docvqa_record).size == (800, 200)
    assert store.get(chart_record).size == (800, 200)


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        get_source("dude")
