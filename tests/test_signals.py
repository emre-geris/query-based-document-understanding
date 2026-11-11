import gzip
import json
from pathlib import Path

import pytest

from qdu.layout import PageLayout, Region, compress, load_layout
from qdu.ocr.base import OcrLine, OcrResult
from qdu.router import (
    EvidenceSignals,
    OcrSignals,
    PageSignals,
    QuestionSignals,
    build_features,
)
from qdu.router.features import RouterInput
from qdu.router.signals import kind, terms

# A 1000 x 1000 page: a text block at the top, a table below it, a chart at the bottom.
LAYOUT = PageLayout(
    "docvqa/1",
    "PP-DocLayout-M",
    0.09,
    "cpu",
    (1000, 1000),
    (
        Region("text", 0.9, (0, 0, 1000, 200)),
        Region("table", 0.9, (0, 300, 1000, 600)),
        Region("chart", 0.8, (0, 700, 500, 1000)),
    ),
)
OCR = OcrResult(
    "Annual Report 2019\nRevenue 1,250\nProfit 300\nSales",
    "e",
    0.7,
    lines=(
        OcrLine("Annual Report 2019", 0.99, (10, 10, 400, 60)),
        OcrLine("Revenue 1,250", 0.95, (10, 350, 300, 380)),
        OcrLine("Profit 300", 0.40, (10, 400, 300, 430)),
        OcrLine("Sales", 0.90, (10, 800, 100, 830)),
    ),
    page_size=(1000, 1000),
)


def item(question: str) -> RouterInput:
    return RouterInput("docvqa/q", "docvqa/1", question, OCR, LAYOUT)


def test_terms_drop_stop_words_and_keep_numbers() -> None:
    assert terms("What is the total Revenue in 2019?") == {"revenue", "2019"}
    assert terms("How many 1,250 units") == {"1250", "units"}


def test_s1_question_cues() -> None:
    s = QuestionSignals()(item("How many bars in the chart are higher than 2019?"))
    assert s["wh_how_many"] and not s["wh_how"]
    assert s["cue_visual"] and s["cue_compare"] and s["cue_numeric"] and s["cue_year"]
    assert not s["cue_layout"]
    assert s["n_words"] == 10


def test_s2_ocr_amount_and_confidence() -> None:
    s = OcrSignals()(item("anything"))
    assert s["n_lines"] == 4
    assert s["mean_conf"] == pytest.approx((0.99 + 0.95 + 0.40 + 0.90) / 4)
    assert s["min_conf"] == 0.40
    assert s["low_conf_share"] == 0.25 and s["very_low_conf_share"] == 0.25
    assert s["numeric_share"] == pytest.approx(3 / 8)
    assert s["words_per_mpx"] == 8.0
    assert not s["empty"]


def test_s3_region_shares() -> None:
    s = PageSignals()(item("anything"))
    assert (s["n_text"], s["n_table"], s["n_figure"]) == (1, 1, 1)
    assert s["share_text"] == pytest.approx(0.2)
    assert s["share_table"] == pytest.approx(0.3)
    assert s["share_figure"] == pytest.approx(0.15)
    assert s["log_mpx"] == pytest.approx(0.0)
    assert kind("paragraph_title") == "text" and kind("seal") == "figure"
    assert kind("something-new") == "other"


def test_s4_evidence_location() -> None:
    s = EvidenceSignals()(item("What was the revenue and the profit in 2019?"))
    assert s["match_share"] == 1.0  # revenue, profit, 2019 all found
    assert s["n_matched_lines"] == 3
    assert s["best_line_share"] == pytest.approx(1 / 3)
    assert s["in_table"] == pytest.approx(2 / 3) and s["in_text"] == pytest.approx(1 / 3)
    assert s["matched_min_conf"] == 0.40
    assert s["numbers_found"] == 1.0
    miss = EvidenceSignals()(item("Which colour is the logo?"))
    assert (miss["match_share"], miss["n_matched_lines"], miss["numbers_found"]) == (0, 0, -1)
    assert EvidenceSignals()(item("What are the sales?"))["in_figure"] == 1.0


def test_router_cost_charges_ocr_layout_and_signal_time() -> None:
    table = build_features([item("What is the revenue?")], [QuestionSignals(), EvidenceSignals()])
    assert (table.needs_ocr, table.needs_layout) == (True, True)
    cost = table.router_cost({"docvqa/1": LAYOUT}, {"docvqa/q": "docvqa/1"})
    assert cost["uses_ocr"] == {"docvqa/q": True}
    assert cost["page_s"] == {"docvqa/q": 0.09}
    assert cost["question_s"]["docvqa/q"] == pytest.approx(sum(table.seconds["docvqa/q"].values()))
    with pytest.raises(ValueError, match="layout"):
        build_features([RouterInput("q", "d", "x", OCR, None)], [PageSignals()])


def test_layout_file_round_trip_and_deterministic_gzip(tmp_path: Path) -> None:
    work = tmp_path / "layout.jsonl"
    row = {
        "doc_id": "docvqa/1",
        "model": "PP-DocLayout-M",
        "latency_s": 0.09,
        "hardware": "cpu",
        "page_size": [1000, 1000],
        "regions": [["table", 0.9, 0, 300, 1000, 600]],
    }
    work.write_text(json.dumps(row) + "\n", encoding="utf-8")
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    assert compress(work, a) == 1
    compress(work, b)
    assert a.read_bytes() == b.read_bytes()
    page = load_layout(a)["docvqa/1"]
    assert page.regions == (Region("table", 0.9, (0, 300, 1000, 600)),)
    with gzip.open(a, "rt") as f:
        assert json.loads(f.read()) == row
