import json

import pytest

from qdu.data.splits import assign_split, router_split
from qdu.data.tatdqa import read_reports, report_of_documents, write_reports

REPORTS = {"aaa": "acme_2019.pdf", "bbb": "acme_2019.pdf", "ccc": "zeta_2019.pdf"}


def test_v1_keeps_the_manifest_split():
    assert router_split("tatdqa", "tatdqa/aaa_p1", "test", "v1") == "test"
    assert router_split("docvqa", "docvqa/1", "val", "v1") == "val"


def test_v2_changes_only_tatdqa():
    assert router_split("docvqa", "docvqa/1", "val", "v2", REPORTS) == "val"
    assert router_split("chartqa", "chartqa/x", "train", "v2") == "train"


def test_v2_groups_tatdqa_documents_by_report():
    a = router_split("tatdqa", "tatdqa/aaa_p1", "test", "v2", REPORTS)
    b = router_split("tatdqa", "tatdqa/bbb_p3", "test", "v2", REPORTS)
    assert a == b == assign_split("tatdqa", "tatdqa/acme_2019.pdf")
    assert router_split("tatdqa", "tatdqa/ccc_p1", "test", "v2", REPORTS) == assign_split(
        "tatdqa", "tatdqa/zeta_2019.pdf"
    )


def test_v2_needs_the_report_of_every_tatdqa_document():
    with pytest.raises(ValueError, match="no report"):
        router_split("tatdqa", "tatdqa/zzz_p1", "test", "v2", REPORTS)
    with pytest.raises(ValueError, match="no report"):
        router_split("tatdqa", "tatdqa/aaa_p1", "test", "v2")
    with pytest.raises(ValueError, match="page id"):
        router_split("tatdqa", "tatdqa/aaa", "test", "v2", REPORTS)
    with pytest.raises(ValueError, match="scheme"):
        router_split("docvqa", "docvqa/1", "val", "v3")


def test_reports_round_trip(tmp_path):
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            [
                {"doc": {"uid": u, "page": 1, "source": s}, "questions": []}
                for u, s in REPORTS.items()
            ]
        ),
        encoding="utf-8",
    )
    reports = report_of_documents(gold)
    assert reports == REPORTS
    out = tmp_path / "reports.csv"
    write_reports(reports, out)
    assert read_reports(out) == REPORTS
    assert out.read_text(encoding="utf-8").splitlines()[0] == "doc_uid,report"


def test_committed_mapping_covers_every_tatdqa_page():
    import csv
    from pathlib import Path

    reports = read_reports(Path("benchmarks/tatdqa_reports.csv"))
    with Path("benchmarks/full.csv").open(encoding="utf-8") as f:
        pages = {r["doc_id"] for r in csv.DictReader(f) if r["dataset"] == "tatdqa"}
    assert len(reports) == 277 and len(set(reports.values())) == 126
    splits = {router_split("tatdqa", p, "test", "v2", reports) for p in pages}
    assert splits == {"train", "val", "test"}
