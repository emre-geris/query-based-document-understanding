import io
import json
import zipfile
from pathlib import Path

import pytest
from conftest import png_bytes, render_page

from qdu.data import ImageStore, iter_source_records
from qdu.data.tatdqa import build_tatdqa, format_answer


def question(uid: str, answer, blocks: list[str], answer_type: str = "span") -> dict:
    return {
        "uid": uid,
        "question": f"Question {uid}?",
        "answer": answer,
        "answer_type": answer_type,
        "scale": "thousand" if answer_type == "arithmetic" else "",
        "block_mapping": [{b: [0, 5]} for b in blocks],
    }


@pytest.fixture
def tatdqa_files(tmp_path: Path) -> tuple[Path, Path]:
    gold = [
        {
            "doc": {"uid": "a" * 32, "page": 1, "source": "one.pdf"},
            "questions": [question("q1", ["645,791"], ["b1"])],
        },
        {
            "doc": {"uid": "b" * 32, "page": 2, "source": "two.pdf"},
            "questions": [
                question("q2", 12.345, ["p2-block"], "arithmetic"),
                question("q3", ["x", "y"], ["p1-block", "p2-block"], "multi-span"),
            ],
        },
    ]
    layouts = {
        "a" * 32: {"pages": [{"bbox": [], "blocks": [{"uuid": "b1"}]}]},
        "b" * 32: {
            "pages": [
                {"bbox": [], "blocks": [{"uuid": "p1-block"}]},
                {"bbox": [], "blocks": [{"uuid": "p2-block"}]},
            ]
        },
    }
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    zip_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for uid, layout in layouts.items():
            z.writestr(f"test/{uid}.json", json.dumps(layout))
            pages = [render_page(f"{uid[0]} p{n}") for n in range(1, len(layout["pages"]) + 1)]
            for n, page in enumerate(pages, start=1):  # the release's 224 x 224 thumbnails
                z.writestr(f"test/{uid}_{n}.png", png_bytes(page.resize((224, 224))))
            pdf = io.BytesIO()  # 800 x 200 px pages at 72 dpi -> 800 x 200 pt
            pages[0].save(pdf, format="PDF", save_all=True, append_images=pages[1:])
            z.writestr(f"test/{uid}.pdf", pdf.getvalue())
    return gold_path, zip_path


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (["645,791"], ["645,791"]),
        (["a", "b"], ["a, b"]),
        (12.345, ["12.35"]),
        (3.0, ["3"]),
        (7, ["7"]),
    ],
)
def test_format_answer(answer, expected) -> None:
    assert format_answer(answer) == expected


def test_questions_map_to_their_evidence_page(tatdqa_files) -> None:
    dataset, stats = build_tatdqa(*tatdqa_files)
    assert stats == {"questions": 2, "skipped_multi_page": 1}
    rows = {row["uid"]: row for row in dataset}
    assert rows["q1"]["page"] == 1
    assert rows["q2"]["page"] == 2
    assert rows["q2"]["answers"] == ["12.35"]


def test_tatdqa_records_are_test_only(tatdqa_files) -> None:
    dataset, _ = build_tatdqa(*tatdqa_files)
    records = list(iter_source_records("tatdqa", dataset))
    assert {r.split for r in records} == {"test"}
    q2 = next(r for r in records if r.question_id == "tatdqa/q2")
    assert q2.doc_id == f"tatdqa/{'b' * 32}_p2"
    assert q2.question_types == ("arithmetic", "thousand")
    image = ImageStore({"tatdqa": dataset}).get(q2)
    # rendered from the PDF at 200 dpi, not the 224 x 224 thumbnail
    assert abs(image.size[0] - 800 * 200 / 72) <= 2 and abs(image.size[1] - 200 * 200 / 72) <= 2
    page1 = ImageStore({"tatdqa": dataset}).get(
        next(r for r in records if r.question_id == "tatdqa/q1")
    )
    assert page1.size == image.size


def test_built_dataset_is_cached_on_disk(tmp_path, monkeypatch) -> None:
    import huggingface_hub
    from datasets import Dataset

    import qdu.data.tatdqa as tat

    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda r, f, **k: str(snapshot / f))
    monkeypatch.setattr(tat, "built_cache_dir", lambda docs: tmp_path / f"built-{docs.parent.name}")
    calls = []

    def build(gold, docs):
        calls.append(docs)
        return Dataset.from_list([{"uid": "q1", "page": 1}]), {"questions": 1}

    monkeypatch.setattr(tat, "build_tatdqa", build)
    first = tat.load_tatdqa()
    second = tat.load_tatdqa()
    assert len(calls) == 1  # the second call reads the saved copy
    assert first[0] == second[0] == {"uid": "q1", "page": 1}
    assert (tmp_path / "built-abc123" / "dataset_info.json").exists()
    assert not (tmp_path / "built-abc123.partial").exists()


def test_cache_dir_names_snapshot_dpi_and_version() -> None:
    from qdu.data.tatdqa import CACHE_VERSION, RENDER_DPI, built_cache_dir

    name = built_cache_dir(Path("x/snapshots/abc123/tatdqa_docs_test.zip")).name
    assert name == f"tatdqa-test-abc123-dpi{RENDER_DPI}-v{CACHE_VERSION}"
