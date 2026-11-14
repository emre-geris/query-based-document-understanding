import numpy as np
import pytest
from PIL import Image

from qdu.cli import main
from qdu.data import QARecord
from qdu.router import EXTRACTORS, build_features, router_inputs
from qdu.router.embed import (
    PageEmbeddingSignals,
    QuestionEmbeddingSignals,
    compress_pages,
    compress_questions,
    decode_vector,
    embed_pages,
    embed_questions,
    encode_vector,
    load_page_embeddings,
    load_question_embeddings,
)
from qdu.router.two_stage import train_two_stage

DIM = 8


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer: strings by their characters, images by
    their colour; counts calls so tests can check the warm-up."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, items, normalize_embeddings=True, **kwargs):
        self.calls += 1
        out = []
        for item in items:
            if isinstance(item, str):
                v = np.array([(sum(map(ord, item)) * (i + 1)) % 97 + 1 for i in range(DIM)])
            else:
                r, g, b = item.getpixel((0, 0))
                v = np.array([r + 1, g + 1, b + 1, *range(1, DIM - 2)])
            v = v.astype(np.float32)
            out.append(v / np.linalg.norm(v) if normalize_embeddings else v)
        return np.stack(out)


def record(i: int, doc: int) -> QARecord:
    return QARecord(f"docvqa/{i}", f"docvqa/doc{doc}", f"what is item {i}?", ("x",))


RECORDS = [record(0, 0), record(1, 0), record(2, 1)]


def image_of(r: QARecord) -> Image.Image:
    colour = (200, 10, 10) if r.doc_id.endswith("0") else (10, 10, 200)
    return Image.new("RGB", (4, 4), colour)


@pytest.fixture
def embedded(tmp_path):
    pages_work, questions_work = tmp_path / "pages.jsonl", tmp_path / "questions.jsonl"
    clip = FakeEncoder()
    assert embed_pages(RECORDS, image_of, clip, pages_work, hardware="test CPU") == 2
    pages = load_page_embeddings(pages_work)
    assert embed_questions(RECORDS, pages, FakeEncoder(), clip, questions_work) == 3
    pages_gz, questions_gz = tmp_path / "pages.jsonl.gz", tmp_path / "questions.jsonl.gz"
    assert compress_pages(pages_work, pages_gz) == 2
    assert compress_questions(questions_work, questions_gz) == 3
    return tmp_path, pages_gz, questions_gz


def test_vectors_round_trip_as_float16():
    v = np.array([0.1, -0.25, 0.5, 1.0], dtype=np.float32)
    assert np.allclose(decode_vector(encode_vector(v)), v, atol=1e-3)


def test_embed_pages_is_resumable_and_records_time(tmp_path):
    out = tmp_path / "pages.jsonl"
    clip = FakeEncoder()
    assert embed_pages(RECORDS[:2], image_of, clip, out, hardware="test CPU") == 1
    assert clip.calls == 2  # warm-up + one page
    assert embed_pages(RECORDS, image_of, clip, out, hardware="test CPU") == 1  # only doc1 left
    assert embed_pages(RECORDS, image_of, clip, out) == 0
    pages = load_page_embeddings(out)
    assert sorted(pages) == ["docvqa/doc0", "docvqa/doc1"]
    page = pages["docvqa/doc0"]
    assert page.hardware == "test CPU" and page.latency_s >= 0.0
    assert page.vector.shape == (DIM,)
    assert np.linalg.norm(page.vector) == pytest.approx(1.0, abs=1e-2)


def test_question_similarity_is_the_clip_cosine(embedded):
    _, pages_gz, questions_gz = embedded
    pages, questions = load_page_embeddings(pages_gz), load_question_embeddings(questions_gz)
    clip = FakeEncoder()
    for r in RECORDS:
        q = questions[r.question_id]
        text = clip.encode([r.question])[0]
        assert q.qsim == pytest.approx(float(text @ pages[r.doc_id].vector), abs=1e-2)
        assert q.vector.shape == (DIM,) and q.latency_s >= 0.0 and q.qsim_s >= 0.0


def test_embed_questions_needs_every_page(tmp_path):
    with pytest.raises(ValueError, match="no page embedding"):
        embed_questions(RECORDS, {}, FakeEncoder(), FakeEncoder(), tmp_path / "q.jsonl")


def test_compressed_files_are_byte_identical_on_rewrite(embedded):
    tmp_path, pages_gz, questions_gz = embedded
    again_p, again_q = tmp_path / "again_p.jsonl.gz", tmp_path / "again_q.jsonl.gz"
    compress_pages(tmp_path / "pages.jsonl", again_p)
    compress_questions(tmp_path / "questions.jsonl", again_q)
    assert pages_gz.read_bytes() == again_p.read_bytes()
    assert questions_gz.read_bytes() == again_q.read_bytes()


def test_extractors_read_the_files_and_charge_recorded_time(embedded):
    _, pages_gz, questions_gz = embedded
    pages, questions = load_page_embeddings(pages_gz), load_question_embeddings(questions_gz)
    s5 = QuestionEmbeddingSignals(questions_gz)
    s6 = PageEmbeddingSignals(pages_gz, questions_gz)
    assert not (s5.needs_ocr or s5.needs_layout or s6.needs_ocr or s6.needs_layout)
    items = router_inputs(_label_rows())
    table = build_features(items, [s5, s6])
    for r in RECORDS:
        f = table.features[r.question_id]
        assert len([k for k in f if k.startswith("s5_e")]) == DIM
        assert len([k for k in f if k.startswith("s6_e")]) == DIM
        assert f["s6_qsim"] == pytest.approx(questions[r.question_id].qsim)
        # recorded embedding time is part of the per-question time; the page's CLIP time is
        # page-level
        assert table.seconds[r.question_id]["s5"] >= questions[r.question_id].latency_s
        assert table.seconds[r.question_id]["s6"] >= questions[r.question_id].qsim_s
        assert table.page_seconds[r.question_id] == {"s6": pages[r.doc_id].latency_s}
    cost = table.router_cost()
    for r in RECORDS:
        assert cost["page_s"][r.question_id] == pytest.approx(pages[r.doc_id].latency_s)
        assert not cost["uses_ocr"][r.question_id]


def test_missing_embedding_is_an_error(embedded):
    _, _, questions_gz = embedded
    item = router_inputs([_label_rows()[0]])[0]
    item.question_id = "docvqa/unknown"
    with pytest.raises(KeyError, match="no question embedding"):
        QuestionEmbeddingSignals(questions_gz)(item)


def test_s5_s6_are_registered():
    assert EXTRACTORS["s5"] is QuestionEmbeddingSignals
    assert EXTRACTORS["s6"] is PageEmbeddingSignals


def _label_rows(split: str = "train"):
    from qdu.router.labels import LabelRow

    rows = []
    for r in RECORDS:
        rows.append(
            LabelRow(
                question_id=r.question_id,
                doc_id=r.doc_id,
                dataset="docvqa",
                split=split,
                stratum="docvqa/x",
                question=r.question,
                questions_on_page=1,
                ocr_s=0.5,
                score={"text": 1.0, "vision": 1.0, "hybrid": 1.0},
                model_s={"text": 0.2, "vision": 2.0, "hybrid": 2.2},
                cost_c={"text": 0.7, "vision": 2.0, "hybrid": 2.7},
                cost_b={"text": 0.7, "vision": 2.0, "hybrid": 2.7},
                tokens={"text": 500.0, "vision": 3000.0, "hybrid": 3500.0},
                oracle_latency="text",
                oracle_tokens="text",
            )
        )
    return rows


def test_two_stage_charges_page_embedding_time_where_it_is_used():
    """Stage 1 uses s6 (page time 0.07 s), stage 2 adds layout: a question decided in stage 1
    pays the page embedding only; one in stage 2 pays it once, plus layout and OCR."""
    from qdu.router.labels import LabelRow

    layout_time, page_time = 0.1, 0.07
    rows, features, seconds = [], {}, {}
    for i in range(320):
        kind = ("chart", "plain", "plain", "dense")[i % 4]
        score = {
            "text": 1.0 if kind == "plain" else 0.0,
            "vision": 0.0 if kind == "dense" else 1.0,
            "hybrid": 1.0,
        }
        q = f"docvqa/{i}"
        rows.append(
            LabelRow(
                question_id=q,
                doc_id=f"doc/{i}",
                dataset="docvqa",
                split="train" if i < 240 else "val",
                stratum="docvqa/x",
                question="what is the reference code",
                questions_on_page=1,
                ocr_s=0.5,
                score=score,
                model_s={"text": 0.2, "vision": 2.0, "hybrid": 2.2},
                cost_c={"text": 0.7, "vision": 2.0, "hybrid": 2.7},
                cost_b={"text": 0.7, "vision": 2.0, "hybrid": 2.7},
                tokens={"text": 500.0, "vision": 3000.0, "hybrid": 3500.0},
                oracle_latency="vision",
                oracle_tokens="vision",
            )
        )
        features[q] = {"s6_e0": float(kind == "chart"), "s2_dense": float(kind == "dense")}
        seconds[q] = {"s6": 0.002, "s2": 0.002, "s3": 0.003}
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "val"]
    layout_s = {r.question_id: layout_time for r in rows}
    page_seconds = {q: {"s6": page_time} for q in features}
    router = train_two_stage(
        train,
        val,
        features,
        seconds,
        layout_s,
        stage1=("s6",),
        stage2=("s2", "s3", "s6"),
        needs_layout={"s2": False, "s3": True, "s6": False},
        budget=10.0,
        page_seconds=page_seconds,
    )
    policy = router.policy(val, features, seconds, layout_s, page_seconds=page_seconds)
    stage1 = [r for r in val if not policy.uses_ocr[r.question_id]]
    assert stage1, "chart questions should be decided in stage 1"
    for r in val:
        q = r.question_id
        expected = page_time if not policy.uses_ocr[q] else page_time + layout_time
        assert policy.page_s[q] == pytest.approx(expected)
    # without page_seconds the same policy pays no embedding time
    bare = router.policy(val, features, seconds, layout_s)
    assert all(bare.page_s[r.question_id] == pytest.approx(0.0) for r in stage1)


def test_cli_learned_cascade_rejects_s6(capsys):
    assert main(["router", "learned-cascade", "--signals", "s1,s6"]) == 2
    assert "does not support s6" in capsys.readouterr().err
