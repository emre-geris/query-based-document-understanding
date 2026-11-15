import numpy as np
import pytest
from PIL import Image

from qdu.layout import PageLayout, Region
from qdu.ocr.base import OcrLine, OcrResult
from qdu.router import EXTRACTORS, build_features, router_inputs
from qdu.router.labels import LabelRow
from qdu.router.live import (
    LivePageEmbeddingSignals,
    LiveQuestionEmbeddingSignals,
    LiveRouter,
    live_extractors,
)
from qdu.router.two_stage import train_two_stage

QUESTIONS = {
    "chart": "which bar in the chart is highest",
    "plain": "what is the reference code",
    "dense": "what is the reference code",
}
SCORES = {
    "chart": {"text": 0.0, "vision": 1.0, "hybrid": 1.0},
    "plain": {"text": 1.0, "vision": 1.0, "hybrid": 1.0},
    "dense": {"text": 0.0, "vision": 0.0, "hybrid": 1.0},
}
KINDS = ("chart", "plain", "plain", "dense")
SIZE = (1000, 1400)


def page_ocr(kind: str) -> OcrResult:
    n = 60 if kind == "dense" else 6
    lines = tuple(
        OcrLine(f"line {i} total 12{i}", 0.95, (50, 20 * i, 600, 20 * i + 15)) for i in range(n)
    )
    return OcrResult(" ".join(ln.text for ln in lines), "fake", 0.5, lines=lines, page_size=SIZE)


def page_layout(doc_id: str, kind: str) -> PageLayout:
    label, box = ("chart", (50, 50, 950, 1300)) if kind == "chart" else ("text", (50, 50, 950, 300))
    return PageLayout(doc_id, "fake", 0.1, "test", SIZE, (Region(label, 0.9, box),))


def label_row(i: int, kind: str) -> LabelRow:
    return LabelRow(
        question_id=f"docvqa/{i}",
        doc_id=f"doc/{i}",
        dataset="docvqa",
        split="train" if i < 240 else "val",
        stratum="docvqa/x",
        question=QUESTIONS[kind],
        questions_on_page=1,
        ocr_s=0.5,
        score=dict(SCORES[kind]),
        model_s={"text": 0.2, "vision": 2.0, "hybrid": 2.2},
        cost_c={"text": 0.7, "vision": 2.0, "hybrid": 2.7},
        cost_b={"text": 0.7, "vision": 2.0, "hybrid": 2.7},
        tokens={"text": 500.0, "vision": 3000.0, "hybrid": 3500.0},
        oracle_latency="vision",
        oracle_tokens="vision",
    )


class FakeOcr:
    name = "fake"

    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def extract(self, image, key):
        self.calls.append(key)
        return self.pages[key]


class FakeLayout:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def detect(self, image, doc_id):
        self.calls.append(doc_id)
        return self.pages[doc_id]


@pytest.fixture(scope="module")
def trained():
    kinds = {f"doc/{i}": KINDS[i % len(KINDS)] for i in range(320)}
    rows = [label_row(i, kinds[f"doc/{i}"]) for i in range(320)]
    ocr = {d: page_ocr(k) for d, k in kinds.items()}
    layout = {d: page_layout(d, k) for d, k in kinds.items()}
    extractors = [EXTRACTORS[e]() for e in ("s1", "s2", "s3")]
    table = build_features(router_inputs(rows, ocr, layout), extractors)
    layout_s = {r.question_id: layout[r.doc_id].latency_s for r in rows}
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "val"]
    router = train_two_stage(
        train,
        val,
        table.features,
        table.seconds,
        layout_s,
        stage1=("s1", "s3"),
        stage2=("s1", "s2", "s3"),
        needs_layout={"s1": False, "s2": False, "s3": True},
        budget=10.0,
    )
    policy = router.policy(val, table.features, table.seconds, layout_s)
    return router, val, ocr, layout, policy, kinds


def live(router, ocr, layout):
    extractors = {k: EXTRACTORS[k]() for k in ("s1", "s2", "s3")}
    return LiveRouter(router, extractors, FakeOcr(ocr), FakeLayout(layout))


def test_live_router_chooses_like_the_evaluation(trained):
    router, val, ocr, layout, policy, _ = trained
    lr = live(router, ocr, layout)
    image = Image.new("RGB", SIZE, "white")
    paths = set()
    for r in val:
        d = lr.decide(r.question, image, r.doc_id, r.question_id)
        assert d.path == policy.choice[r.question_id]
        assert (d.stage == 1) == (not policy.uses_ocr[r.question_id])
        paths.add(d.path)
    assert paths == {"text", "vision", "hybrid"}  # all three paths occur


def test_ocr_runs_only_in_stage_two_and_timings_are_reported(trained):
    router, val, ocr, layout, _, kinds = trained
    lr = live(router, ocr, layout)
    image = Image.new("RGB", SIZE, "white")
    chart = next(r for r in val if kinds[r.doc_id] == "chart")
    d = lr.decide(chart.question, image, chart.doc_id)
    assert (d.path, d.stage, d.ocr, d.ocr_s) == ("vision", 1, None, 0.0)
    assert lr.ocr_engine.calls == [] and lr.layout_detector.calls == [chart.doc_id]
    assert d.layout_s == pytest.approx(0.1) and d.p_image_stage2 is None
    assert d.signal_s > 0 and d.decide_s > 0 and d.router_s >= d.signal_s

    dense = next(r for r in val if kinds[r.doc_id] == "dense")
    d = lr.decide(dense.question, image, dense.doc_id)
    assert (d.path, d.stage) == ("hybrid", 2)
    assert lr.ocr_engine.calls == [dense.doc_id]
    assert d.ocr is ocr[dense.doc_id] and d.ocr_s == pytest.approx(0.5)
    assert d.p_image_stage2 is not None and d.p_hybrid is not None
    assert len(d.thresholds) == 3 and d.reused == ()


def test_second_question_on_a_page_reuses_layout_and_ocr(trained):
    router, val, ocr, layout, _, kinds = trained
    lr = live(router, ocr, layout)
    image = Image.new("RGB", SIZE, "white")
    plain = next(r for r in val if kinds[r.doc_id] == "plain")
    first = lr.decide(plain.question, image, plain.doc_id)
    second = lr.decide(plain.question, image, plain.doc_id)
    assert first.reused == () and set(second.reused) == {"layout", "ocr"}
    assert lr.ocr_engine.calls == [plain.doc_id] and lr.layout_detector.calls == [plain.doc_id]
    assert second.ocr_s == first.ocr_s  # scenario C still charges the page's measured time
    lr.forget(plain.doc_id)
    lr.decide(plain.question, image, plain.doc_id)
    assert lr.ocr_engine.calls == [plain.doc_id, plain.doc_id]


def test_missing_extractor_or_detector_is_an_error(trained):
    router, _, ocr, layout, _, _ = trained
    with pytest.raises(ValueError, match="no extractor"):
        LiveRouter(router, {"s1": EXTRACTORS["s1"]()}, FakeOcr(ocr), FakeLayout(layout))
    extractors = {k: EXTRACTORS[k]() for k in ("s1", "s2", "s3")}
    with pytest.raises(ValueError, match="need page layout"):
        LiveRouter(router, extractors, FakeOcr(ocr), None)


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, items, normalize_embeddings=True, **kwargs):
        self.calls += 1
        out = []
        for item in items:
            seed = sum(map(ord, item)) if isinstance(item, str) else 7
            v = np.arange(1, 5, dtype=np.float32) * (seed % 5 + 1)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def test_live_embedding_signals_match_the_precomputed_names():
    from qdu.router.features import RouterInput

    image = Image.new("RGB", (10, 10), "white")
    item = RouterInput("q1", "p1", "what is it", None, None, lambda: image)
    s5 = LiveQuestionEmbeddingSignals(FakeEncoder())
    assert sorted(s5(item)) == ["e0", "e1", "e2", "e3"] and s5.name == "s5"
    enc = FakeEncoder()
    s6 = LivePageEmbeddingSignals(enc)
    f = s6(item)
    assert sorted(f) == ["e0", "e1", "e2", "e3", "qsim"] and s6.name == "s6"
    assert -1.0 <= f["qsim"] <= 1.0 + 1e-6
    calls = enc.calls
    s6(item)  # the page vector is computed once per page
    assert enc.calls == calls + 1
    q_s, page_s = s6.cost(item)
    assert q_s == 0.0 and page_s >= 0.0
    extractors = live_extractors(FakeEncoder(), FakeEncoder())
    assert sorted(extractors) == ["s1", "s2", "s3", "s4", "s5", "s6"]
