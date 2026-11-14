import pytest
from PIL import Image
from test_router_live import SIZE, FakeLayout, FakeOcr, trained  # noqa: F401 (fixture)

from qdu.cli import main
from qdu.demo import DemoService, SharedOcr, demo_fake_model, describe
from qdu.router import EXTRACTORS
from qdu.router.live import LiveRouter


def service(trained):  # noqa: F811
    router, _, ocr, layout, _, _ = trained
    shared = SharedOcr(FakeOcr(ocr))
    extractors = {k: EXTRACTORS[k]() for k in ("s1", "s2", "s3")}
    live = LiveRouter(router, extractors, shared, FakeLayout(layout))
    return DemoService(live, demo_fake_model(), shared)


def first(trained, kind):  # noqa: F811
    _, val, _, _, _, kinds = trained
    return next(r for r in val if kinds[r.doc_id] == kind)


IMAGE = Image.new("RGB", SIZE, "white")


def test_vision_in_stage_one_never_reads_the_page(trained):  # noqa: F811
    s = service(trained)
    r = first(trained, "chart")
    a = s.ask(r.question, IMAGE, r.doc_id)
    assert (a.path, a.decision.stage) == ("vision", 1)
    assert a.ocr_s == 0.0 and s.ocr.peek(r.doc_id) is None and s.ocr.engine.calls == []
    assert a.layout_s == pytest.approx(0.1) and a.router_s > 0
    assert a.run.answer == "[fake model] answer from the vision path"
    assert a.total_c_s == pytest.approx(a.layout_s + a.page_signal_s + a.router_s + a.model_s)


def test_hybrid_reads_the_page_once_for_router_and_path(trained):  # noqa: F811
    s = service(trained)
    r = first(trained, "dense")
    a = s.ask(r.question, IMAGE, r.doc_id)
    assert (a.path, a.decision.stage) == ("hybrid", 2)
    assert s.ocr.engine.calls == [r.doc_id]  # router and hybrid path share one OCR run
    assert a.ocr_s == pytest.approx(0.5) and a.run.ocr_s == pytest.approx(0.5)
    assert a.run.answer == "[fake model] answer from the hybrid path"
    assert a.tokens[0] > 0


def test_compare_runs_every_path_with_one_ocr(trained):  # noqa: F811
    s = service(trained)
    r = first(trained, "plain")
    a = s.ask(r.question, IMAGE, r.doc_id, compare=True)
    assert set(a.compare) == {"text", "vision", "hybrid"}
    assert a.compare[a.path] is a.run
    assert s.ocr.engine.calls == [r.doc_id]
    assert a.compare["vision"].ocr_s == 0.0 and a.compare["text"].ocr_s == pytest.approx(0.5)
    assert a.compare["text"].answer == "[fake model] answer from the text path"


def test_forced_path_skips_the_router(trained):  # noqa: F811
    s = service(trained)
    r = first(trained, "chart")
    a = s.ask(r.question, IMAGE, r.doc_id, mode="text")
    assert (a.path, a.decision) == ("text", None)
    assert a.router_s == 0.0 and a.layout_s == 0.0 and a.ocr_s == pytest.approx(0.5)
    with pytest.raises(ValueError, match="unknown mode"):
        s.ask(r.question, IMAGE, r.doc_id, mode="audio")


def test_scenario_b_shares_page_work_between_questions(trained):  # noqa: F811
    s = service(trained)
    r = first(trained, "plain")
    one = s.ask(r.question, IMAGE, r.doc_id)
    two = s.ask(r.question, IMAGE, r.doc_id)
    assert (one.questions_on_page, two.questions_on_page) == (1, 2)
    assert one.total_b_s == pytest.approx(one.total_c_s)
    assert two.total_b_s < two.total_c_s
    assert set(two.decision.reused) == {"layout", "ocr"}
    s.forget(r.doc_id)
    assert s.ask(r.question, IMAGE, r.doc_id).questions_on_page == 1


def test_router_and_paths_must_share_the_ocr(trained):  # noqa: F811
    router, _, ocr, layout, _, _ = trained
    extractors = {k: EXTRACTORS[k]() for k in ("s1", "s2", "s3")}
    live = LiveRouter(router, extractors, SharedOcr(FakeOcr(ocr)), FakeLayout(layout))
    with pytest.raises(ValueError, match="same SharedOcr"):
        DemoService(live, demo_fake_model(), SharedOcr(FakeOcr(ocr)))


def test_describe_reports_decision_answer_and_costs(trained):  # noqa: F811
    s = service(trained)
    r = first(trained, "dense")
    text = describe(s.ask(r.question, IMAGE, r.doc_id, compare=True))
    for part in ("Path: hybrid (router, decided in stage 2)", "hybrid head", "scenario C",
                 "scenario B", "Tokens (scenario A)", "always-vision", "<- shown"):  # fmt: skip
        assert part in text


def test_cli_demo_ask_reports_a_missing_router(tmp_path, capsys):
    image = tmp_path / "page.png"
    Image.new("RGB", (20, 20), "white").save(image)
    code = main(
        ["demo", "ask", "--image", str(image), "--question", "q", "--router", str(tmp_path / "x")]
    )
    assert code == 2 and "error" in capsys.readouterr().err
