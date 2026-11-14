import pytest
from PIL import Image
from test_router_live import SIZE, FakeLayout, FakeOcr, trained  # noqa: F401 (fixture)

from qdu.data import QARecord
from qdu.demo import DemoService, Example, SharedOcr, demo_fake_model, render
from qdu.demo.service import DemoAnswer, PathRun
from qdu.demo.session import DemoSession
from qdu.pipelines import PipelineOutput
from qdu.router import EXTRACTORS
from qdu.router.labels import LabelRow
from qdu.router.live import LiveDecision, LiveRouter

# --- render -------------------------------------------------------------------------------


def out(path: str, answer: str, latency: float = 1.0) -> PipelineOutput:
    return PipelineOutput(path, answer, answer, "m", 100, 3, latency)


def answer(chosen: str, scores: dict[str, float], stage: int = 2, question: str = "q?"):
    cost = {"text": 0.3, "vision": 1.4, "hybrid": 1.6}
    runs = {
        p: PathRun(
            p,
            out(p, f"ans {p}", cost[p]),
            cost[p],
            0.5 if p != "vision" else 0.0,
            100,
            3,
            score=scores[p],
        )
        for p in ("text", "vision", "hybrid")
    }
    decision = LiveDecision(
        chosen, stage, 0.2 if stage == 2 else 0.6, 0.3 if stage == 2 else None,
        0.001 if stage == 2 else None, (0.25, 0.4, 0.0067), None, 0.1, 0.5 if stage == 2 else 0.0,
        0.0, 0.01, 0.001,
    )  # fmt: skip
    return DemoAnswer(
        question, "p", "auto", chosen, runs[chosen], decision, runs, 1,
        0.5 if stage == 2 else 0.0, 0.1, 0.0, 0.011, ("gold",), "anls", "OCR TEXT",
    )  # fmt: skip


def test_story_when_the_router_picks_the_cheapest_correct_path():
    a = answer("text", {"text": 1.0, "vision": 1.0, "hybrid": 1.0})
    assert render.cheapest_correct(a) == "text"
    title, text = render.step_story(a, 1, 3)
    assert title == "1. soru → Metin yolu"
    assert "2. aşamada" in text and "doğru ve en ucuz yol" in text
    assert "En ucuz doğru yol seçildi" in render.panel_html(a)


def test_story_when_the_router_is_wrong():
    a = answer("text", {"text": 0.0, "vision": 1.0, "hybrid": 1.0})
    _, text = render.step_story(a, 2, 3)
    assert "Cevap yanlış; görüntü ve hibrit yolu doğru bilirdi" in text
    assert "Daha iyi bir yol vardı" in render.panel_html(a)
    b = answer("vision", {"text": 0.0, "vision": 0.0, "hybrid": 1.0}, stage=1)
    _, text = render.step_story(b, 3, 3)
    assert "metni okumadan, 1. aşamada" in text
    assert "yalnızca görüntü ve metin birlikte" in text


def test_panel_sections_and_scenarios():
    a = answer("hybrid", {"text": 0.0, "vision": 0.0, "hybrid": 1.0})
    closed, opened = render.panel_html(a), render.panel_html(a, "B", present=True)
    assert "<details>" in closed and "<details open>" in opened and "qdu-present" in opened
    assert "senaryo B" in opened and "soruya bölünür" in opened
    assert "Hibrit (seçilen)" in closed and "OCR TEXT" in closed
    assert "Doğru cevap: <b>gold</b>" in closed
    assert render.panel_html(None).count("Henüz") == 1


def test_html_escapes_text_from_users_documents_and_models():
    a = answer("text", {"text": 1.0, "vision": 1.0, "hybrid": 1.0}, question="<script>x</script>")
    html = render.chat_html([a], 0) + render.panel_html(a)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_number_formats_are_turkish():
    assert render.s(1.234) == "1,23 s" and render.tokens(4051) == "4.051"


# --- session ------------------------------------------------------------------------------

QUESTIONS = ("what is the reference code", "which bar in the chart is highest", "total amount")


def example(i: int, doc: str, question: str) -> Example:
    label = LabelRow(
        question_id=f"ex/{i}", doc_id=doc, dataset="docvqa", split="val", stratum="docvqa/x",
        question=question, questions_on_page=3, ocr_s=0.5,
        score={"text": 1.0, "vision": 1.0, "hybrid": 1.0},
        model_s={"text": 0.2, "vision": 1.4, "hybrid": 1.6},
        cost_c={"text": 0.7, "vision": 1.4, "hybrid": 2.1},
        cost_b={"text": 0.7, "vision": 1.4, "hybrid": 2.1},
        tokens={"text": 500.0, "vision": 3000.0, "hybrid": 3500.0},
        oracle_latency="text", oracle_tokens="text",
    )  # fmt: skip
    return Example(QARecord(f"ex/{i}", doc, question, (f"gold {i}",)), label)


@pytest.fixture
def session(trained):  # noqa: F811
    router, val, ocr, layout, _, kinds = trained
    doc = next(r.doc_id for r in val if kinds[r.doc_id] == "plain")
    page = [example(i, doc, q) for i, q in enumerate(QUESTIONS)]
    recorded = {
        e.question_id: {p: out(p, f"gold {i}" if p != "text" else "wrong") for p in render.LABEL}
        for i, e in enumerate(page)
    }
    shared = SharedOcr(FakeOcr(ocr))
    extractors = {k: EXTRACTORS[k]() for k in ("s1", "s2", "s3")}
    service = DemoService(LiveRouter(router, extractors, shared, FakeLayout(layout)),
                          demo_fake_model(), shared)  # fmt: skip
    image = Image.new("RGB", SIZE, "white")
    return DemoSession(service, [page], lambda e: image, recorded)


def test_example_page_answers_are_scored_with_recorded_outputs(session):
    assert session.example_choices() == [f"DocVQA · {session.showcase[0][0].doc_id}"]
    with pytest.raises(ValueError, match="sayfa"):
        session.ask("q")
    session.open_example(0)
    assert "Örnek sayfa (val)" in session.header_html()
    assert "GPU koşusunun kayıtları" in session.header_html()
    assert session.question_choices() == list(QUESTIONS)
    a = session.ask(QUESTIONS[0], "Otomatik", compare=True)
    assert a.run.recorded and a.gold == ("gold 0",) and a.score is not None
    assert a.compare["text"].score == 0.0 and a.compare["vision"].score == 1.0
    assert session.answer_choices() == ["1. cevap"]
    assert "Cevap 1" in session.chat_html() and "ANALİZ" in session.panel_html()
    with pytest.raises(ValueError, match="boş"):
        session.ask("   ")


def test_forced_mode_and_scenario(session):
    session.open_example(0)
    a = session.ask(QUESTIONS[0], "Görüntü", compare=False)
    assert a.path == "vision" and a.decision is None and a.compare == {}
    session.set_scenario("B (sayfa paylaşımlı)")
    assert session.scenario == "B" and "senaryo B" in session.panel_html()


def test_presentation_walks_through_the_page(session):
    session.start_presentation(0)
    assert session.present and len(session.answers) == 1
    assert "adım 1 / 3" in session.story_html() and "qdu-present" in session.chat_html()
    session.next_step()
    session.next_step()
    assert len(session.answers) == 3 and session.at_last_step
    assert "adım 3 / 3" in session.story_html() and "genel başarı" in session.story_html()
    session.next_step()  # past the end: nothing happens
    assert len(session.answers) == 3
    session.previous_step()
    assert session.selected == 1 and session.answer_choices() == ["1. cevap", "2. cevap"]
    session.stop_presentation()
    assert session.story_html() == "" and len(session.answer_choices()) == 3


def test_uploaded_page_has_no_gold(session):
    session.open_upload(Image.new("RGB", (40, 60), "white"), "benim.png")
    assert session.page.page_id.startswith("upload/") and not session.page.is_example
    assert "doğru cevap bilinmiyor" in session.header_html()
    assert session.question_choices() == []


def test_session_view_is_what_the_page_draws(session):
    from qdu.demo.web import session_view

    v = session_view(session, "durum")
    assert v["page"] is None and v["answers"] == [] and v["status"] == "durum"
    assert v["examples"][0]["dataset"] == "DocVQA" and v["examples"][0]["questions"] == 3
    session.start_presentation(0)
    v = session_view(session)
    a = v["answers"][0]
    assert v["present"] and v["story"]["n"] == 1 and v["story"]["of"] == 3
    assert v["page"]["example"] and v["page"]["recorded"] and len(v["questions"]) == 3
    assert {"question", "answer", "path", "label", "total", "steps", "timing", "compare"} <= set(a)
    assert [c["path"] for c in a["compare"]] == ["text", "vision", "hybrid"]
    assert a["steps"][-1]["title"].startswith("Seçim: ") and "p" not in a["steps"][-1]
    assert sum(t["value"] for t in a["timing"]) == pytest.approx(a["total"])
    session.set_scenario("B")
    assert session_view(session)["scenario"] == "B"


def test_web_api_drives_the_session(session):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from qdu.demo.web import create_app

    client = TestClient(create_app(session, "durum"))
    page = client.get("/")
    assert page.status_code == 200 and "Doküman Asistanı" in page.text
    assert client.get("/api/state").json()["page"] is None
    assert client.get("/api/page.png").status_code == 404
    v = client.post("/api/example", json={"index": 0}).json()
    assert v["page"]["name"].startswith("DocVQA")
    assert client.get("/api/page.png").headers["content-type"] == "image/png"
    v = client.post(
        "/api/ask", json={"question": QUESTIONS[0], "mode": "auto", "compare": True}
    ).json()
    assert len(v["answers"]) == 1 and v["answers"][0]["score"] is not None
    bad = client.post("/api/ask", json={"question": " ", "mode": "auto"})
    assert bad.status_code == 400 and "boş" in bad.json()["error"]
    v = client.post("/api/present", json={"index": 0}).json()
    assert v["present"] and v["story"]["n"] == 1
    v = client.post("/api/next").json()
    assert v["story"]["n"] == 2 and v["can_prev"]
    assert client.post("/api/stop").json()["present"] is False
    assert client.post("/api/reset").json()["page"] is None
