"""The demo in the browser: the design draft as a web page, served by a small FastAPI app.

``qdu demo serve`` starts it. The page (``static/index.html``) is the design draft made live:
it draws everything itself and asks this server for data. The server keeps one `DemoSession`
(a demo is used by one presenter at a time) and answers each button with the session's state
as JSON (`session_view`), which is tested without a browser or FastAPI.

Endpoints (all JSON except the page and the page image):
``GET /`` the page · ``GET /api/state`` · ``POST /api/reset`` · ``POST /api/example`` {index}
· ``POST /api/upload`` (multipart ``file``) · ``POST /api/ask`` {question, mode, compare}
· ``POST /api/select`` {index} · ``POST /api/scenario`` {scenario} · ``POST /api/present``
{index} · ``POST /api/next`` · ``POST /api/prev`` · ``POST /api/stop`` · ``GET /api/page.png``.
"""

import io
import threading
from pathlib import Path
from typing import Any

from qdu.demo import render
from qdu.demo.service import PATHS, DemoAnswer
from qdu.demo.session import DATASET_NAME, MODE_OF, DemoSession

STATIC = Path(__file__).parent / "static"
PAGE_PREVIEW_SIDE = 900
PAGE_FULL_SIDE = 4000
"""Longer side of the enlarged page; DocVQA / TAT-DQA pages (~1,700 x 2,400) stay full size."""
MODE_NAMES = {v: k for k, v in MODE_OF.items()}


def answer_view(a: DemoAnswer, scenario: str = "C") -> dict[str, Any]:
    """One answer as the page draws it: chips, summary card, decision steps, timing, compare."""
    d = a.decision
    share = a.questions_on_page if scenario == "B" else 1
    steps: list[dict[str, Any]] = []
    if d is not None:
        t1, t2, t3 = d.thresholds
        steps.append(
            {"title": "1. aşama · metin okumadan karar", "name": "p(görüntü gerekli)",
             "p": d.p_image_stage1, "t": t1, "passed": d.p_image_stage1 >= t1, "scale": 1.0,
             "text": f"Bakılan: {render.stage1_inputs(d.stage1_signals)}; "
                     "sayfanın metni okunmadı."}
        )  # fmt: skip
        if d.stage == 2:
            if d.p_hybrid is not None and t3 < 1.01:
                steps.append(
                    {"title": "2. aşama · hibrit kararı", "name": "p(hibrit > görüntü)",
                     "p": d.p_hybrid, "t": t3, "passed": d.p_hybrid >= t3,
                     "scale": max(0.02, 2 * t3, d.p_hybrid * 1.2),
                     "text": "Sayfanın metni okundu; hibritin görüntüden iyi olma olasılığı."}
                )  # fmt: skip
            if a.path != "hybrid" and d.p_image_stage2 is not None:
                steps.append(
                    {"title": "2. aşama · görüntü kararı", "name": "p(görüntü gerekli)",
                     "p": d.p_image_stage2, "t": t2, "passed": d.p_image_stage2 >= t2,
                     "scale": 1.0, "text": "Okunan metinle birlikte görüntü gerekiyor mu?"}
                )  # fmt: skip
        ending = (
            "Sayfanın metni hiç okunmadı; maliyeti ödenmedi."
            if d.stage == 1
            else {
                "text": "En ucuz yol: okunan metin + soru; görüntü modele gönderilmedi.",
                "vision": "Sayfa görüntüsü + soru.",
                "hybrid": "Görüntü + okunan metin birlikte; metin zaten okunduğu için ek "
                "maliyet sadece modelin süresi.",
            }[a.path]
        )
        steps.append({"title": f"Seçim: {render.LABEL[a.path]}", "text": ending})
    best = render.cheapest_correct(a)
    message = render.choice_message(a)
    vision = a.compare.get("vision")
    return {
        "question": a.question,
        "answer": a.run.answer,
        "path": a.path,
        "label": render.LABEL[a.path],
        "stage": d.stage if d else None,
        "forced": d is None,
        "recorded": a.run.recorded,
        "reused": list(d.reused) if d else [],
        "total": a.total_b_s if scenario == "B" else a.total_c_s,
        "total_c": a.total_c_s,
        "tokens_in": a.tokens[0],
        "tokens_out": a.tokens[1],
        "score": a.score,
        "metric": render.METRIC.get(a.metric or "", a.metric),
        "verdict": render.verdict_of(a.score) if a.score is not None else None,
        "gold": list(a.gold),
        "best": best,
        "quality": message[0] if message else None,
        "quality_text": message[1] if message else None,
        "best_label": render.LABEL[best] if best else None,
        "best_cost": a.compare[best].cost_c_s if best else None,
        "vs_vision": (vision.cost_c_s - a.total_c_s) if vision else None,
        "why": (
            "Yol elle seçildi; yönlendirici çalışmadı."
            if d is None
            else "Metin okunmadan, 1. aşamada karar verildi."
            if d.stage == 1
            else "Sayfanın metni okunduktan sonra, 2. aşamada karar verildi."
        ),
        "steps": steps,
        "timing": [
            {"key": "ocr", "label": "Metin okuma (OCR)", "value": a.ocr_s / share},
            {"key": "layout", "label": "Sayfa düzeni", "value": a.layout_s / share},
            {"key": "page", "label": "Sayfa sinyalleri", "value": a.page_signal_s / share},
            {"key": "router", "label": "Yönlendirme kararı", "value": a.router_s},
            {"key": "model", "label": "Modelin cevabı", "value": a.model_s},
        ],
        "questions_on_page": a.questions_on_page,
        "compare": [
            {
                "path": p,
                "label": render.LABEL[p],
                "answer": a.compare[p].answer,
                "score": a.compare[p].score,
                "time": a.compare[p].cost_c_s,
                "tokens": a.compare[p].tokens_in + a.compare[p].tokens_out,
                "chosen": p == a.path,
            }
            for p in PATHS
            if p in a.compare
        ],
        "ocr_text": (a.ocr_text or "")[:4000],
    }


def session_view(sess: DemoSession, status: str = "", error: str = "") -> dict[str, Any]:
    """Everything the page draws, as JSON-ready data."""
    page = sess.page
    story = sess.story()
    shown = sess.shown()
    return {
        "status": status,
        "error": error,
        "examples": [
            {
                "label": label,
                "dataset": DATASET_NAME.get(p[0].dataset, p[0].dataset),
                "doc_id": p[0].doc_id,
                "questions": len(p),
            }
            for label, p in zip(sess.example_choices(), sess.showcase, strict=True)
        ],
        "page": None
        if page is None
        else {
            "name": page.name,
            "info": page.info,
            "example": page.is_example,
            "recorded": page.is_example and bool(sess.recorded),
            "id": page.page_id,
        },
        "questions": sess.question_choices(),
        "answers": [answer_view(a, sess.scenario) for a in shown],
        "selected": sess.selected
        if sess.selected is not None and sess.selected < len(shown)
        else (len(shown) - 1 if shown else None),
        "scenario": sess.scenario,
        "present": sess.present,
        "story": None
        if story is None
        else {"title": story[0], "text": story[1], "n": story[2], "of": story[3]},
        "at_last": sess.at_last_step,
        "can_prev": sess.present and sess.step > 0,
    }


def create_app(session: DemoSession, status: str = ""):
    """The FastAPI app around one session (needs the ``demo`` extra)."""
    try:
        from fastapi import FastAPI, File, UploadFile
        from fastapi.responses import FileResponse, JSONResponse, Response
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError("the demo needs the 'demo' extra: uv sync --extra demo") from exc

    app = FastAPI(title="QDU Doküman Asistanı")
    lock = threading.Lock()  # one presenter, but the browser may send two clicks at once

    def reply(action=None, *args) -> JSONResponse:
        with lock:
            try:
                if action is not None:
                    action(*args)
            except Exception as exc:  # shown on the page; the session keeps working
                return JSONResponse(session_view(session, status, f"Hata: {exc}"), 400)
            return JSONResponse(session_view(session, status))

    def index_of(body: dict, key: str = "index") -> int:
        return int(body.get(key, 0))

    @app.get("/")
    def page():
        return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8")

    @app.get("/api/state")
    def state():
        return reply()

    @app.post("/api/reset")
    def reset():
        return reply(session.reset)

    @app.post("/api/example")
    def example(body: dict):
        def run():
            session.stop_presentation()
            session.open_example(index_of(body))

        return reply(run)

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):  # noqa: B008 (FastAPI's idiom)
        from PIL import Image

        data = await file.read()

        def run():
            session.stop_presentation()
            image = Image.open(io.BytesIO(data))
            session.open_upload(image, file.filename or "yüklenen sayfa")

        return reply(run)

    @app.post("/api/ask")
    def ask(body: dict):
        def run():
            session.stop_presentation()
            mode = MODE_NAMES.get(body.get("mode", "auto"), body.get("mode", "Otomatik"))
            session.ask(str(body.get("question", "")), mode, bool(body.get("compare", True)))

        return reply(run)

    @app.post("/api/select")
    def select(body: dict):
        return reply(session.select, index_of(body))

    @app.post("/api/scenario")
    def scenario(body: dict):
        return reply(session.set_scenario, str(body.get("scenario", "C")))

    @app.post("/api/present")
    def present(body: dict):
        return reply(session.start_presentation, index_of(body))

    @app.post("/api/next")
    def next_step():
        return reply(session.next_step)

    @app.post("/api/prev")
    def prev_step():
        return reply(session.previous_step)

    @app.post("/api/stop")
    def stop():
        return reply(session.stop_presentation)

    @app.get("/api/page.png")
    def page_image(full: int = 0):
        """The current page: a 900-pixel preview, or with ``full=1`` the page itself (capped at
        PAGE_FULL_SIDE pixels on the longer side) for the enlarged viewer."""
        with lock:
            if session.page is None:
                return Response(status_code=404)
            image = session.page.image.copy()
        side = PAGE_FULL_SIDE if full else PAGE_PREVIEW_SIDE
        image.thumbnail((side, side))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return Response(
            buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    return app


def serve(app, *, host: str = "127.0.0.1", port: int = 7860) -> None:
    import uvicorn

    print(f"QDU demo: http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
