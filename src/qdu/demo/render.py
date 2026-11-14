"""Wording and HTML for the demo, in Turkish (the demo is presented in class).

Pure functions of `DemoAnswer` values, so they are tested without a browser or Gradio. Every
piece of text that comes from a user, a document or a model is HTML-escaped. The look follows
the design draft (QDU Demo Arayüzü Taslağı): a chat column, and an analysis panel with a
summary card and collapsible sections (`<details>`, open in presentation mode).

The presentation narrative (`step_story`) is generated from the answer itself, never written
by hand: it says which path the router chose, whether that was right, and what the other
paths would have cost, so it cannot claim more than the run shows.
"""

from html import escape

from qdu.demo.service import PATHS, DemoAnswer

LABEL = {"text": "Metin", "vision": "Görüntü", "hybrid": "Hibrit"}
COLOR = {"text": "#0F766E", "vision": "#1D4ED8", "hybrid": "#9A4A07"}
TINT = {"text": "#E3F3F0", "vision": "#E8EEFC", "hybrid": "#FCEEDF"}
METRIC = {"anls": "ANLS", "relaxed_accuracy": "relaxed accuracy", "f1": "F1"}

CSS = """
.qdu { font-family: Inter, system-ui, sans-serif; color: #1A1A19; }
.qdu * { box-sizing: border-box; }
.qdu-present { font-size: 125%; }
.qdu-badge { display: inline-block; font-weight: 600; border-radius: 999px; padding: 3px 10px;
  font-size: 0.85em; }
.qdu-chip { display: inline-block; border-radius: 999px; padding: 3px 10px; font-size: 0.85em;
  background: #F0F0EB; color: #3A3A36; font-family: ui-monospace, monospace; }
.qdu-user { display: flex; justify-content: flex-end; margin: 14px 0 8px; }
.qdu-user p { margin: 0; max-width: 85%; background: #1A1A19; color: #FFF; padding: 10px 14px;
  border-radius: 16px 16px 4px 16px; line-height: 1.5; }
.qdu-bot { background: #FFF; border: 1px solid #DADAD3; border-radius: 4px 16px 16px 16px;
  padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; max-width: 92%; }
.qdu-bot.qdu-selected { border: 2px solid #1A1A19; }
.qdu-bot .qdu-answer { font-size: 1.05em; font-weight: 500; line-height: 1.5; }
.qdu-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.qdu-muted { color: #55554F; font-size: 0.85em; line-height: 1.45; }
.qdu-card { border: 1px solid #E3E3DC; border-radius: 14px; padding: 14px; display: flex;
  flex-direction: column; gap: 12px; background: #FFF; }
.qdu-grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.qdu-stat span { display: block; }
.qdu-stat .qdu-k { color: #55554F; font-size: 0.8em; margin-bottom: 4px; }
.qdu-stat .qdu-v { font-weight: 600; font-size: 1.1em; font-family: ui-monospace, monospace; }
.qdu-ok { background: #E3F3F0; color: #134E4A; border-radius: 10px; padding: 10px 12px; }
.qdu-warn { background: #FCEEDF; color: #6B3005; border-radius: 10px; padding: 10px 12px; }
.qdu details { border-top: 1px solid #E3E3DC; padding: 6px 0; }
.qdu summary { cursor: pointer; font-weight: 600; min-height: 36px; display: flex;
  align-items: center; justify-content: space-between; gap: 8px; }
.qdu summary .qdu-muted { font-weight: 400; }
.qdu table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
.qdu th, .qdu td { padding: 6px 8px; text-align: left; border-top: 1px solid #E3E3DC; }
.qdu td.qdu-num, .qdu th.qdu-num { text-align: right; font-family: ui-monospace, monospace; }
.qdu-bar { display: flex; height: 12px; border-radius: 4px; overflow: hidden; gap: 2px; }
.qdu-steps li { margin-bottom: 8px; line-height: 1.45; }
.qdu-story { background: #1A1A19; color: #FFF; border-radius: 16px; padding: 16px 18px; }
.qdu-story .qdu-n { font-size: 0.8em; letter-spacing: 0.04em; text-transform: uppercase;
  color: #D6D6CF; }
.qdu-story .qdu-t { font-size: 1.25em; font-weight: 600; margin: 6px 0; }
.qdu-story p { margin: 0; line-height: 1.5; color: #E9E9E4; }
.qdu-ocr { white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.8em;
  background: #F6F6F3; border-radius: 8px; padding: 10px; max-height: 260px; overflow: auto; }
.qdu-note { font-size: 0.8em; color: #8A3B06; background: #FDF0E3; border-radius: 999px;
  padding: 4px 10px; display: inline-block; }
"""


def s(x: float) -> str:
    """Seconds with two decimals, Turkish decimal comma."""
    return f"{x:.2f} s".replace(".", ",")


def num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def tokens(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def badge(path: str) -> str:
    return (
        f'<span class="qdu-badge" style="color:{COLOR[path]};background:{TINT[path]}">'
        f"{LABEL[path]}</span>"
    )


def verdict_of(score: float) -> str:
    return "Doğru" if score >= 1 else ("Kısmen doğru" if score > 0 else "Yanlış")


def score_chip(score: float | None, metric: str | None) -> str:
    if score is None:
        return ""
    color = "#0F5F58" if score >= 1 else ("#8A3B06" if score > 0 else "#9F1239")
    bg = "#E3F3F0" if score >= 1 else ("#FCEEDF" if score > 0 else "#FDE8EC")
    return (
        f'<span class="qdu-badge" style="color:{color};background:{bg}">'
        f"{verdict_of(score)} · {METRIC.get(metric or '', metric or '')} {num(score)}</span>"
    )


def total(a: DemoAnswer, scenario: str = "C") -> float:
    return a.total_b_s if scenario == "B" else a.total_c_s


def cheapest_correct(a: DemoAnswer) -> str | None:
    """Among the compared paths with the best score, the cheapest (scenario C); None without a
    comparison or gold answers."""
    if not a.compare or a.score is None:
        return None
    best = max(r.score or 0.0 for r in a.compare.values())
    if best <= 0:
        return None
    tied = [p for p, r in a.compare.items() if (r.score or 0.0) == best]
    return min(tied, key=lambda p: a.compare[p].cost_c_s)


def chat_html(answers: list[DemoAnswer], selected: int | None, present: bool = False) -> str:
    if not answers:
        return (
            '<div class="qdu"><p class="qdu-muted">Bir örnek sayfa seçin ya da kendi sayfanızı '
            "yükleyin, sonra soru sorun.</p></div>"
        )
    parts = [f'<div class="qdu{" qdu-present" if present else ""}">']
    for i, a in enumerate(answers):
        sel = " qdu-selected" if i == selected else ""
        recorded = (
            '<span class="qdu-muted">kayıtlı cevap (GPU koşusu)</span>' if a.run.recorded else ""
        )
        forced = "" if a.decision else '<span class="qdu-muted">yol elle seçildi</span>'
        parts.append(
            f'<div class="qdu-user"><p>{escape(a.question)}</p></div>'
            f'<div class="qdu-bot{sel}"><span class="qdu-muted">Cevap {i + 1}</span>'
            f'<span class="qdu-answer">{escape(a.run.answer) or "(boş cevap)"}</span>'
            f'<div class="qdu-row">{badge(a.path)}<span class="qdu-chip">{s(a.total_c_s)}</span>'
            f'<span class="qdu-chip">{tokens(sum(a.tokens))} token</span>'
            f"{score_chip(a.score, a.metric)}{recorded}{forced}</div></div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _decision_steps(a: DemoAnswer) -> str:
    d = a.decision
    if d is None:
        return f"<p>Yol elle seçildi ({LABEL[a.path]}); yönlendirici çalıştırılmadı.</p>"
    t1, t2, t3 = d.thresholds
    passed = "geçti" if d.p_image_stage1 >= t1 else "geçmedi"
    items = [
        f"<li><b>1. aşama, metin okumadan</b> (soru, sayfa düzeni, soru anlamı): görüntü gerekme "
        f"olasılığı {num(d.p_image_stage1, 3)}, eşik {num(t1)} → {passed}.</li>"
    ]
    if d.stage == 1:
        items.append(f"<li><b>Seçim: {LABEL[a.path]}</b>. Sayfanın metni hiç okunmadı.</li>")
    else:
        items.append("<li>Sayfanın metni okundu (OCR).</li>")
        if d.p_hybrid is not None and t3 < 1.01:
            h = "geçti" if d.p_hybrid >= t3 else "geçmedi"
            items.append(
                f"<li><b>2. aşama, hibrit kararı</b>: hibritin görüntüden iyi olma olasılığı "
                f"{num(d.p_hybrid, 4)}, eşik {num(t3, 4)} → {h}.</li>"
            )
        if a.path != "hybrid":
            v = "geçti" if (d.p_image_stage2 or 0) >= t2 else "geçmedi"
            items.append(
                f"<li><b>2. aşama, görüntü kararı</b>: görüntü gerekme olasılığı "
                f"{num(d.p_image_stage2 or 0, 3)}, eşik {num(t2)} → {v}.</li>"
            )
        items.append(f"<li><b>Seçim: {LABEL[a.path]}</b>.</li>")
    if d.reused:
        names = {"layout": "sayfa düzeni", "ocr": "metin okuma"}
        reused = ", ".join(names.get(x, x) for x in d.reused)
        items.append(
            f"<li class='qdu-muted'>Bu sayfadaki önceki sorudan kullanıldı: {reused}.</li>"
        )
    return f'<ol class="qdu-steps">{"".join(items)}</ol>'


def _timing(a: DemoAnswer, scenario: str) -> str:
    share = a.questions_on_page if scenario == "B" else 1
    parts = [
        ("Metin okuma (OCR)", a.ocr_s / share, "#0F766E"),
        ("Sayfa düzeni", a.layout_s / share, "#7BA9A2"),
        ("Sayfa sinyalleri", a.page_signal_s / share, "#B9B9B0"),
        ("Yönlendirme kararı", a.router_s, "#8A8A82"),
        ("Modelin cevabı", a.model_s, "#1D4ED8"),
    ]
    bar = "".join(
        f'<div style="flex:{v} 0 0;background:{c}" title="{n}"></div>' for n, v, c in parts if v > 0
    )
    rows = "".join(
        f'<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:3px;'
        f'background:{c};margin-right:6px"></span>{n}</td><td class="qdu-num">'
        f"{s(v) if v > 0 else '—'}</td></tr>"
        for n, v, c in parts
    )
    note = (
        f"B: sayfa bir kez okunur; okuma ve sayfa düzeni süresi bu sayfadaki {a.questions_on_page} "
        "soruya bölünür."
        if scenario == "B"
        else "C: her soru metin okuma ve sayfa düzeni süresini kendisi öder "
        "(projenin ana senaryosu)."
    )
    return (
        f'<div class="qdu-bar" role="img" aria-label="Süre dökümü">{bar}</div>'
        f"<table>{rows}<tr><th>Toplam</th><th class='qdu-num'>{s(total(a, scenario))}</th></tr>"
        f'</table><p class="qdu-muted">{note} Token sayısı senaryo A\'ya karşılık gelir.</p>'
    )


def _compare(a: DemoAnswer) -> str:
    if not a.compare:
        return (
            '<p class="qdu-muted">Karşılaştırma kapalı. "Üç yolu karşılaştır" seçiliyken her '
            "soru üç yolla da cevaplanır.</p>"
        )
    rows = []
    for p in PATHS:
        r = a.compare[p]
        mark = " (seçilen)" if p == a.path else ""
        sc = num(r.score) if r.score is not None else "—"
        rows.append(
            f'<tr style="background:{TINT[p] if p == a.path else "#FFF"}">'
            f'<th style="color:{COLOR[p]}">{LABEL[p]}{mark}</th><td>{escape(r.answer)}</td>'
            f'<td class="qdu-num">{sc}</td><td class="qdu-num">{s(r.cost_c_s)}</td>'
            f'<td class="qdu-num">{tokens(r.tokens_in + r.tokens_out)}</td></tr>'
        )
    head = (
        "<tr><th>Yol</th><th>Cevap</th><th class='qdu-num'>Puan</th>"
        "<th class='qdu-num'>Süre</th><th class='qdu-num'>Token</th></tr>"
    )
    return (
        f'<div style="overflow-x:auto"><table>{head}{"".join(rows)}</table></div>'
        '<p class="qdu-muted">Her yol tek başına kullanılsaydı (senaryo C). Karşılaştırma üç yolu '
        "da çalıştırır; gerçek kullanımda kapalıdır.</p>"
    )


def _ocr_text(a: DemoAnswer) -> str:
    if not a.ocr_text:
        return '<p class="qdu-muted">Bu soru için sayfanın metni okunmadı.</p>'
    return f'<div class="qdu-ocr">{escape(a.ocr_text[:4000])}</div>'


def panel_html(a: DemoAnswer | None, scenario: str = "C", present: bool = False) -> str:
    if a is None:
        return (
            '<div class="qdu"><p class="qdu-muted">Henüz analiz edilecek bir cevap yok.</p></div>'
        )
    opened = " open" if present else ""
    best = cheapest_correct(a)
    if best is None:
        box = ""
    elif best == a.path:
        box = (
            '<div class="qdu-ok"><b>En ucuz doğru yol seçildi.</b><br>'
            f"Bu soruda en ucuz doğru yol {LABEL[best].lower()} yoluydu.</div>"
        )
    else:
        box = (
            f'<div class="qdu-warn"><b>Daha iyi bir yol vardı.</b><br>Yönlendirici '
            f"{LABEL[a.path].lower()} yolunu seçti; en ucuz doğru yol {LABEL[best].lower()} "
            f"yoluydu ({s(a.compare[best].cost_c_s)}).</div>"
        )
    vision = a.compare.get("vision")
    vs = ""
    if vision is not None:
        diff = vision.cost_c_s - total(a, "C")
        faster = "daha hızlı" if diff >= 0 else "daha yavaş"
        vs = f" · hep görüntüye göre {s(abs(diff))} {faster}"
    gold = f"Doğru cevap: <b>{escape(' | '.join(a.gold))}</b> · " if a.gold else ""
    score = (
        f'<span class="qdu-v">{verdict_of(a.score)} · {num(a.score)}</span>'
        if a.score is not None
        else '<span class="qdu-v">—</span>'
    )
    metric = METRIC.get(a.metric or "", "puan")
    stage = f"{a.decision.stage}. aşamada karar" if a.decision else "elle seçildi"
    return (
        f'<div class="qdu{" qdu-present" if present else ""}">'
        f'<p class="qdu-muted" style="margin:0 0 6px">ANALİZ</p>'
        f'<p style="margin:0 0 12px;font-weight:600;line-height:1.4">{escape(a.question)}</p>'
        f'<div class="qdu-card"><div class="qdu-grid3">'
        f'<div class="qdu-stat"><span class="qdu-k">Seçilen yol</span>{badge(a.path)}</div>'
        f'<div class="qdu-stat"><span class="qdu-k">Toplam süre ({scenario})</span>'
        f'<span class="qdu-v">{s(total(a, scenario))}</span></div>'
        f'<div class="qdu-stat"><span class="qdu-k">Puan ({metric})</span>{score}</div></div>'
        f"{box}"
        f'<p class="qdu-muted" style="margin:0">{gold}Token: {tokens(a.tokens[0])} giden / '
        f"{tokens(a.tokens[1])} gelen{vs}</p></div>"
        f"<details{opened}><summary>Karar adımları <span class='qdu-muted'>{stage}</span>"
        f"</summary>{_decision_steps(a)}</details>"
        f"<details{opened}><summary>Süre dökümü <span class='qdu-muted'>senaryo {scenario}"
        f"</span></summary>{_timing(a, scenario)}</details>"
        f"<details{opened}><summary>Üç yolun karşılaştırması <span class='qdu-muted'>"
        f"{'açık' if a.compare else 'kapalı'}</span></summary>{_compare(a)}</details>"
        f"<details{opened}><summary>Sayfadan okunan metin (OCR)</summary>{_ocr_text(a)}</details>"
        "</div>"
    )


def step_story(a: DemoAnswer, n: int, of: int) -> tuple[str, str]:
    """Title and text for a presentation step, generated from the answer itself."""
    where = (
        "metni okumadan, 1. aşamada"
        if a.decision and a.decision.stage == 1
        else "sayfanın metnini okuduktan sonra, 2. aşamada"
    )
    title = f"{n}. soru → {LABEL[a.path]} yolu"
    text = f"Yönlendirici {where} {LABEL[a.path].lower()} yolunu seçti ({s(a.total_c_s)})."
    best = cheapest_correct(a)
    if a.score is not None and a.compare:
        right = [p for p in PATHS if (a.compare[p].score or 0) >= 1]
        if best == a.path:
            others = [p for p in PATHS if p != a.path]
            costs = ", ".join(f"{LABEL[p].lower()} {s(a.compare[p].cost_c_s)}" for p in others)
            text += f" Bu doğru ve en ucuz yoldu; diğerleri: {costs}."
        elif (a.score or 0) >= 1:
            text += f" Cevap doğru, ama {LABEL[best].lower()} yolu da doğruydu ve daha ucuzdu."
        elif right:
            names = " ve ".join(LABEL[p].lower() for p in right)
            text += f" Cevap yanlış; {names} yolu doğru bilirdi."
        else:
            text += " Bu soruyu hiçbir yol doğru bilemedi."
        if right == ["hybrid"]:
            text += " Doğru cevabı yalnızca görüntü ve metin birlikte verdi."
    return title, text


def story_html(title: str, text: str, n: int, of: int, present: bool = True) -> str:
    return (
        f'<div class="qdu{" qdu-present" if present else ""}"><div class="qdu-story">'
        f'<div class="qdu-n">Sunum · adım {n} / {of}</div><div class="qdu-t">{escape(title)}</div>'
        f"<p>{escape(text)}</p></div></div>"
    )


def page_header_html(name: str, info: str, example: bool, recorded: bool) -> str:
    notes = []
    if example:
        notes.append('<span class="qdu-note" style="color:#134E4A;background:#E3F3F0">'
                     "Örnek sayfa (val) · doğru cevaplar biliniyor</span>")  # fmt: skip
    if recorded:
        notes.append('<span class="qdu-note">Cevaplar: GPU koşusunun kayıtları</span>')
    return (
        f'<div class="qdu"><p style="margin:0;font-weight:600">{escape(name)}</p>'
        f'<p class="qdu-muted" style="margin:2px 0 6px">{escape(info)}</p>'
        f'<div class="qdu-row">{"".join(notes)}</div></div>'
    )
