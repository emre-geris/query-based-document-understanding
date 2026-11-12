"""Learned text-then-vision cascade: answer from OCR text, escalate when a classifier doubts it.

Like the confidence cascade (`qdu.router.cascade`, after SAFE-Cascade, CIKM 2026 demo), every
question is first answered by the text path, and the image is consulted only when the text
answer looks unreliable; an escalated question pays for both calls. The difference is the
escalation rule: instead of a threshold on the answer's probability, a LightGBM classifier
fitted on ``train`` predicts whether vision would score higher than the text answer, from

- **answer features** (``ans_``), computed from the text answer the cascade already has: its
  probability, mean and least token probability (from the log-probability run), length, number
  of words, whether it is a number, whether it is empty or says the answer is missing, and
  whether it appears verbatim in the page's OCR text;
- optionally the router signals S1-S4 (the text path ran OCR anyway, so S2 and S4 cost no extra
  OCR, only their extraction time and, for S3 and S4, the page's layout).

The escalation threshold is chosen on ``val`` within the always-vision scenario-C budget, like
the other routers (`qdu.router.baselines.select_point`). Answers whose confidence is missing
(the log-probability run answered differently, see `read_confidence`) get NaN confidences,
which LightGBM handles, instead of being escalated by rule.
"""

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qdu.ocr.base import OcrResult
from qdu.router.baselines import GridPoint, grid_point, select_point, vision_budget
from qdu.router.cascade import CONFIDENCE, read_confidence
from qdu.router.evaluate import Policy
from qdu.router.labels import LabelRow
from qdu.router.two_stage import NeedsImage
from qdu.runner import read_predictions

THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))
"""Escalate when P(vision beats the text answer) reaches the threshold."""

_MISSING = re.compile(
    r"not (?:provided|mentioned|given|available|specified|visible|stated|shown)|unanswerable"
    r"|cannot be determined|can't be determined|no information|unknown|\bn/?a\b"
)
_NUMBER = re.compile(r"[-+$€£]?\s*\d[\d.,\s]*%?")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def text_answers(labelled_run: Path) -> dict[str, str]:
    """The text path's answer per question in the labelled run (the answer that was scored)."""
    answers = {}
    for r in read_predictions(labelled_run / "predictions.jsonl"):
        if r["error"] is None and r["pipeline"] == "text":
            answers[r["question_id"]] = r["prediction"] or ""
    return answers


def answer_features(
    rows: Sequence[LabelRow],
    answers: Mapping[str, str],
    confidence: Mapping[str, Mapping[str, float]],
    ocr: Mapping[str, OcrResult],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """``ans_`` features per question and the time each took to compute.

    ``confidence`` maps a kind of `qdu.router.cascade.CONFIDENCE` to question id -> value."""
    features, seconds = {}, {}
    nan = float("nan")
    for r in rows:
        start = time.perf_counter()
        answer = answers.get(r.question_id, "")
        a = _norm(answer)
        features[r.question_id] = {
            "ans_chars": float(len(answer)),
            "ans_words": float(len(answer.split())),
            "ans_number": float(bool(answer) and bool(_NUMBER.fullmatch(answer.strip()))),
            "ans_empty": float(not a),
            "ans_missing": float(bool(_MISSING.search(answer.lower()))),
            "ans_in_ocr": float(bool(a) and a in _norm(ocr[r.doc_id].text)),
            **{f"ans_conf_{k}": confidence.get(k, {}).get(r.question_id, nan) for k in CONFIDENCE},
        }
        seconds[r.question_id] = time.perf_counter() - start
    return features, seconds


def read_answer_inputs(
    labelled_run: Path, logprob_run: Path
) -> tuple[dict[str, str], dict[str, dict[str, float]], int]:
    """Text answers, confidences of every kind, and how many confidences were left out."""
    confidence, differ = {}, []
    for kind in CONFIDENCE:
        confidence[kind], differ = read_confidence(logprob_run, labelled_run, kind)
    return text_answers(labelled_run), confidence, len(differ)


@dataclass
class LearnedCascade:
    classifier: NeedsImage
    signals: tuple[str, ...]
    signals_need_layout: bool
    threshold: float = 1.0
    grid: list[GridPoint] = field(default_factory=list)
    budget: float = 0.0
    met_budget: bool = False

    def policy(
        self,
        rows: Sequence[LabelRow],
        features: Mapping[str, Mapping[str, float]],
        seconds: Mapping[str, float],
        layout_s: Mapping[str, float],
        threshold: float | None = None,
        name: str = "learned-cascade",
    ) -> Policy:
        """Text kept unless P(vision is better) reaches the threshold; escalations pay both.

        ``seconds`` is each question's feature time (signals + answer features)."""
        t = self.threshold if threshold is None else threshold
        p = self.classifier.predict(rows, features)
        choice, before = {}, {}
        for r in rows:
            if p[r.question_id] >= t:
                choice[r.question_id] = "vision"
                before[r.question_id] = ("text",)
            else:
                choice[r.question_id] = "text"
        ids = [r.question_id for r in rows]
        return Policy(
            name,
            choice,
            uses_ocr=dict.fromkeys(ids, True),  # the text path runs OCR for every question
            page_s={q: layout_s[q] if self.signals_need_layout else 0.0 for q in ids},
            question_s={q: seconds[q] for q in ids},
            called_before=before,
        )


def train_learned_cascade(
    train: Sequence[LabelRow],
    val: Sequence[LabelRow],
    features: Mapping[str, Mapping[str, float]],
    seconds: Mapping[str, float],
    layout_s: Mapping[str, float],
    *,
    signals: Sequence[str] = ("s1", "s2", "s3", "s4"),
    signals_need_layout: bool = True,
    thresholds: Sequence[float] = THRESHOLDS,
    seed: int = 0,
    budget: float | None = None,
) -> LearnedCascade:
    """Fit the escalation classifier on ``train`` (answer features + ``signals``) and choose
    the threshold on ``val``."""
    prefixes = ("ans_", *(f"{s}_" for s in signals))
    cascade = LearnedCascade(
        NeedsImage(prefixes, seed).fit(train, features), tuple(signals), signals_need_layout
    )
    cascade.grid = [
        grid_point(val, cascade.policy(val, features, seconds, layout_s, t), (t,))
        for t in thresholds
    ]
    cascade.budget = vision_budget(val) if budget is None else budget
    best, cascade.met_budget = select_point(cascade.grid, cascade.budget)
    cascade.threshold = best.params[0]
    return cascade
