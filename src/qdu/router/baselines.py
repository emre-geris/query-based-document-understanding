"""Literature baselines for the router, re-implemented on this project's data.

- **question-only** ("Text, Pixels, or Both?", arXiv:2609.22628): TF-IDF of the question and a
  logistic regression with balanced class weights predict whether the image is needed (vision
  scores higher than text). The router sends a question to vision when that probability reaches
  a threshold, otherwise to text. It never looks at the page and never chooses hybrid.
- **ecodoc** (EcoDoc, ACL 2025 Industry): two rules. (1) Query intent: the question is compared
  with a fixed list of text-answerable and image-requiring example queries; if it is closer to
  the image examples (by more than a margin) it goes to vision without OCR. (2) Text relevance:
  otherwise the page is OCR'd, and the question goes to text only if enough of its terms appear
  in the OCR text, else to vision. EcoDoc embeds queries with a sentence encoder and uses
  LLM-written examples; this version uses TF-IDF cosine similarity and the hand-written examples
  below (written before looking at validation questions), and its relevance score is the share
  of question terms found in the OCR. No hybrid.

Both are fitted on ``train`` (the classifier, the TF-IDF vocabulary) and their thresholds are
chosen on ``val`` exactly like the learned router's λ: the most accurate setting whose
scenario-C cost stays within always-vision's (`qdu.router.train`). Each declares its own cost:
the question-only router never runs OCR to decide; EcoDoc runs it only after rule 1 says text.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from qdu.ocr.base import OcrResult
from qdu.router.evaluate import Policy
from qdu.router.labels import PATHS, LabelRow
from qdu.router.signals import terms

NAMES = ("question-only", "ecodoc")

# EcoDoc rule 1: example queries answerable from text vs. needing the page image.
TEXT_EXAMPLES = (
    "What is the invoice number?",
    "What is the date mentioned in the letter?",
    "Who is the letter addressed to?",
    "What is the name of the company?",
    "What is the title of the document?",
    "What is the phone number given?",
    "Who signed the report?",
    "What is the total amount due?",
    "What is the address of the office?",
    "What is the subject of the memo?",
)
IMAGE_EXAMPLES = (
    "What colour is the largest bar in the chart?",
    "Which line in the graph is the highest?",
    "What is shown in the image at the top?",
    "How many icons are shown in the figure?",
    "What is the value in the second row of the table?",
    "Which category has the smallest share in the pie chart?",
    "What is written inside the logo?",
    "What percentage is shown for the year with the lowest value?",
    "Which region is highlighted on the map?",
    "What is the trend shown in the line graph?",
)

QUESTION_ONLY_THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.05, 1.0, 0.05))
ECODOC_MARGINS = (-0.05, 0.0, 0.05, 0.1, 0.2, 1.0)
ECODOC_RELEVANCE = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class GridPoint:
    """One setting of a baseline's thresholds and its validation score and cost."""

    params: tuple[float, ...]
    score: float
    cost: float
    shares: dict[str, float]


def select_point(points: Sequence[GridPoint], budget: float) -> tuple[GridPoint, bool]:
    """The most accurate point within ``budget`` (ties: cheaper); else the most accurate, False."""
    if not points:
        raise ValueError("empty grid")
    within = [p for p in points if p.cost <= budget]
    pool = within or list(points)
    return max(pool, key=lambda p: (round(p.score, 12), -p.cost)), bool(within)


def _vision_budget(rows: Sequence[LabelRow], scenario: str = "C") -> float:
    vision = Policy("always-vision", {r.question_id: "vision" for r in rows})
    return sum(vision.cost(r, scenario) for r in rows) / len(rows)


def _point(rows: Sequence[LabelRow], policy: Policy, params: tuple[float, ...]) -> GridPoint:
    n = len(rows)
    score = sum(r.score[policy.choice[r.question_id]] for r in rows) / n
    cost = sum(policy.cost(r, "C") for r in rows) / n
    shares = {p: sum(policy.choice[r.question_id] == p for r in rows) / n for p in PATHS}
    return GridPoint(params, score, cost, shares)


@dataclass
class Baseline:
    """A fitted baseline: its thresholds (chosen on val) and how it decides."""

    name: str
    params: tuple[float, ...]
    grid: list[GridPoint]
    budget: float
    met_budget: bool
    _decide: object  # callable(rows) -> Policy, with the chosen params

    def policy(self, rows: Sequence[LabelRow]) -> Policy:
        return self._decide(rows, self.params)  # type: ignore[operator]


class QuestionOnly:
    """TF-IDF + logistic regression on the question: P(vision scores higher than text)."""

    name = "question-only"

    def __init__(self, seed: int = 0) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)

    def fit(self, rows: Sequence[LabelRow]) -> "QuestionOnly":
        x = self.vectorizer.fit_transform([r.question for r in rows])
        y = [int(r.score["vision"] > r.score["text"]) for r in rows]
        if len(set(y)) < 2:
            raise ValueError("question-only baseline needs both labels in the training rows")
        self.model.fit(x, y)
        return self

    def probabilities(self, rows: Sequence[LabelRow]) -> tuple[dict[str, float], dict[str, float]]:
        """P(vision needed) per question and the decision time per question."""
        if not rows:
            return {}, {}
        start = time.perf_counter()
        p = self.model.predict_proba(self.vectorizer.transform([r.question for r in rows]))[:, 1]
        per_question = (time.perf_counter() - start) / len(rows)
        ids = [r.question_id for r in rows]
        return dict(zip(ids, map(float, p), strict=True)), dict.fromkeys(ids, per_question)

    def decide(self, rows: Sequence[LabelRow], params: tuple[float, ...]) -> Policy:
        (threshold,) = params
        prob, seconds = self.probabilities(rows)
        choice = {q: "vision" if p >= threshold else "text" for q, p in prob.items()}
        return Policy(self.name, choice, question_s=seconds)


def train_question_only(
    train: Sequence[LabelRow],
    val: Sequence[LabelRow],
    *,
    thresholds: Sequence[float] = QUESTION_ONLY_THRESHOLDS,
    seed: int = 0,
    budget: float | None = None,
) -> Baseline:
    router = QuestionOnly(seed).fit(train)
    grid = [_point(val, router.decide(val, (t,)), (t,)) for t in thresholds]
    budget = _vision_budget(val) if budget is None else budget
    best, met = select_point(grid, budget)
    return Baseline(router.name, best.params, grid, budget, met, router.decide)


class EcoDocRules:
    """EcoDoc-style rules: query intent by similarity to example queries, then OCR relevance."""

    name = "ecodoc"

    def __init__(self, ocr: Mapping[str, OcrResult]) -> None:
        self.ocr = ocr
        self.vectorizer = None
        self._text = self._image = None

    def fit(self, rows: Sequence[LabelRow]) -> "EcoDocRules":
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [r.question for r in rows] + list(TEXT_EXAMPLES) + list(IMAGE_EXAMPLES)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(corpus)
        self._text = self.vectorizer.transform(TEXT_EXAMPLES)
        self._image = self.vectorizer.transform(IMAGE_EXAMPLES)
        return self

    def intent(self, rows: Sequence[LabelRow]) -> tuple[dict[str, float], dict[str, float]]:
        """Image minus text similarity (mean cosine to each example list) and time per question."""
        if not rows:
            return {}, {}
        start = time.perf_counter()
        q = self.vectorizer.transform([r.question for r in rows])  # rows are L2-normalised
        gap = np.asarray((q @ self._image.T).mean(axis=1) - (q @ self._text.T).mean(axis=1))
        per_question = (time.perf_counter() - start) / len(rows)
        ids = [r.question_id for r in rows]
        return dict(zip(ids, gap.ravel().tolist(), strict=True)), dict.fromkeys(ids, per_question)

    def relevance(self, row: LabelRow) -> tuple[float, float]:
        """Share of the question's terms found in the page's OCR text, and the time it took."""
        start = time.perf_counter()
        q = terms(row.question)
        share = len(q & terms(self.ocr[row.doc_id].text)) / len(q) if q else 0.0
        return share, time.perf_counter() - start

    def decide(self, rows: Sequence[LabelRow], params: tuple[float, ...]) -> Policy:
        margin, min_relevance = params
        gap, seconds = self.intent(rows)
        choice, uses_ocr, question_s = {}, {}, {}
        for r in rows:
            q = r.question_id
            question_s[q] = seconds[q]
            if gap[q] > margin:  # rule 1: an image question, no OCR needed
                choice[q], uses_ocr[q] = "vision", False
                continue
            share, spent = self.relevance(r)  # rule 2 reads the OCR, so the page pays for it
            question_s[q] += spent
            uses_ocr[q] = True
            choice[q] = "text" if share >= min_relevance else "vision"
        return Policy(self.name, choice, uses_ocr=uses_ocr, question_s=question_s)


def train_ecodoc(
    train: Sequence[LabelRow],
    val: Sequence[LabelRow],
    ocr: Mapping[str, OcrResult],
    *,
    margins: Sequence[float] = ECODOC_MARGINS,
    relevance: Sequence[float] = ECODOC_RELEVANCE,
    budget: float | None = None,
) -> Baseline:
    missing = sorted({r.doc_id for r in [*train, *val]} - set(ocr))
    if missing:
        raise ValueError(f"no OCR for {len(missing)} pages, e.g. {missing[:3]}")
    router = EcoDocRules(ocr).fit(train)
    grid = [_point(val, router.decide(val, (m, t)), (m, t)) for m in margins for t in relevance]
    budget = _vision_budget(val) if budget is None else budget
    best, met = select_point(grid, budget)
    return Baseline(router.name, best.params, grid, budget, met, router.decide)
