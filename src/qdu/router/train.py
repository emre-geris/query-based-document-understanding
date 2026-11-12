"""Per-path quality models and the λ choice: the learned router.

One regressor per path predicts that path's score (the dataset's primary metric, 0-1) from the
router signals and, optionally, the question's TF-IDF. The TF-IDF vocabulary, the imputation
values and the scaler are fitted on ``train`` only, inside the model, so nothing leaks from
validation or test. The router then picks ``argmax_p q̂_p - λ·ĉ_p`` (`qdu.router.policy`).

Decision cost ``ĉ_p`` is each path's mean cost on ``train``. When the router's own signals
already needed OCR, OCR is paid whatever path it picks, so only the model time differs between
paths (``decision_costs``). λ is chosen on ``val`` (``select_lambda``): the most accurate point
of the cost-quality curve whose scenario-C cost stays within a budget, by default the cost of
always-vision, the project's success baseline. Test is never used to choose anything.

Kinds: ``ridge`` (linear, standardised signals + TF-IDF) and ``lgbm`` (LightGBM gradient
boosting, missing values handled natively).
"""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from qdu.router.evaluate import Policy
from qdu.router.labels import PATHS, LabelRow
from qdu.router.policy import cost_quality_curve, route

KINDS = ("ridge", "lgbm")
DEFAULT_LAMBDAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)

Features = Mapping[str, Mapping[str, float]]
"""question id -> feature name -> value (`qdu.router.features.FeatureTable.features`)."""


def _lgbm(seed: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )


@dataclass
class QualityModel:
    """One fitted regressor per path, with the preprocessing fitted on the training rows."""

    kind: str
    feature_names: list[str]
    tfidf: bool
    seed: int
    paths: tuple[str, ...] = PATHS
    _models: dict[str, Any] = field(default_factory=dict, repr=False)
    _vectorizer: Any = field(default=None, repr=False)
    _fill: Any = field(default=None, repr=False)
    _scaler: Any = field(default=None, repr=False)

    @classmethod
    def fit(
        cls,
        rows: Sequence[LabelRow],
        features: Features,
        *,
        kind: str = "lgbm",
        tfidf: bool = True,
        seed: int = 0,
        paths: Sequence[str] = PATHS,
    ) -> "QualityModel":
        if kind not in KINDS:
            raise ValueError(f"unknown model kind {kind!r}; choose one of {KINDS}")
        if not rows:
            raise ValueError("no training rows")
        names = sorted({k for r in rows for k in features[r.question_id]})
        model = cls(kind, names, tfidf, seed, tuple(paths))
        x = model._design(rows, features, fit=True)
        for p in model.paths:
            y = np.array([r.score[p] for r in rows], dtype=float)
            if kind == "ridge":
                from sklearn.linear_model import Ridge

                reg = Ridge(alpha=3.0)
            else:
                reg = _lgbm(seed)
            model._models[p] = reg.fit(x, y)
        return model

    def _numeric(self, rows: Sequence[LabelRow], features: Features) -> np.ndarray:
        return np.array(
            [[features[r.question_id].get(n, np.nan) for n in self.feature_names] for r in rows],
            dtype=float,
        ).reshape(len(rows), len(self.feature_names))

    def _design(self, rows: Sequence[LabelRow], features: Features, *, fit: bool = False):
        x = self._numeric(rows, features)
        if self.kind == "ridge":
            from sklearn.preprocessing import StandardScaler

            if fit:
                med = np.nanmedian(x, axis=0) if len(x) else np.zeros(x.shape[1])
                self._fill = np.where(np.isnan(med), 0.0, med)
            x = np.where(np.isnan(x), self._fill, x)
            if fit:
                self._scaler = StandardScaler().fit(x)
            x = self._scaler.transform(x)
        blocks = [sparse.csr_matrix(x)]
        if self.tfidf:
            from sklearn.feature_extraction.text import TfidfVectorizer

            questions = [r.question for r in rows]
            if fit:
                self._vectorizer = TfidfVectorizer(
                    min_df=3, ngram_range=(1, 2), sublinear_tf=True
                ).fit(questions)
            blocks.append(self._vectorizer.transform(questions))
        return sparse.hstack(blocks, format="csr")

    def predict(self, rows: Sequence[LabelRow], features: Features) -> dict[str, dict[str, float]]:
        """Predicted score per question and path, clipped to [0, 1]."""
        if not rows:
            return {}
        x = self._design(rows, features)
        preds = {p: np.clip(m.predict(x), 0.0, 1.0) for p, m in self._models.items()}
        return {
            r.question_id: {p: float(preds[p][i]) for p in self.paths} for i, r in enumerate(rows)
        }


def decision_costs(
    rows: Sequence[LabelRow], *, ocr_paid: bool, scenario: str = "C"
) -> dict[str, float]:
    """Expected cost of each path at decision time: its mean on ``rows`` (use train).

    With ``ocr_paid`` the router's own signals already ran OCR, so only the model time differs
    between paths; otherwise each path's mean cost under ``scenario`` (which includes OCR for
    the text and hybrid paths in C and B)."""
    if not rows:
        raise ValueError("no rows to average")
    if ocr_paid and scenario in ("C", "B"):
        return {p: sum(r.model_s[p] for r in rows) / len(rows) for p in rows[0].score}
    return {p: sum(r.cost(scenario)[p] for r in rows) / len(rows) for p in rows[0].score}


@dataclass(frozen=True)
class CurvePoint:
    lam: float
    score: float
    cost: float
    shares: dict[str, float]


def select_lambda(curve: Sequence[CurvePoint], budget: float) -> tuple[CurvePoint, bool]:
    """The most accurate point whose cost is within ``budget`` (ties: cheaper, then smaller λ).

    Returns the point and whether it met the budget; if no point does, the most accurate one
    overall is returned with ``False`` so the caller can report that the budget was missed."""
    if not curve:
        raise ValueError("empty cost-quality curve")
    within = [c for c in curve if c.cost <= budget]
    pool = within or list(curve)
    best = max(pool, key=lambda c: (round(c.score, 12), -c.cost, -c.lam))
    return best, bool(within)


@dataclass
class TrainedRouter:
    model: QualityModel
    lam: float
    cost: dict[str, float]
    curve: list[CurvePoint]
    budget: float
    met_budget: bool
    uses_ocr: bool


def train_router(
    train: Sequence[LabelRow],
    val: Sequence[LabelRow],
    features: Features,
    router_cost: Mapping[str, Mapping[str, Any]],
    *,
    uses_ocr: bool,
    kind: str = "lgbm",
    tfidf: bool = True,
    seed: int = 0,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    budget: float | None = None,
    scenario: str = "C",
) -> TrainedRouter:
    """Fit the quality models on ``train`` and choose λ on ``val``.

    ``router_cost`` is `FeatureTable.router_cost(...)` (``uses_ocr``, ``page_s``,
    ``question_s`` per question), so the curve charges the router for its own signals.
    ``budget`` defaults to always-vision's mean cost on ``val`` under ``scenario``."""
    model = QualityModel.fit(train, features, kind=kind, tfidf=tfidf, seed=seed)
    cost = decision_costs(train, ocr_paid=uses_ocr, scenario=scenario)
    predicted = model.predict(val, features)
    val_ids = {r.question_id for r in val}
    rc = {k: {q: v for q, v in d.items() if q in val_ids} for k, d in router_cost.items()}
    if budget is None:
        vision = Policy("always-vision", {r.question_id: "vision" for r in val})
        budget = sum(vision.cost(r, scenario) for r in val) / len(val)
    curve = []
    for lam, score, spent in cost_quality_curve(val, predicted, cost, lambdas, scenario, **rc):
        choice = route(predicted, cost, lam).choice
        shares = {p: sum(choice[r.question_id] == p for r in val) / len(val) for p in PATHS}
        curve.append(CurvePoint(lam, score, spent, shares))
    best, met = select_lambda(curve, budget)
    return TrainedRouter(model, best.lam, cost, curve, budget, met, uses_ocr)


def write_choices(policy: Policy, rows: Sequence[LabelRow], path: Path) -> None:
    """The policy as a choices CSV for ``qdu router evaluate --choices``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id", "pipeline", "uses_ocr", "page_s", "question_s"])
        for r in rows:
            q = r.question_id
            w.writerow(
                [
                    q,
                    policy.choice[q],
                    int(policy.uses_ocr.get(q, False)),
                    f"{policy.page_s.get(q, 0.0):.6f}",
                    f"{policy.question_s.get(q, 0.0):.6f}",
                ]
            )


def write_curve(curve: Sequence[CurvePoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["lambda", "score", "cost", *(f"share_{p}" for p in PATHS)])
        for c in curve:
            w.writerow(
                [c.lam, f"{c.score:.6f}", f"{c.cost:.4f}", *(f"{c.shares[p]:.4f}" for p in PATHS)]
            )
