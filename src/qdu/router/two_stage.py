"""Two-stage router: decide from OCR-free signals first, pay OCR only when unsure.

A router whose signals need OCR pays the page's OCR on every question, even when it then picks
vision (scenario C), so on validation the one-stage S1-S4 router costs more than it saves. This
router splits the decision:

1. **Stage 1** sees only OCR-free signals (by default S1 question cues and S3 page layout) and
   the question's TF-IDF. If the predicted probability that the image is needed (vision scores
   higher than text) reaches ``t1``, the question goes to vision and no OCR is run.
2. **Stage 2** runs only for the remaining questions: OCR is paid, every signal (S1-S4) is
   available, and the question goes to vision if its own probability reaches ``t2``, else to
   text.

Both stages are LightGBM classifiers fitted on ``train``; ``(t1, t2)`` is chosen on ``val`` like
the other routers' λ and thresholds: the most accurate setting whose scenario-C cost stays
within always-vision's (`qdu.router.baselines.select_point`). Hybrid is not a choice: on the
full run it ties with vision in score at a higher cost. Each question is charged its stage-1
signal time, the layout time if a used signal needs layout, and, if it reached stage 2, the
page's OCR and the stage-2 signal time.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from qdu.router.baselines import GridPoint, grid_point, select_point, vision_budget
from qdu.router.evaluate import Policy
from qdu.router.labels import LabelRow

Features = Mapping[str, Mapping[str, float]]


def lgbm_params(seed: int = 0, **overrides) -> dict:
    """LightGBM settings shared by the two-stage and variant classifiers (deterministic)."""
    return {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "verbose": -1,
        **overrides,
    }


T1 = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.01)
T2 = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01)
"""Threshold grids; 1.01 switches a stage's vision decision off."""


@dataclass
class NeedsImage:
    """LightGBM classifier for P(vision scores higher than text) from signals + question TF-IDF."""

    prefixes: tuple[str, ...]
    """Feature-name prefixes this stage may use, e.g. ``("s1_", "s3_")``."""
    seed: int = 0
    feature_names: list[str] = field(default_factory=list)
    _model: Any = field(default=None, repr=False)
    _vectorizer: Any = field(default=None, repr=False)

    def _design(self, rows: Sequence[LabelRow], features: Features, *, fit: bool = False):
        x = np.array(
            [[features[r.question_id].get(n, np.nan) for n in self.feature_names] for r in rows],
            dtype=float,
        ).reshape(len(rows), len(self.feature_names))
        questions = [r.question for r in rows]
        if fit:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True).fit(
                questions
            )
        return sparse.hstack(
            [sparse.csr_matrix(x), self._vectorizer.transform(questions)], format="csr"
        )

    def fit(self, rows: Sequence[LabelRow], features: Features) -> "NeedsImage":
        import lightgbm as lgb

        self.feature_names = sorted(
            {k for r in rows for k in features[r.question_id] if k.startswith(self.prefixes)}
        )
        y = [int(r.score["vision"] > r.score["text"]) for r in rows]
        if len(set(y)) < 2:
            raise ValueError("two-stage router needs both labels in the training rows")
        self._model = lgb.LGBMClassifier(**lgbm_params(self.seed)).fit(
            self._design(rows, features, fit=True), y
        )
        return self

    def predict(self, rows: Sequence[LabelRow], features: Features) -> dict[str, float]:
        if not rows:
            return {}
        p = self._model.predict_proba(self._design(rows, features))[:, 1]
        return {r.question_id: float(v) for r, v in zip(rows, p, strict=True)}


@dataclass
class TwoStageRouter:
    stage1: NeedsImage
    stage2: NeedsImage
    stage1_extractors: tuple[str, ...]
    stage2_extractors: tuple[str, ...]
    stage1_needs_layout: bool
    any_needs_layout: bool
    params: tuple[float, float] = (1.01, 1.01)
    grid: list[GridPoint] = field(default_factory=list)
    budget: float = 0.0
    met_budget: bool = False

    def policy(
        self,
        rows: Sequence[LabelRow],
        features: Features,
        seconds: Mapping[str, Mapping[str, float]],
        layout_s: Mapping[str, float],
        params: tuple[float, float] | None = None,
        name: str = "two-stage",
    ) -> Policy:
        """Choices and the router's own cost for ``rows``.

        ``seconds`` is `FeatureTable.seconds` (extractor -> time per question), ``layout_s``
        the layout time of each question's page."""
        t1, t2 = params or self.params
        p1 = self.stage1.predict(rows, features)
        p2 = self.stage2.predict(rows, features)
        choice, uses_ocr, page_s, question_s = {}, {}, {}, {}
        for r in rows:
            q = r.question_id
            spent = seconds[q]
            question_s[q] = sum(spent[e] for e in self.stage1_extractors)
            if p1[q] >= t1:  # stage 1: the image is needed, decided without OCR
                choice[q], uses_ocr[q] = "vision", False
                page_s[q] = layout_s[q] if self.stage1_needs_layout else 0.0
                continue
            uses_ocr[q] = True  # stage 2 reads the OCR, so the page pays for it
            page_s[q] = layout_s[q] if self.any_needs_layout else 0.0
            question_s[q] += sum(
                spent[e] for e in self.stage2_extractors if e not in self.stage1_extractors
            )
            choice[q] = "vision" if p2[q] >= t2 else "text"
        return Policy(name, choice, uses_ocr, page_s, question_s)


def train_two_stage(
    train: Sequence[LabelRow],
    val: Sequence[LabelRow],
    features: Features,
    seconds: Mapping[str, Mapping[str, float]],
    layout_s: Mapping[str, float],
    *,
    stage1: Sequence[str] = ("s1", "s3"),
    stage2: Sequence[str] = ("s1", "s2", "s3", "s4"),
    needs_layout: Mapping[str, bool],
    t1: Sequence[float] = T1,
    t2: Sequence[float] = T2,
    seed: int = 0,
    budget: float | None = None,
) -> TwoStageRouter:
    """Fit both stages on ``train`` and choose ``(t1, t2)`` on ``val``.

    ``needs_layout`` maps each extractor name to whether it reads the page layout."""
    router = TwoStageRouter(
        NeedsImage(tuple(f"{e}_" for e in stage1), seed).fit(train, features),
        NeedsImage(tuple(f"{e}_" for e in stage2), seed).fit(train, features),
        tuple(stage1),
        tuple(stage2),
        any(needs_layout[e] for e in stage1),
        any(needs_layout[e] for e in (*stage1, *stage2)),
    )
    router.grid = [
        grid_point(val, router.policy(val, features, seconds, layout_s, (a, b)), (a, b))
        for a in t1
        for b in t2
    ]
    router.budget = vision_budget(val) if budget is None else budget
    best, router.met_budget = select_point(router.grid, router.budget)
    router.params = best.params  # type: ignore[assignment]
    return router
