"""Two-stage router: decide from OCR-free signals first, pay OCR only when unsure.

A router whose signals need OCR pays the page's OCR on every question, even when it then picks
vision (scenario C), so on validation the one-stage S1-S4 router costs more than it saves. This
router splits the decision:

1. **Stage 1** sees only OCR-free signals (by default S1 question cues and S3 page layout) and
   the question's TF-IDF. If the predicted probability that the image is needed (vision scores
   higher than text) reaches ``t1``, the question goes to vision and no OCR is run.
2. **Stage 2** runs only for the remaining questions: OCR is paid and every signal (S1-S4) is
   available. In the three-way router (the default) a **hybrid head** decides first: if the
   predicted probability that hybrid scores higher than vision reaches ``t3``, the question goes
   to hybrid. Otherwise it goes to vision if its image-needed probability reaches ``t2``, else to
   text. Hybrid needs the OCR text, which stage 2 has already paid for, so its extra cost is
   only the model-time difference; stage 1 never chooses hybrid, because it has not run OCR.

The hybrid head decides before the text/vision head because most questions that only hybrid
answers are ones that text alone also fails (the image head, trained on vision > text, cannot
see them); on val this order is also slightly more accurate. With ``t3 = 1.01`` (or ``paths =
("text", "vision")``) the router is the earlier two-way one, so its three-way grid contains every
two-way setting and the old result stays reproducible.

All heads are LightGBM classifiers fitted on ``train``; ``(t1, t2[, t3])`` is chosen on ``val``
like the other routers' λ and thresholds: the most accurate setting whose scenario-C cost stays
within a budget, by default always-vision's (`qdu.router.baselines.select_point`). The hybrid
head is badly calibrated in absolute terms (P(hybrid > vision) is rare), so its thresholds are
quantiles of its val outputs (`T3_QUANTILES`), as in `qdu.router.variants`. Each question is
charged its stage-1 signal time, the layout time if a used signal needs layout, and, if it
reached stage 2, the page's OCR and the stage-2 signal time (the hybrid head reads the same
stage-2 signals, so it adds no signal time).
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

T3_QUANTILES = (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)
"""Hybrid-head thresholds: these quantiles of its val outputs, plus 1.01 (never hybrid)."""

OFF = 1.01

ROUTER_PATHS = ("text", "vision", "hybrid")
"""Paths the two-stage router can choose; text and vision are always among them."""

TARGETS = {
    "image": ("vision", "text"),
    "hybrid": ("hybrid", "vision"),
}
"""A head's training target: 1 when the first path scores strictly higher than the second.
``image``: the page image is needed (vision beats text); ``hybrid``: adding the OCR text to the
image helps (hybrid beats vision)."""


@dataclass
class NeedsImage:
    """LightGBM classifier from signals + question TF-IDF for P(vision scores higher than text)
    or, with ``target="hybrid"``, P(hybrid scores higher than vision)."""

    prefixes: tuple[str, ...]
    """Feature-name prefixes this stage may use, e.g. ``("s1_", "s3_")``."""
    seed: int = 0
    target: str = "image"
    """Key of `TARGETS`."""
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
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}; choose from {sorted(TARGETS)}")
        better, worse = TARGETS[self.target]
        y = [int(r.score[better] > r.score[worse]) for r in rows]
        if len(set(y)) < 2:
            raise ValueError(
                f"two-stage router needs both labels in the training rows ({better} > {worse}); "
                "without hybrid gains, train the two-way router (paths text, vision)"
            )
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
    params: tuple[float, ...] = (OFF, OFF)
    hybrid: NeedsImage | None = None
    """Hybrid head (target ``hybrid``, stage-2 signals); None for the two-way router."""
    grid: list[GridPoint] = field(default_factory=list)
    budget: float = 0.0
    met_budget: bool = False

    def probabilities(
        self, rows: Sequence[LabelRow], features: Features
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Stage-1, stage-2 and hybrid-head probabilities (the last empty without the head)."""
        p3 = self.hybrid.predict(rows, features) if self.hybrid is not None else {}
        return self.stage1.predict(rows, features), self.stage2.predict(rows, features), p3

    def policy(
        self,
        rows: Sequence[LabelRow],
        features: Features,
        seconds: Mapping[str, Mapping[str, float]],
        layout_s: Mapping[str, float],
        params: tuple[float, ...] | None = None,
        name: str = "two-stage",
        probabilities: tuple[Mapping[str, float], ...] | None = None,
        page_seconds: Mapping[str, Mapping[str, float]] | None = None,
    ) -> Policy:
        """Choices and the router's own cost for ``rows``.

        ``params`` is ``(t1, t2)`` or ``(t1, t2, t3)``; a missing ``t3`` means never hybrid.
        ``seconds`` is `FeatureTable.seconds` (extractor -> time per question), ``layout_s``
        the layout time of each question's page. ``probabilities`` (from `probabilities`)
        skips the models when many settings are scored on the same rows. ``page_seconds``
        (`FeatureTable.page_seconds`) is page-level signal time per extractor, e.g. the CLIP
        page embedding of S6; a stage pays it for the extractors it uses."""
        t1, t2, *rest = params or self.params
        t3 = rest[0] if rest else OFF
        p1, p2, p3 = probabilities or self.probabilities(rows, features)
        choice, uses_ocr, page_s, question_s = {}, {}, {}, {}
        both = (*self.stage1_extractors, *self.stage2_extractors)
        for r in rows:
            q = r.question_id
            spent = seconds[q]
            on_page = (page_seconds or {}).get(q, {})
            question_s[q] = sum(spent[e] for e in self.stage1_extractors)
            if p1[q] >= t1:  # stage 1: the image is needed, decided without OCR
                choice[q], uses_ocr[q] = "vision", False
                page_s[q] = (layout_s[q] if self.stage1_needs_layout else 0.0) + sum(
                    t for e, t in on_page.items() if e in self.stage1_extractors
                )
                continue
            uses_ocr[q] = True  # stage 2 reads the OCR, so the page pays for it
            page_s[q] = (layout_s[q] if self.any_needs_layout else 0.0) + sum(
                t for e, t in on_page.items() if e in both
            )
            question_s[q] += sum(
                spent[e] for e in self.stage2_extractors if e not in self.stage1_extractors
            )
            if t3 < OFF and q in p3 and p3[q] >= t3:  # OCR already paid: only model time
                choice[q] = "hybrid"
            elif p2[q] >= t2:
                choice[q] = "vision"
            else:
                choice[q] = "text"
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
    paths: Sequence[str] = ROUTER_PATHS,
    t1: Sequence[float] = T1,
    t2: Sequence[float] = T2,
    t3: Sequence[float] | None = None,
    seed: int = 0,
    budget: float | None = None,
    page_seconds: Mapping[str, Mapping[str, float]] | None = None,
) -> TwoStageRouter:
    """Fit the heads on ``train`` and choose ``(t1, t2)``, or ``(t1, t2, t3)`` when ``paths``
    includes hybrid, on ``val``.

    ``needs_layout`` maps each extractor name to whether it reads the page layout; ``t3`` defaults
    to `T3_QUANTILES` of the hybrid head's val outputs plus 1.01; ``budget`` (seconds, scenario
    C) to always-vision's mean cost on ``val``."""
    paths = tuple(dict.fromkeys(paths))
    if not {"text", "vision"} <= set(paths) or not set(paths) <= set(ROUTER_PATHS):
        raise ValueError(f"paths must include text and vision, from {ROUTER_PATHS}; got {paths}")
    prefixes2 = tuple(f"{e}_" for e in stage2)
    three_way = "hybrid" in paths
    router = TwoStageRouter(
        NeedsImage(tuple(f"{e}_" for e in stage1), seed).fit(train, features),
        NeedsImage(prefixes2, seed).fit(train, features),
        tuple(stage1),
        tuple(stage2),
        any(needs_layout[e] for e in stage1),
        any(needs_layout[e] for e in (*stage1, *stage2)),
        hybrid=NeedsImage(prefixes2, seed, target="hybrid").fit(train, features)
        if three_way
        else None,
    )
    cached = router.probabilities(val, features)
    if three_way and t3 is None:
        t3 = sorted({*(float(v) for v in np.quantile(list(cached[2].values()), T3_QUANTILES)), OFF})
    settings = (
        [(a, b, c) for a in t1 for b in t2 for c in t3]  # type: ignore[union-attr]
        if three_way
        else [(a, b) for a in t1 for b in t2]
    )
    router.grid = [
        grid_point(
            val,
            router.policy(
                val,
                features,
                seconds,
                layout_s,
                ps,
                probabilities=cached,
                page_seconds=page_seconds,
            ),
            ps,
        )
        for ps in settings
    ]
    router.budget = vision_budget(val) if budget is None else budget
    best, router.met_budget = select_point(router.grid, router.budget)
    router.params = best.params  # type: ignore[assignment]
    return router
