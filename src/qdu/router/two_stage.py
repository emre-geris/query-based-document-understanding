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
from pathlib import Path
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

ROUTER_PATHS = ("text", "vision", "hybrid", "vision-low")
"""Paths the two-stage router can choose; text and vision are always among them.
``vision-low`` (the page at 1/4 of its pixels, Phase 7 A) needs a label table that
includes it (`qdu router labels --extra-run`)."""

DEFAULT_PATHS = ("text", "vision", "hybrid")
"""The three-way router's paths (the default); add ``vision-low`` explicitly."""

T4_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""Detail-head thresholds: these quantiles of its val outputs, plus 0 (always full
resolution) and 1.01 (always vision-low when the image is chosen)."""

TARGETS = {
    "image": ("vision", "text"),
    "hybrid": ("hybrid", "vision"),
    "detail": ("vision", "vision-low"),
}
"""A head's training target: 1 when the first path scores strictly higher than the second.
``image``: the page image is needed (vision beats text); ``hybrid``: adding the OCR text to the
image helps (hybrid beats vision); ``detail``: the full-resolution page beats the page at 1/4
of its pixels."""


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
    detail: NeedsImage | None = None
    """Detail head (target ``detail``, stage-1 signals, no OCR); when the router chooses the
    image, it sends the full-resolution page only if P(vision > vision-low) reaches t4,
    else the page at 1/4 of its pixels. None when vision-low is not a path."""
    grid: list[GridPoint] = field(default_factory=list)
    budget: float = 0.0
    met_budget: bool = False

    def probabilities(
        self, rows: Sequence[LabelRow], features: Features
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Stage-1, stage-2 and hybrid-head probabilities (the last empty without the head)."""
        p3 = self.hybrid.predict(rows, features) if self.hybrid is not None else {}
        return self.stage1.predict(rows, features), self.stage2.predict(rows, features), p3

    def choose(
        self,
        p1: float,
        p2: float | None = None,
        p3: float | None = None,
        params: tuple[float, ...] | None = None,
        p4: float | None = None,
    ) -> tuple[str, int]:
        """The path for one question and the stage that decided it, from its head
        probabilities. ``p2`` (and ``p3``) are needed only when stage 1 does not decide; the
        live router (`qdu.router.live`) runs OCR only then. The single place the rule lives,
        so evaluation and the demo choose identically."""
        t1, t2, *rest = params or self.params
        t3 = rest[0] if rest else OFF
        t4 = rest[1] if len(rest) > 1 else 0.0  # 0: always full resolution

        def image() -> str:
            if self.detail is None or p4 is None or p4 >= t4:
                return "vision"
            return "vision-low"

        if p1 >= t1:  # stage 1: the image is needed, decided without OCR
            return image(), 1
        if p2 is None:
            raise ValueError("stage 1 did not decide: the stage-2 probability is needed")
        if t3 < OFF and p3 is not None and p3 >= t3:  # OCR already paid: only model time
            return "hybrid", 2
        return (image() if p2 >= t2 else "text"), 2

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
        detail: Mapping[str, float] | None = None,
    ) -> Policy:
        """Choices and the router's own cost for ``rows``.

        ``params`` is ``(t1, t2)`` or ``(t1, t2, t3)``; a missing ``t3`` means never hybrid.
        ``seconds`` is `FeatureTable.seconds` (extractor -> time per question), ``layout_s``
        the layout time of each question's page. ``probabilities`` (from `probabilities`)
        skips the models when many settings are scored on the same rows. ``page_seconds``
        (`FeatureTable.page_seconds`) is page-level signal time per extractor, e.g. the CLIP
        page embedding of S6; a stage pays it for the extractors it uses."""
        p1, p2, p3 = probabilities or self.probabilities(rows, features)
        if detail is None and self.detail is not None:
            detail = self.detail.predict(rows, features)
        p4 = detail or {}
        choice, uses_ocr, page_s, question_s = {}, {}, {}, {}
        both = (*self.stage1_extractors, *self.stage2_extractors)
        for r in rows:
            q = r.question_id
            spent = seconds[q]
            on_page = (page_seconds or {}).get(q, {})
            question_s[q] = sum(spent[e] for e in self.stage1_extractors)
            choice[q], stage = self.choose(p1[q], p2[q], p3.get(q), params, p4.get(q))
            if stage == 1:  # decided without OCR
                uses_ocr[q] = False
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
    paths: Sequence[str] = DEFAULT_PATHS,
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
    with_low = "vision-low" in paths
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
    if with_low:  # OCR-free, like stage 1: deciding the resolution must not need OCR
        router.detail = NeedsImage(tuple(f"{e}_" for e in stage1), seed, target="detail").fit(
            train, features
        )
    cached = router.probabilities(val, features)
    p4 = router.detail.predict(val, features) if router.detail is not None else None
    if three_way and t3 is None:
        t3 = sorted({*(float(v) for v in np.quantile(list(cached[2].values()), T3_QUANTILES)), OFF})
    if with_low:
        t4 = sorted(
            {0.0, *(float(v) for v in np.quantile(list(p4.values()), T4_QUANTILES)), OFF}  # type: ignore[union-attr]
        )
        settings = [
            (a, b, c, d)
            for a in t1
            for b in t2
            for c in (t3 if three_way else (OFF,))  # type: ignore[union-attr]
            for d in t4
        ]
    elif three_way:
        settings = [(a, b, c) for a in t1 for b in t2 for c in t3]  # type: ignore[union-attr]
    else:
        settings = [(a, b) for a in t1 for b in t2]
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
                detail=p4,
            ),
            ps,
        )
        for ps in settings
    ]
    router.budget = vision_budget(val) if budget is None else budget
    best, router.met_budget = select_point(router.grid, router.budget)
    router.params = best.params  # type: ignore[assignment]
    return router


MODEL_FILE = "model.joblib"
"""A trained two-stage router inside its run folder (``runs/routers/<name>/``)."""


def save_two_stage(router: TwoStageRouter, out_dir: Path, meta: Mapping[str, Any]) -> Path:
    """Write the trained router (all heads, their TF-IDF vocabularies and the chosen thresholds)
    with its run metadata, so the demo can route new questions without retraining."""
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MODEL_FILE
    joblib.dump({"router": router, "meta": dict(meta)}, path)
    return path


def load_two_stage(path: Path) -> tuple[TwoStageRouter, dict[str, Any]]:
    """The router and metadata written by `save_two_stage`; ``path`` is the file or its folder.

    joblib files are pickles: load only routers this project trained (never a downloaded one)."""
    import joblib

    path = path / MODEL_FILE if path.is_dir() else path
    saved = joblib.load(path)
    if not isinstance(saved, dict) or not isinstance(saved.get("router"), TwoStageRouter):
        raise ValueError(f"{path} is not a two-stage router written by qdu router two-stage")
    return saved["router"], saved["meta"]
