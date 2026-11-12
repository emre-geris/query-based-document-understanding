"""Training variants of the two-stage router: does better training close the gap?

The two-stage router (`qdu.router.two_stage`) stays the design; only how its two LightGBM
models are trained changes. Every variant is fitted on ``train`` and judged on ``val`` the same
way:

- **default**: classifier for "vision scores higher than text", fixed settings (`lgbm_params`)
- **weighted**: the same classifier, each training question weighted by how much the choice
  matters there, ``|score_vision - score_text|`` (questions where both paths tie barely count)
- **regress**: a regressor for the score difference ``score_vision - score_text`` itself, so the
  model predicts how much is lost by answering from text
- **tuned**: the default classifier with its settings chosen by grouped cross-validation on
  ``train`` (folds by document, ROC AUC), over ``TUNE_GRID``

Because a regressor's outputs are not probabilities, thresholds are taken as quantiles of each
stage's validation outputs for every variant alike (stage 1 may also be switched off), and the
most accurate setting within the always-vision scenario-C budget is kept
(`qdu.router.baselines.select_point`). Reported per variant: the validation AUC of stage 2 for
"text is as good as vision", and the router's score and cost.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qdu.router.baselines import grid_point, select_point, vision_budget
from qdu.router.labels import LabelRow
from qdu.router.two_stage import NeedsImage, TwoStageRouter, lgbm_params

Features = Mapping[str, Mapping[str, float]]

VARIANTS = ("default", "weighted", "regress", "tuned")
TUNE_GRID = tuple(
    {"num_leaves": nl, "min_child_samples": mc, "learning_rate": lr, "n_estimators": ne}
    for nl in (7, 31)
    for mc in (20, 100)
    for lr, ne in ((0.03, 300), (0.05, 600))
)
STAGE1_QUANTILES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
STAGE2_QUANTILES = tuple(round(0.05 * i, 2) for i in range(1, 20))


@dataclass
class StageModel(NeedsImage):
    """A two-stage stage model trained as one of ``VARIANTS`` (``tuned`` uses ``params``)."""

    objective: str = "classify"
    """``classify``, ``weighted`` or ``regress``."""
    params: dict[str, Any] = field(default_factory=dict)

    def fit(self, rows: Sequence[LabelRow], features: Features) -> "StageModel":
        import lightgbm as lgb

        self.feature_names = sorted(
            {k for r in rows for k in features[r.question_id] if k.startswith(self.prefixes)}
        )
        x = self._design(rows, features, fit=True)
        diff = np.array([r.score["vision"] - r.score["text"] for r in rows], dtype=float)
        settings = lgbm_params(self.seed, **self.params)
        if self.objective == "regress":
            self._model = lgb.LGBMRegressor(**settings).fit(x, diff)
            return self
        y = (diff > 0).astype(int)
        if len(set(y)) < 2:
            raise ValueError("training rows need both labels")
        weight = np.abs(diff) + 1e-3 if self.objective == "weighted" else None
        if self.objective not in ("classify", "weighted"):
            raise ValueError(f"unknown objective {self.objective!r}")
        self._model = lgb.LGBMClassifier(**settings).fit(x, y, sample_weight=weight)
        return self

    def predict(self, rows: Sequence[LabelRow], features: Features) -> dict[str, float]:
        if not rows:
            return {}
        x = self._design(rows, features)
        if self.objective == "regress":
            values = self._model.predict(x)
        else:
            values = self._model.predict_proba(x)[:, 1]
        return {r.question_id: float(v) for r, v in zip(rows, values, strict=True)}


def tune(
    rows: Sequence[LabelRow],
    features: Features,
    prefixes: tuple[str, ...],
    *,
    grid: Sequence[Mapping[str, Any]] = TUNE_GRID,
    folds: int = 3,
    seed: int = 0,
) -> tuple[dict[str, Any], float]:
    """Settings from ``grid`` with the best mean ROC AUC over ``folds`` document-grouped folds
    of ``rows`` (ties: the earlier setting), and that AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    rows = list(rows)
    y = np.array([r.score["vision"] > r.score["text"] for r in rows], dtype=int)
    splits = list(GroupKFold(folds).split(rows, y, [r.doc_id for r in rows]))
    best: tuple[float, dict[str, Any]] | None = None
    for params in grid:
        aucs = []
        for fit_idx, test_idx in splits:
            fit_rows = [rows[i] for i in fit_idx]
            test_rows = [rows[i] for i in test_idx]
            model = StageModel(prefixes, seed, params=dict(params)).fit(fit_rows, features)
            p = model.predict(test_rows, features)
            aucs.append(roc_auc_score(y[test_idx], [p[r.question_id] for r in test_rows]))
        mean = float(np.mean(aucs))
        if best is None or mean > best[0]:
            best = (mean, dict(params))
    assert best is not None
    return best[1], best[0]


@dataclass(frozen=True)
class VariantResult:
    variant: str
    auc_text_ok: float
    score: float
    cost: float
    met_budget: bool
    stage1_share: float
    """Share of validation questions decided in stage 1 (no OCR)."""
    params: dict[str, Any]


def _quantiles(values: Mapping[str, float], qs: Sequence[float]) -> list[float]:
    return [float(v) for v in np.quantile(list(values.values()), qs)]


def run_variants(
    train: Sequence[LabelRow],
    val: Sequence[LabelRow],
    features: Features,
    seconds: Mapping[str, Mapping[str, float]],
    layout_s: Mapping[str, float],
    *,
    stage1: Sequence[str] = ("s1", "s3"),
    stage2: Sequence[str] = ("s1", "s2", "s3", "s4"),
    needs_layout: Mapping[str, bool],
    variants: Sequence[str] = VARIANTS,
    tune_grid: Sequence[Mapping[str, Any]] = TUNE_GRID,
    seed: int = 0,
) -> list[VariantResult]:
    from sklearn.metrics import roc_auc_score

    p1 = tuple(f"{e}_" for e in stage1)
    p2 = tuple(f"{e}_" for e in stage2)
    budget = vision_budget(val)
    text_ok = [int(r.score["text"] >= r.score["vision"]) for r in val]
    out = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        params: dict[str, dict[str, Any]] = defaultdict(dict)
        if variant == "tuned":
            params["stage1"], _ = tune(train, features, p1, grid=tune_grid, seed=seed)
            params["stage2"], _ = tune(train, features, p2, grid=tune_grid, seed=seed)
        objective = {"weighted": "weighted", "regress": "regress"}.get(variant, "classify")
        router = TwoStageRouter(
            StageModel(p1, seed, objective=objective, params=params["stage1"]).fit(train, features),
            StageModel(p2, seed, objective=objective, params=params["stage2"]).fit(train, features),
            tuple(stage1),
            tuple(stage2),
            any(needs_layout[e] for e in stage1),
            any(needs_layout[e] for e in (*stage1, *stage2)),
        )
        s1 = router.stage1.predict(val, features)
        s2 = router.stage2.predict(val, features)
        auc = float(roc_auc_score(text_ok, [-s2[r.question_id] for r in val]))
        t1s = [float("inf"), *_quantiles(s1, STAGE1_QUANTILES)]
        t2s = _quantiles(s2, STAGE2_QUANTILES)
        grid = [
            grid_point(val, router.policy(val, features, seconds, layout_s, (a, b)), (a, b))
            for a in t1s
            for b in t2s
        ]
        best, met = select_point(grid, budget)
        policy = router.policy(val, features, seconds, layout_s, best.params)
        stage1_share = sum(not used for used in policy.uses_ocr.values()) / len(val)
        out.append(
            VariantResult(variant, auc, best.score, best.cost, met, stage1_share, dict(params))
        )
    return out


def format_variants(results: Sequence[VariantResult]) -> str:
    lines = [f"  {'variant':<10}{'AUC text_ok':>12}{'score':>8}{'C s':>8}{'stage 1':>9}"]
    for r in results:
        lines.append(
            f"  {r.variant:<10}{r.auc_text_ok:>12.3f}{r.score * 100:>8.1f}{r.cost:>8.3f}"
            f"{r.stage1_share:>9.2f}{'' if r.met_budget else '  (over budget)'}"
        )
    return "\n".join(lines)
