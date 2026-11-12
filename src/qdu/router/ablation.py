"""Signal ablation: what each group of router signals adds, measured on validation.

Every configuration (a set of signal groups, with or without the question's TF-IDF) is scored
two ways, both fitted on ``train`` and measured on ``val``:

- **Predictability** (LightGBM classifiers, ROC AUC; 0.5 = chance): can the signals tell, before
  answering, whether text is as good as vision (``text_ok``, the cost lever), whether the text
  answer is right (``text_right``), and whether text or hybrid beats vision (``beats_vision``,
  the accuracy lever)? ``text_ok`` is also reported as the mean AUC *within* each dataset, which
  a router cannot reach by recognising the dataset.
- **Routing**: the learned per-path router (`qdu.router.train`) on that configuration, with λ
  chosen within the always-vision C budget, as score and C cost.

The ``dataset`` group is the dataset name as one-hot features. A deployed router does not know
it, so it only appears in configurations labelled as a leakage check (CLAUDE.md, *Router work*).
"""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from qdu.router.labels import LabelRow
from qdu.router.train import train_router

Features = Mapping[str, Mapping[str, float]]

TARGETS = {
    "text_ok": lambda r: r.score["text"] >= r.score["vision"],
    "text_right": lambda r: r.score["text"] >= 0.5,
    "beats_vision": lambda r: max(r.score["text"], r.score["hybrid"]) > r.score["vision"],
}


@dataclass(frozen=True)
class Config:
    name: str
    groups: tuple[str, ...]
    """Signal groups (``s1``..``s4``, ``dataset``); feature names start with ``<group>_``."""
    tfidf: bool = True


DEFAULT_CONFIGS = (
    Config("tfidf only", ()),
    Config("S1", ("s1",)),
    Config("S1+S2", ("s1", "s2")),
    Config("S1+S2+S3", ("s1", "s2", "s3")),
    Config("S1-S4", ("s1", "s2", "s3", "s4")),
    Config("S1-S4, no tfidf", ("s1", "s2", "s3", "s4"), tfidf=False),
    Config("S1+S3 (no OCR)", ("s1", "s3")),
    Config("dataset only (leakage check)", ("dataset",), tfidf=False),
    Config("S1-S4 + dataset (leakage check)", ("s1", "s2", "s3", "s4", "dataset")),
)


@dataclass(frozen=True)
class AblationResult:
    config: Config
    auc: dict[str, float]
    auc_within_text_ok: float
    score: float
    cost: float
    lam: float
    uses_ocr: bool


def with_dataset(rows: Sequence[LabelRow], features: Features) -> dict[str, dict[str, float]]:
    """``features`` plus one-hot ``dataset_<name>`` columns (for leakage checks only)."""
    names = sorted({r.dataset for r in rows})
    out = {q: dict(f) for q, f in features.items()}
    for r in rows:
        out[r.question_id].update({f"dataset_{n}": float(r.dataset == n) for n in names})
    return out


def _subset(features: Features, groups: Sequence[str]) -> dict[str, dict[str, float]]:
    prefixes = tuple(f"{g}_" for g in groups)
    return {q: {k: v for k, v in f.items() if k.startswith(prefixes)} for q, f in features.items()}


def _design(rows, features, names, tfidf, vectorizer, fit):
    x = np.array(
        [[features[r.question_id].get(n, np.nan) for n in names] for r in rows], dtype=float
    ).reshape(len(rows), len(names))
    blocks = [sparse.csr_matrix(x)]
    if tfidf:
        questions = [r.question for r in rows]
        if fit:
            vectorizer.fit(questions)
        blocks.append(vectorizer.transform(questions))
    return sparse.hstack(blocks, format="csr")


def predictability(
    train: Sequence[LabelRow], val: Sequence[LabelRow], features: Features, tfidf: bool, seed: int
) -> tuple[dict[str, float], float]:
    """Validation AUC per target, and the mean within-dataset AUC of ``text_ok``."""
    import lightgbm as lgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_auc_score

    names = sorted({k for r in train for k in features[r.question_id]})
    vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True)
    x_train = _design(train, features, names, tfidf, vectorizer, True)
    x_val = _design(val, features, names, tfidf, vectorizer, False)
    auc, within = {}, float("nan")
    for target, label in TARGETS.items():
        y_train = np.array([label(r) for r in train], dtype=int)
        y_val = np.array([label(r) for r in val], dtype=int)
        model: Any = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        ).fit(x_train, y_train)
        p = model.predict_proba(x_val)[:, 1]
        auc[target] = float(roc_auc_score(y_val, p)) if len(set(y_val)) == 2 else float("nan")
        if target == "text_ok":
            per = []
            for d in sorted({r.dataset for r in val}):
                mask = np.array([r.dataset == d for r in val])
                if len(set(y_val[mask])) == 2:
                    per.append(roc_auc_score(y_val[mask], p[mask]))
            within = float(np.mean(per)) if per else float("nan")
    return auc, within


def config_cost(
    config: Config,
    question_ids: Sequence[str],
    seconds: Mapping[str, Mapping[str, float]],
    layout_s: Mapping[str, float],
    needs_ocr: Mapping[str, bool],
    needs_layout: Mapping[str, bool],
) -> dict[str, dict[str, Any]]:
    """What a router on ``config``'s signals pays per question (``uses_ocr``, ``page_s``,
    ``question_s``), from each extractor's timing and needs; ``dataset`` is free."""
    groups = [g for g in config.groups if g != "dataset"]
    ocr = any(needs_ocr[g] for g in groups)
    layout = any(needs_layout[g] for g in groups)
    return {
        "uses_ocr": dict.fromkeys(question_ids, ocr),
        "page_s": {q: layout_s[q] if layout else 0.0 for q in question_ids},
        "question_s": {q: sum(seconds[q][g] for g in groups) for q in question_ids},
    }


def run_ablation(
    train: Sequence[LabelRow],
    val: Sequence[LabelRow],
    features: Features,
    seconds: Mapping[str, Mapping[str, float]],
    layout_s: Mapping[str, float],
    needs_ocr: Mapping[str, bool],
    needs_layout: Mapping[str, bool],
    *,
    configs: Sequence[Config] = DEFAULT_CONFIGS,
    seed: int = 0,
) -> list[AblationResult]:
    """Predictability and routing result for every configuration.

    ``features`` must contain every group the configurations use (``with_dataset`` for the
    leakage checks); ``seconds`` is `FeatureTable.seconds`, ``layout_s`` each question's page
    layout time, ``needs_ocr`` / ``needs_layout`` say which extractor reads what."""
    ids = [r.question_id for r in [*train, *val]]
    out = []
    for config in configs:
        feats = _subset(features, config.groups)
        auc, within = predictability(train, val, feats, config.tfidf, seed)
        cost = config_cost(config, ids, seconds, layout_s, needs_ocr, needs_layout)
        uses_ocr = any(cost["uses_ocr"].values())
        trained = train_router(
            train, val, feats, cost, uses_ocr=uses_ocr, tfidf=config.tfidf, seed=seed
        )
        best = next(c for c in trained.curve if c.lam == trained.lam)
        out.append(
            AblationResult(config, auc, within, best.score, best.cost, trained.lam, uses_ocr)
        )
    return out


def format_ablation(results: Sequence[AblationResult]) -> str:
    lines = [
        f"  {'configuration':<34}{'text_ok':>8}{'within':>8}{'right':>8}{'beats':>8}"
        f"{'score':>8}{'C s':>7}{'λ':>6}"
    ]
    for r in results:
        a = r.auc
        lines.append(
            f"  {r.config.name:<34}{a['text_ok']:>8.3f}{r.auc_within_text_ok:>8.3f}"
            f"{a['text_right']:>8.3f}{a['beats_vision']:>8.3f}{r.score * 100:>8.1f}"
            f"{r.cost:>7.2f}{r.lam:>6.2f}"
        )
    return "\n".join(lines)


def write_ablation(results: Sequence[AblationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "configuration", "groups", "tfidf", "auc_text_ok", "auc_text_ok_within_dataset",
                "auc_text_right", "auc_beats_vision", "router_score", "router_cost_c_s",
                "router_lambda", "router_uses_ocr",
            ]
        )  # fmt: skip
        for r in results:
            w.writerow(
                [
                    r.config.name, "+".join(r.config.groups), int(r.config.tfidf),
                    f"{r.auc['text_ok']:.4f}", f"{r.auc_within_text_ok:.4f}",
                    f"{r.auc['text_right']:.4f}", f"{r.auc['beats_vision']:.4f}",
                    f"{r.score:.6f}", f"{r.cost:.4f}", r.lam, int(r.uses_ocr),
                ]
            )  # fmt: skip
