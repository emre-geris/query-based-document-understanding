"""Routing-oracle analysis.

Given every pipeline's score on every question, the *oracle* picks the best pipeline per
question. Its mean score (each dataset's primary metric, see qdu.evaluation.primary) is the
ceiling any router can reach with these pipelines; the gap to the best single pipeline is the
most a router can add. The oracle's choice, breaking ties
towards the cheaper pipeline, is the training label for a learned router.

Ties are broken by each question's **measured** cost (seconds or tokens) when every tied
pipeline has one, and only otherwise by ``DEFAULT_COST_ORDER``. The pilot showed why: with OCR
time included, the text path was on average slower than the vision path on a Colab L4, so a
fixed text < vision < hybrid order mislabelled questions that every pipeline answered.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

# Fallback only, when measured costs are missing (e.g. a provider that reports no tokens).
DEFAULT_COST_ORDER: tuple[str, ...] = ("text", "vision", "hybrid")


@dataclass(frozen=True)
class ScoredPrediction:
    question_id: str
    stratum: str
    pipeline: str
    score: float
    """The dataset's primary metric (see qdu.evaluation.primary)."""
    cost: float | None = None
    """Measured cost of this call (seconds or tokens); used to break ties."""


@dataclass
class GroupSummary:
    n_questions: int
    mean_score: dict[str, float]
    oracle_score: float
    label_counts: dict[str, int]

    @property
    def best_single(self) -> str:
        return max(self.mean_score, key=lambda p: (self.mean_score[p], p))

    @property
    def headroom(self) -> float:
        """Oracle minus best single pipeline."""
        return self.oracle_score - self.mean_score[self.best_single]


@dataclass
class OracleSummary:
    pipelines: tuple[str, ...]
    overall: GroupSummary
    by_stratum: dict[str, GroupSummary]
    labels: dict[str, str] = field(repr=False)
    incomplete_questions: int = 0


def oracle_label(
    scores: dict[str, float],
    cost_order: Sequence[str],
    tolerance: float,
    costs: dict[str, float | None] | None = None,
) -> str:
    """Cheapest pipeline whose score is within ``tolerance`` of the best score.

    "Cheapest" uses the measured ``costs`` when every candidate has one, else ``cost_order``.
    """
    best = max(scores.values())
    rank = {p: i for i, p in enumerate(cost_order)}
    candidates = [p for p, s in scores.items() if s >= best - tolerance]
    if costs is not None and all(costs.get(p) is not None for p in candidates):
        return min(candidates, key=lambda p: (costs[p], rank.get(p, len(rank)), p))
    return min(candidates, key=lambda p: (rank.get(p, len(rank)), p))


def _summarize(
    per_question: dict[str, dict[str, float]],
    labels: dict[str, str],
    pipelines: tuple[str, ...],
) -> GroupSummary:
    n = len(per_question)
    mean = {p: sum(s[p] for s in per_question.values()) / n for p in pipelines}
    oracle = sum(max(s.values()) for s in per_question.values()) / n
    counts = {p: 0 for p in pipelines}
    for qid in per_question:
        counts[labels[qid]] += 1
    return GroupSummary(n, mean, oracle, counts)


def summarize(
    rows: Sequence[ScoredPrediction],
    pipelines: Sequence[str] | None = None,
    *,
    cost_order: Sequence[str] = DEFAULT_COST_ORDER,
    tolerance: float = 0.0,
) -> OracleSummary:
    """Summarize over questions that have a score for every pipeline (complete cases only)."""
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    costs: dict[str, dict[str, float | None]] = defaultdict(dict)
    stratum: dict[str, str] = {}
    for r in rows:
        scores[r.question_id][r.pipeline] = r.score
        costs[r.question_id][r.pipeline] = r.cost
        stratum[r.question_id] = r.stratum
    names = tuple(pipelines) if pipelines else tuple(sorted({r.pipeline for r in rows}))
    complete = {q: s for q, s in scores.items() if all(p in s for p in names)}
    if not complete:
        raise ValueError("no question has results for all pipelines")
    complete = {q: {p: s[p] for p in names} for q, s in complete.items()}

    labels = {q: oracle_label(s, cost_order, tolerance, costs[q]) for q, s in complete.items()}
    groups: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for q, s in complete.items():
        groups[stratum[q]][q] = s

    return OracleSummary(
        pipelines=names,
        overall=_summarize(complete, labels, names),
        by_stratum={g: _summarize(qs, labels, names) for g, qs in sorted(groups.items())},
        labels=labels,
        incomplete_questions=len(scores) - len(complete),
    )
