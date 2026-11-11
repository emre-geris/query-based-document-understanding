"""The decision rule: pick the path maximising predicted quality minus λ times its cost.

At decision time the router does not know a call's latency or token count, only an estimate;
the simplest is each path's mean cost on the training split (`mean_costs`). λ trades quality
for cost and is chosen on the validation split from the cost-quality curve
(`cost_quality_curve`), never on test.
"""

from collections.abc import Iterable, Mapping, Sequence

from qdu.router.evaluate import Policy
from qdu.router.labels import LabelRow


def choose(quality: Mapping[str, float], cost: Mapping[str, float], lam: float) -> str:
    """``argmax_p quality[p] - lam * cost[p]``; ties go to the cheaper path."""
    return max(quality, key=lambda p: (quality[p] - lam * cost[p], -cost[p], p))


def mean_costs(rows: Iterable[LabelRow], scenario: str = "C") -> dict[str, float]:
    """Mean cost of each path over ``rows`` (use the training split)."""
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to average")
    paths = rows[0].score.keys()
    return {p: sum(r.cost(scenario)[p] for r in rows) / len(rows) for p in paths}


def route(
    predicted: Mapping[str, Mapping[str, float]],
    cost: Mapping[str, float] | Mapping[str, Mapping[str, float]],
    lam: float,
    name: str = "router",
    **router_cost: Mapping,
) -> Policy:
    """A policy from predicted quality per question and path.

    ``cost`` is one estimate per path, or one per question and path. ``router_cost`` passes
    ``uses_ocr``, ``page_s`` and ``question_s`` on to the policy (see qdu.router.evaluate).
    """
    per_question = isinstance(next(iter(cost.values())), Mapping)
    choice = {
        qid: choose(quality, cost[qid] if per_question else cost, lam)  # type: ignore[arg-type]
        for qid, quality in predicted.items()
    }
    return Policy(name, choice, **router_cost)


def cost_quality_curve(
    rows: Sequence[LabelRow],
    predicted: Mapping[str, Mapping[str, float]],
    cost: Mapping[str, float],
    lambdas: Iterable[float],
    scenario: str = "C",
    **router_cost: Mapping,
) -> list[tuple[float, float, float]]:
    """(λ, mean score, mean measured cost under ``scenario``) for each λ, on ``rows``."""
    curve = []
    for lam in lambdas:
        policy = route(predicted, cost, lam, **router_cost)
        n = len(rows)
        score = sum(r.score[policy.choice[r.question_id]] for r in rows) / n
        spent = sum(policy.cost(r, scenario) for r in rows) / n
        curve.append((lam, score, spent))
    return curve
