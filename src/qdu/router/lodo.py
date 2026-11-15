"""Leave one dataset out: does the router work on a kind of document it never saw?

Under split scheme v2 every dataset, TAT-DQA included, has train / val / test questions, so the
unseen-data test is no longer "TAT-DQA was never seen" but this one, for each dataset ``D`` in
turn: the two-stage router is fitted on the train questions of the other datasets, its
thresholds are chosen on their val questions within the same budget rule as the in-domain
router (`qdu.router.baselines.Budget`, measured on those val questions), and it is scored on
``D``, which it never saw in fitting or tuning. ``D``'s train and val questions (``dev``) can be
used freely for this, since the held-out router never saw them; ``D``'s test questions are kept
for the one-time final evaluation like every other test number.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qdu.router.baselines import Budget
from qdu.router.evaluate import Policy
from qdu.router.labels import LabelRow
from qdu.router.two_stage import TwoStageRouter, train_two_stage

Features = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class HeldOut:
    dataset: str
    router: TwoStageRouter
    policy: Policy
    rows: list[LabelRow]
    """The held-out dataset's questions the policy is scored on."""
    budget_s: float
    """The budget on the other datasets' val questions, in C seconds per question."""


def leave_one_out(
    train: Sequence[LabelRow],
    val: Sequence[LabelRow],
    evaluated: Sequence[LabelRow],
    features: Features,
    seconds: Mapping[str, Mapping[str, float]],
    layout_s: Mapping[str, float],
    *,
    budget: Budget,
    datasets: Sequence[str] | None = None,
    page_seconds: Mapping[str, Mapping[str, float]] | None = None,
    **two_stage: object,
) -> list[HeldOut]:
    """One router per held-out dataset (default: every dataset in ``evaluated``).

    ``two_stage`` is passed to `train_two_stage` (stage signals, paths, needs_layout, seed)."""
    out = []
    for dataset in datasets or sorted({r.dataset for r in evaluated}):
        fit = [r for r in train if r.dataset != dataset]
        tune = [r for r in val if r.dataset != dataset]
        scored = [r for r in evaluated if r.dataset == dataset]
        if not fit or not tune or not scored:
            raise ValueError(f"leaving out {dataset!r} leaves no train, val or evaluated rows")
        limit = budget.on(tune)
        router = train_two_stage(
            fit,
            tune,
            features,
            seconds,
            layout_s,
            budget=limit,
            page_seconds=page_seconds,
            **two_stage,  # type: ignore[arg-type]
        )
        policy = router.policy(
            scored, features, seconds, layout_s, name=f"lodo-{dataset}", page_seconds=page_seconds
        )
        out.append(HeldOut(dataset, router, policy, scored, limit))
    return out
