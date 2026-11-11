"""Paired bootstrap confidence intervals that resample documents, not questions.

DocVQA asks ~4 questions per page and they are not independent (one hard page fails several
of them), so resampling questions would make intervals too narrow. Each resample draws
documents with replacement and keeps all questions of a drawn document; every method is scored
on the same resample, so differences between methods are paired.

The percentile interval is taken from ``resamples`` draws of a seeded ``random.Random``, with
documents in sorted order, so the same inputs always give the same interval.
"""

import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Difference:
    method: str
    baseline: str
    estimate: float
    low: float
    high: float
    n_questions: int
    n_documents: int

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_differences(
    values: Mapping[str, Mapping[str, float]],
    document: Mapping[str, str],
    baseline: str,
    *,
    resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Difference]:
    """Mean of each method minus the mean of ``baseline``, with a percentile CI.

    ``values[method][question_id]`` is a per-question value (a score or a cost); every method
    must cover the same questions. ``document[question_id]`` groups questions for resampling.
    """
    if baseline not in values:
        raise ValueError(f"baseline {baseline!r} not among methods {sorted(values)}")
    questions = sorted(values[baseline])
    for method, v in values.items():
        if set(v) != set(questions):
            raise ValueError(f"method {method!r} does not cover the baseline's questions")
    if not questions:
        raise ValueError("no questions to compare")

    by_doc: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by_doc[document[q]].append(q)
    docs = sorted(by_doc)
    # Per document: number of questions and each method's sum minus the baseline's sum.
    sizes = [len(by_doc[d]) for d in docs]
    sums = {
        m: [sum(v[q] - values[baseline][q] for q in by_doc[d]) for d in docs]
        for m, v in values.items()
        if m != baseline
    }

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {m: [] for m in sums}
    for _ in range(resamples):
        picks = [rng.randrange(len(docs)) for _ in docs]
        n = sum(sizes[i] for i in picks)
        for m, s in sums.items():
            draws[m].append(sum(s[i] for i in picks) / n)

    alpha = (1 - level) / 2
    lo, hi = int(alpha * resamples), int((1 - alpha) * resamples) - 1
    out = {}
    for m, d in draws.items():
        d.sort()
        out[m] = Difference(
            method=m,
            baseline=baseline,
            estimate=sum(sums[m]) / len(questions),
            low=d[lo],
            high=d[hi],
            n_questions=len(questions),
            n_documents=len(docs),
        )
    return out
