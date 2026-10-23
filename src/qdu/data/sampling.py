"""Stratified, reproducible question sampling.

DocVQA asks several questions about the same page, so questions from one page are not
independent evidence. The sampler therefore takes at most one question per document by
default and stratifies by question type so rare types (e.g. ``figure/diagram``) are present
in sufficient numbers to be analysed separately.

Questions carry several type labels. Each question is assigned to the stratum of its
**rarest** label, so a question tagged ``["table/list", "figure/diagram"]`` counts towards the
rarer ``figure/diagram`` stratum.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace

from qdu.data.records import QARecord

UNKNOWN_STRATUM = "unknown"


def label_counts(records: Sequence[QARecord]) -> Counter[str]:
    return Counter(label for r in records for label in r.question_types)


def assign_stratum(question_types: Sequence[str], counts: Counter[str]) -> str:
    if not question_types:
        return UNKNOWN_STRATUM
    return min(question_types, key=lambda label: (counts[label], label))


def _allocate(available: dict[str, int], n: int, min_per_stratum: int) -> dict[str, int]:
    """Minimum per stratum first, the remainder proportionally (largest remainder method)."""
    allocation = {s: min(a, min_per_stratum) for s, a in available.items()}
    remaining = n - sum(allocation.values())
    if remaining < 0:
        raise ValueError(
            f"n={n} is smaller than min_per_stratum x strata ({sum(allocation.values())})"
        )
    capacity = {s: available[s] - allocation[s] for s in available}
    while remaining > 0 and any(capacity.values()):
        total = sum(capacity.values())
        quotas = {s: remaining * c / total for s, c in capacity.items()}
        grant = {s: min(capacity[s], int(q)) for s, q in quotas.items()}
        leftover = remaining - sum(grant.values())
        by_fraction = sorted(capacity, key=lambda s: (-(quotas[s] - int(quotas[s])), s))
        for s in by_fraction:
            if leftover == 0:
                break
            if capacity[s] > grant[s]:
                grant[s] += 1
                leftover -= 1
        for s, g in grant.items():
            allocation[s] += g
            capacity[s] -= g
        remaining -= sum(grant.values())
    return allocation


def stratified_sample(
    records: Sequence[QARecord],
    n: int,
    *,
    min_per_stratum: int = 0,
    one_per_document: bool = True,
    seed: int = 0,
) -> list[QARecord]:
    """Return ``n`` records with ``stratum`` filled in, deterministic for a given seed."""
    counts = label_counts(records)
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: r.question_id)
    rng.shuffle(pool)

    by_stratum: dict[str, list[QARecord]] = defaultdict(list)
    for r in pool:
        by_stratum[assign_stratum(r.question_types, counts)].append(r)

    def docs(rs: list[QARecord]) -> int:
        return len({r.doc_id for r in rs}) if one_per_document else len(rs)

    available = {s: docs(rs) for s, rs in by_stratum.items()}
    total_available = docs(pool)
    if n > total_available:
        raise ValueError(f"requested n={n} but only {total_available} eligible questions")
    allocation = _allocate(available, n, min_per_stratum)

    chosen: list[QARecord] = []
    used_docs: set[str] = set()
    chosen_ids: set[str] = set()

    def take(r: QARecord, stratum: str) -> None:
        chosen.append(replace(r, stratum=stratum))
        chosen_ids.add(r.question_id)
        used_docs.add(r.doc_id)

    # Rarest strata first, so they get first pick of shared documents.
    for stratum in sorted(by_stratum, key=lambda s: (available[s], s)):
        quota = allocation.get(stratum, 0)
        for r in by_stratum[stratum]:
            if quota == 0:
                break
            if one_per_document and r.doc_id in used_docs:
                continue
            take(r, stratum)
            quota -= 1

    # Document conflicts can leave a shortfall; fill it from any remaining stratum.
    for r in pool:
        if len(chosen) >= n:
            break
        if r.question_id in chosen_ids or (one_per_document and r.doc_id in used_docs):
            continue
        take(r, assign_stratum(r.question_types, counts))

    return sorted(chosen, key=lambda r: (r.stratum, r.question_id))
