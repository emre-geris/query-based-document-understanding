"""Deterministic document-level train / val / test assignment.

Each document is assigned by hashing ``(salt, dataset, doc_id)``, so the assignment never
depends on row order, sample size or which other documents exist: a document always lands
in the same split, and no document can appear in two splits.
"""

import hashlib
from collections.abc import Sequence

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.6, 0.2, 0.2)
DEFAULT_SALT = "qdu-split-v1"


def assign_split(
    dataset: str,
    doc_id: str,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    salt: str = DEFAULT_SALT,
) -> str:
    if len(ratios) != len(SPLITS) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be {len(SPLITS)} numbers summing to 1, got {ratios}")
    digest = hashlib.sha256(f"{salt}:{dataset}:{doc_id}".encode()).hexdigest()
    position = int(digest[:12], 16) / 16**12
    cumulative = 0.0
    for split, ratio in zip(SPLITS, ratios, strict=True):
        cumulative += ratio
        if position < cumulative:
            return split
    return SPLITS[-1]
