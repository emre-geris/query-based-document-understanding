"""Deterministic document-level train / val / test assignment.

Each document is assigned by hashing ``(salt, dataset, doc_id)``, so the assignment never
depends on row order, sample size or which other documents exist: a document always lands
in the same split, and no document can appear in two splits.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence

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


SPLIT_SCHEMES = ("v1", "v2")
"""Router split schemes of the label table.

- ``v1``: the manifest's split: TAT-DQA all ``test`` (unseen data), the others by document.
- ``v2`` (Decided, Emre, 15 Nov 2026): the same for DocVQA, InfographicVQA and ChartQA; TAT-DQA
  is split 60/20/20 like them, grouped by annual report, since one report gives several
  documents. Unseen-data generalisation is tested by leaving one dataset out instead
  (`qdu.router.lodo`).
"""
DEFAULT_SCHEME = "v2"
_TATDQA_PAGE = re.compile(r"^tatdqa/(?P<uid>.+)_p\d+$")


def router_split(
    dataset: str,
    doc_id: str,
    manifest_split: str,
    scheme: str = DEFAULT_SCHEME,
    tatdqa_reports: Mapping[str, str] | None = None,
) -> str:
    """The router split of a question under ``scheme``, from its manifest split.

    ``tatdqa_reports`` maps a TAT-DQA document uid to its report (``qdu.data.tatdqa``); it is
    needed for TAT-DQA questions under ``v2``."""
    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"unknown split scheme {scheme!r}; choose from {SPLIT_SCHEMES}")
    if scheme == "v1" or dataset != "tatdqa":
        return manifest_split
    match = _TATDQA_PAGE.match(doc_id)
    if match is None:
        raise ValueError(f"not a TAT-DQA page id: {doc_id!r}")
    if tatdqa_reports is None or match["uid"] not in tatdqa_reports:
        raise ValueError(f"no report for TAT-DQA document {match['uid']!r}")
    return assign_split("tatdqa", f"tatdqa/{tatdqa_reports[match['uid']]}")
