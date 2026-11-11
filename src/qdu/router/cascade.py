"""Text-then-vision confidence cascade: answer from OCR text, escalate to the image when unsure.

Each question first goes to the text path; if the model's confidence in its text answer is
below a threshold, the vision path answers instead and the question pays for both calls (and
the OCR). The confidence is the probability of the generated answer, ``exp(sum of token
log-probabilities)``, or the mean token probability, from a run of the text path with
``QDU_LOGPROBS=1`` (`results/full-text-logprobs-qwen3.5-4b`). The threshold is chosen on the
validation split.

The logprob run repeats the text calls of the labelled run with the same greedy decoding. When
its answer differs from the labelled answer (non-deterministic GPU arithmetic), the confidence
belongs to another answer, so `read_confidence` reports how many differ and leaves them out.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from qdu.router.evaluate import Policy
from qdu.router.labels import LabelRow
from qdu.runner import read_predictions

CONFIDENCE = ("sequence", "mean_token", "min_token")


def answer_confidence(token_logprobs: Sequence[float], kind: str = "sequence") -> float:
    """Probability of the whole answer, the mean token probability or the least likely token's."""
    if not token_logprobs:
        return 0.0
    if kind == "sequence":
        return math.exp(sum(token_logprobs))
    if kind == "mean_token":
        return sum(math.exp(lp) for lp in token_logprobs) / len(token_logprobs)
    if kind == "min_token":
        return math.exp(min(token_logprobs))
    raise ValueError(f"unknown confidence {kind!r}; choose from {CONFIDENCE}")


def read_confidence(
    logprob_run: Path, labelled_run: Path, kind: str = "sequence", pipeline: str = "text"
) -> tuple[dict[str, float], list[str]]:
    """(question id -> confidence, ids whose answer differs from the labelled run's)."""

    def latest(run: Path) -> dict[str, dict]:
        rows = {}
        for r in read_predictions(run / "predictions.jsonl"):
            if r["error"] is None and r["pipeline"] == pipeline:
                rows[r["question_id"]] = r
        return rows

    labelled = latest(labelled_run)
    confidence, differ = {}, []
    for qid, r in latest(logprob_run).items():
        if r.get("token_logprobs") is None:
            continue
        if qid in labelled and labelled[qid]["prediction"] != r["prediction"]:
            differ.append(qid)
            continue
        confidence[qid] = answer_confidence(r["token_logprobs"], kind)
    return confidence, sorted(differ)


def cascade(
    rows: Iterable[LabelRow],
    confidence: Mapping[str, float],
    threshold: float,
    *,
    first: str = "text",
    fallback: str = "vision",
    name: str | None = None,
) -> Policy:
    """``first`` when its confidence is at least ``threshold``, else ``fallback`` (paying both).
    A question without a confidence escalates."""
    choice, before = {}, {}
    for r in rows:
        if confidence.get(r.question_id, -1.0) >= threshold:
            choice[r.question_id] = first
        else:
            choice[r.question_id] = fallback
            before[r.question_id] = (first,)
    return Policy(name or f"cascade@{threshold:g}", choice, called_before=before)


def sweep(
    rows: Sequence[LabelRow],
    confidence: Mapping[str, float],
    thresholds: Iterable[float],
    scenario: str = "C",
) -> list[tuple[float, float, float, float]]:
    """(threshold, mean score, mean cost, share answered by text) for each threshold."""
    out = []
    for t in thresholds:
        policy = cascade(rows, confidence, t)
        n = len(rows)
        score = sum(r.score[policy.choice[r.question_id]] for r in rows) / n
        cost = sum(policy.cost(r, scenario) for r in rows) / n
        kept = sum(policy.choice[r.question_id] == "text" for r in rows) / n
        out.append((t, score, cost, kept))
    return out
