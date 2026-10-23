"""Average Normalized Levenshtein Similarity (ANLS), the standard DocVQA metric.

For one question with gold answers ``g_1..g_k`` and a prediction ``p``::

    NL_i  = levenshtein(norm(p), norm(g_i)) / max(len(norm(p)), len(norm(g_i)))
    s_i   = 1 - NL_i   if NL_i < threshold else 0
    score = max_i s_i

The dataset score is the mean of per-question scores. ``norm`` lowercases, strips and
collapses whitespace, following the official DocVQA evaluation (Biten et al., 2019;
Mathew et al., 2021). The threshold is 0.5.
"""

from collections.abc import Iterable, Sequence

ANLS_THRESHOLD = 0.5


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def levenshtein(a: str, b: str) -> int:
    """Edit distance with unit costs for insertion, deletion and substitution."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (char_a != char_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def anls_score(prediction: str, answers: Iterable[str], threshold: float = ANLS_THRESHOLD) -> float:
    """ANLS for a single question: best similarity over all gold answers."""
    pred = _normalize(prediction)
    best = 0.0
    for answer in answers:
        gold = _normalize(answer)
        longest = max(len(pred), len(gold))
        if longest == 0:
            similarity = 1.0
        else:
            nl = levenshtein(pred, gold) / longest
            similarity = 1.0 - nl if nl < threshold else 0.0
        best = max(best, similarity)
    return best


def mean_anls(
    predictions: Sequence[str],
    answers: Sequence[Iterable[str]],
    threshold: float = ANLS_THRESHOLD,
) -> float:
    """Dataset-level ANLS: mean of per-question scores."""
    if len(predictions) != len(answers):
        raise ValueError(f"{len(predictions)} predictions but {len(answers)} answer lists")
    if not predictions:
        raise ValueError("cannot compute ANLS over zero questions")
    scores = [anls_score(p, a, threshold) for p, a in zip(predictions, answers, strict=True)]
    return sum(scores) / len(scores)
