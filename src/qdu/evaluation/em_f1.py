"""Exact match and token F1 in the style of the TAT-QA / TAT-DQA evaluation (and DROP).

Text is lowercased, punctuation and English articles are removed, and every token that is a
number is rewritten to a canonical form rounded to two decimals, so ``"645,791"`` and
``"645791.0"`` compare equal. F1 is the bag-of-tokens overlap, maximised over gold answers.

Deviation from the official script: TAT-DQA also scores the predicted *scale* (thousand,
million, percent). Our pipelines return a single answer string without a separate scale, so
scale is not scored; this is stated wherever TAT-DQA numbers are reported.
"""

import re
import string
from collections import Counter
from collections.abc import Iterable

from qdu.evaluation.relaxed import parse_number

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation.replace(".", "").replace("-", ""))


def _canonical_number(token: str) -> str | None:
    value = parse_number(token)
    if value is None:
        return None
    rounded = round(value, 2)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.2f}".rstrip("0")


def normalize_tokens(text: str) -> list[str]:
    tokens = []
    for raw in _ARTICLES.sub(" ", text.lower()).split():
        number = _canonical_number(raw)
        if number is not None:
            tokens.append(number)
            continue
        cleaned = raw.translate(_PUNCT).strip(".-")
        if cleaned:
            tokens.append(cleaned)
    return tokens


def contains_answer(text: str, answers: Iterable[str]) -> bool:
    """True if some gold answer occurs in ``text`` as a contiguous run of normalized tokens.

    Used to measure OCR *coverage*: an answer missing from the OCR text cannot be produced by
    the text path, whatever the language model does.
    """
    tokens = normalize_tokens(text)
    for answer in answers:
        target = normalize_tokens(answer)
        n = len(target)
        if n and any(tokens[i : i + n] == target for i in range(len(tokens) - n + 1)):
            return True
    return False


def exact_match(prediction: str, answers: Iterable[str]) -> bool:
    pred = normalize_tokens(prediction)
    return any(pred == normalize_tokens(answer) for answer in answers)


def token_f1(prediction: str, answers: Iterable[str]) -> float:
    pred = Counter(normalize_tokens(prediction))
    best = 0.0
    for answer in answers:
        gold = Counter(normalize_tokens(answer))
        if not pred and not gold:
            return 1.0
        overlap = sum((pred & gold).values())
        if overlap == 0:
            continue
        precision = overlap / sum(pred.values())
        recall = overlap / sum(gold.values())
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
