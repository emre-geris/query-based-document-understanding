"""Relaxed accuracy, the standard ChartQA metric (Masry et al., 2022).

A numeric prediction is correct when it is within 5% of a numeric gold answer; anything else
must match exactly (case-insensitive, surrounding whitespace ignored).
"""

import re
from collections.abc import Iterable

RELAXED_TOLERANCE = 0.05
_NUMBER = re.compile(r"^[-+]?\d*\.?\d+(e[-+]?\d+)?$", re.IGNORECASE)


def parse_number(text: str) -> float | None:
    """Parse '1,234.5', '45%', '$3.2' or '-7' as a float; None if not a plain number."""
    cleaned = text.strip().lower().replace(",", "").replace("$", "").rstrip("%").strip()
    return float(cleaned) if _NUMBER.match(cleaned) else None


def relaxed_match(
    prediction: str, answers: Iterable[str], tolerance: float = RELAXED_TOLERANCE
) -> bool:
    pred = prediction.strip().lower()
    pred_number = parse_number(prediction)
    for answer in answers:
        gold_number = parse_number(answer)
        if pred_number is not None and gold_number is not None:
            if gold_number == 0:
                if pred_number == 0:
                    return True
            elif abs(pred_number - gold_number) / abs(gold_number) <= tolerance:
                return True
        elif pred == answer.strip().lower():
            return True
    return False
