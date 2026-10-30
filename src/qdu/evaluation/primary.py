"""The primary metric of each dataset, used for the oracle and the router's labels.

Each benchmark has a conventional headline metric, and the oracle must pick "the best path"
by that metric, not by one metric for all: on ChartQA an answer of "2.0" for a gold "1.98"
is correct under relaxed accuracy but scores 0 under ANLS.

- DocVQA, InfographicVQA: ANLS (Biten et al., 2019; Mathew et al., 2021/2022)
- ChartQA: relaxed accuracy, 5% numeric tolerance (Masry et al., 2022)
- TAT-DQA: token F1 (Zhu et al., 2022; scale not scored, see qdu.evaluation.em_f1)
"""

from collections.abc import Iterable

from qdu.evaluation.anls import anls_score
from qdu.evaluation.em_f1 import token_f1
from qdu.evaluation.relaxed import relaxed_match

PRIMARY_METRIC = {
    "docvqa": "anls",
    "infographicvqa": "anls",
    "chartqa": "relaxed_accuracy",
    "tatdqa": "f1",
}
DEFAULT_METRIC = "anls"


def primary_metric(dataset: str) -> str:
    return PRIMARY_METRIC.get(dataset, DEFAULT_METRIC)


def primary_score(dataset: str, prediction: str, answers: Iterable[str]) -> float:
    answers = list(answers)
    metric = primary_metric(dataset)
    if metric == "relaxed_accuracy":
        return float(relaxed_match(prediction, answers))
    if metric == "f1":
        return token_f1(prediction, answers)
    return anls_score(prediction, answers)
