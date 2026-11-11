"""Query router: label table, signals, decision rule and evaluation against the baselines."""

from qdu.router.evaluate import BASELINE, Policy, baselines, evaluate, read_policy
from qdu.router.features import (
    FeatureExtractor,
    QuestionShape,
    RouterInput,
    build_features,
    router_inputs,
)
from qdu.router.labels import PATHS, LabelRow, build_labels, read_labels, write_labels
from qdu.router.ocr import DEFAULT_OCR, load_ocr, pack_ocr
from qdu.router.policy import choose, cost_quality_curve, mean_costs, route

# Extractors available to `qdu router features --extractors`, by name.
EXTRACTORS: dict[str, type] = {"q": QuestionShape}

__all__ = [
    "BASELINE",
    "DEFAULT_OCR",
    "EXTRACTORS",
    "PATHS",
    "FeatureExtractor",
    "LabelRow",
    "Policy",
    "QuestionShape",
    "RouterInput",
    "baselines",
    "build_features",
    "build_labels",
    "choose",
    "cost_quality_curve",
    "evaluate",
    "load_ocr",
    "mean_costs",
    "pack_ocr",
    "read_labels",
    "read_policy",
    "route",
    "router_inputs",
    "write_labels",
]
