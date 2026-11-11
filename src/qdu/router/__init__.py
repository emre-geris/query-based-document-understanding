"""Query router: label table, signals, decision rule and evaluation against the baselines."""

from qdu.router.evaluate import BASELINE, Policy, baselines, evaluate, read_policy
from qdu.router.features import FeatureExtractor, RouterInput, build_features, router_inputs
from qdu.router.labels import PATHS, LabelRow, build_labels, read_labels, write_labels
from qdu.router.ocr import DEFAULT_OCR, load_ocr, pack_ocr
from qdu.router.policy import choose, cost_quality_curve, mean_costs, route
from qdu.router.signals import EvidenceSignals, OcrSignals, PageSignals, QuestionSignals

# Extractors available to `qdu router features --extractors`, by name.
EXTRACTORS: dict[str, type] = {
    "s1": QuestionSignals,
    "s2": OcrSignals,
    "s3": PageSignals,
    "s4": EvidenceSignals,
}

__all__ = [
    "BASELINE",
    "DEFAULT_OCR",
    "EXTRACTORS",
    "PATHS",
    "EvidenceSignals",
    "FeatureExtractor",
    "LabelRow",
    "OcrSignals",
    "PageSignals",
    "Policy",
    "QuestionSignals",
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
