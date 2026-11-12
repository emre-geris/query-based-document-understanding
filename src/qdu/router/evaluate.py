"""Score a routing policy against the fixed baselines, with costs that include the router's work.

A policy is a path per question plus what the router itself spent deciding. Its cost per
question, by scenario (see qdu.evaluation.metrics):

- **C**: model latency of the chosen path + the page's OCR time if the path *or the router*
  needed OCR + the router's page-level signal time (e.g. layout) + its per-question time
- **B**: as C, but OCR and page-level signal time are divided among the page's questions
- **A**: tokens of the chosen path; signals are free

A router that computes OCR signals for every question therefore pays OCR even when it then
picks vision, which is why the oracle (no signal cost) is an optimistic ceiling.

Baselines are always-text, always-vision, always-hybrid and the oracle (latency ties). Every
method is compared with always-vision, the project's success baseline, by a paired bootstrap
over documents (qdu.evaluation.bootstrap).
"""

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qdu.evaluation.bootstrap import Difference, paired_differences
from qdu.router.labels import PATHS, LabelRow

PATH_USES_OCR = {"text": True, "vision": False, "hybrid": True}
BASELINE = "always-vision"


@dataclass(frozen=True)
class Policy:
    """A router's decision per question and the cost of making it."""

    name: str
    choice: Mapping[str, str]
    uses_ocr: Mapping[str, bool] = field(default_factory=dict)
    """Question ids for which the router ran OCR to compute its signals (missing: False)."""
    page_s: Mapping[str, float] = field(default_factory=dict)
    """Per-page signal time other than OCR, e.g. layout detection (shared by the page in B)."""
    question_s: Mapping[str, float] = field(default_factory=dict)
    """Per-question signal and decision time."""
    called_before: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Paths a cascade called (and paid for) before the one whose answer it kept."""

    def cost(self, row: LabelRow, scenario: str) -> float:
        qid = row.question_id
        paths = (*self.called_before.get(qid, ()), self.choice[qid])
        if scenario == "A":
            return sum(row.tokens[p] for p in paths)
        ocr = any(PATH_USES_OCR[p] for p in paths) or self.uses_ocr.get(qid, False)
        page = (row.ocr_s if ocr else 0.0) + self.page_s.get(qid, 0.0)  # OCR runs once
        if scenario == "B":
            page /= row.questions_on_page
        elif scenario != "C":
            raise ValueError(f"unknown scenario {scenario!r}; choose C, B or A")
        return sum(row.model_s[p] for p in paths) + page + self.question_s.get(qid, 0.0)


def baselines(rows: Iterable[LabelRow], paths: Sequence[str] = PATHS) -> list[Policy]:
    rows = list(rows)
    fixed = [Policy(f"always-{p}", {r.question_id: p for r in rows}) for p in paths]
    return [*fixed, Policy("oracle", {r.question_id: r.oracle_latency for r in rows})]


def read_policy(path: Path, name: str | None = None) -> Policy:
    """A policy from a CSV with ``question_id`` and ``pipeline`` and, optionally, the router's
    own cost: ``uses_ocr`` (0/1), ``page_s`` and ``question_s`` (seconds), and for a cascade
    ``called_before``: the paths it paid for before the kept one, separated by ``;``."""
    choice, uses_ocr, page_s, question_s, before = {}, {}, {}, {}, {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qid = r["question_id"]
            if r["pipeline"] not in PATH_USES_OCR:
                raise ValueError(f"{path}: unknown pipeline {r['pipeline']!r} for {qid}")
            choice[qid] = r["pipeline"]
            if r.get("uses_ocr"):
                uses_ocr[qid] = r["uses_ocr"].strip().lower() in ("1", "true", "yes")
            if r.get("page_s"):
                page_s[qid] = float(r["page_s"])
            if r.get("question_s"):
                question_s[qid] = float(r["question_s"])
            if r.get("called_before"):
                paths = tuple(p for p in r["called_before"].split(";") if p)
                unknown = [p for p in paths if p not in PATH_USES_OCR]
                if unknown:
                    raise ValueError(f"{path}: unknown called_before {unknown} for {qid}")
                before[qid] = paths
    return Policy(name or path.stem, choice, uses_ocr, page_s, question_s, before)


@dataclass(frozen=True)
class MethodResult:
    group: str
    method: str
    n: int
    score: float
    cost_c: float
    cost_b: float
    tokens: float
    score_vs_baseline: Difference | None
    cost_c_vs_baseline: Difference | None


def evaluate(
    rows: Iterable[LabelRow],
    policies: Sequence[Policy] = (),
    *,
    baseline: str = BASELINE,
    resamples: int = 1000,
    seed: int = 0,
) -> list[MethodResult]:
    """Baselines and ``policies`` per dataset and over all rows (group ``ALL``)."""
    rows = list(rows)
    if not rows:
        raise ValueError("no questions to evaluate (check the split and dataset filters)")
    methods = [*baselines(rows), *policies]
    names = [m.name for m in methods]
    for policy in policies:
        if names.count(policy.name) > 1:
            raise ValueError(f"policy name {policy.name!r} is reserved or used twice; rename it")
        missing = [r.question_id for r in rows if r.question_id not in policy.choice]
        if missing:
            raise ValueError(
                f"policy {policy.name!r} has no choice for {len(missing)} questions, "
                f"e.g. {missing[:3]}"
            )
    groups = {"ALL": rows}
    for dataset in sorted({r.dataset for r in rows}):
        groups[dataset] = [r for r in rows if r.dataset == dataset]

    out = []
    for group, rs in groups.items():
        document = {r.question_id: r.doc_id for r in rs}
        score = {
            m.name: {r.question_id: r.score[m.choice[r.question_id]] for r in rs} for m in methods
        }
        cost_c = {m.name: {r.question_id: m.cost(r, "C") for r in rs} for m in methods}
        score_diff = paired_differences(score, document, baseline, resamples=resamples, seed=seed)
        cost_diff = paired_differences(cost_c, document, baseline, resamples=resamples, seed=seed)
        n = len(rs)
        for m in methods:
            out.append(
                MethodResult(
                    group=group,
                    method=m.name,
                    n=n,
                    score=sum(score[m.name].values()) / n,
                    cost_c=sum(cost_c[m.name].values()) / n,
                    cost_b=sum(m.cost(r, "B") for r in rs) / n,
                    tokens=sum(m.cost(r, "A") for r in rs) / n,
                    score_vs_baseline=score_diff.get(m.name),
                    cost_c_vs_baseline=cost_diff.get(m.name),
                )
            )
    return out


def _ci(d: Difference | None, scale: float, digits: int) -> str:
    if d is None:
        return "-"
    est, low, high = (f"{v * scale:+.{digits}f}" for v in (d.estimate, d.low, d.high))
    return f"{est} [{low}, {high}]"


def format_results(results: Sequence[MethodResult], baseline: str = BASELINE) -> str:
    """Score in points (primary metric x 100), C and B seconds and tokens per question, and
    the differences to ``baseline`` with their 95% CIs."""
    lines = [
        f"  {'group':<16}{'method':<16}{'n':>6}{'score':>8}{'C s':>7}{'B s':>7}{'tokens':>8}"
        f"   {'score - ' + baseline:<24}{'C s - ' + baseline}"
    ]
    for r in results:
        lines.append(
            f"  {r.group:<16}{r.method:<16}{r.n:>6}{r.score * 100:>8.1f}{r.cost_c:>7.2f}"
            f"{r.cost_b:>7.2f}{r.tokens:>8.0f}   {_ci(r.score_vs_baseline, 100, 1):<24}"
            f"{_ci(r.cost_c_vs_baseline, 1, 2)}"
        )
    return "\n".join(lines)


def _diff_cells(d: Difference | None, digits: int) -> list[str]:
    return [f"{v:.{digits}f}" for v in (d.estimate, d.low, d.high)] if d else ["", "", ""]


def write_results(results: Sequence[MethodResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "group", "method", "n", "score", "cost_c_s", "cost_b_s", "tokens",
                "score_diff", "score_low", "score_high", "cost_c_diff", "cost_c_low", "cost_c_high",
            ]
        )  # fmt: skip
        for r in results:
            s, c = r.score_vs_baseline, r.cost_c_vs_baseline
            w.writerow(
                [
                    r.group, r.method, r.n, f"{r.score:.6f}", f"{r.cost_c:.4f}",
                    f"{r.cost_b:.4f}", f"{r.tokens:.1f}",
                    *_diff_cells(s, 6), *_diff_cells(c, 4),
                ]
            )  # fmt: skip
