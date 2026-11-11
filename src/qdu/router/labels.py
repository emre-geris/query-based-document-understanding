"""The label table: one row per question with every path's score and cost.

It is what router training and evaluation start from, derived from a finished run
(``predictions.jsonl``) and its manifest (router split per question), without calling a model:

- ``score_<path>``: the dataset's primary metric (qdu.evaluation.primary)
- ``model_s_<path>``: measured model latency; ``ocr_s``: the page's OCR time, paid by the
  text and hybrid paths (and by a router that computes OCR signals)
- ``cost_c_<path>``, ``cost_b_<path>``, ``tokens_<path>``: cost of the path under scenarios C
  (OCR per question), B (OCR shared by the page's questions) and A (input + output tokens)
- ``oracle_latency``, ``oracle_tokens``: the cheapest path within ``tolerance`` of the best
  score, ties broken by C seconds or by tokens (as in ``oracle_labels.csv`` of the report)

Only questions with a successful row for every path are included.
"""

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from qdu.data import read_manifest
from qdu.evaluation.metrics import dataset_of
from qdu.evaluation.oracle import DEFAULT_COST_ORDER, oracle_label
from qdu.evaluation.primary import primary_score
from qdu.runner import read_predictions

PATHS: tuple[str, ...] = DEFAULT_COST_ORDER
SCENARIOS = ("C", "B", "A")
_PER_PATH = ("score", "model_s", "cost_c", "cost_b", "tokens")


@dataclass(frozen=True)
class LabelRow:
    question_id: str
    doc_id: str
    dataset: str
    split: str
    stratum: str
    question: str
    questions_on_page: int
    ocr_s: float
    score: dict[str, float]
    model_s: dict[str, float]
    cost_c: dict[str, float]
    cost_b: dict[str, float]
    tokens: dict[str, float]
    oracle_latency: str
    oracle_tokens: str

    def cost(self, scenario: str) -> dict[str, float]:
        """Cost of each path under scenario C, B or A."""
        return {"C": self.cost_c, "B": self.cost_b, "A": self.tokens}[scenario]

    @property
    def ocr_share_s(self) -> float:
        """The page's OCR time divided among its questions (scenario B)."""
        return self.ocr_s / self.questions_on_page


def build_labels(
    run_dir: Path, manifest: Path, *, tolerance: float = 0.0, paths: Sequence[str] = PATHS
) -> list[LabelRow]:
    records = {r.question_id: r for r in read_manifest(manifest)}
    latest: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_predictions(run_dir / "predictions.jsonl"):
        if row["error"] is None:
            latest[row["question_id"]][row["pipeline"]] = row
    complete = {q: rs for q, rs in latest.items() if all(p in rs for p in paths)}
    missing = sorted(set(complete) - set(records))
    if missing:
        raise ValueError(f"{len(missing)} questions are not in {manifest}, e.g. {missing[:3]}")

    on_page: dict[str, int] = defaultdict(int)
    for rs in complete.values():
        on_page[rs[paths[0]]["doc_id"]] += 1

    out = []
    for qid in sorted(complete):
        rs = complete[qid]
        record = records[qid]
        first = rs[paths[0]]
        n = on_page[first["doc_id"]]
        ocr = next((r["ocr_latency_s"] for r in rs.values() if r.get("ocr_latency_s")), 0.0)
        uses_ocr = {p: rs[p].get("ocr_latency_s") is not None for p in paths}
        score = {
            p: primary_score(dataset_of(qid), rs[p]["prediction"], rs[p]["answers"]) for p in paths
        }
        model = {p: rs[p]["model_latency_s"] for p in paths}
        cost_c = {p: model[p] + (ocr if uses_ocr[p] else 0.0) for p in paths}
        cost_b = {p: model[p] + (ocr / n if uses_ocr[p] else 0.0) for p in paths}
        tokens = {p: float(rs[p]["input_tokens"] + rs[p]["output_tokens"]) for p in paths}
        out.append(
            LabelRow(
                question_id=qid,
                doc_id=first["doc_id"],
                dataset=dataset_of(qid),
                split=record.split,
                stratum=first["stratum"],
                question=first["question"],
                questions_on_page=n,
                ocr_s=ocr,
                score=score,
                model_s=model,
                cost_c=cost_c,
                cost_b=cost_b,
                tokens=tokens,
                oracle_latency=oracle_label(score, paths, tolerance, cost_c),
                oracle_tokens=oracle_label(score, paths, tolerance, tokens),
            )
        )
    return out


def _columns(paths: Sequence[str]) -> list[str]:
    flat = [f.name for f in fields(LabelRow) if f.name not in _PER_PATH]
    per_path = [f"{name}_{p}" for name in _PER_PATH for p in paths]
    i = flat.index("oracle_latency")
    return [*flat[:i], *per_path, *flat[i:]]


def write_labels(rows: Iterable[LabelRow], path: Path, paths: Sequence[str] = PATHS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_columns(paths), lineterminator="\n")
        w.writeheader()
        for row in rows:
            d = asdict(row)
            flat = {k: v for k, v in d.items() if k not in _PER_PATH}
            for name in _PER_PATH:
                for p in paths:
                    value = d[name][p]
                    flat[f"{name}_{p}"] = f"{value:.0f}" if name == "tokens" else f"{value:.6f}"
            flat["ocr_s"] = f"{row.ocr_s:.6f}"
            w.writerow(flat)


def read_labels(
    path: Path, *, splits: Iterable[str] | None = None, datasets: Iterable[str] | None = None
) -> list[LabelRow]:
    """Rows of a label table, optionally only some router splits and datasets."""
    keep_splits = set(splits) if splits is not None else None
    keep_datasets = set(datasets) if datasets is not None else None
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        paths = [
            c.removeprefix("score_") for c in reader.fieldnames or () if c.startswith("score_")
        ]
        rows = []
        for r in reader:
            if keep_splits is not None and r["split"] not in keep_splits:
                continue
            if keep_datasets is not None and r["dataset"] not in keep_datasets:
                continue
            per = {name: {p: float(r[f"{name}_{p}"]) for p in paths} for name in _PER_PATH}
            rows.append(
                LabelRow(
                    question_id=r["question_id"],
                    doc_id=r["doc_id"],
                    dataset=r["dataset"],
                    split=r["split"],
                    stratum=r["stratum"],
                    question=r["question"],
                    questions_on_page=int(r["questions_on_page"]),
                    ocr_s=float(r["ocr_s"]),
                    oracle_latency=r["oracle_latency"],
                    oracle_tokens=r["oracle_tokens"],
                    **per,
                )
            )
    return rows
