"""Command-line entry point: ``uv run qdu --help``."""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from qdu import benchmark
from qdu.data.splits import DEFAULT_SCHEME, SPLIT_SCHEMES
from qdu.pipelines import ALL_PIPELINES, PIPELINE_NAMES, PipelineInput, build_pipeline


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def _ocr(spec: str, psm: int, cache_dir: Path | None):
    from qdu.ocr import create_ocr

    return create_ocr(spec, cache_dir, psm=psm)


def cmd_answer(args: argparse.Namespace) -> int:
    from PIL import Image

    from qdu.models import create_model

    needs_ocr = args.pipeline in ("text", "hybrid")
    pipeline = build_pipeline(
        args.pipeline,
        text_model=create_model(args.text_model) if args.pipeline == "text" else None,
        vision_model=create_model(args.vision_model) if args.pipeline != "text" else None,
        ocr=_ocr(args.ocr, args.psm, None) if needs_ocr else None,
    )
    with Image.open(args.image) as image:
        out = pipeline.run(PipelineInput(args.question, image.convert("RGB"), args.image.stem))
    print(json.dumps(asdict(out), ensure_ascii=False, indent=2) if args.json else out.answer)
    return 0


def _names(text: str) -> list[str]:
    return [n.strip() for n in text.split(",") if n.strip()]


def cmd_data_download(args: argparse.Namespace) -> int:
    from qdu.data import get_source, iter_source_records, load_source

    for name in _names(args.datasets):
        spec = get_source(name)
        print(f"{name}: {spec.hf_id} [{spec.hf_config or 'default'}] {spec.hf_split} ...")
        records = list(iter_source_records(name, load_source(name)))
        splits = {s: sum(r.split == s for r in records) for s in ("train", "val", "test")}
        docs = len({r.doc_id for r in records})
        print(f"  {len(records)} questions, {docs} documents, router splits {splits}")
    return 0


def cmd_data_tatdqa_reports(args: argparse.Namespace) -> int:
    from qdu.data.tatdqa import gold_file, report_of_documents, write_reports

    reports = report_of_documents(gold_file())
    write_reports(reports, args.out)
    print(f"{len(reports)} documents from {len(set(reports.values()))} reports -> {args.out}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    names = _names(args.datasets)
    split = None if args.split == "all" else args.split
    per_dataset = None if args.all_questions else args.per_dataset
    amount = "all" if per_dataset is None else per_dataset
    print(f"Taking {amount} questions per dataset from {', '.join(names)}...")
    info = benchmark.build_sample(
        args.out,
        datasets=names,
        per_dataset=per_dataset,
        split=split,
        min_per_stratum=args.min_per_stratum,
        seed=args.seed,
    )
    for name, source in info["sources"].items():
        print(
            f"\n{name}: {source['population_questions']} questions / "
            f"{source['population_documents']} documents in split {args.split}"
        )
        for stratum, n in source["sample_strata"].items():
            print(f"  {stratum:<32}{n:>5}")
    print(f"\n{info['questions']} questions -> {args.out} and {benchmark.info_path(args.out)}")
    return 0


def cmd_largest(args: argparse.Namespace) -> int:
    info = benchmark.build_largest(args.out, args.source, n=args.n)
    for doc, size in info["pages"].items():
        print(f"  {doc:<44}{size:>12}")
    print(f"\n{info['questions']} questions -> {args.out} and {benchmark.info_path(args.out)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from qdu.models import create_model

    names = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    unknown = [p for p in names if p not in ALL_PIPELINES]
    if unknown:
        print(f"unknown pipelines {unknown}; choose from {list(ALL_PIPELINES)}", file=sys.stderr)
        return 2
    if "crop" in names and args.layout is None:
        print("the crop pipeline needs --layout (a page-layout file)", file=sys.stderr)
        return 2
    run_dir = args.runs_dir / _slug(args.name or f"{args.text_model}__{args.vision_model}")
    needs_ocr = bool({"text", "hybrid", "crop"} & set(names))
    if not needs_ocr:
        ocr = None  # vision alone needs no OCR engine (nor PaddleOCR installed)
    elif args.stored_ocr is not None:
        from qdu.demo.examples import StoredOcr
        from qdu.router import load_ocr

        ocr = StoredOcr(load_ocr(args.stored_ocr))  # the labelled run's OCR, its L4 times
    else:
        ocr = _ocr(args.ocr, args.psm, args.ocr_cache)

    def on_row(row) -> None:
        status = f"ANLS {row.anls:.2f}" if row.error is None else f"ERROR {row.error}"
        print(f"  {row.question_id:<28} {row.pipeline:<7} {status}")

    stats = benchmark.run(
        run_dir,
        text_model=create_model(args.text_model),
        vision_model=create_model(args.vision_model),
        ocr=ocr,
        pipelines=names,
        manifest=args.manifest,
        limit=args.limit,
        on_row=on_row,
        layout=args.layout,
    )
    print(f"\ncompleted {stats.completed}, skipped {stats.skipped}, failed {stats.failed}")
    print(f"Run directory: {run_dir}")
    if stats.failed:
        print("Re-run the same command to retry failed calls.")
    return 1 if stats.failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    result = benchmark.report(args.run_dir, tolerance=args.tolerance, cost=args.cost)
    print(f"Quality and cost per question (oracle ties broken by {args.cost})")
    print(benchmark.format_metrics(result.metrics))
    print("\nPrimary score and oracle per dataset (ANLS; ChartQA: relaxed accuracy)")
    print(benchmark.format_summary(result.by_dataset, "dataset"))
    print("\nPrimary score and oracle per question type")
    print(benchmark.format_summary(result.by_stratum))
    print(
        "\nWrote summary.json, metrics.csv, by_dataset.csv, by_stratum.csv, oracle_labels.csv"
        f" to {benchmark.report_dir(args.run_dir, args.cost)}"
    )
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from qdu.audit import audit

    lines, problems = audit(args.run_dir, examples=args.examples)
    print("\n".join(lines))
    print()
    if not problems:
        print("All checks passed.")
        return 0
    print(f"{len(problems)} problems:")
    for problem in problems[:50]:
        print(f"  - {problem}")
    return 1


def cmd_rescore(args: argparse.Namespace) -> int:
    stats = benchmark.rescore(args.run_dir)
    print(
        f"{stats.rows} predictions re-derived from raw_output, {stats.changed} changed; "
        f"mean ANLS {stats.anls_before:.4f} -> {stats.anls_after:.4f}"
    )
    print("Run `qdu benchmark report` again to refresh the reports.")
    return 0


def cmd_ocr_compare(args: argparse.Namespace) -> int:
    from qdu.data import ImageStore, read_manifest
    from qdu.ocr import create_ocr
    from qdu.ocr.compare import (
        compare_engines,
        format_comparison,
        summarize_engines,
        write_comparison,
    )
    from qdu.provenance import file_sha256, run_metadata

    records = read_manifest(args.manifest)[: args.limit]
    specs = _names(args.engines)
    engines = {spec: create_ocr(spec, args.ocr_cache, psm=args.psm) for spec in specs}
    images = ImageStore()

    def on_result(r) -> None:
        mark = "yes" if r.answer_in_text else "no "
        print(f"  {r.question_id:<28} {r.engine:<14} answer {mark} {r.latency_s:6.2f} s")

    results = compare_engines(records, images.get, engines, on_result)
    summary = summarize_engines(results)
    info = run_metadata(
        engines={spec: engine.name for spec, engine in engines.items()},
        manifest=args.manifest.as_posix(),
        manifest_sha256=file_sha256(args.manifest),
        questions=len(records),
    )
    write_comparison(args.out, results, summary, info)
    print("\n" + format_comparison(summary))
    print(f"\nWrote pages.csv, summary.csv, run_info.json to {args.out}")
    return 0


def cmd_ocr_warm(args: argparse.Namespace) -> int:
    import time

    from qdu.data import ImageStore, read_manifest
    from qdu.ocr.warm import warm_cache

    records = read_manifest(args.manifest)[: args.limit]
    engine = _ocr(args.engine, args.psm, args.ocr_cache)
    pages = len({r.doc_id for r in records})
    print(f"{engine.name} on {getattr(engine, 'hardware', '?')}: {pages} pages -> {args.ocr_cache}")
    start, done = time.perf_counter(), 0

    def on_page(record, result) -> None:
        nonlocal done
        done += 1
        if not result.cached or done % 100 == 0:
            source = "cache" if result.cached else f"{result.latency_s:6.2f} s"
            elapsed = time.perf_counter() - start
            print(f"  {done:>5}/{pages} {record.doc_id:<40} {source}  [{elapsed / 60:.1f} min]")

    stats = warm_cache(records, ImageStore().get, engine, on_page)
    mean = stats.seconds / stats.extracted if stats.extracted else 0.0
    print(
        f"\n{stats.pages} pages: {stats.extracted} extracted ({mean:.2f} s/page), "
        f"{stats.cached} already cached"
    )
    return 0


DEFAULT_LABELS = Path("results/full-qwen3.5-4b/labels.csv")
SPLIT_CHOICES = ("train", "val", "test", "all")


def _splits(split: str) -> list[str] | None:
    return None if split == "all" else [split]


def cmd_router_labels(args: argparse.Namespace) -> int:
    from qdu.data.tatdqa import read_reports
    from qdu.router import build_labels, write_labels
    from qdu.router.labels import PATHS

    reports = read_reports(args.tatdqa_reports) if args.split_scheme != "v1" else None
    paths = [*PATHS, *_names(args.extra_paths)] if args.extra_paths else list(PATHS)
    rows = build_labels(
        args.run_dir,
        args.manifest,
        tolerance=args.tolerance,
        split_scheme=args.split_scheme,
        tatdqa_reports=reports,
        extra_runs=args.extra_run,
        paths=paths,
        chartqa_exact_years=args.chartqa_exact_years,
    )
    out = args.out or args.run_dir / "labels.csv"
    write_labels(rows, out, paths)
    by_split: dict[str, int] = {}
    for r in rows:
        by_split[r.split] = by_split.get(r.split, 0) + 1
    splits = ", ".join(f"{k} {v}" for k, v in sorted(by_split.items()))
    print(f"{len(rows)} questions ({splits}) -> {out}")
    return 0


def cmd_router_evaluate(args: argparse.Namespace) -> int:
    from qdu.router import read_labels, read_policy
    from qdu.router.evaluate import evaluate, format_results, write_results

    datasets = _names(args.datasets) if args.datasets else None
    rows = read_labels(args.labels, splits=_splits(args.split), datasets=datasets)
    if args.split in ("test", "all"):
        print(f"Note: split {args.split!r} includes test questions: for the final evaluation only.")
    try:
        policies = [read_policy(p) for p in args.choices]
        results = evaluate(
            rows,
            policies,
            baseline=args.baseline,
            resamples=args.resamples,
            seed=args.seed,
            units=_units(rows, args.tatdqa_reports),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Split {args.split}: {len(rows)} questions; score = primary metric x 100; "
        "95% CIs resample documents"
    )
    print(format_results(results, args.baseline))
    if args.out:
        write_results(results, args.out)
        print(f"\nWrote {args.out}")
    return 0


def cmd_router_features(args: argparse.Namespace) -> int:
    from qdu.router import EXTRACTORS, build_features, load_ocr, read_labels, router_inputs

    unknown = [e for e in _names(args.extractors) if e not in EXTRACTORS]
    if unknown:
        print(f"unknown extractors {unknown}; available: {sorted(EXTRACTORS)}", file=sys.stderr)
        return 2
    extractors = [EXTRACTORS[e]() for e in _names(args.extractors)]
    rows = read_labels(args.labels, splits=_splits(args.split))
    from qdu.layout import load_layout

    ocr = load_ocr(args.ocr) if any(e.needs_ocr for e in extractors) else None
    layout = load_layout(args.layout) if any(e.needs_layout for e in extractors) else None
    table = build_features(router_inputs(rows, ocr, layout), extractors)
    table.write(args.out)
    n_features = len(next(iter(table.features.values()), {}))
    print(f"{len(table.features)} questions x {n_features} features -> {args.out}")
    return 0


def cmd_router_train(args: argparse.Namespace) -> int:
    import joblib

    from qdu.layout import load_layout
    from qdu.provenance import file_sha256, run_metadata
    from qdu.router import EXTRACTORS, build_features, load_ocr, read_labels, route, router_inputs
    from qdu.router.train import train_router, write_choices, write_curve

    names = _names(args.extractors)
    unknown = [e for e in names if e not in EXTRACTORS]
    if unknown:
        print(f"unknown extractors {unknown}; available: {sorted(EXTRACTORS)}", file=sys.stderr)
        return 2
    extractors = [EXTRACTORS[e]() for e in names]
    train = read_labels(args.labels, splits=["train"])
    val = read_labels(args.labels, splits=["val"])
    target = val if args.split == "val" else read_labels(args.labels, splits=["test"])
    if args.split == "test":
        print("Note: writing test choices; λ and the model are fixed on train/val beforehand.")
    rows = {r.question_id: r for r in [*train, *val, *target]}.values()
    uses_ocr = any(e.needs_ocr for e in extractors)
    needs_layout = any(e.needs_layout for e in extractors)
    ocr = load_ocr(args.ocr) if uses_ocr else None
    layout = load_layout(args.layout) if needs_layout else None
    table = build_features(router_inputs(rows, ocr, layout), extractors)
    router_cost = table.router_cost(layout, {r.question_id: r.doc_id for r in rows})
    try:
        trained = train_router(
            train,
            val,
            table.features,
            router_cost,
            uses_ocr=uses_ocr,
            kind=args.model,
            tfidf=not args.no_tfidf,
            seed=args.seed,
            lambdas=[float(x) for x in _names(args.lambdas)],
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    name = args.name or f"{args.model}-{'-'.join(names)}{'' if args.no_tfidf else '-tfidf'}"
    out = args.out_dir / name
    print(f"Router {name}: {args.model}, signals {','.join(names)}, tfidf {not args.no_tfidf}")
    print(f"Validation curve (C cost budget = always-vision {trained.budget:.3f} s):")
    print(f"  {'lambda':>7}{'score':>8}{'C s':>8}   paths text/vision/hybrid")
    for c in trained.curve:
        mark = "  <- chosen" if c.lam == trained.lam else ""
        shares = "/".join(f"{c.shares[p]:.2f}" for p in ("text", "vision", "hybrid"))
        print(f"  {c.lam:>7.3f}{c.score * 100:>8.1f}{c.cost:>8.3f}   {shares}{mark}")
    if not trained.met_budget:
        print("  No λ stays within the budget: the most accurate point is kept.")

    target_ids = {r.question_id for r in target}
    predicted = trained.model.predict(target, table.features)
    cost_args = {k: {q: v for q, v in d.items() if q in target_ids} for k, d in router_cost.items()}
    policy = route(predicted, trained.cost, trained.lam, name=name, **cost_args)
    # the file stem is the method name in `qdu router evaluate`
    choices = out / (f"{name}.csv" if args.split == "val" else f"{name}-test.csv")
    write_choices(policy, target, choices)
    write_curve(trained.curve, out / "val_curve.csv")
    joblib.dump(trained.model, out / "model.joblib")
    meta = run_metadata(
        router=name,
        model=args.model,
        extractors=names,
        tfidf=not args.no_tfidf,
        seed=args.seed,
        lam=trained.lam,
        met_budget=trained.met_budget,
        budget_c_s=trained.budget,
        decision_cost=trained.cost,
        labels=str(args.labels),
        labels_sha256=file_sha256(args.labels),
    )
    (out / "router.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nλ = {trained.lam}; wrote {choices}, model and metadata in {out}")
    print(f"Evaluate: uv run qdu router evaluate --split {args.split} --choices {choices}")
    return 0


def cmd_router_baseline(args: argparse.Namespace) -> int:
    from qdu.provenance import file_sha256, run_metadata
    from qdu.router import load_ocr, read_labels
    from qdu.router.baselines import train_ecodoc, train_question_only, write_grid
    from qdu.router.train import write_choices

    train = read_labels(args.labels, splits=["train"])
    val = read_labels(args.labels, splits=["val"])
    target = val if args.split == "val" else read_labels(args.labels, splits=["test"])
    if args.split == "test":
        print("Note: writing test choices; thresholds are fixed on train/val beforehand.")
    try:
        if args.name == "question-only":
            baseline = train_question_only(train, val, seed=args.seed)
            labels = ("threshold",)
        else:
            ocr = load_ocr(args.ocr)
            missing = sorted({r.doc_id for r in target} - set(ocr))
            if missing:
                raise ValueError(f"no OCR for {len(missing)} pages, e.g. {missing[:3]}")
            baseline = train_ecodoc(train, val, ocr)
            labels = ("margin", "min_relevance")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = args.out_dir / args.name
    print(f"Baseline {args.name}: thresholds chosen on val")
    print(f"Validation grid (C cost budget = always-vision {baseline.budget:.3f} s):")
    print(f"  {'  '.join(f'{x:>13}' for x in labels)}{'score':>8}{'C s':>8}   text/vision/hybrid")
    for g in baseline.grid:
        mark = "  <- chosen" if g.params == baseline.params else ""
        shares = "/".join(f"{g.shares[p]:.2f}" for p in ("text", "vision", "hybrid"))
        params = "  ".join(f"{v:>13.2f}" for v in g.params)
        print(f"  {params}{g.score * 100:>8.1f}{g.cost:>8.3f}   {shares}{mark}")
    if not baseline.met_budget:
        print("  No setting stays within the budget: the most accurate one is kept.")

    policy = baseline.policy(target)
    choices = out / (f"{args.name}.csv" if args.split == "val" else f"{args.name}-test.csv")
    write_choices(policy, target, choices)
    write_grid(baseline.grid, labels, out / "val_grid.csv")
    meta = run_metadata(
        router=args.name,
        params=dict(zip(labels, baseline.params, strict=True)),
        met_budget=baseline.met_budget,
        budget_c_s=baseline.budget,
        seed=args.seed,
        labels=str(args.labels),
        labels_sha256=file_sha256(args.labels),
    )
    (out / "router.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    chosen = ", ".join(f"{k} = {v}" for k, v in zip(labels, baseline.params, strict=True))
    print(f"\n{chosen}; wrote {choices}")
    print(f"Evaluate: uv run qdu router evaluate --split {args.split} --choices {choices}")
    return 0


def _stage_extractors(stage1: Sequence[str], stage2: Sequence[str]) -> list | None:
    """The extractors of a two-stage router, or None (after printing why) if invalid."""
    from qdu.router import EXTRACTORS

    unknown = [e for e in (*stage1, *stage2) if e not in EXTRACTORS]
    if unknown:
        print(f"unknown extractors {unknown}; available: {sorted(EXTRACTORS)}", file=sys.stderr)
        return None
    first_ocr = [e for e in stage1 if EXTRACTORS[e]().needs_ocr]
    if first_ocr:
        print(f"stage 1 must not need OCR, but {first_ocr} do", file=sys.stderr)
        return None
    return [EXTRACTORS[e]() for e in dict.fromkeys((*stage1, *stage2))]


def _signal_table(rows, extractors, ocr_path: Path, layout_path: Path):
    """Feature table of ``rows`` and each question's page layout time."""
    from qdu.layout import load_layout
    from qdu.router import build_features, load_ocr, router_inputs

    ocr = load_ocr(ocr_path) if any(e.needs_ocr for e in extractors) else None
    layout = load_layout(layout_path) if any(e.needs_layout for e in extractors) else None
    table = build_features(router_inputs(rows, ocr, layout), extractors)
    layout_s = {r.question_id: layout[r.doc_id].latency_s if layout else 0.0 for r in rows}
    return table, layout_s


def _units(rows, reports_path: Path) -> dict[str, str]:
    """Bootstrap units: TAT-DQA pages resample by annual report."""
    from qdu.data.tatdqa import read_reports
    from qdu.router.evaluate import resampling_units

    if not any(r.dataset == "tatdqa" for r in rows):
        return {}
    return resampling_units(rows, read_reports(reports_path))


def _ci(d, scale: float, digits: int) -> str:
    if d is None:
        return "-"
    return (
        f"{d.estimate * scale:+.{digits}f} "
        f"[{d.low * scale:+.{digits}f}, {d.high * scale:+.{digits}f}]"
    )


def cmd_router_lodo(args: argparse.Namespace) -> int:
    import csv

    from qdu.router import read_labels
    from qdu.router.baselines import parse_budget
    from qdu.router.evaluate import evaluate
    from qdu.router.lodo import leave_one_out
    from qdu.router.two_stage import ROUTER_PATHS

    stage1, stage2, paths = _names(args.stage1), _names(args.stage2), _names(args.paths)
    try:
        budget = parse_budget(args.budget)
    except ValueError as exc:
        print(f"--budget must be valid: {exc}", file=sys.stderr)
        return 2
    extractors = _stage_extractors(stage1, stage2)
    if extractors is None:
        return 2
    train = read_labels(args.labels, splits=["train"])
    val = read_labels(args.labels, splits=["val"])
    if args.on == "test":
        print("Note: scoring held-out test questions: for the final evaluation only.")
        scored = read_labels(args.labels, splits=["test"])
    else:
        scored = [*train, *val]
    rows = list({r.question_id: r for r in [*train, *val, *scored]}.values())
    table, layout_s = _signal_table(rows, extractors, args.ocr, args.layout)
    try:
        held = leave_one_out(
            train,
            val,
            scored,
            table.features,
            table.seconds,
            layout_s,
            budget=budget,
            datasets=_names(args.datasets) if args.datasets else None,
            page_seconds=table.page_seconds,
            stage1=stage1,
            stage2=stage2,
            needs_layout={e.name: e.needs_layout for e in extractors},
            paths=paths,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Leave one dataset out ({args.on} questions of the held-out dataset); router fitted on "
        f"the others' train, thresholds on their val within {budget.name}. Score x 100, C s; "
        "95% CIs resample documents"
    )
    print(
        f"  {'held out':<16}{'method':<16}{'n':>6}{'score':>7}{'C s':>7}   "
        f"{'score - hybrid':<21}{'C s - hybrid':<21}score - vision"
    )
    units = _units(scored, args.tatdqa_reports)
    lines = []
    for h in held:
        res = {
            b: {
                m.method: m
                for m in evaluate(
                    h.rows,
                    [h.policy],
                    baseline=b,
                    resamples=args.resamples,
                    seed=args.seed,
                    units=units,
                )
                if m.group == "ALL"
            }
            for b in ("always-hybrid", "always-vision")
        }
        n = len(h.rows)
        shares = {p: sum(c == p for c in h.policy.choice.values()) / n for p in ROUTER_PATHS}
        for method in ("always-vision", "always-hybrid", h.policy.name):
            m, v = res["always-hybrid"][method], res["always-vision"][method]
            is_router = method == h.policy.name
            print(
                f"  {h.dataset:<16}{'router' if is_router else method:<16}{m.n:>6}"
                f"{m.score * 100:>7.1f}{m.cost_c:>7.2f}   "
                f"{_ci(m.score_vs_baseline, 100, 1):<21}{_ci(m.cost_c_vs_baseline, 1, 2):<21}"
                f"{_ci(v.score_vs_baseline, 100, 1)}"
            )
            line = {
                "held_out": h.dataset,
                "method": "router" if is_router else method,
                "n": m.n,
                "score": f"{m.score:.6f}",
                "cost_c_s": f"{m.cost_c:.4f}",
                "cost_b_s": f"{m.cost_b:.4f}",
                "tokens": f"{m.tokens:.1f}",
            }
            for b, r in res.items():
                short = b.removeprefix("always-")
                for key, d in (
                    ("score", r[method].score_vs_baseline),
                    ("cost_c", r[method].cost_c_vs_baseline),
                ):
                    for part in ("estimate", "low", "high"):
                        value = "" if d is None else f"{getattr(d, part):.6f}"
                        line[f"{key}_vs_{short}_{part}"] = value
            for p in ROUTER_PATHS:
                line[f"share_{p}"] = f"{shares[p]:.4f}" if is_router else ""
            line["params"] = " ".join(f"{t:g}" for t in h.router.params) if is_router else ""
            line["budget_c_s"] = f"{h.budget_s:.4f}"
            lines.append(line)
        print(
            f"  {'':<16}router paths: "
            + ", ".join(f"{p} {v:.0%}" for p, v in shares.items())
            + f"; budget on the others' val {h.budget_s:.2f} s"
        )
    out = args.out or Path(f"runs/lodo/lodo_{args.on}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(lines)
    print(f"Wrote {out}")
    return 0


def cmd_router_two_stage(args: argparse.Namespace) -> int:
    from qdu.provenance import file_sha256, run_metadata
    from qdu.router import read_labels
    from qdu.router.baselines import parse_budget, select_point, write_grid
    from qdu.router.train import write_choices
    from qdu.router.two_stage import MODEL_FILE, ROUTER_PATHS, save_two_stage, train_two_stage

    stage1, stage2 = _names(args.stage1), _names(args.stage2)
    paths = _names(args.paths)
    if not {"text", "vision"} <= set(paths) or not set(paths) <= set(ROUTER_PATHS):
        print(f"--paths must include text and vision, from {list(ROUTER_PATHS)}", file=sys.stderr)
        return 2
    three_way = "hybrid" in paths
    try:
        spec = parse_budget(args.budget)
    except ValueError as exc:
        print(f"--budget must be valid: {exc}", file=sys.stderr)
        return 2
    extractors = _stage_extractors(stage1, stage2)
    if extractors is None:
        return 2
    needs_layout = {e.name: e.needs_layout for e in extractors}
    train = read_labels(args.labels, splits=["train"])
    val = read_labels(args.labels, splits=["val"])
    target = val if args.split == "val" else read_labels(args.labels, splits=["test"])
    if args.split == "test":
        print("Note: writing test choices; thresholds are fixed on train/val beforehand.")
    rows = list({r.question_id: r for r in [*train, *val, *target]}.values())
    table, layout_s = _signal_table(rows, extractors, args.ocr, args.layout)
    budget_name, budget = spec.name, spec.on(val)
    try:
        router = train_two_stage(
            train,
            val,
            table.features,
            table.seconds,
            layout_s,
            stage1=stage1,
            stage2=stage2,
            needs_layout=needs_layout,
            paths=paths,
            seed=args.seed,
            budget=budget,
            page_seconds=table.page_seconds,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    name = args.name
    out = args.out_dir / name
    labels = (
        ("t1", "t2", "t3", "t4")
        if "vision-low" in paths
        else ("t1", "t2", "t3")
        if three_way
        else ("t1", "t2")
    )
    print(
        f"Router {name}: paths {'/'.join(paths)}; stage 1 {','.join(stage1)} (no OCR), "
        f"stage 2 {','.join(stage2)}"
        + (" + hybrid head" if three_way else "")
        + (" + detail head (stage 1)" if "vision-low" in paths else "")
    )
    print(f"Validation grid (C cost budget = {budget_name} {router.budget:.3f} s):")
    head = "".join(f"{k:>6}" for k in labels)
    print(f"  {head}{'score':>8}{'C s':>8}   text/vision/hybrid")

    def show(g, mark=""):
        values = "".join(f"{v:>6.2f}" for v in g.params)
        shares = "/".join(f"{g.shares[p]:.2f}" for p in ("text", "vision", "hybrid"))
        print(f"  {values}{g.score * 100:>8.1f}{g.cost:>8.3f}   {shares}{mark}")

    if three_way:  # hundreds of settings: the best within budget per t3 (full grid in the CSV)
        for t3 in sorted({g.params[2] for g in router.grid}):
            pool = [g for g in router.grid if g.params[2] == t3]
            best, within = select_point(pool, router.budget)
            mark = "  <- chosen" if best.params == router.params else ""
            show(best, mark + ("" if within else "  (over budget)"))
    else:
        for g in router.grid:
            show(g, "  <- chosen" if g.params == router.params else "")
    if not router.met_budget:
        print("  No setting stays within the budget: the most accurate one is kept.")

    policy = router.policy(
        target, table.features, table.seconds, layout_s, name=name, page_seconds=table.page_seconds
    )
    choices = out / (f"{name}.csv" if args.split == "val" else f"{name}-test.csv")
    write_choices(policy, target, choices)
    write_grid(router.grid, labels, out / "val_grid.csv")
    stage2_share = sum(policy.uses_ocr.values()) / len(target)
    shares = {p: sum(c == p for c in policy.choice.values()) / len(target) for p in ROUTER_PATHS}
    meta = run_metadata(
        router=name,
        paths=list(paths),
        stage1=stage1,
        stage2=stage2,
        **dict(zip(labels, router.params, strict=True)),
        shares=shares,
        met_budget=router.met_budget,
        budget=budget_name,
        budget_c_s=router.budget if router.budget != float("inf") else None,
        seed=args.seed,
        labels=str(args.labels),
        labels_sha256=file_sha256(args.labels),
    )
    meta["model"] = MODEL_FILE
    (out / "router.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    model_path = save_two_stage(router, out, meta)
    chosen = ", ".join(f"{k} = {v}" for k, v in zip(labels, router.params, strict=True))
    split_shares = ", ".join(f"{p} {v:.0%}" for p, v in shares.items())
    print(
        f"\n{chosen}; {stage2_share:.0%} of {args.split} questions reach stage 2 (pay OCR); "
        f"chosen paths: {split_shares}; wrote {choices} and {model_path}"
    )
    print(f"Evaluate: uv run qdu router evaluate --split {args.split} --choices {choices}")
    return 0


def cmd_router_embed(args: argparse.Namespace) -> int:
    import time

    from qdu.data import ImageStore, read_manifest
    from qdu.router.embed import (
        PAGE_MODEL,
        QUESTION_MODEL,
        compress_pages,
        compress_questions,
        device_hardware,
        embed_pages,
        embed_questions,
        load_encoder,
        load_page_embeddings,
    )

    try:
        clip = load_encoder(PAGE_MODEL, args.device)
        minilm = load_encoder(QUESTION_MODEL, args.device)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    records = read_manifest(args.manifest)
    pages = len({r.doc_id for r in records})
    hardware = device_hardware(args.device)
    work_pages = args.work_dir / "clip_pages.jsonl"
    work_questions = args.work_dir / "question_embeddings.jsonl"
    print(f"S6 {PAGE_MODEL} on {hardware}: {pages} pages -> {work_pages}")
    start, done = time.perf_counter(), 0

    def on_page(page) -> None:
        nonlocal done
        done += 1
        if done % 200 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"  {done:>5} pages, last {page.latency_s * 1000:.0f} ms  [{elapsed / 60:.1f} min]"
            )

    n = embed_pages(
        records,
        ImageStore().get,
        clip,
        work_pages,
        model=PAGE_MODEL,
        hardware=hardware,
        on_page=on_page,
    )
    total_pages = compress_pages(work_pages, args.pages_out)
    print(f"{n} pages embedded now; {total_pages} pages -> {args.pages_out}")
    print(f"S5 {QUESTION_MODEL} + CLIP text: {len(records)} questions -> {work_questions}")
    m = embed_questions(records, load_page_embeddings(work_pages), minilm, clip, work_questions)
    total_questions = compress_questions(work_questions, args.questions_out)
    print(f"{m} questions embedded now; {total_questions} questions -> {args.questions_out}")
    return 0 if (total_pages, total_questions) == (pages, len(records)) else 1


def cmd_router_ablation(args: argparse.Namespace) -> int:
    from qdu.layout import load_layout
    from qdu.router import EXTRACTORS, build_features, load_ocr, read_labels, router_inputs
    from qdu.router.ablation import (
        auc_comparisons,
        format_ablation,
        format_comparisons,
        run_ablation,
        with_dataset,
        write_ablation,
    )

    extractors = [EXTRACTORS[e]() for e in ("s1", "s2", "s3", "s4")]
    train = read_labels(args.labels, splits=["train"])
    val = read_labels(args.labels, splits=["val"])
    rows = [*train, *val]
    layout = load_layout(args.layout)
    table = build_features(router_inputs(rows, load_ocr(args.ocr), layout), extractors)
    results = run_ablation(
        train,
        val,
        with_dataset(rows, table.features),
        table.seconds,
        {r.question_id: layout[r.doc_id].latency_s for r in rows},
        {e.name: e.needs_ocr for e in extractors},
        {e.name: e.needs_layout for e in extractors},
        seed=args.seed,
    )
    print(
        f"Signal ablation on val ({len(val)} questions; models fitted on train). AUC: 0.5 = "
        "chance. text_ok: text as good as vision; within: mean AUC inside each dataset; right: "
        "text answer correct; beats: text or hybrid beats vision. score / C s / λ: learned "
        "router within the always-vision C budget."
    )
    print(format_ablation(results))
    write_ablation(results, args.out)
    comparisons = auc_comparisons(results, val, resamples=args.resamples, seed=args.seed)
    print("\ntext_ok AUC differences, paired bootstrap over documents:")
    print(format_comparisons(comparisons))
    ci_path = args.out.with_name(args.out.stem + "_ci.csv")
    with ci_path.open("w", newline="", encoding="utf-8") as f:
        import csv

        w = csv.writer(f, lineterminator="\n")
        w.writerow(["comparison", "method", "baseline", "auc_diff", "low", "high"])
        for c in comparisons:
            w.writerow([c.label, c.method, c.baseline, f"{c.estimate:.4f}", f"{c.low:.4f}",
                        f"{c.high:.4f}"])  # fmt: skip
    print(f"\nWrote {args.out} and {ci_path}")
    return 0


def cmd_router_learned_cascade(args: argparse.Namespace) -> int:
    from qdu.layout import load_layout
    from qdu.provenance import file_sha256, run_metadata
    from qdu.router import EXTRACTORS, build_features, load_ocr, read_labels, router_inputs
    from qdu.router.baselines import write_grid
    from qdu.router.learned_cascade import (
        answer_features,
        read_answer_inputs,
        train_learned_cascade,
    )
    from qdu.router.train import write_choices

    signals = [] if args.signals == "none" else _names(args.signals)
    unknown = [e for e in signals if e not in EXTRACTORS]
    if unknown:
        print(f"unknown extractors {unknown}; available: {sorted(EXTRACTORS)}", file=sys.stderr)
        return 2
    if "s6" in signals:  # its page-level CLIP time is not charged by the cascade's cost
        print(
            "learned-cascade does not support s6 (page-level cost); use two-stage", file=sys.stderr
        )
        return 2
    extractors = [EXTRACTORS[e]() for e in signals]
    train = read_labels(args.labels, splits=["train"])
    val = read_labels(args.labels, splits=["val"])
    target = val if args.split == "val" else read_labels(args.labels, splits=["test"])
    if args.split == "test":
        print("Note: writing test choices; the threshold is fixed on train/val beforehand.")
    rows = list({r.question_id: r for r in [*train, *val, *target]}.values())
    ocr = load_ocr(args.ocr)
    needs_layout = any(e.needs_layout for e in extractors)
    layout = load_layout(args.layout) if needs_layout else None
    answers, confidence, left_out = read_answer_inputs(args.labelled_run, args.logprob_run)
    features, ans_seconds = answer_features(rows, answers, confidence, ocr)
    if extractors:
        table = build_features(router_inputs(rows, ocr, layout), extractors)
        for q, f in table.features.items():
            features[q].update(f)
        seconds = {q: ans_seconds[q] + sum(table.seconds[q].values()) for q in ans_seconds}
    else:
        seconds = ans_seconds
    layout_s = {r.question_id: layout[r.doc_id].latency_s if layout else 0.0 for r in rows}
    try:
        cascade = train_learned_cascade(
            train,
            val,
            features,
            seconds,
            layout_s,
            signals=signals,
            signals_need_layout=needs_layout,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    name = args.name
    out = args.out_dir / name
    print(
        f"Router {name}: text first, escalate to vision; answer features + {signals or 'no'}"
        f" signals ({left_out} confidences left out: answer differs in the logprob run)"
    )
    print(f"Validation grid (C cost budget = always-vision {cascade.budget:.3f} s):")
    print(f"  {'threshold':>9}{'score':>8}{'C s':>8}   kept text/escalated")
    for g in cascade.grid:
        mark = "  <- chosen" if g.params[0] == cascade.threshold else ""
        print(
            f"  {g.params[0]:>9.2f}{g.score * 100:>8.1f}{g.cost:>8.3f}"
            f"   {g.shares['text']:.2f}/{g.shares['vision']:.2f}{mark}"
        )
    if not cascade.met_budget:
        print("  No threshold stays within the budget: the most accurate one is kept.")

    policy = cascade.policy(target, features, seconds, layout_s, name=name)
    choices = out / (f"{name}.csv" if args.split == "val" else f"{name}-test.csv")
    write_choices(policy, target, choices)
    write_grid(cascade.grid, ("threshold",), out / "val_grid.csv")
    meta = run_metadata(
        router=name,
        signals=signals,
        threshold=cascade.threshold,
        met_budget=cascade.met_budget,
        budget_c_s=cascade.budget,
        confidences_left_out=left_out,
        seed=args.seed,
        labels=str(args.labels),
        labels_sha256=file_sha256(args.labels),
    )
    (out / "router.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nthreshold = {cascade.threshold}; wrote {choices}")
    print(f"Evaluate: uv run qdu router evaluate --split {args.split} --choices {choices}")
    return 0


def cmd_router_variants(args: argparse.Namespace) -> int:
    import csv

    from qdu.layout import load_layout
    from qdu.router import EXTRACTORS, build_features, load_ocr, read_labels, router_inputs
    from qdu.router.variants import format_variants, run_variants

    names = ("s1", "s2", "s3", "s4")
    extractors = [EXTRACTORS[e]() for e in names]
    train = read_labels(args.labels, splits=["train"])
    val = read_labels(args.labels, splits=["val"])
    rows = [*train, *val]
    layout = load_layout(args.layout)
    table = build_features(router_inputs(rows, load_ocr(args.ocr), layout), extractors)
    variants = _names(args.variants)
    if "tuned" in variants:
        print("Tuning by grouped cross-validation on train: this takes a few minutes.")
    try:
        results = run_variants(
            train,
            val,
            table.features,
            table.seconds,
            {r.question_id: layout[r.doc_id].latency_s for r in rows},
            needs_layout={e.name: e.needs_layout for e in extractors},
            variants=variants,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "Two-stage router, training variants on val (fitted on train; thresholds as quantiles "
        "of each stage's outputs, chosen within the always-vision C budget). AUC text_ok: "
        "stage 2 separating text-as-good-as-vision; stage 1: share decided without OCR."
    )
    print(format_variants(results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "auc_text_ok", "score", "cost_c_s", "met_budget", "stage1_share",
                    "params"])  # fmt: skip
        for r in results:
            w.writerow([r.variant, f"{r.auc_text_ok:.4f}", f"{r.score:.6f}", f"{r.cost:.4f}",
                        int(r.met_budget), f"{r.stage1_share:.4f}",
                        json.dumps(r.params, sort_keys=True)])  # fmt: skip
    print(f"\nWrote {args.out}")
    return 0


def cmd_router_pack_ocr(args: argparse.Namespace) -> int:
    from qdu.router import pack_ocr, read_labels

    try:
        n = pack_ocr(read_labels(args.labels), args.ocr_cache, args.out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{n} pages -> {args.out}")
    return 0


def cmd_layout_detect(args: argparse.Namespace) -> int:
    import time

    from qdu.data import ImageStore, read_manifest
    from qdu.layout import PaddleLayout, compress, detect_pages

    records = read_manifest(args.manifest)
    pages = len({r.doc_id for r in records})
    detector = PaddleLayout(args.model)
    print(f"{args.model} on {detector.hardware}: {pages} pages -> {args.work}")
    start, done = time.perf_counter(), 0

    def on_page(page) -> None:
        nonlocal done
        done += 1
        if done % 100 == 0:
            elapsed = time.perf_counter() - start
            print(f"  {done:>5} pages, last {page.latency_s:.3f} s  [{elapsed / 60:.1f} min]")

    n = detect_pages(records, ImageStore().get, detector, args.work, on_page)
    total = compress(args.work, args.out)
    print(f"{n} pages detected now; {total} pages -> {args.out}")
    return 0 if total == pages else 1


def build_demo_service(args: argparse.Namespace):
    """The demo's router, OCR, layout and model from command-line options (``qdu demo``).

    Example pages use the run's stored OCR and layout; other pages get the live OCR engine and
    layout detector, built only when first needed. ``--model replay`` answers example questions
    with the labelled run's recorded outputs and other questions with the fake model."""
    from qdu.demo import DemoService, SharedOcr, StoredLayout, StoredOcr, demo_fake_model
    from qdu.layout import load_layout
    from qdu.models import create_model
    from qdu.router import load_ocr
    from qdu.router.live import LiveRouter, live_extractors
    from qdu.router.two_stage import load_two_stage

    router, meta = load_two_stage(args.router)
    names = {*router.stage1_extractors, *router.stage2_extractors}
    question_encoder = clip_encoder = None
    if names & {"s5", "s6"}:
        from qdu.router.embed import PAGE_MODEL, QUESTION_MODEL, load_encoder

        question_encoder = load_encoder(QUESTION_MODEL) if "s5" in names else None
        clip_encoder = load_encoder(PAGE_MODEL) if "s6" in names else None
    extractors = live_extractors(question_encoder, clip_encoder)

    def live_layout():
        from qdu.layout import PaddleLayout

        return PaddleLayout(args.layout_model)

    stored = args.run / "ocr.jsonl.gz", args.run / "layout.jsonl.gz"
    ocr = SharedOcr(
        StoredOcr(
            load_ocr(stored[0]) if stored[0].exists() else {}, lambda: _ocr(args.ocr, 3, None)
        )
    )
    layout = StoredLayout(load_layout(stored[1]) if stored[1].exists() else {}, live_layout)
    if args.model == "replay":  # recorded answers for examples, the router alone otherwise
        model = None
    else:
        model = demo_fake_model() if args.model == "fake" else create_model(args.model)
    return DemoService(LiveRouter(router, extractors, ocr, layout), model, ocr), meta


def cmd_demo_ask(args: argparse.Namespace) -> int:
    from PIL import Image

    from qdu.data import ImageStore
    from qdu.demo import describe, load_examples, load_recorded

    if bool(args.image) == bool(args.example):
        print("give either --image with --question, or --example", file=sys.stderr)
        return 2
    if args.image and not args.question:
        print("--image needs at least one --question", file=sys.stderr)
        return 2
    try:
        service, meta = build_demo_service(args)
    except (ImportError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Router {meta.get('router', args.router)} ({'/'.join(meta.get('paths', []))}), "
        f"model {args.model}, OCR {args.ocr} (stored for example pages)"
    )
    if args.model == "replay":
        print(
            "Note: example questions show the labelled run's recorded answers (real); other "
            "questions get fake-model answers, whose answers and times mean nothing."
        )
    elif args.model == "fake":
        print("Note: fake-model answers and times mean nothing; never report them.")
    if args.example:
        examples = {e.question_id: e for e in load_examples(args.run, args.manifest)}
        unknown = [q for q in args.example if q not in examples]
        if unknown:
            print(f"not val questions of the label table: {unknown}", file=sys.stderr)
            return 2
        recorded = (
            load_recorded(args.example, args.run, _extra_runs(args))
            if args.model == "replay"
            else {}
        )
        store = ImageStore()
        for qid in args.example:
            e = examples[qid]
            page = store.get(e.record).convert("RGB")
            answer = service.ask(
                e.record.question, page, e.doc_id, args.mode, args.compare,
                answers=e.record.answers, dataset=e.dataset, recorded=recorded.get(qid),
            )  # fmt: skip
            print(f"\n[{qid}, page {e.doc_id}, needs: {e.kind}]")
            print(describe(answer))
        return 0
    with Image.open(args.image) as image:
        page = image.convert("RGB")
    for question in args.question:
        answer = service.ask(question, page, args.image.stem, args.mode, args.compare)
        print()
        print(describe(answer))
    return 0


def _recorded_model(run: Path) -> str:
    """Which model, GPU and OCR produced a run's recorded answers, from its run_info.json."""
    try:
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "GPU koşusunun"
    model = (info.get("vision_model") or info.get("text_model") or "?").split("/")[-1]
    gpu = ", ".join(g.get("name", "?").removeprefix("NVIDIA ") for g in info.get("gpu") or [])
    return f"{model} ({gpu})" if gpu else model


def cmd_demo_serve(args: argparse.Namespace) -> int:
    from qdu.data import ImageStore
    from qdu.demo import load_examples, load_recorded, showcase
    from qdu.demo.session import DemoSession
    from qdu.demo.web import create_app, serve

    try:
        service, meta = build_demo_service(args)
        pages = showcase(load_examples(args.run, args.manifest), per_dataset=args.per_dataset)
    except (ImportError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ids = [e.question_id for page in pages for e in page]
    recorded = load_recorded(ids, args.run, _extra_runs(args)) if args.model == "replay" else {}
    store = ImageStore()
    model = {
        "replay": f"{_recorded_model(args.run)}, kayıtlı cevaplar; yeni sorular ve yüklenen "
        "sayfalar: yalnızca yönlendirme (model GPU'da çalışır)",
        "fake": "deneme modeli (cevaplar anlamsız)",
    }.get(args.model, args.model)
    status = f"Model: {model} · Yönlendirici: {meta.get('router', args.router)}"
    session = DemoSession(service, pages, lambda e: store.get(e.record).convert("RGB"), recorded)

    def warm() -> None:  # load each example page's dataset before anyone clicks it
        for page in pages:
            try:
                store.get(page[0].record)
            except Exception as exc:  # the click will show the error; keep warming the rest
                print(f"warning: could not preload {page[0].record.doc_id}: {exc}")

    import threading

    threading.Thread(target=warm, name="demo-warm", daemon=True).start()
    try:
        app = create_app(session, status)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{len(pages)} example pages; {status}")
    serve(app, host=args.host, port=args.port)
    return 0


def cmd_demo_examples(args: argparse.Namespace) -> int:
    from qdu.demo import load_examples, showcase

    pages = showcase(load_examples(args.run, args.manifest), per_dataset=args.per_dataset)
    print(f"{len(pages)} val pages whose questions need text / vision / hybrid:")
    for page in pages:
        print(f"\n{page[0].doc_id}  ({page[0].dataset})")
        for e in page:
            s = e.label.score
            print(
                f"  {e.kind:<12} {e.question_id:<24} {e.record.question}\n"
                f"  {'':<12} gold: {' | '.join(e.record.answers)}   "
                f"score text {s['text']:.2f} vision {s['vision']:.2f} hybrid {s['hybrid']:.2f}"
            )
    print("\nTry one: uv run qdu demo ask --compare --example <question id>")
    return 0


def _add_ocr_options(parser: argparse.ArgumentParser) -> None:
    from qdu.ocr import OCR_SPECS

    parser.add_argument(
        "--ocr", choices=OCR_SPECS, default="tesseract", help="OCR engine for text/hybrid"
    )
    parser.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")


def _extra_runs(args: argparse.Namespace) -> list[Path]:
    """Runs whose recorded answers add paths (default: the vision-low run, if present)."""
    if args.extra_run:
        return list(args.extra_run)
    default = Path("results/full-vision-low-qwen3.5-4b")
    return [default] if (default / "predictions.jsonl").exists() else []


def _add_demo_options(parser: argparse.ArgumentParser) -> None:
    from qdu.demo.service import MODES
    from qdu.ocr import OCR_SPECS

    parser.add_argument(
        "--router",
        type=Path,
        default=Path("runs/routers/two-stage"),
        help="folder of a router saved by qdu router two-stage (default: its default --name)",
    )
    parser.add_argument(
        "--model",
        default="replay",
        help="replay (example pages: the labelled run's answers; other questions: the router's "
        "choice only, no answer), fake, or "
        "openai:<model> for the vLLM server",
    )
    parser.add_argument("--run", type=Path, default=Path("results/full-qwen3.5-4b"))
    parser.add_argument(
        "--extra-run",
        type=Path,
        action="append",
        default=[],
        help="another run whose recorded answers add a path (default: the vision-low run)",
    )
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/full.csv"))
    parser.add_argument(
        "--ocr", choices=OCR_SPECS, default="rapid-mobile", help="OCR engine (CPU by default)"
    )
    parser.add_argument("--layout-model", default="PP-DocLayout-M")
    parser.add_argument("--mode", choices=MODES, default="auto", help="auto: the router chooses")
    parser.add_argument("--compare", action="store_true", help="also run the other two paths")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdu", description="Query-aware document understanding")
    sub = parser.add_subparsers(dest="command", required=True)

    answer = sub.add_parser("answer", help="answer a question about a document page image")
    answer.add_argument("--image", type=Path, required=True)
    answer.add_argument("--question", required=True)
    answer.add_argument("--pipeline", choices=PIPELINE_NAMES, required=True)
    answer.add_argument("--text-model", default="fake", help="model spec for the text pipeline")
    answer.add_argument("--vision-model", default="fake", help="model spec for vision/hybrid")
    _add_ocr_options(answer)
    answer.add_argument("--json", action="store_true", help="print the full pipeline output")
    answer.set_defaults(func=cmd_answer)

    data = sub.add_parser("data", help="download and inspect datasets")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    download = data_sub.add_parser("download", help="download datasets to the HF cache")
    download.add_argument(
        "--datasets",
        default="docvqa,infographicvqa,chartqa,tatdqa",
        help="comma-separated: docvqa, infographicvqa, chartqa, tatdqa (default: all)",
    )
    download.set_defaults(func=cmd_data_download)
    reports = data_sub.add_parser(
        "tatdqa-reports", help="TAT-DQA document -> annual report, for report-grouped splits"
    )
    reports.add_argument("--out", type=Path, default=Path("benchmarks/tatdqa_reports.csv"))
    reports.set_defaults(func=cmd_data_tatdqa_reports)

    ocr = sub.add_parser("ocr", help="compare OCR engines")
    ocr_sub = ocr.add_subparsers(dest="ocr_command", required=True)
    compare = ocr_sub.add_parser("compare", help="answer coverage and time per OCR engine")
    compare.add_argument("--engines", default="tesseract,paddle-mobile,paddle-server")
    compare.add_argument("--manifest", type=Path, default=benchmark.DEFAULT_MANIFEST)
    compare.add_argument("--limit", type=int, help="only the first N questions")
    compare.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")
    compare.add_argument("--out", type=Path, default=Path("runs/ocr-compare"))
    compare.add_argument("--ocr-cache", type=Path, default=benchmark.DEFAULT_OCR_CACHE)
    compare.set_defaults(func=cmd_ocr_compare)
    warm = ocr_sub.add_parser("warm", help="OCR every page of a manifest into the cache")
    warm.add_argument("--engine", default="paddle-mobile", help="OCR spec, e.g. paddle-mobile")
    warm.add_argument("--manifest", type=Path, default=benchmark.DEFAULT_MANIFEST)
    warm.add_argument("--limit", type=int, help="only the first N questions")
    warm.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")
    warm.add_argument("--ocr-cache", type=Path, default=benchmark.DEFAULT_OCR_CACHE)
    warm.set_defaults(func=cmd_ocr_warm)

    demo = sub.add_parser("demo", help="the demo: route, answer and cost one question")
    demo_sub = demo.add_subparsers(dest="demo_command", required=True)
    ask = demo_sub.add_parser(
        "ask", help="route and answer questions about one page image, printing every cost"
    )
    ask.add_argument("--image", type=Path, help="a page image of your own")
    ask.add_argument("--question", action="append", help="repeat for several questions")
    ask.add_argument(
        "--example", action="append", help="a val question id instead (repeatable); see examples"
    )
    _add_demo_options(ask)
    ask.set_defaults(func=cmd_demo_ask)
    sv = demo_sub.add_parser("serve", help="open the demo in the browser (needs the demo extra)")
    _add_demo_options(sv)
    sv.add_argument("--per-dataset", type=int, default=1, help="example pages per dataset")
    sv.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines")
    sv.add_argument("--port", type=int, default=7860)
    sv.set_defaults(func=cmd_demo_serve)
    ex = demo_sub.add_parser("examples", help="val pages whose questions need different paths")
    ex.add_argument("--per-dataset", type=int, default=2)
    ex.add_argument("--run", type=Path, default=Path("results/full-qwen3.5-4b"))
    ex.add_argument("--manifest", type=Path, default=Path("benchmarks/full.csv"))
    ex.set_defaults(func=cmd_demo_examples)

    layout = sub.add_parser("layout", help="layout regions per page (PP-DocLayout)")
    layout_sub = layout.add_subparsers(dest="layout_command", required=True)
    detect = layout_sub.add_parser(
        "detect", help="detect every page of a manifest (resumable), then write one gzip"
    )
    detect.add_argument("--manifest", type=Path, default=Path("benchmarks/full.csv"))
    detect.add_argument("--model", default="PP-DocLayout-M")
    detect.add_argument("--work", type=Path, default=Path("runs/layout.jsonl"))
    detect.add_argument("--out", type=Path, default=Path("results/full-qwen3.5-4b/layout.jsonl.gz"))
    detect.set_defaults(func=cmd_layout_detect)

    bench = sub.add_parser("benchmark", help="evaluate the pipelines")
    bench_sub = bench.add_subparsers(dest="benchmark_command", required=True)

    sample = bench_sub.add_parser("sample", help="build a fixed question sample (manifest)")
    sample.add_argument("--out", type=Path, default=benchmark.DEFAULT_MANIFEST)
    sample.add_argument("--datasets", default=",".join(benchmark.PILOT_DATASETS))
    sample.add_argument("--per-dataset", type=int, default=100)
    sample.add_argument(
        "--all-questions",
        action="store_true",
        help="every question in the split (several per document) instead of a sample",
    )
    sample.add_argument(
        "--split", choices=("train", "val", "test", "all"), default="train", help="router split"
    )
    sample.add_argument("--min-per-stratum", type=int, default=5)
    sample.add_argument("--seed", type=int, default=42)
    sample.set_defaults(func=cmd_sample)
    largest = bench_sub.add_parser(
        "largest", help="one question on each of the largest pages of a manifest (smoke test)"
    )
    largest.add_argument("--source", type=Path, required=True, help="manifest to pick from")
    largest.add_argument("--n", type=int, default=15)
    largest.add_argument("--out", type=Path, required=True)
    largest.set_defaults(func=cmd_largest)
    run = bench_sub.add_parser("run", help="run pipelines on the sample (resumable)")
    run.add_argument("--text-model", required=True)
    run.add_argument("--vision-model", required=True)
    run.add_argument(
        "--pipelines",
        default=",".join(PIPELINE_NAMES),
        help=f"comma-separated, from {', '.join(ALL_PIPELINES)}",
    )
    run.add_argument(
        "--layout", type=Path, help="page-layout file for the crop path (qdu layout detect)"
    )
    run.add_argument(
        "--stored-ocr",
        type=Path,
        help="use a packed OCR file (qdu router pack-ocr) instead of running OCR; its times "
        "are the ones measured when it was made",
    )
    run.add_argument("--manifest", type=Path, default=benchmark.DEFAULT_MANIFEST)
    run.add_argument("--limit", type=int, help="only the first N questions")
    _add_ocr_options(run)
    run.add_argument("--ocr-cache", type=Path, default=benchmark.DEFAULT_OCR_CACHE)
    run.add_argument("--name", help="run name, default <text-model>__<vision-model>")
    run.add_argument("--runs-dir", type=Path, default=benchmark.DEFAULT_RUNS_DIR)
    run.set_defaults(func=cmd_run)

    rep = bench_sub.add_parser("report", help="summarize a run")
    rep.add_argument("run_dir", type=Path)
    aud = bench_sub.add_parser(
        "audit", help="re-derive scores and costs from the raw rows and check their consistency"
    )
    aud.add_argument("run_dir", type=Path)
    aud.add_argument("--examples", type=int, default=3, help="example questions per dataset")
    aud.set_defaults(func=cmd_audit)
    resc = bench_sub.add_parser(
        "rescore", help="re-derive answers from raw model output with the current cleaning"
    )
    resc.add_argument("run_dir", type=Path)
    resc.set_defaults(func=cmd_rescore)

    rep.add_argument("--tolerance", type=float, default=0.0, help="score tie tolerance")
    rep.add_argument(
        "--cost",
        choices=("latency", "tokens"),
        default="latency",
        help="cost used to break oracle ties (tokens: written to <run>/cost-tokens/)",
    )
    rep.set_defaults(func=cmd_report)

    router = sub.add_parser("router", help="label table, signals and router evaluation")
    router_sub = router.add_subparsers(dest="router_command", required=True)
    lab = router_sub.add_parser("labels", help="label table of a finished run (question x path)")
    lab.add_argument("run_dir", type=Path)
    lab.add_argument("--manifest", type=Path, default=Path("benchmarks/full.csv"))
    lab.add_argument("--tolerance", type=float, default=0.0, help="score tie tolerance (oracle)")
    lab.add_argument("--out", type=Path, help="default <run_dir>/labels.csv")
    lab.add_argument(
        "--chartqa-exact-years",
        action="store_true",
        help="sensitivity check: ChartQA years must match exactly (the official relaxed "
        "accuracy gives them 5%% tolerance); write it to another --out",
    )
    lab.add_argument(
        "--extra-run",
        type=Path,
        action="append",
        default=[],
        help="another run whose paths are merged in (repeatable), e.g. a vision-low run",
    )
    lab.add_argument(
        "--extra-paths", help="paths to add from the extra runs, e.g. vision-low (comma-separated)"
    )
    lab.add_argument(
        "--split-scheme",
        choices=SPLIT_SCHEMES,
        default=DEFAULT_SCHEME,
        help="v2 (default): TAT-DQA split by annual report like the others; v1: TAT-DQA test-only",
    )
    lab.add_argument(
        "--tatdqa-reports",
        type=Path,
        default=Path("benchmarks/tatdqa_reports.csv"),
        help="TAT-DQA document -> report (qdu data tatdqa-reports)",
    )
    lab.set_defaults(func=cmd_router_labels)
    ev = router_sub.add_parser(
        "evaluate", help="baselines and routing choices vs always-hybrid, with document-level CIs"
    )
    ev.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ev.add_argument("--split", choices=SPLIT_CHOICES, default="val")
    ev.add_argument("--datasets", help="comma-separated, e.g. docvqa,chartqa (default: all)")
    ev.add_argument(
        "--choices",
        type=Path,
        nargs="*",
        default=[],
        help="CSV per router: question_id,pipeline[,uses_ocr,page_s,question_s]",
    )
    ev.add_argument(
        "--baseline",
        default="always-hybrid",
        help="method the CIs compare against (default always-hybrid, the primary baseline; "
        "always-text|vision, oracle or a router)",
    )
    ev.add_argument("--resamples", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out", type=Path, help="write the results as CSV")
    ev.add_argument(
        "--tatdqa-reports",
        type=Path,
        default=Path("benchmarks/tatdqa_reports.csv"),
        help="TAT-DQA document -> report: the bootstrap resamples TAT-DQA by report",
    )
    ev.set_defaults(func=cmd_router_evaluate)
    feat = router_sub.add_parser("features", help="run signal extractors into a feature table")
    feat.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    feat.add_argument("--split", choices=SPLIT_CHOICES, default="all")
    feat.add_argument("--extractors", default="s1,s2,s3,s4", help="comma-separated names")
    feat.add_argument("--ocr", type=Path, default=Path("results/full-qwen3.5-4b/ocr.jsonl.gz"))
    feat.add_argument(
        "--layout", type=Path, default=Path("results/full-qwen3.5-4b/layout.jsonl.gz")
    )
    feat.add_argument("--out", type=Path, required=True)
    feat.set_defaults(func=cmd_router_features)
    tr = router_sub.add_parser(
        "train", help="fit per-path quality models on train, choose λ on val, write choices"
    )
    tr.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    tr.add_argument("--extractors", default="s1,s2,s3,s4", help="comma-separated signal names")
    tr.add_argument("--model", choices=("ridge", "lgbm"), default="lgbm")
    tr.add_argument("--no-tfidf", action="store_true", help="leave out the question's TF-IDF")
    tr.add_argument("--lambdas", default="0,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.5")
    tr.add_argument(
        "--split", choices=("val", "test"), default="val", help="split to write choices for"
    )
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--name", help="router name (default: model, signals, tfidf)")
    tr.add_argument("--out-dir", type=Path, default=Path("runs/routers"))
    tr.add_argument("--ocr", type=Path, default=Path("results/full-qwen3.5-4b/ocr.jsonl.gz"))
    tr.add_argument("--layout", type=Path, default=Path("results/full-qwen3.5-4b/layout.jsonl.gz"))
    tr.set_defaults(func=cmd_router_train)
    bl = router_sub.add_parser(
        "baseline", help="literature baselines: question-only TF-IDF + LR, EcoDoc-style rules"
    )
    bl.add_argument("name", choices=("question-only", "ecodoc"))
    bl.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    bl.add_argument(
        "--split", choices=("val", "test"), default="val", help="split to write choices for"
    )
    bl.add_argument("--seed", type=int, default=0)
    bl.add_argument("--out-dir", type=Path, default=Path("runs/routers"))
    bl.add_argument("--ocr", type=Path, default=Path("results/full-qwen3.5-4b/ocr.jsonl.gz"))
    bl.set_defaults(func=cmd_router_baseline)
    ts = router_sub.add_parser(
        "two-stage",
        help="three-way router: OCR-free stage 1, OCR signals and a hybrid head only when unsure",
    )
    ts.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ts.add_argument("--stage1", default="s1,s3", help="OCR-free signals for stage 1")
    ts.add_argument("--stage2", default="s1,s2,s3,s4", help="signals for stage 2 (after OCR)")
    ts.add_argument(
        "--paths",
        default="text,vision,hybrid",
        help="paths the router may choose; 'text,vision' reproduces the earlier two-way router",
    )
    ts.add_argument(
        "--budget",
        default="hybrid:0.75",
        help="val C-cost budget for the thresholds: hybrid:0.75 (default: 25%% below "
        "always-hybrid, the success criterion), vision, hybrid, <path>:<fraction>, none or seconds",
    )
    ts.add_argument(
        "--split", choices=("val", "test"), default="val", help="split to write choices for"
    )
    ts.add_argument("--seed", type=int, default=0)
    ts.add_argument("--name", default="two-stage")
    ts.add_argument("--out-dir", type=Path, default=Path("runs/routers"))
    ts.add_argument("--ocr", type=Path, default=Path("results/full-qwen3.5-4b/ocr.jsonl.gz"))
    ts.add_argument("--layout", type=Path, default=Path("results/full-qwen3.5-4b/layout.jsonl.gz"))
    ts.set_defaults(func=cmd_router_two_stage)
    lo = router_sub.add_parser(
        "lodo", help="leave one dataset out: the two-stage router on a dataset it never saw"
    )
    lo.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    lo.add_argument("--stage1", default="s1,s3", help="OCR-free signals for stage 1")
    lo.add_argument("--stage2", default="s1,s2,s3,s4", help="signals for stage 2 (after OCR)")
    lo.add_argument("--paths", default="text,vision,hybrid")
    lo.add_argument("--budget", default="hybrid:0.75", help="as for two-stage")
    lo.add_argument("--datasets", help="held-out datasets, comma-separated (default: all)")
    lo.add_argument(
        "--on",
        choices=("dev", "test"),
        default="dev",
        help="score the held-out dataset's train+val questions (dev, default) or its test ones",
    )
    lo.add_argument("--resamples", type=int, default=1000)
    lo.add_argument("--seed", type=int, default=0)
    lo.add_argument("--out", type=Path, help="default runs/lodo/lodo_<on>.csv")
    lo.add_argument("--ocr", type=Path, default=Path("results/full-qwen3.5-4b/ocr.jsonl.gz"))
    lo.add_argument("--layout", type=Path, default=Path("results/full-qwen3.5-4b/layout.jsonl.gz"))
    lo.add_argument(
        "--tatdqa-reports",
        type=Path,
        default=Path("benchmarks/tatdqa_reports.csv"),
        help="TAT-DQA document -> report: the bootstrap resamples TAT-DQA by report",
    )
    lo.set_defaults(func=cmd_router_lodo)
    em = router_sub.add_parser(
        "embed",
        help="S5/S6 signals: CLIP page and MiniLM question embeddings (resumable; needs the "
        "'embed' extra and the page images)",
    )
    em.add_argument("--manifest", type=Path, default=Path("benchmarks/full.csv"))
    em.add_argument("--device", default="cpu", help="cpu, or cuda to time on the GPU")
    em.add_argument("--work-dir", type=Path, default=Path("runs/embed"))
    em.add_argument(
        "--pages-out", type=Path, default=Path("results/full-qwen3.5-4b/clip_pages.jsonl.gz")
    )
    em.add_argument(
        "--questions-out",
        type=Path,
        default=Path("results/full-qwen3.5-4b/question_embeddings.jsonl.gz"),
    )
    em.set_defaults(func=cmd_router_embed)
    ab = router_sub.add_parser(
        "ablation", help="what each signal group adds: AUC and learned-router result on val"
    )
    ab.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ab.add_argument("--seed", type=int, default=0)
    ab.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples for the CIs")
    ab.add_argument("--out", type=Path, default=Path("runs/ablation/ablation_val.csv"))
    ab.add_argument("--ocr", type=Path, default=Path("results/full-qwen3.5-4b/ocr.jsonl.gz"))
    ab.add_argument("--layout", type=Path, default=Path("results/full-qwen3.5-4b/layout.jsonl.gz"))
    ab.set_defaults(func=cmd_router_ablation)
    lc = router_sub.add_parser(
        "learned-cascade", help="text first; a classifier on the text answer decides escalation"
    )
    lc.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    lc.add_argument(
        "--signals", default="s1,s2,s3,s4", help="router signals next to the answer, or 'none'"
    )
    lc.add_argument(
        "--split", choices=("val", "test"), default="val", help="split to write choices for"
    )
    lc.add_argument("--seed", type=int, default=0)
    lc.add_argument("--name", default="learned-cascade")
    lc.add_argument("--out-dir", type=Path, default=Path("runs/routers"))
    lc.add_argument("--labelled-run", type=Path, default=Path("results/full-qwen3.5-4b"))
    lc.add_argument(
        "--logprob-run", type=Path, default=Path("results/full-text-logprobs-qwen3.5-4b")
    )
    lc.add_argument("--ocr", type=Path, default=Path("results/full-qwen3.5-4b/ocr.jsonl.gz"))
    lc.add_argument("--layout", type=Path, default=Path("results/full-qwen3.5-4b/layout.jsonl.gz"))
    lc.set_defaults(func=cmd_router_learned_cascade)
    va = router_sub.add_parser(
        "variants", help="two-stage router trained four ways: default, weighted, regress, tuned"
    )
    va.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    va.add_argument("--variants", default="default,weighted,regress,tuned")
    va.add_argument("--seed", type=int, default=0)
    va.add_argument("--out", type=Path, default=Path("runs/variants/variants_val.csv"))
    va.add_argument("--ocr", type=Path, default=Path("results/full-qwen3.5-4b/ocr.jsonl.gz"))
    va.add_argument("--layout", type=Path, default=Path("results/full-qwen3.5-4b/layout.jsonl.gz"))
    va.set_defaults(func=cmd_router_variants)
    pack = router_sub.add_parser(
        "pack-ocr", help="collect the run's OCR of every labelled page into one file"
    )
    pack.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    pack.add_argument(
        "--ocr-cache", type=Path, required=True, help="engine folder, e.g. <cache>/rapid-mobile-gpu"
    )
    pack.add_argument("--out", type=Path, default=Path("results/full-qwen3.5-4b/ocr.jsonl.gz"))
    pack.set_defaults(func=cmd_router_pack_ocr)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles may default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
