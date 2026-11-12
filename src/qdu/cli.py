"""Command-line entry point: ``uv run qdu --help``."""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from qdu import benchmark
from qdu.pipelines import PIPELINE_NAMES, PipelineInput, build_pipeline


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
    run_dir = args.runs_dir / _slug(args.name or f"{args.text_model}__{args.vision_model}")

    def on_row(row) -> None:
        status = f"ANLS {row.anls:.2f}" if row.error is None else f"ERROR {row.error}"
        print(f"  {row.question_id:<28} {row.pipeline:<7} {status}")

    stats = benchmark.run(
        run_dir,
        text_model=create_model(args.text_model),
        vision_model=create_model(args.vision_model),
        # vision alone needs no OCR engine (nor PaddleOCR installed)
        ocr=_ocr(args.ocr, args.psm, args.ocr_cache) if {"text", "hybrid"} & set(names) else None,
        pipelines=names,
        manifest=args.manifest,
        limit=args.limit,
        on_row=on_row,
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
    from qdu.router import build_labels, write_labels

    rows = build_labels(args.run_dir, args.manifest, tolerance=args.tolerance)
    out = args.out or args.run_dir / "labels.csv"
    write_labels(rows, out)
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
            rows, policies, baseline=args.baseline, resamples=args.resamples, seed=args.seed
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
    import csv

    from qdu.provenance import file_sha256, run_metadata
    from qdu.router import load_ocr, read_labels
    from qdu.router.baselines import train_ecodoc, train_question_only
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
    with (out / "val_grid.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([*labels, "score", "cost", "share_text", "share_vision", "share_hybrid"])
        for g in baseline.grid:
            shares = [f"{g.shares[p]:.4f}" for p in ("text", "vision", "hybrid")]
            w.writerow([*g.params, f"{g.score:.6f}", f"{g.cost:.4f}", *shares])
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


def _add_ocr_options(parser: argparse.ArgumentParser) -> None:
    from qdu.ocr import OCR_SPECS

    parser.add_argument(
        "--ocr", choices=OCR_SPECS, default="tesseract", help="OCR engine for text/hybrid"
    )
    parser.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")


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
    run.add_argument("--pipelines", default=",".join(PIPELINE_NAMES))
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
    lab.set_defaults(func=cmd_router_labels)
    ev = router_sub.add_parser(
        "evaluate", help="baselines and routing choices vs always-vision, with document-level CIs"
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
        default="always-vision",
        help="method the CIs compare against (always-text|vision|hybrid, oracle or a router)",
    )
    ev.add_argument("--resamples", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out", type=Path, help="write the results as CSV")
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
