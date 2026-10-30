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


def _ocr(psm: int, cache_dir: Path | None):
    from qdu.ocr import TesseractOcr

    return TesseractOcr(psm=psm, cache_dir=cache_dir)


def cmd_answer(args: argparse.Namespace) -> int:
    from PIL import Image

    from qdu.models import create_model

    needs_ocr = args.pipeline in ("text", "hybrid")
    pipeline = build_pipeline(
        args.pipeline,
        text_model=create_model(args.text_model) if args.pipeline == "text" else None,
        vision_model=create_model(args.vision_model) if args.pipeline != "text" else None,
        ocr=_ocr(args.psm, None) if needs_ocr else None,
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
    print(f"Sampling {args.per_dataset} questions per dataset from {', '.join(names)}...")
    info = benchmark.build_sample(
        args.out,
        datasets=names,
        per_dataset=args.per_dataset,
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
        ocr=_ocr(args.psm, benchmark.DEFAULT_OCR_CACHE),
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
    print("\nANLS and oracle per dataset")
    print(benchmark.format_summary(result.by_dataset, "dataset"))
    print("\nANLS and oracle per question type")
    print(benchmark.format_summary(result.by_stratum))
    print(
        "\nWrote summary.json, metrics.csv, by_dataset.csv, by_stratum.csv, oracle_labels.csv"
        f" to {benchmark.report_dir(args.run_dir, args.cost)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdu", description="Query-aware document understanding")
    sub = parser.add_subparsers(dest="command", required=True)

    answer = sub.add_parser("answer", help="answer a question about a document page image")
    answer.add_argument("--image", type=Path, required=True)
    answer.add_argument("--question", required=True)
    answer.add_argument("--pipeline", choices=PIPELINE_NAMES, required=True)
    answer.add_argument("--text-model", default="fake", help="model spec for the text pipeline")
    answer.add_argument("--vision-model", default="fake", help="model spec for vision/hybrid")
    answer.add_argument("--psm", type=int, default=3)
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

    bench = sub.add_parser("benchmark", help="evaluate the pipelines")
    bench_sub = bench.add_subparsers(dest="benchmark_command", required=True)

    sample = bench_sub.add_parser("sample", help="build a fixed question sample (manifest)")
    sample.add_argument("--out", type=Path, default=benchmark.DEFAULT_MANIFEST)
    sample.add_argument("--datasets", default=",".join(benchmark.PILOT_DATASETS))
    sample.add_argument("--per-dataset", type=int, default=100)
    sample.add_argument(
        "--split", choices=("train", "val", "test", "all"), default="train", help="router split"
    )
    sample.add_argument("--min-per-stratum", type=int, default=5)
    sample.add_argument("--seed", type=int, default=42)
    sample.set_defaults(func=cmd_sample)
    run = bench_sub.add_parser("run", help="run pipelines on the sample (resumable)")
    run.add_argument("--text-model", required=True)
    run.add_argument("--vision-model", required=True)
    run.add_argument("--pipelines", default=",".join(PIPELINE_NAMES))
    run.add_argument("--manifest", type=Path, default=benchmark.DEFAULT_MANIFEST)
    run.add_argument("--limit", type=int, help="only the first N questions")
    run.add_argument("--psm", type=int, default=3)
    run.add_argument("--name", help="run name, default <text-model>__<vision-model>")
    run.add_argument("--runs-dir", type=Path, default=benchmark.DEFAULT_RUNS_DIR)
    run.set_defaults(func=cmd_run)

    rep = bench_sub.add_parser("report", help="summarize a run")
    rep.add_argument("run_dir", type=Path)
    rep.add_argument("--tolerance", type=float, default=0.0, help="ANLS tie tolerance")
    rep.add_argument(
        "--cost",
        choices=("latency", "tokens"),
        default="latency",
        help="cost used to break oracle ties (tokens: written to <run>/cost-tokens/)",
    )
    rep.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles may default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
