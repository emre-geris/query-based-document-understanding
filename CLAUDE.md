# CLAUDE.md

Guidance for Claude (and other coding agents) working in this repository.

## Project

Capstone project: a document understanding system that routes each query to a **text**
(OCR + LLM), **vision** (VLM on the page image) or **hybrid** pipeline. The goal is accuracy
close to the best single pipeline at lower cost. Benchmark numbers feed the thesis and a
journal paper, so **reproducibility matters more than speed**.

## Commands

```bash
uv sync                    # install locked dependencies (Python 3.13)
uv run pytest              # tests
uv run ruff check --fix .  # lint
uv run ruff format .       # format
uv run qdu --help          # CLI: answer, benchmark sample/run/report
uv add <pkg>               # runtime dependency; uv add --dev <pkg> for tooling
```

Never use `pip install` or edit `uv.lock` by hand.

## Layout

- All code lives in the `qdu` package under `src/qdu/`, with tests in `tests/`. There are no standalone scripts; new entry points are `qdu` CLI subcommands in `cli.py`.
- `benchmarks/` holds committed question manifests written by `qdu benchmark sample`; never edit them by hand.
- `runs/` (benchmark outputs) and `data/` (caches) are gitignored.

## Conventions

- English identifiers, docstrings and docs.
- Pipelines talk to models only through `qdu.models.LanguageModel`. A new provider is a module in `src/qdu/models/` registered in `registry.py`; it must report token usage.
- Keep the answer instruction in `qdu.pipelines.prompts` identical across pipelines; score differences must come from the input modality.
- ANLS lives in `qdu.evaluation`; reuse it instead of re-implementing normalization.
- Every benchmark output gets run metadata (`qdu.provenance`): git commit, versions, manifest hash.
- Report sample size honestly: DocVQA asks several questions per page, so samples take at most one question per document.
- Runs are resumable JSONL (`qdu.runner`); never delete a predictions file to "start clean" after paid API calls — use a new `--name`.
- The `fake` model is for tests and wiring checks only; its numbers must never be reported.
- Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`), one logical change per commit.

## Environment notes

- Primary development is on Windows. Tesseract is expected on `PATH` or at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- DocVQA is downloaded from Hugging Face (`lmms-lab/DocVQA`) into the user's HF cache, outside the repo.
