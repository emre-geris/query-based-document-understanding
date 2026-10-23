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
- No local GPU. Qwen2.5-VL runs behind a vLLM OpenAI-compatible server (Colab / Kaggle / university GPU); the code reaches it over HTTP, so everything else runs on a laptop.

## Roadmap

Source of truth for what to build next. Design follows the MFK420 pre-presentation (Oct 2026); deviations are allowed when they improve the system, but record them under *Decisions*. Mark items `[x]` in the same commit that completes them. Owners: **E** Emre (router & ML), **H** Hayrunnisa (data & signals), **R** Ramazan (model infra & demo). Weeks refer to the 14-week plan (H1 = 5 Oct).

**Target system.** Per question: OCR + layout → signals S1–S4 (no LM call, < 50 ms) → router picks the path maximizing `predicted_quality − λ·cost` → Qwen2.5-VL answers via text / vision / hybrid (same model, one call) → answer, ANLS, latency, tokens logged.

### Phase 0 — Foundation (done)
- [x] Repo, uv, ruff, pre-commit, pytest, CI
- [x] ANLS metric; routing-oracle analysis (ceiling, headroom, labels)
- [x] DocVQA loader, document-level stratified sampling, manifests
- [x] Tesseract OCR with cache; provider-independent `LanguageModel`; fake model
- [x] Text / vision / hybrid pipelines with a shared answer instruction
- [x] Resumable runner, provenance, `qdu benchmark sample|run|report`, `qdu answer`

### Phase 1 — Pilot & go/no-go (H3–H4, decision **30 Oct**)
- [ ] R · vLLM server for Qwen2.5-VL (3B first, 7B if memory allows) + setup notes in README
- [ ] R/E · `openai_compat` provider in `qdu.models` (vLLM endpoint, token usage, timeouts, retries)
- [ ] H · Dataset registry beyond DocVQA: InfographicVQA, ChartQA loaders on the same `QARecord`
- [ ] H · Pilot manifest: ~300 questions (100 per dataset, one per document) → `benchmarks/pilot_300.csv`
- [ ] E · Cost = OCR time + model latency (+ tokens) per path, reported side by side with ANLS
- [ ] Ortak · Run 3 paths on the pilot, `qdu benchmark report`, per-dataset breakdown
- [ ] Ortak · Go/no-go: oracle headroom over best single path and over "always hybrid" at equal cost. No usable gap → pivot to a measurement study on the same data

### Phase 2 — Data & labels (H4–H7)
- [ ] H · Document-level train / val / test splits for DocVQA, InfographicVQA, ChartQA subsets (no document in two splits)
- [ ] H · TAT-DQA loader, test-only (never used for training or λ selection)
- [ ] R · Batch labelling: every train/val question × 3 paths, resumable, cost logged
- [ ] E · Label table: per question and path → ANLS, cost; oracle label at chosen tolerance

### Phase 3 — Signals (H5–H7)
- [ ] H · OCR with word boxes + confidences (`image_to_data`); optional PaddleOCR backend
- [ ] H · Layout detection (DocLayout-YOLO): text / table / figure regions per page, cached
- [ ] H · S1 question features (wh-type, numeric/visual cue words, length, TF-IDF)
- [ ] H · S2 OCR-text features (word count, mean/low confidence share, numeric density)
- [ ] H · S3 page features (region area shares, table/figure counts, image size)
- [ ] H · **S4 evidence location**: match question terms to OCR words → where the likely evidence is (text / table / figure region) and how confidently it was read
- [ ] E · Feature assembly into one table keyed by question id; decision-time measurement

### Phase 4 — Router & baselines (H6–H10)
- [ ] E · Per-path quality regressors (Ridge, LightGBM) on S1–S4
- [ ] E · Decision rule `argmax_p q̂_p − λ·cost_p`; λ selected on validation (cost–quality curve)
- [ ] H · Baselines: always text / vision / hybrid, oracle, EcoDoc-style rules, question-only router (TF-IDF + logistic regression), text→vision confidence cascade
- [ ] E · Ablations: S1, S1+S2, S1+S2+S3, S1–S4
- [ ] E · `qdu router train|evaluate` CLI; router artefact versioned with its training manifest hash

### Phase 5 — Evaluation (H9–H11)
- [ ] E · Paired bootstrap 95% CIs (router vs each baseline)
- [ ] E/H · Unseen-data test on TAT-DQA
- [ ] E · Success criteria table (cost ≥ 25% below always-hybrid with ≤ 1 ANLS point loss; S4 gain CI excludes 0; TAT-DQA: more accurate than always-text, cheaper than always-hybrid; decision < 50 ms)

### Phase 6 — Demo & deliverables (H9–H14)
- [ ] R · Gradio demo: upload page + question → chosen path, answer, latency (`qdu demo`)
- [ ] Ortak · Academic activity documented by **14 Dec** (TÜBİTAK 2209-A or paper)
- [ ] Ortak · Vize (16 Nov upload), final (14 Dec), report + demo (28 Dec), poster (5 Jan)

### Decisions
- Same model (Qwen2.5-VL) serves all three paths; the text path sends no image. Keeps differences attributable to input modality.
- Samples take at most one question per document; splits are by document.
