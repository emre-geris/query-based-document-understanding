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
- `notebooks/` holds GPU runbooks (Colab) that only call `qdu` commands; no logic lives in notebooks.
- `benchmarks/` holds committed question manifests written by `qdu benchmark sample`; never edit them by hand.
- `results/<run>/` holds GPU runs worth keeping (predictions, run_info, reports, a README with findings); they cannot be regenerated without a GPU, so they are committed. Reports inside are regenerated with `qdu benchmark report`.
- `runs/` (scratch benchmark outputs) and `data/` (caches) are gitignored.

## Conventions

- English identifiers, docstrings and docs.
- Pipelines talk to models only through `qdu.models.LanguageModel`. A new provider is a module in `src/qdu/models/` registered in `registry.py`; it must report token usage.
- Keep the answer instruction in `qdu.pipelines.prompts` identical across pipelines; score differences must come from the input modality.
- ANLS lives in `qdu.evaluation`; reuse it instead of re-implementing normalization.
- Every benchmark output gets run metadata (`qdu.provenance`): git commit, versions, manifest hash.
- Report sample size honestly: DocVQA asks several questions per page, so samples take at most one question per document, and analyses over all questions (full labelling) resample by document, not by question.
- Runs are resumable JSONL (`qdu.runner`); never delete a predictions file to "start clean" after paid API calls — use a new `--name`.
- The `fake` model is for tests and wiring checks only; its numbers must never be reported.
- Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`), one logical change per commit.

## Environment notes

- Primary development is on Windows. Tesseract is expected on `PATH` or at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- Datasets are downloaded by `qdu data download` into the user's Hugging Face cache (`~/.cache/huggingface`), never into the repo.
- Keep the virtual environment outside OneDrive: set `UV_PROJECT_ENVIRONMENT` (e.g. `C:\Users\<you>\venvs\qdu`) before `uv sync`.
- No local GPU. Qwen3.5 runs behind a vLLM OpenAI-compatible server on a **Colab L4** (paid compute units) via `notebooks/gpu_pilot.ipynb`; benchmark runs execute on that machine against `localhost` so measured latency excludes network time. Everything else runs on a laptop.
- Qwen3.5 on vLLM 0.32 **failed to start on a T4** in our attempt (engine init error on compute capability 7.5, 30 Oct 2026; root cause not isolated), so we use an L4.
- Qwen3.5 thinks by default. Every request must disable it (`chat_template_kwargs: {enable_thinking: false}`); otherwise answers, latency and token cost are all distorted.
- Stop the vLLM server with `pkill -f "^/usr/bin/python3 /usr/local/bin/vllm"`; an unanchored `pkill -f "vllm serve"` also matches and kills the shell that runs it.

## Datasets

Each source uses its **labelled evaluation split**, not its training split: official training splits are standard VLM training data, so their questions would overstate every path. Router splits are cut from these questions **by document** with a salted hash (`qdu.data.splits`, 60/20/20), so a document never appears in two splits. Ids are dataset-prefixed (`docvqa/49153`).

| Name | Source (HF) | Split used | Size | Role | Licence |
|---|---|---|---|---|---|
| `docvqa` | `lmms-lab/DocVQA` / DocVQA | validation | 5,349 q, ~1.1 GB | router train/val/test | RRC DocVQA terms (research) |
| `infographicvqa` | `lmms-lab/DocVQA` / InfographicVQA | validation | 2,801 q, ~0.3 GB | router train/val/test | RRC InfographicVQA terms (research) |
| `chartqa` | `HuggingFaceM4/ChartQA` | test | 2,500 q, ~0.1 GB | router train/val/test | GPL-3.0 |
| `tatdqa` | `next-tat/TAT-DQA` (docs zip with page PNGs + gold JSON) | test | 1,655 q on 291 pages (8 cross-page q skipped) | unseen-data test only | CC BY 4.0 |
| DUDE | — | — | — | optional extra test | — |

## Roadmap

Source of truth for what to build next. Design follows the MFK420 pre-presentation (Oct 2026); deviations are allowed when they improve the system, but record them under *Decisions*. Mark items `[x]` in the same commit that completes them. Owners: **E** Emre (router & ML), **H** Hayrunnisa (data & signals), **R** Ramazan (model infra & demo). Weeks refer to the 14-week plan (H1 = 5 Oct).

**Ordering rule.** Phases group work by goal and deadline, not strict sequence. Work that only pays off if the pilot shows a usable gap (Phase 3–4: signals, router, full labelling) does not start before the go/no-go on 30 Oct. Small items needed in either outcome (data, splits, tooling) may be pulled forward; move such an item into the phase where it is actually needed.

**Target system.** Per question: OCR + layout → signals S1–S4 (no LM call, < 50 ms) → router picks the path maximizing `predicted_quality − λ·cost` → Qwen3.5 answers via text / vision / hybrid (same model, one call) → answer, ANLS, latency, tokens logged.

### Phase 0 — Foundation (done)
- [x] Repo, uv, ruff, pre-commit, pytest, CI
- [x] ANLS metric; routing-oracle analysis (ceiling, headroom, labels)
- [x] DocVQA loader, document-level stratified sampling, manifests
- [x] Tesseract OCR with cache; provider-independent `LanguageModel`; fake model
- [x] Text / vision / hybrid pipelines with a shared answer instruction
- [x] Resumable runner, provenance, `qdu benchmark sample|run|report`, `qdu answer`

### Phase 1 — Pilot & go/no-go (H3–H4, decision **30 Oct**)
- [x] R · vLLM server for **Qwen3.5-4B** on a Colab L4 (bf16, thinking off, caches off); starts in ~7 min. Qwen3.5-9B only if the pilot justifies it
- [x] R/E · `openai_compat` provider in `qdu.models` (vLLM endpoint, thinking disabled, token usage, timeouts, retries)
- [x] R/E · GPU runbook `notebooks/gpu_pilot.ipynb` (Colab L4/A100; a T4 failed): vLLM + Qwen3.5-4B, data, smoke test, run on `MANIFEST`, report
- [x] H · Dataset registry (`qdu.data.sources`): DocVQA, InfographicVQA, ChartQA on the same `QARecord`; `qdu data download`
- [x] H · Document-level train / val / test splits (hash-based, `qdu.data.splits`); prerequisite for drawing the pilot from the router-train split
- [x] H · Download the pilot datasets (`qdu data download`): DocVQA 5,349 q / 1,286 docs, InfographicVQA 2,801 / 500, ChartQA 2,500 / 1,509
- [x] H · Pilot manifest: 300 questions (100 per dataset, one per document, router-train split) → `benchmarks/pilot_300.csv`
- [x] E · Cost = OCR time + model latency (+ tokens) per path, reported side by side with ANLS (`metrics.csv`)
- [x] Ortak · Run 3 paths on the pilot (Colab L4, 30 Oct): `results/pilot-qwen3.5-4b/` with README
- [x] E · Oracle ties broken by measured per-question cost; reports for latency and token cost
- [x] E · Pilot audit (bootstrap CIs, single-path wins, image-size effect); fixes before the full run: oracle by each dataset's primary metric (`qdu.evaluation.primary`), list answers joined, image cap = model limit, vLLM flags recorded (`QDU_SERVER_ARGS`), `sample --all-questions`
- [ ] Ortak · Go/no-go (team + advisor). Pilot evidence (primary metric): oracle +4.3 over always-vision at equal latency, 95% CI [+2.2, +6.5]; +2.6 over always-hybrid [+1.0, +4.4]; vision and hybrid statistically tied (+1.7 [-0.6, +4.1]). Only 20/300 questions have a single best path (hybrid 10, vision 7, text 3), so much of the value is cost saving on easy questions; ChartQA has almost no accuracy headroom (+1.0). Recommendation: **go**, judged against always-vision

### Phase 2 — Data & labels (H4–H7)
- [x] H · Download TAT-DQA test files (`tatdqa_docs_test.zip`, `tatdqa_dataset_test_gold.json`) into the HF cache
- [x] H · TAT-DQA loader (`qdu.data.tatdqa`): each question mapped to its evidence page from `block_mapping`; cross-page questions skipped; router split fixed to `test`
- [ ] R · Batch labelling: every question of DocVQA, InfographicVQA, ChartQA (all splits: test labels are needed to score baselines) and TAT-DQA × 3 paths on a Colab L4; manifests from `qdu benchmark sample --all-questions --split all`, new `--name` per run, resumable, cost logged. Pilot rate ≈ 26 calls/min at 4 MP, so ~12k questions ≈ 36k calls ≈ 25 h of L4 or more with the model-native image cap; start with a smoke test on the largest pages
- [ ] E · Label table: per question and path → primary score, cost; oracle label at chosen tolerance

### Phase 3 — Signals (H5–H7)
- [ ] H · Compare Tesseract vs PP-OCRv5 on the 300 pilot pages (answer coverage in OCR text, time); pilot text ANLS is low partly because of OCR quality
- [ ] H · OCR with word boxes + confidences: Tesseract 5 (`image_to_data`, cheap baseline) and a PaddleOCR 3.x / PP-OCRv5 backend; choose by accuracy vs. time, since OCR time counts as cost
- [ ] H · Layout detection with PP-DocLayout (PaddleOCR 3.x; S/M variant first for speed, V3 if accuracy needs it): text / table / figure regions per page, cached; its time is reported next to the < 50 ms router decision time
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
- [ ] E · Paired bootstrap 95% CIs (router vs each baseline), resampling documents, since full labelling has several questions per document
- [ ] E/H · Unseen-data test on TAT-DQA
- [ ] E · Success criteria table — revised after the pilot: the router must beat **always-vision** (more accurate at equal or lower cost), since always-vision is statistically tied with always-hybrid (pilot: -1.7 points, CI includes 0) at half the latency. Original: (cost ≥ 25% below always-hybrid with ≤ 1 ANLS point loss; S4 gain CI excludes 0; TAT-DQA: more accurate than always-text, cheaper than always-hybrid; decision < 50 ms)

### Phase 6 — Demo & deliverables (H9–H14)
- [ ] R · Gradio demo: upload page + question → chosen path, answer, latency (`qdu demo`)
- [ ] Ortak · Academic activity documented by **14 Dec** (TÜBİTAK 2209-A or paper)
- [ ] Ortak · Vize (16 Nov upload), final (14 Dec), report + demo (28 Dec), poster (5 Jan)

### Decisions
- Same model serves all three paths; the text path sends no image. Keeps differences attributable to input modality.
- Splits are by document. Samples (pilot) take at most one question per document; full labelling takes every question and resamples by document for confidence intervals.
- Use each dataset's labelled evaluation split (DocVQA/InfographicVQA validation, ChartQA test) because official training splits are standard VLM training data.
- Model: **Qwen3.5-4B** instead of the presentation's Qwen2.5-VL (Oct 2026). Newer (Mar 2026), natively multimodal, runs on one 24 GB L4 in bf16, stronger on OCR (OCRBench 85.0). Tell the advisor; cite the Qwen3.5 model card instead of the Qwen2.5-VL report.
- Tools: PP-DocLayout (PaddleOCR 3.x) instead of DocLayout-YOLO for layout; PP-OCRv5 as the stronger OCR backend next to Tesseract (Oct 2026).
- Each dataset is judged by its conventional metric (`qdu.evaluation.primary`): ANLS for DocVQA and InfographicVQA, relaxed accuracy (5% numeric tolerance, exact match for text) for ChartQA, F1 for TAT-DQA. Oracle labels use the same metric. ANLS, relaxed accuracy, EM and F1 are still reported for every dataset. Exact match and token F1 (TAT-QA style, numbers rounded to two decimals; `qdu.evaluation.em_f1`) are reported too and are the TAT-DQA metric; unlike the official script they do not score the predicted scale, because pipelines return one answer string.
- Images sent to the model are capped at Qwen3.5's own processor limit, 16,777,216 pixels (aspect ratio kept; `QDU_MAX_IMAGE_PIXELS`), for both vision and hybrid paths; OCR uses the full-resolution page. The pilot used 4 MP, which downscaled a quarter of DocVQA/InfographicVQA pages and may have favoured hybrid. The server needs `--max-model-len 32768`. Every run records GPU, vLLM version, launch flags (`QDU_SERVER_ARGS`) and request settings in `run_info.json`.
- vLLM runs with prefix caching and the multimodal processor cache **off** (`--no-enable-prefix-caching --mm-processor-cache-gb 0`). The smoke test showed hybrid calls 3x faster than vision calls on the same page despite more input tokens, because the image was cached by the preceding vision call; with caches on, measured latency is not a fair cost.
- Cost is reported two ways: latency (model + OCR seconds; hardware-specific, what self-hosting pays) and tokens (input + output; API-price proxy, OCR free). Conclusions that hold under only one measure must say so.
- Answers are normalised by `clean_answer` for every path (multi-line lists joined with commas, prefixes, quotes, final period, space thousands separators). After changing it, re-derive finished runs with `qdu benchmark rescore` instead of re-running the GPU.
- The pilot samples from the router-train split, so pilot labels can be reused for router training without leaking val/test documents.
