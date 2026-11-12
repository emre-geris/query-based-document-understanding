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
- `qdu.router` holds the label table, signal extractors, decision rule and evaluation; `docs/router.md` is the hand-off guide for router work (set-up, label-table columns, split and cost rules).
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

## Router work

The GPU work is done; the router is built on a laptop from committed files. Read `docs/router.md` before starting. The rules that keep its numbers valid:

- **Inputs:** `results/full-qwen3.5-4b/labels.csv` (`qdu.router.read_labels`), the run's OCR `results/full-qwen3.5-4b/ocr.jsonl.gz` (`qdu.router.load_ocr`), the page layouts `results/full-qwen3.5-4b/layout.jsonl.gz` (`qdu.layout.load_layout`) and, for the cascade, the text-path confidences in `results/full-text-logprobs-qwen3.5-4b`. Never re-OCR pages on a laptop for signals: the text would differ from what the paths saw and the time from what the label table charges. Page images come from `uv run qdu data download`.
- **Splits:** fit on `train`; choose features, models and λ on `val`; run `test` once at the end. TAT-DQA is test-only (unseen data). Never re-split by question.
- **Signals** see only the question, page image and OCR (`RouterInput`), never gold answers, a path's answer or score, and not the dataset name (a deployed router does not know it; allowed only in a labelled ablation). Each signal is a `FeatureExtractor` registered in `qdu.router.EXTRACTORS`, declares `needs_ocr`, and is timed per question (decision < 50 ms).
- **Cost:** a router pays for itself. A router's choices CSV declares per question `uses_ocr`, `page_s` (layout etc.) and `question_s`; `qdu router evaluate` charges OCR to every question whose signals used it, even when vision is chosen.
- **Evaluation** goes through `qdu router evaluate` (baselines, scenarios C/B/A, paired bootstrap over documents against always-vision or `--baseline`); never report a router number computed another way.

## Environment notes

- Primary development is on Windows. Tesseract is expected on `PATH` or at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- Datasets are downloaded by `qdu data download` into the user's Hugging Face cache (`~/.cache/huggingface`), never into the repo.
- Keep the virtual environment outside OneDrive: set `UV_PROJECT_ENVIRONMENT` (e.g. `C:\Users\<you>\venvs\qdu`) before `uv sync`.
- No local GPU. Qwen3.5 runs behind a vLLM OpenAI-compatible server on a **Colab L4** (paid compute units) via `notebooks/gpu_run.ipynb`; benchmark runs execute on that machine against `localhost` so measured latency excludes network time. OCR (PP-OCRv5 via RapidOCR, `--ocr rapid-mobile-gpu`) runs on the same GPU, with vLLM limited to 65% of its memory and `--max-num-seqs 16` (at 80% OCR and then vLLM ran out of memory in the full run; at 65% vLLM's default 256 sequences need more Mamba cache blocks than fit, and calls are sequential anyway); `qdu ocr warm` fills the OCR cache before the model calls. Outputs go to Google Drive, since a Colab disk is lost on disconnect. Everything else runs on a laptop.
- PaddleOCR needs **paddlepaddle 3.2.2** (pinned in the `paddle` extra): with 3.3.x the oneDNN CPU backend crashes on PP-OCRv5 (Windows and Linux), and without oneDNN OCR is 5-8x slower (Colab: 21-32 s vs 2.7-3.0 s per DocVQA page). Measured OCR time is a cost, so never run it with oneDNN off (`QDU_PADDLE_MKLDNN=0`, engine name `-nomkldnn`) in a reported run.
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

Source of truth for what to build next. **Where things stand (11 Nov 2026):** Phases 0–3 are done: label table, the run's OCR and page layouts in single files, signals S1–S4 (`qdu router features`), cost accounting, bootstrap, the cascade baseline and `qdu router evaluate`. Next: the Phase 4 quality models, λ, the two-stage router and the remaining baselines. Open items are the unchecked boxes below. Design follows the MFK420 pre-presentation (Oct 2026); deviations are allowed when they improve the system, but record them under *Decisions*. Mark items `[x]` in the same commit that completes them. Weeks refer to the 14-week plan (H1 = 5 Oct).

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
- [x] vLLM server for **Qwen3.5-4B** on a Colab L4 (bf16, thinking off, caches off); starts in ~7 min. Qwen3.5-9B only if the pilot justifies it
- [x] `openai_compat` provider in `qdu.models` (vLLM endpoint, thinking disabled, token usage, timeouts, retries)
- [x] GPU runbook `notebooks/gpu_run.ipynb` (Colab L4/A100; a T4 failed): vLLM + Qwen3.5-4B, data, smoke test, run on `MANIFEST`, report
- [x] Dataset registry (`qdu.data.sources`): DocVQA, InfographicVQA, ChartQA on the same `QARecord`; `qdu data download`
- [x] Document-level train / val / test splits (hash-based, `qdu.data.splits`); prerequisite for drawing the pilot from the router-train split
- [x] Download the pilot datasets (`qdu data download`): DocVQA 5,349 q / 1,286 docs, InfographicVQA 2,801 / 500, ChartQA 2,500 / 1,509
- [x] Pilot manifest: 300 questions (100 per dataset, one per document, router-train split) → `benchmarks/pilot_300.csv`
- [x] Cost = OCR time + model latency (+ tokens) per path, reported side by side with ANLS (`metrics.csv`)
- [x] Run 3 paths on the pilot (Colab L4, 30 Oct): `results/pilot-qwen3.5-4b/` with README
- [x] Oracle ties broken by measured per-question cost; reports for latency and token cost
- [x] Pilot audit (bootstrap CIs, single-path wins, image-size effect); fixes before the full run: oracle by each dataset's primary metric (`qdu.evaluation.primary`), list answers joined, image cap = model limit, vLLM flags recorded (`QDU_SERVER_ARGS`), `sample --all-questions`
- [x] Go/no-go: **go** (30 Oct 2026). The project lead (Emre) decides; the advisor follows progress. Pilot evidence (primary metric): oracle +4.3 over always-vision at equal latency, 95% CI [+2.2, +6.5]; +2.6 over always-hybrid [+1.0, +4.4]; vision and hybrid statistically tied (+1.7 [-0.6, +4.1]). Only 20/300 questions have a single best path (hybrid 10, vision 7, text 3), so much of the value is cost saving on easy questions; ChartQA has almost no accuracy headroom (+1.0). Success is judged against always-vision

### Phase 2 — Data & labels (H4–H7)
- [x] Download TAT-DQA test files (`tatdqa_docs_test.zip`, `tatdqa_dataset_test_gold.json`) into the HF cache
- [x] TAT-DQA loader (`qdu.data.tatdqa`): each question mapped to its evidence page from `block_mapping`; cross-page questions skipped; router split fixed to `test`
- [x] Full labelling manifest `benchmarks/full.csv`: every question of the four datasets, all router splits (test labels are needed to score baselines): 12,305 questions on 3,586 pages; smoke manifest of the 15 largest pages `benchmarks/smoke_largest.csv` (`qdu benchmark largest`)
- [x] Run preparation: model latency excludes client-side image encoding; OCR engine per run (`--ocr`), OCR pre-pass (`qdu ocr warm`) with the CPU recorded per page; vision-only sessions need no OCR; one `run_history.jsonl` entry per session; runbooks keep everything on Google Drive
- [x] Check run on Colab L4 (10 Nov 2026), `results/check52v2-qwen3.5-4b/`: 52 questions × 3 paths with PP-OCRv5; audit passes and reports reproduce byte-identically. Found and fixed: TAT-DQA page PNGs are 224 × 224 thumbnails (now rendered from the PDFs), paddlepaddle 3.3 breaks oneDNN (pinned 3.2.2). Finding: CPU OCR (~3 s/page) costs more than a vision call (~1.4 s), so under scenario C text is costlier than vision; the OCR choice for the full run (CPU PP-OCRv5, GPU OCR or Tesseract) is the project lead's decision
- [x] GPU OCR test (Colab L4, 10 Nov 2026), 52 check pages, PP-OCRv5 mobile: paddlepaddle-gpu could not be installed (Paddle index 502, BOS mirror 404); the same models via RapidOCR on ONNX Runtime CUDA (`--ocr rapid-mobile-gpu`, cuDNN search HEURISTIC) take **0.85 s/page** (ChartQA 0.24, DocVQA 0.96, InfographicVQA 1.11, TAT-DQA 1.08) against 2.87 s for RapidOCR and 2.99 s for PaddleOCR on the same VM's 12 CPU cores, with identical answer coverage (0.654) and near-identical text (36/52 pages identical, mean similarity 0.998). Recognition dominates GPU time (~5 ms per text line). With GPU OCR, DocVQA text costs 0.27 + 0.96 s against 1.57 s for vision, so the text path is cheaper than vision again under scenario C. Decided (Emre, 10 Nov 2026): OCR on the GPU, shared with vLLM as in a single-GPU deployment
- [x] Batch labelling on one Colab L4 (`notebooks/gpu_run.ipynb`), 10–11 Nov 2026: `results/full-qwen3.5-4b/`, 36,915 calls, all (question, path) pairs complete, audit passes, reports reproduce byte-identically. Four sessions: at 80% GPU memory OCR and then vLLM ran out of memory (1,037 failed calls, all retried), so vLLM now gets 65% with `--max-num-seqs 16`; one Colab disconnect. Primary metric over all questions: text 0.607, vision 0.833, hybrid 0.832, oracle 0.875 (+4.2 over vision, 95% CI [+3.9, +4.6], resampling documents) at 1.15 s vs 1.45 s per question (scenario C) and 1,611 vs 3,040 input tokens. Vision and hybrid tie overall (−0.1 [−0.5, +0.3]); TAT-DQA is the exception (hybrid +2.2 [+0.9, +3.6]). Only 7.6% of questions have a single best path, so most of the oracle's gain is cost (text is the cheapest path with GPU OCR). Colab L4 bills ~1.94 compute units per hour
- [x] Label table (`qdu router labels`, `results/full-qwen3.5-4b/labels.csv`): per question its router split, each path's primary score, model time, C / B / A cost, and the oracle label by latency and by tokens (tolerance 0; identical to the reports' `oracle_labels.csv`)

### Phase 3 — Signals (H5–H7)
- [x] Compare Tesseract vs PP-OCRv5 on the 300 pilot pages (`qdu ocr compare`, laptop CPU). Answer coverage in the OCR text: Tesseract 0.42, PP-OCRv5 mobile 0.63 (DocVQA 0.65 → 0.72, InfographicVQA 0.34 → 0.51, ChartQA 0.28 → 0.67); 80 pages where only PP-OCRv5 keeps the answer, 17 where only Tesseract does. Time per page 0.6 s vs 24 s, the latter with oneDNN off (paddle 3.3.1), so not representative; with paddle 3.2.2 and oneDNN it is ~1 s per page on the laptop and ~3 s on the Colab CPU. PP-OCRv5 server skipped: several times slower on CPU, and mobile already settles the choice
- [x] OCR with word boxes + confidences: PP-OCRv5 lines (text, confidence, box) and page size for every page of the full run, packed into `results/full-qwen3.5-4b/ocr.jsonl.gz` (`qdu router pack-ocr`; the per-page cache files clash on Windows and in Drive zips); Tesseract 5 now uses one `image_to_data` call, giving the same lines with mean word confidence (text words unchanged from `image_to_string` on pilot pages, only whitespace differs)
- [x] Layout detection with PP-DocLayout (`qdu layout detect`, `qdu.layout`): PP-DocLayout-M on all 3,586 pages, `results/full-qwen3.5-4b/layout.jsonl.gz` (regions with label, score, box; page size identical to the OCR's). Laptop CPU (16 threads): 0.087 s per page (-S: 0.05 s, less accurate; some pages were timed while tests ran alongside, medians 0.081–0.090 s across the run, so up to ~5% high), against ~0.8 s for OCR on the L4; charged as page-level router cost (`page_s`)
- [x] S1 question features (`qdu.router.signals.QuestionSignals`): length, content terms, wh-type, cue words (numeric, visual, layout, comparison, year), quotes. TF-IDF is fitted on train inside the model, not an extractor (its vocabulary would leak otherwise)
- [x] S2 OCR-text features (`OcrSignals`): lines, chars, words, words per megapixel, numeric share, mean/min confidence, low-confidence shares, median line height, empty page
- [x] S3 page features (`PageSignals`, from the layout file, no image needed): megapixels, aspect, region counts and area shares per kind (text / table / figure / formula / other), largest region
- [x] **S4 evidence location** (`EvidenceSignals`): question terms found in the OCR, matched lines, best single line, confidence of matched lines, numbers found, and the share of matches inside text / table / figure regions. All four signals take 0.15 ms per question (median; max 4.9 ms) on the laptop, far under 50 ms; on train, text scores 0.79 when every question term is in the OCR against 0.50 when only some are
- [x] Feature assembly into one table keyed by question id; decision-time measurement (`qdu router features`: extractor interface `qdu.router.features`, each extractor timed per question and marked if it needs OCR; S1–S4 are in `qdu.router.signals`)

### Phase 4 — Router & baselines (H6–H10)
- [x] Per-path quality regressors (Ridge, LightGBM) on S1–S4 + question TF-IDF (`qdu.router.train`, `qdu router train`; TF-IDF, imputation and scaling fitted on train only; artefacts in `runs/routers/<name>/`). Val, 12 Nov 2026: neither beats always-vision at any λ. At λ = 0 both reach 87.5 (vision 88.0) at 2.1–2.2 s C, since the models cannot tell vision from hybrid and the signals already paid OCR; within the always-vision C budget (λ = 0.3) they reach 75.3–75.4 at 1.36 s. Prototype checks: a perfect text-vs-vision chooser that always pays OCR would reach 90.3 at 1.35 s, but S1–S4 + TF-IDF separate "text is as good as vision" with AUC 0.66 and "text or hybrid beats vision" with AUC 0.58 (LightGBM, val)
- [x] Decision rule `argmax_p q̂_p − λ·cost_p`; λ selected on validation (`qdu.router.train.select_lambda`): the most accurate point whose C cost stays within always-vision's, else the most accurate point, flagged. Decision costs are the mean model time per path when the signals already paid OCR, otherwise the mean C cost
- [x] Two-stage router: stage 1 uses OCR-free signals (S1, image size) and decides when confident; only uncertain questions pay OCR + layout for S2–S4. Its decision cost is part of every router cost figure. `qdu.router.two_stage` / `qdu router two-stage`: stage 1 = LightGBM on S1 + S3 (layout, no OCR) + question TF-IDF sends likely image questions to vision without OCR; stage 2 = LightGBM on S1–S4 + TF-IDF chooses text or vision after OCR; (t1, t2) chosen on val within the always-vision C budget. Val, 12 Nov 2026: 84.5 at 1.43 s (−3.5 [−4.6, −2.6]), 33% of questions reach stage 2. +9.2 over the one-stage S1–S4 router within the budget, but below always-vision and the question-only baseline; stage 2's OCR signals add little (+0.8 over sending every stage-2 question to text)
- [x] Cost accounting in reports: scenarios A (tokens), B (OCR once per document, shared by its questions) and C (everything per question, incl. the router's own OCR/layout); computed from logged times, no GPU re-run (`qdu.router.evaluate.Policy.cost`: a router declares per question whether it ran OCR, its page-level and per-question signal time)
- [x] Baselines: always text / vision / hybrid, oracle, EcoDoc-style rules, question-only router (TF-IDF + logistic regression), text→vision confidence cascade. Always-* and oracle are built into `qdu router evaluate`; the cascade is `qdu.router.cascade` (text answer confidence from the logprob run `results/full-text-logprobs-qwen3.5-4b`, L4, 11 Nov 2026; a policy pays for every path it called, `Policy.called_before`). On val it never reaches always-vision: 78.4 vs 88.0 points at equal C cost, 85.3 at t = 0.8 (cheaper under B and A only); the model is confident about many wrong text answers. 2.3% of the re-run's text answers differ from the labelled run (bf16 GPU non-determinism) and are escalated. EcoDoc-style rules and the question-only router (TF-IDF + logistic regression, "Text, Pixels, or Both?") are `qdu.router.baselines` / `qdu router baseline question-only|ecodoc`, thresholds chosen on val like λ. Val, 12 Nov 2026: within the C budget question-only collapses to always-vision (88.0 at 1.44 s, threshold 0.10; it sends no question to text without losing points), EcoDoc reaches 87.1 at 1.43 s (−0.9 [−1.4, −0.5]). Off the budget, question-only dominates the S1–S4 router at every cost (e.g. 85.4 at 1.33 s vs 75.3 at 1.36 s): it never pays OCR to decide, which argues for the two-stage router
- [x] Ablations: S1, S1+S2, S1+S2+S3, S1–S4 (`qdu.router.ablation`, `qdu router ablation` → `runs/ablation/ablation_val.csv`): per configuration, val ROC AUC of LightGBM classifiers for text-as-good-as-vision (overall and mean within dataset), text-correct and text-or-hybrid-beats-vision, plus the learned router within the always-vision C budget. Val, 12 Nov 2026 (AUC text_ok / within / beats): TF-IDF only 0.583 / 0.562 / 0.541; S1 0.592 / 0.572 / 0.539; S1+S2 0.646 / 0.615 / 0.582; S1+S2+S3 0.658 / 0.630 / 0.579; S1–S4 0.659 / 0.626 / 0.584. OCR signals (S2) carry most of what is predictable; **S4 adds nothing measurable** over S1–S3 (+0.001 AUC). Leakage check: the dataset name alone reaches 0.585 (0.500 within datasets) and adding it to S1–S4 changes nothing (0.659), so the signals do not just recognise the dataset. Routers that pay OCR fall to ~75 within the budget; OCR-free ones stay near always-vision (87.6–87.7)
- [x] `qdu router train|evaluate` CLI; router artefact (`model.joblib`, `router.json` with the label table's SHA-256, git commit and λ, `val_curve.csv`, choices CSV) in `runs/routers/<name>/`

### Phase 5 — Evaluation (H9–H11)
- [x] Paired bootstrap 95% CIs (router vs each baseline), resampling documents, since full labelling has several questions per document (`qdu.evaluation.bootstrap`; `qdu router evaluate --baseline`)
- [ ] Unseen-data test on TAT-DQA
- [ ] Success criteria table — revised after the pilot: the router must beat **always-vision** (more accurate at equal or lower cost), since always-vision is statistically tied with always-hybrid (pilot: -1.7 points, CI includes 0) at half the latency. Original: (cost ≥ 25% below always-hybrid with ≤ 1 ANLS point loss; S4 gain CI excludes 0; TAT-DQA: more accurate than always-text, cheaper than always-hybrid; decision < 50 ms)

### Phase 6 — Demo & deliverables (H9–H14)
- [ ] Gradio demo: upload page + question → chosen path, answer, latency (`qdu demo`)
- [ ] Academic activity documented by **14 Dec** (TÜBİTAK 2209-A or paper)
- [ ] Vize (16 Nov upload), final (14 Dec), report + demo (28 Dec), poster (5 Jan)

### Decisions
- Same model serves all three paths; the text path sends no image. Keeps differences attributable to input modality.
- Splits are by document. Samples (pilot) take at most one question per document; full labelling takes every question and resamples by document for confidence intervals.
- Use each dataset's labelled evaluation split (DocVQA/InfographicVQA validation, ChartQA test) because official training splits are standard VLM training data.
- Model: **Qwen3.5-4B** instead of the presentation's Qwen2.5-VL (Oct 2026). Newer (Mar 2026), natively multimodal, runs on one 24 GB L4 in bf16, stronger on OCR (OCRBench 85.0). Cite the Qwen3.5 model card instead of the Qwen2.5-VL report.
- Tools: PP-DocLayout (PaddleOCR 3.x) instead of DocLayout-YOLO for layout; PP-OCRv5 as the stronger OCR backend next to Tesseract (Oct 2026).
- Each dataset is judged by its conventional metric (`qdu.evaluation.primary`): ANLS for DocVQA and InfographicVQA, relaxed accuracy (5% numeric tolerance, exact match for text) for ChartQA, F1 for TAT-DQA. Oracle labels use the same metric. ANLS, relaxed accuracy, EM and F1 are still reported for every dataset. Exact match and token F1 (TAT-QA style, numbers rounded to two decimals; `qdu.evaluation.em_f1`) are reported too and are the TAT-DQA metric; unlike the official script they do not score the predicted scale, because pipelines return one answer string.
- Images sent to the model are capped at Qwen3.5's own processor limit, 16,777,216 pixels (aspect ratio kept; `QDU_MAX_IMAGE_PIXELS`), for both vision and hybrid paths; OCR uses the full-resolution page. The pilot used 4 MP, which downscaled a quarter of DocVQA/InfographicVQA pages and may have favoured hybrid. The server needs `--max-model-len 32768`. Every run records GPU, vLLM version, launch flags (`QDU_SERVER_ARGS`) and request settings in `run_info.json`.
- vLLM runs with prefix caching and the multimodal processor cache **off** (`--no-enable-prefix-caching --mm-processor-cache-gb 0`). The smoke test showed hybrid calls 3x faster than vision calls on the same page despite more input tokens, because the image was cached by the preceding vision call; with caches on, measured latency is not a fair cost.
- Cost is reported two ways: latency (model + OCR seconds; hardware-specific, what self-hosting pays) and tokens (input + output; API-price proxy, OCR free). Conclusions that hold under only one measure must say so.
- Router cost includes the router's own work. S2–S4 need OCR and layout, so a router that runs them on every question pays OCR even when it then picks vision; the pilot oracle (no signal cost) is therefore optimistic. Three scenarios are reported for every method: **C** end-to-end per question, including signal computation (primary: the presentation claims OCR-inclusive cost as a gap we fill); **B** OCR and layout run once per document at ingestion and shared by its questions (realistic for document archives); **A** tokens only (API pricing, comparable to the literature). Success claims state which scenario they hold in; the target is C. Decided 30 Oct 2026.
- OCR for the full run is **PP-OCRv5 mobile** instead of Tesseract (Nov 2026): it keeps the gold answer on 63% of pilot pages against 42%, and the text path cannot answer what OCR misses. Tesseract stays available (`--ocr tesseract`) as the cheap baseline. It runs as RapidOCR on ONNX Runtime CUDA on the same GPU as vLLM (Decided 10 Nov 2026: a single-GPU deployment; CPU OCR at ~3 s/page cost more than a vision call, GPU OCR ~0.85 s/page with the same output). OCR time is measured once per page (`qdu ocr warm`, vLLM loaded but idle) and every row records the hardware (`ocr_hardware`); scenario C charges it to every question, B divides it among the page's questions (`total_shared_s` in metrics.csv).
- Model latency is the server round trip only: resizing and PNG-encoding the page on the client is done before the clock starts (it inflated vision and hybrid latency in the pilot by the encoding time).
- Answers are normalised by `clean_answer` for every path (multi-line lists joined with commas, prefixes, quotes, final period, space thousands separators). After changing it, re-derive finished runs with `qdu benchmark rescore` instead of re-running the GPU.
- The pilot samples from the router-train split, so pilot labels can be reused for router training without leaking val/test documents.
