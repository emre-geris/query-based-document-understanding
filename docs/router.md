# Router work: hand-off guide

> Final phase (from 14 Nov 2026): see `docs/phase7_handoff.md` for what to build next and why.

Everything needed to build and evaluate the router **on a laptop, without a GPU**. The GPU
work is finished: every question of the four datasets was answered by all three paths
(`results/full-qwen3.5-4b/`, 10–11 Nov 2026), and the label table, the run's OCR, page
layouts, signals S1–S4, cost accounting, bootstrap, the cascade baseline and the evaluation
command are in place. What remains is the router itself: the per-path quality models, λ, the
two-stage design, the remaining baselines and ablations (CLAUDE.md, Phases 4–5).

## 1. Set-up

```powershell
git clone https://github.com/emre-geris/query-based-document-understanding C:\Users\<you>\source\qdu
cd C:\Users\<you>\source\qdu
uv sync                        # Python 3.13 and the locked dependencies
uv run pytest                  # all tests should pass
uv run qdu router evaluate     # baselines on the validation split (needs only the repo)
```

Keep the clone and the virtual environment outside OneDrive (README, *Setup*).

Everything the signals read is in the repo, so no download is needed:

| File | What | Load with |
|---|---|---|
| `results/full-qwen3.5-4b/ocr.jsonl.gz` | the run's own PP-OCRv5 output for all 3,586 pages: text, every line with confidence and box, page size, OCR time | `qdu.router.load_ocr()` |
| `results/full-qwen3.5-4b/layout.jsonl.gz` | PP-DocLayout-M regions of every page (label, score, box), page size, detection time on a laptop CPU | `qdu.layout.load_layout()` |
| `results/full-text-logprobs-qwen3.5-4b/` | the text path re-run with token log-probabilities (confidence for the cascade) | `qdu.router.cascade.read_confidence()` |

Do not re-run OCR on a laptop: its text would differ slightly from what the text and hybrid
paths saw, and its time is not the cost the label table charges. Page images
(`uv run qdu data download`, ~1.6 GB, Hugging Face cache) are only needed for new image-based
signals; `qdu layout detect` re-creates the layout file from them (needs `uv sync --extra paddle`).

## 2. What you start from

`results/full-qwen3.5-4b/labels.csv` (`qdu router labels`): one row per question, 12,305 rows.

| Column | Meaning |
|---|---|
| `question_id`, `doc_id`, `dataset`, `stratum`, `question` | ids are dataset-prefixed (`docvqa/1012`); `doc_id` is the page |
| `split` | router split, fixed per document (scheme v2, below): train 7,426, val 2,558, test 2,321 |
| `questions_on_page`, `ocr_s` | questions asked about this page; the page's OCR time on the L4 |
| `score_<path>` | the dataset's primary metric for that path's answer (ANLS / relaxed accuracy / F1), 0–1 |
| `model_s_<path>` | model latency, seconds |
| `cost_c_<path>`, `cost_b_<path>`, `tokens_<path>` | cost of the path in scenarios C, B and A (below) |
| `oracle_latency`, `oracle_tokens` | the best path, ties broken by C seconds or by tokens: the classification target |

Paths are `text` (OCR text only), `vision` (page image only) and `hybrid` (both).

Cost scenarios (CLAUDE.md, *Decisions*):

- **C** (primary): every question pays its page's OCR when OCR is used, plus the router's own work.
- **B**: OCR and page-level signals run once per page and are divided among its questions.
- **A**: tokens only; OCR is free.

## 3. Rules

1. **Splits.** Fit on `train`. Choose features, models and λ on `val`. Run `test` **once**, at
   the end, for the reported numbers. Splits are by document, so never re-split by question.
   Scheme v2 (default since 15 Nov 2026) splits TAT-DQA 60/20/20 like the other datasets,
   **grouped by annual report** (its 277 documents come from 126 reports;
   `benchmarks/tatdqa_reports.csv`); until then TAT-DQA was test-only (scheme v1,
   `qdu router labels --split-scheme v1` reproduces that table). The unseen-data test is now
   *leave one dataset out* (§4).
2. **Inputs.** A router sees the question, the page image and its OCR (`RouterInput`), never
   the gold answer, a path's answer or score. Do not use the dataset name as a feature: a
   deployed router does not know it (an ablation may show its effect, labelled as such).
3. **The router pays for itself.** If it computes OCR signals for a question, that question
   pays OCR even when the router then picks vision. Layout time is page-level, decision time
   per question. Declare these in the choices file (§4); the evaluation adds them. The
   decision itself must stay under 50 ms per question (`time_s_<extractor>` in the feature table).
4. **Success** (Decided, Emre, 15 Nov 2026; the October pre-presentation's criterion): against
   **always-hybrid**, the primary baseline (a document-QA system that sends both the OCR text
   and the page), the router costs **at least 25% less in scenario C** and loses **at most 1
   point** (the lower end of the score CI above −1). `qdu router two-stage` therefore chooses
   its thresholds within `--budget hybrid:0.75` by default. **Always-vision is reported next to
   it in every table** (`--baseline always-vision`): on our data it is as accurate as hybrid
   and cheaper, and a reader will ask. Report B and A too.
5. **Confidence intervals** resample documents (`qdu.evaluation.bootstrap`), never questions;
   TAT-DQA pages resample by annual report (`qdu router evaluate` does this itself).
6. **Reproducibility.** Fix seeds; commit what produces a reported number; never edit
   `labels.csv`, manifests or predictions by hand (regenerate with `qdu`).

## 4. Evaluating a router

A router's output is a CSV, one row per question of the split being evaluated:

```csv
question_id,pipeline,uses_ocr,page_s,question_s
docvqa/1012,text,1,0.031,0.004
chartqa/test-00000,vision,0,0,0.002
```

`uses_ocr` (0/1), `page_s` (page-level signal time other than OCR, e.g. layout) and
`question_s` (per-question signal + decision time) are optional and default to 0.

```powershell
uv run qdu router evaluate --split val --choices runs/my_router.csv [--out runs/my_router_eval.csv]
```

It prints, per dataset and overall, the score (primary metric × 100) and C/B/A costs of
always-text, always-vision, always-hybrid, the oracle and each router, with paired-bootstrap
95% CIs against always-hybrid (default; `--baseline always-vision` for the other table) for
the score and the C cost. Validation baselines (scheme v2, incl. 384 TAT-DQA questions):

| val (2,558 q) | score | C s | B s | tokens | score − hybrid | score − vision |
|---|---|---|---|---|---|---|
| always-text | 61.2 | 1.07 | 0.49 | 621 | −21.4 [−24.1, −19.1] | −21.9 [−24.6, −19.3] |
| always-vision | 83.1 | 1.48 | 1.48 | 3,108 | +0.4 [−0.4, +1.3] | – |
| always-hybrid | 82.6 | 2.34 | 1.77 | 3,673 | – | −0.4 [−1.3, +0.4] |
| oracle | 87.4 | 1.16 | 0.86 | 1,618 | +4.8 [+4.0, +5.6] | +4.3 [+3.5, +5.2] |

The oracle shows the room: +4.8 points over always-hybrid at half its C cost. Only ~8% of
questions have a single best path, so most of the gain is sending easy questions to text.

**Unseen data: leave one dataset out.** `uv run qdu router lodo` fits the two-stage router on
the other datasets' train questions, chooses its thresholds on their val questions (same
budget rule) and scores each held-out dataset on its train + val questions, which it never
saw (`--on test` for the final run). Results in `runs/lodo/lodo_dev.csv`.

## 5. Writing code

```python
from pathlib import Path
from qdu.layout import load_layout
from qdu.router import (
    EvidenceSignals,
    OcrSignals,
    PageSignals,
    QuestionSignals,
    build_features,
    cost_quality_curve,
    evaluate,
    load_ocr,
    mean_costs,
    read_labels,
    route,
    router_inputs,
)

labels = Path("results/full-qwen3.5-4b/labels.csv")
train = read_labels(labels, splits=["train"])
val = read_labels(labels, splits=["val"])
layout = load_layout()

signals = [QuestionSignals(), OcrSignals(), PageSignals(), EvidenceSignals()]
features = build_features(router_inputs(train + val, load_ocr(), layout), signals)
# ... fit one quality model per path on train: features -> score_<path> ...
predicted = {qid: {"text": 0.7, "vision": 0.9, "hybrid": 0.9} for qid in features.features}

cost = mean_costs(train, "C")  # expected cost per path at decision time
curve = cost_quality_curve(val, predicted, cost, [0, 0.05, 0.1, 0.2])  # pick λ on val
policy = route(
    {q: predicted[q] for q in (r.question_id for r in val)},
    cost,
    lam=0.1,
    name="my-router",
    **features.router_cost(layout, {r.question_id: r.doc_id for r in train + val}),
)
for result in evaluate(val, [policy]):
    print(result.group, result.method, result.score, result.cost_c, result.score_vs_baseline)
```

- **Signals** are extractors (`qdu.router.features.FeatureExtractor`): a `name` (feature
  prefix), `needs_ocr`, `needs_layout`, and `__call__(item) -> {feature: number}`. S1–S4 are
  in `qdu.router.signals` (see its docstring for each feature) and registered in
  `qdu.router.EXTRACTORS`; `uv run qdu router features --out runs/features.csv` builds the
  table with all four (`--extractors s1,s3` for a subset, e.g. an ablation). New signals go
  next to them and are registered the same way.
- **TF-IDF** of the question is not an extractor: its vocabulary must be fitted on `train`
  only, so it belongs inside the model (e.g. an sklearn pipeline fitted on the training rows).
- `item.ocr.lines` gives every OCR line as `(text, score, (x0, y0, x1, y1))`, `item.ocr.page_size`
  the page size, and `item.layout.regions` the layout regions `(label, score, box)`.
- **Router cost**: `features.router_cost(layout, doc_of)` gives `uses_ocr`, `page_s` (layout
  time) and `question_s` (extractor time) for `route` or a choices CSV. A two-stage router
  passes OCR and layout only for the questions that reached stage 2.
- **Three-way two-stage router** (`qdu router two-stage`): stage 1 (OCR-free) sends to vision;
  stage 2 (after OCR) has a hybrid head (P(hybrid > vision) ≥ t3 → hybrid), then the text/vision
  head. `--paths text,vision` is the earlier two-way router; `--budget vision|hybrid|none|<s>`
  is the val C budget the thresholds must respect (default always-vision). The full grid is in
  `runs/routers/<name>/val_grid.csv`. The trained router is saved as `runs/routers/<name>/model.joblib`
  (`save_two_stage` / `load_two_stage`, heads, TF-IDF vocabularies and thresholds plus the
  run metadata), so the demo routes new questions without retraining; `runs/` is not committed,
  so retrain it with the same command on a new machine.
- **Live routing** (`qdu.router.live.LiveRouter`, for the demo): routes one new question on a
  new page with a saved router, computing only the signals each stage needs: layout and the
  OCR-free signals first, OCR only if stage 1 does not decide. It returns a `LiveDecision`
  with the path, the stage, each head's probability and the time of every step, and keeps
  layout and OCR per page so later questions on the page reuse them. S5/S6 are embedded on
  the spot (`live_extractors`). The path rule is `TwoStageRouter.choose`, shared with the
  evaluation, so the demo decides exactly as the evaluated router does.
- **Demo service** (`qdu.demo.DemoService`): routes with `LiveRouter`, answers with the chosen
  path (or all three with `compare`), and adds up the cost as the evaluation does: scenario C
  (OCR and layout in full), B (page work divided by the questions asked about the page so
  far) and A (tokens). OCR runs once per page (`SharedOcr`), shared by the router and the
  text/hybrid paths. Try it without a GPU (fake model; needs the `embed`, `paddle` and `rapid`
  extras for S5, layout and CPU OCR):
  `uv run qdu demo ask --image page.png --question "..." --compare`.
- **Demo examples** (`qdu.demo.examples`): val pages only (test stays untouched). `qdu demo
  examples` lists pages whose questions need text, vision and only-hybrid respectively;
  `qdu demo ask --example <id> --compare` routes one with the run's stored OCR and layout (so
  the choice equals the evaluation's) and, with the default `--model replay`, shows the
  labelled run's recorded answers, times and tokens, scored against the gold answers. No GPU
  and no OCR or layout install is needed for examples.
- **Demo interface** (`qdu demo serve`, optional `demo` extra: FastAPI + uvicorn): the design
  draft as a web page (`qdu/demo/static/index.html`, plain HTML and JavaScript, light theme)
  served by `qdu.demo.web`, which keeps one `DemoSession` and answers each button with its state
  as JSON (`session_view`). The presentation narrative is generated from each answer
  (`render.step_story`), so it never claims more than the result. Gradio was tried first and
  dropped: it could not follow the design. To run it: `uv sync --extra demo`, train a router
  (`uv run qdu router two-stage`, saved to `runs/routers/two-stage`, the demo's default), then
  `uv run qdu demo serve` and open http://127.0.0.1:7860. Note for presenting ChartQA: its
  relaxed accuracy gives numbers 5% tolerance, years included, so a wrong year (2019 for 2020)
  is shown as correct.
- **Embedding signals S5/S6** (`qdu.router.embed`): precomputed once by `qdu router embed`
  (optional `embed` extra, page images) into two gzips next to the run; the extractors only
  read them. Their measured time is charged through `cost(item) -> (question_s, page_s)`:
  per-question time is added to the extractor's time, per-page time goes to
  `FeatureTable.page_seconds` (like layout). `learned-cascade` rejects `s6`, since it does not
  charge page-level signal time.
- **Cascade baseline**: `qdu.router.cascade` (`read_confidence`, `cascade`, `sweep`); choose
  its threshold on `val` like λ. It is weak: on val it never reaches always-vision (78.4 vs
  88.0 points at equal C cost; `results/full-text-logprobs-qwen3.5-4b/README.md`).
- New commands go in `qdu router ...` (`src/qdu/cli.py`), e.g. `qdu router train`; heavy
  dependencies (scikit-learn, LightGBM) are added with `uv add`, never `pip install`.
- Every module gets tests in `tests/`; `uv run ruff check .` and `uv run pytest` must pass
  before a commit (pre-commit and CI run both). Conventional Commits, one change per commit.
- Mark roadmap items `[x]` in CLAUDE.md in the commit that completes them.

## 6. Open items (CLAUDE.md roadmap)

- Phase 4: per-path quality regressors (Ridge, LightGBM) on S1–S4 (+ question TF-IDF); λ on
  val; two-stage router; baselines (EcoDoc-style rules, question-only TF-IDF + logistic
  regression; the cascade exists); ablations S1 → S1–S4; `qdu router train` with a versioned
  router artefact.
- Phase 5: test-split evaluation (in-domain and `qdu router lodo --on test`), the
  success-criteria table.
