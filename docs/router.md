# Router work: hand-off guide

Everything needed to build and evaluate the router **on a laptop, without a GPU**. The GPU
work is finished: every question of the four datasets was answered by all three paths
(`results/full-qwen3.5-4b/`, 10–11 Nov 2026), and the label table, cost accounting, bootstrap
and evaluation command are in place. What remains is the router itself: signals S1–S4, the
per-path quality models, λ and the two-stage design (CLAUDE.md, Phases 3–5).

## 1. Set-up

```powershell
git clone https://github.com/emre-geris/query-based-document-understanding C:\Users\<you>\source\qdu
cd C:\Users\<you>\source\qdu
uv sync                        # Python 3.13 and the locked dependencies
uv run pytest                  # all tests should pass
uv run qdu router evaluate     # baselines on the validation split (needs only the repo)
```

Keep the clone and the virtual environment outside OneDrive (README, *Setup*).

The OCR every signal should use is in the repo: `results/full-qwen3.5-4b/ocr.jsonl.gz`, the
run's own PP-OCRv5 output for all 3,586 pages (text, every line with its confidence and box,
page size, OCR time), loaded with `qdu.router.load_ocr()`. Do not re-run OCR on a laptop: its
text would differ slightly from what the text and hybrid paths saw, and its time is not the
cost the label table charges. Only page images need a download:

| What | Needed for | How |
|---|---|---|
| Page images (~1.6 GB) | S3 (image size, layout) | `uv run qdu data download` (Hugging Face cache, never into the repo) |

## 2. What you start from

`results/full-qwen3.5-4b/labels.csv` (`qdu router labels`): one row per question, 12,305 rows.

| Column | Meaning |
|---|---|
| `question_id`, `doc_id`, `dataset`, `stratum`, `question` | ids are dataset-prefixed (`docvqa/1012`); `doc_id` is the page |
| `split` | router split, fixed per document: train 6,353, val 2,174, test 3,778 (incl. all 1,655 TAT-DQA) |
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
   the end, for the reported numbers. TAT-DQA is test-only: the router never trains or tunes on
   it. Splits are by document, so never re-split by question.
2. **Inputs.** A router sees the question, the page image and its OCR (`RouterInput`), never
   the gold answer, a path's answer or score. Do not use the dataset name as a feature: a
   deployed router does not know it (an ablation may show its effect, labelled as such).
3. **The router pays for itself.** If it computes OCR signals for a question, that question
   pays OCR even when the router then picks vision. Layout time is page-level, decision time
   per question. Declare these in the choices file (§4); the evaluation adds them. The
   decision itself must stay under 50 ms per question (`time_s_<extractor>` in the feature table).
4. **Success** means beating **always-vision**: higher score at equal or lower cost, in
   scenario C, with the 95% CI of the score difference above 0. Report B and A next to it.
5. **Confidence intervals** resample documents (`qdu.evaluation.bootstrap`), never questions.
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
95% CIs against always-vision for the score and the C cost. Validation baselines:

| val (2,174 q) | score | C s | B s | tokens | score − vision |
|---|---|---|---|---|---|
| always-text | 62.2 | 0.98 | 0.46 | 559 | −25.9 [−28.3, −23.4] |
| always-vision | 88.0 | 1.44 | 1.44 | 2,999 | – |
| always-hybrid | 87.2 | 2.23 | 1.71 | 3,503 | −0.8 [−1.6, −0.0] |
| oracle | 91.5 | 1.08 | 0.84 | 1,573 | +3.5 [+2.8, +4.3] |

The oracle shows the room: +3.5 points at 25% lower C cost on validation. Only ~8% of
questions have a single best path, so most of the gain is sending easy questions to text.

## 5. Writing code

```python
from pathlib import Path
from qdu.router import (
    QuestionShape,
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

features = build_features(router_inputs(train + val, load_ocr()), [QuestionShape()])
# ... fit one quality model per path on train: features -> score_<path> ...
predicted = {qid: {"text": 0.7, "vision": 0.9, "hybrid": 0.9} for qid in features.features}

cost = mean_costs(train, "C")  # expected cost per path at decision time
curve = cost_quality_curve(val, predicted, cost, [0, 0.05, 0.1, 0.2])  # pick λ on val
policy = route(
    {q: predicted[q] for q in (r.question_id for r in val)},
    cost,
    lam=0.1,
    name="my-router",
    uses_ocr={r.question_id: features.needs_ocr for r in val},
)
for result in evaluate(val, [policy]):
    print(result.group, result.method, result.score, result.cost_c, result.score_vs_baseline)
```

- **Signals** are extractors (`qdu.router.features.FeatureExtractor`): a `name` (feature
  prefix), `needs_ocr`, and `__call__(item) -> {feature: number}`. `QuestionShape` is a
  working example (question length, wh-word, cue words). Put S1–S4 in
  `src/qdu/router/signals/` or next to it, register them in `qdu.router.EXTRACTORS`, and
  `uv run qdu router features --extractors q,s1,... --out runs/features.csv` builds the table.
- `item.ocr.lines` gives every OCR line as `(text, score, (x0, y0, x1, y1))` and
  `item.ocr.page_size` the page size, for confidence features and evidence location (S4).
- New commands go in `qdu router ...` (`src/qdu/cli.py`), e.g. `qdu router train`; heavy
  dependencies (scikit-learn, LightGBM) are added with `uv add`, never `pip install`.
- Every module gets tests in `tests/`; `uv run ruff check .` and `uv run pytest` must pass
  before a commit (pre-commit and CI run both). Conventional Commits, one change per commit.
- Mark roadmap items `[x]` in CLAUDE.md in the commit that completes them.

## 6. Open items (CLAUDE.md roadmap)

- Phase 3 signals: Tesseract `image_to_data` for the cheap OCR baseline; PP-DocLayout regions
  (cached, timed); S1 question, S2 OCR text, S3 page, S4 evidence location; feature table and
  decision-time measurement (the table and timing exist: `qdu router features`).
- Phase 4: per-path quality regressors (Ridge, LightGBM); λ on val; two-stage router; baselines
  (EcoDoc-style rules, question-only TF-IDF + logistic regression, text→vision cascade);
  ablations S1 → S1–S4; `qdu router train|evaluate` with a versioned router artefact.
- Phase 5: test-split evaluation, TAT-DQA as unseen data, the success-criteria table.

The text→vision cascade baseline needs a confidence for the text answer; the run did not log
token log-probabilities, so it has to use a proxy (e.g. answer length or OCR confidence of the
matched words) or a short GPU re-run of the text path with logprobs on.
