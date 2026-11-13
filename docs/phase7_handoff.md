# Phase 7 hand-off: what to build for the final, step by step

Written 14 Nov 2026 for whoever implements Phase 7 (Hayrunnisa or Doğukan), possibly in a new
Claude Code session with no memory of the work so far. Read in this order: this file, CLAUDE.md
(Router work, Roadmap → Phase 7, Decisions), `docs/related_work.md`, `docs/router.md`. Nothing
here has been started; every number below comes from the committed results or from the val-split
analyses whose code is in the appendix.

The project lead (Emre) decides; deviations from this plan go to CLAUDE.md *Decisions* with the
reason and the date. Keep the test split untouched until the final evaluation (step 6).

---

## 0. Why Phase 7 exists (read this first)

**What we promised.** The October pre-presentation (Emre's folder `tarama sonuclari`,
`Sunum_Literatur_ve_Ozgun_Deger.pdf`) promised the advisors:

1. a router that chooses **text, image or hybrid** per question on document pages;
2. an **evidence-location signal** (where the question's words are on the page, how well they
   were read) that makes routing better;
3. a fair comparison: every fixed path, **cost including OCR**, an unseen-data test (TAT-DQA).

Success was defined as: quality close to always-hybrid at clearly lower cost, and a significant
gain from the evidence signal over the question-only router and EcoDoc (research question AS3);
otherwise the result is reported as a controlled negative finding.

**Where we are (val split, 2,174 questions, scores = primary metric × 100, cost C = seconds per
question incl. OCR):**

| Method | Score | C s | vs always-hybrid | vs always-vision |
|---|---|---|---|---|
| always-hybrid | 87.2 | 2.23 | – | −0.8 |
| always-vision | 88.0 | 1.44 | +0.8 | – |
| oracle (perfect chooser, ceiling) | 91.5 | 1.08 | +4.3 | +3.5 |
| question-only router (Text/Pixels/Both style) | 88.0 | 1.44 | +0.8, −36% cost | = (it collapses to vision) |
| EcoDoc-style rules | 87.1 | 1.43 | −0.2, −36% cost | −0.9 [−1.4, −0.5] |
| two-stage router (best learned) | 84.5–84.9 | 1.43 | −2.7, −36% cost | −3.5 [−4.6, −2.6] |
| confidence cascade (text → vision) | 78.4 | 1.43 | | −9.6 |

- Promise 3 (fair comparison) and the cost half of the success criterion are met.
- Promise 1 is **not**: every good router chooses only text or vision (`two_stage.py`:
  "Hybrid is not a choice"); only the one-stage `qdu router train` covers three paths and it is
  the weakest (75.3 within budget). Emre decided (14 Nov) the router must offer all three.
- Promise 2 is **negative**: S4 adds +0.001 AUC over S1–S3 (CI includes 0). Report it as the
  promised controlled negative finding; Phase 7 B reuses the evidence location differently.

**What the literature says** (`docs/related_work.md`, full texts read 14 Nov): per-question
text/image routing exists (Text, Pixels, or Both?; EcoDoc; SAFE-Cascade; TableDART for tables),
and nobody beats the best fixed representation either (Text/Pixels/Both router −1.5 vs
always-image; the NeurIPS 2026 web-agent study; Amazon's long-document study with our model,
Qwen3.5-4B). What is missing, and what Phase 7 adds:

- a **three-way** document router that **pays for its own signals** per question (C);
- routing **resolution** together with modality, without retraining the VLM (A);
- **evidence-guided cropping** from OCR + layout, as a cheap path and a cheap escalation (B).

---

## 1. Set-up for a fresh machine or session

```powershell
git clone https://github.com/emre-geris/query-based-document-understanding
cd query-based-document-understanding
$env:UV_PROJECT_ENVIRONMENT = "$env:USERPROFILE\venvs\qdu"   # keep the venv outside OneDrive
uv sync
uv run pytest                        # all tests must pass before you start
uv run qdu router evaluate --split val   # reproduces the always-* and oracle rows above
```

Everything the routers read is committed: `results/full-qwen3.5-4b/labels.csv`, `ocr.jsonl.gz`,
`layout.jsonl.gz`, and `results/full-text-logprobs-qwen3.5-4b/`. Page images (needed for A, B
and S6) come from `uv run qdu data download` (~1.6 GB, Hugging Face cache).

To reproduce the teammate's routers: `uv run qdu router two-stage`, `uv run qdu router baseline
ecodoc`, `uv run qdu router baseline question-only`; each writes a choices CSV under
`runs/routers/<name>/` and prints the `qdu router evaluate` command to score it.

GPU work runs on a Colab L4 through `notebooks/gpu_run.ipynb` (see CLAUDE.md, Environment
notes). Emre has to approve the Google Drive access dialog himself; never do it for him.

---

## 2. Step 1 — Direction C: the promised three-way router (required)

**Goal.** A router that chooses text, vision or hybrid per question, decides first from
OCR-free signals and buys OCR only when its expected value exceeds its cost.

### 2.1 Add two OCR-free signals (S5, S6)

- **S5 – question embedding**: `sentence-transformers/all-MiniLM-L6-v2`, 384-d, normalised;
  ~1.2 ms per question on the laptop CPU (batched).
- **S6 – page image embedding**: `sentence-transformers/clip-ViT-B-32`, 512-d image vector,
  plus `pg_qsim` = cosine between the CLIP text embedding of the question and the page vector.
  74 ms per page median on the laptop CPU (16 threads); measure on the L4 (step 2.4).

Implementation (follow the existing patterns):

1. `uv add --optional embed sentence-transformers` (pulls torch; keep it out of the base install).
2. New module `src/qdu/router/embed.py`:
   - `embed_pages(manifest, out)`: one row per page, `{doc_id, model, latency_s, hardware,
     vector}`; resumable JSONL then a gzip written with `qdu.gz.open_gzip_text` (deterministic),
     like `qdu.layout.detect_pages` / `compress`. Commit the result as
     `results/full-qwen3.5-4b/clip_pages.jsonl.gz` (3,586 × 512 floats; round to 4 decimals or
     store float16 base64 to keep it a few MB).
   - `load_page_embeddings(path) -> dict[doc_id, (vector, latency_s)]`.
3. Extractors in `src/qdu/router/signals.py`, registered in `qdu.router.EXTRACTORS`:
   `QuestionEmbedding` (name `s5`, `needs_ocr=False`, `needs_layout=False`) and `PageEmbedding`
   (name `s6`, page-level: its per-page time must be charged as `page_s`, like layout; add a
   `page_seconds` hook or pass the CLIP latency into `FeatureTable.router_cost`).
4. CLI: `qdu router embed-pages --manifest benchmarks/full.csv` (and `--extractors` support for
   s5/s6 in `qdu router features`).
5. Tests: fake encoder objects (no model download in CI), determinism of the gzip, cost
   charging of S6 in `router_cost`.

**Expected (val, from the appendix code, LightGBM, text_ok AUC):** S1–S4 + TF-IDF 0.659;
+ S5 0.670; + S6 (512-d) 0.692. An OCR-free router (S1 + S3 + TF-IDF + S5 + S6, text if
p ≥ t else vision, t chosen on val within always-vision's C cost): 86.7 vs 88.0 (−1.4
[−2.0, −0.8]) if CLIP costs ~5 ms per page; with the measured 83 ms CPU time it drops to 83.1.

### 2.2 Make the two-stage router three-way

`src/qdu/router/two_stage.py`, class `TwoStageRouter.policy`: today stage 1 sends "image
needed" questions to vision and stage 2 chooses text vs vision. Change it to:

- Stage 1 (OCR-free: S1, S3, S5, S6, TF-IDF): if p(image needed) ≥ t1 → vision, no OCR.
- Stage 2 (after OCR: + S2, S4): three classes or two binary heads:
  `p_text_ok` (text as good as vision) and `p_hybrid_gain` (hybrid beats vision; target
  `score_hybrid > score_vision`). Choose text if p_text_ok ≥ t2, else hybrid if
  p_hybrid_gain ≥ t3, else vision. Hybrid already paid OCR in stage 2, so its extra cost is only
  the model-time difference (hybrid ≈ +0.12 s over vision; see labels.csv `model_s_*`).
- Thresholds (t1, t2, t3) chosen on val by `qdu.router.baselines.select_point` within the
  always-vision C budget, as now. Keep the CLI name `qdu router two-stage`, add `--paths
  text,vision,hybrid` (default three) so the old two-way result stays reproducible.

**Why hybrid matters:** a perfect vision-vs-hybrid chooser gains +2.7 over always-vision (all
datasets: DocVQA +1.4, InfographicVQA +3.9, ChartQA +2.1, TAT-DQA +5.8). Hybrid wins when the
page is text-dense (OCR median 310 words vs 167 when tied) and the question numeric (55% vs 48%),
but "beats vision" is hard to predict (AUC 0.58), so expect most of the gain on TAT-DQA-like
pages. Hybrid is the only path that can raise accuracy above always-vision.

### 2.3 Evaluate

`uv run qdu router evaluate --split val --choices runs/routers/two-stage/two-stage.csv` and
with `--baseline always-hybrid` and `--baseline question-only`. Report C, B and A, with the
rerun floor (section 6).

### 2.4 Measure CLIP on the L4

Next Colab session (~10 min, ~0.3 compute units): time `clip-ViT-B-32` image encoding per page
on the L4 next to the loaded vLLM (same conditions as OCR), for all 3,586 pages or a 300-page
sample. Record it in the page-embedding file's `latency_s`/`hardware` and use it in the cost.

---

## 3. Step 2 — Direction A: route resolution as well as modality

**Idea.** Add a path `vision-low`: the page image at ~1/4 of its pixels. Router chooses text /
vision-low / vision / hybrid.

**Evidence.** VisionThink and AdaptVision (Qwen2.5-VL-7B): at 1/4 resolution DocVQA loses only
0.8 points (95.1 → 94.3) but ChartQA 16.9 and OCRBench 12.7; VisionThink needs full resolution
for 6.5% of DocVQA questions. Those methods retrain the VLM with RL and have no OCR path; we
keep the VLM untouched. Our vision path averages 3,040 input tokens (DocVQA 3,738); ChartQA pages
are small (518 tokens), so the saving should come from DocVQA / InfographicVQA / TAT-DQA.

**Code changes (laptop):**

1. `src/qdu/models/openai_compat.py`: add a fraction-based cap next to the absolute one, e.g.
   `QDU_IMAGE_PIXEL_FRACTION` (default 1.0): the page is downscaled to
   `min(max_image_pixels, fraction × width × height)` pixels, aspect kept (`limit_pixels`).
   Record it in `describe()` so `run_info.json` keeps it.
2. `src/qdu/pipelines/factory.py`: a pipeline name `vision-low` that builds `VisionPipeline`
   on a model created with fraction 0.25, so prediction rows carry `pipeline = "vision-low"`
   (the label table keys on the pipeline name). Add `"vision-low"` to `PIPELINE_NAMES`.
3. `src/qdu/router/labels.py` (`build_labels`) must accept several runs and a path list:
   `qdu router labels results/full-qwen3.5-4b --extra-run results/full-vision-low-qwen3.5-4b`
   merges the latest successful row per (question, path) across runs; `PATHS` becomes the union.
4. `src/qdu/router/evaluate.py`: `PATH_USES_OCR["vision-low"] = False`; `baselines()` adds
   `always-vision-low`.
5. Runbook preset in `notebooks/gpu_run.ipynb` cell 1, next to `"full"` and `"text-logprobs"`:
   `"vision-low": ("full-vision-low-qwen3.5-4b", "vision-low", False)`, and set
   `QDU_IMAGE_PIXEL_FRACTION=0.25` for that preset. Skip cell 8 (OCR): vision needs no OCR.

**Cheap check before any router work (gate A):** run the preset on `full.csv` (12,305 vision
calls at ~1/4 tokens; estimate 2–4 h on the L4, ~4–8 units), commit the run under `results/`
with a README like the others, rebuild labels with the extra run, then compare on val:

- always-vision-low vs always-vision per dataset (score and tokens);
- the oracle over {text, vision-low, vision, hybrid} vs the current oracle: **continue only if it
  is clearly cheaper at equal score** (rule of thumb: ≥10% lower C cost or tokens, outside the
  rerun floor). Otherwise record the negative result and stop A.

---

## 4. Step 3 — Direction B: the evidence location as an action (crop)

**Idea.** S4 (`EvidenceSignals`) already finds the OCR lines that contain the question's terms
and the layout region they sit in. As a routing feature it added nothing; use it to decide
*what to send*:

- **B1, crop path:** send only that region (union of the matched lines' boxes, grown to the
  enclosing layout region from `layout.jsonl.gz`, plus a margin) at full detail instead of the
  whole page; optionally with that region's OCR lines as text.
- **B2, cheap escalation:** answer with text first; if the answer looks unreliable (learned
  cascade features in `qdu.router.learned_cascade.answer_features`), send only the crop instead
  of the whole page. Our cascades failed because each escalation paid a full vision call; a crop
  costs a fraction. This follows "Modality Relevance is not Modality Utility" (escalate only
  where the evidence gap is).

Prior cropping work localizes with model attention or trained heads (ViCrop, Q-Zoom,
C²-Routing); OCR + layout localization, training-free, was not found.

**Code changes:**

1. `src/qdu/pipelines/crop.py`: `CropPipeline(model, ocr, layout)`; needs the page's OCR (from
   the cache) and layout (from `layout.jsonl.gz` or `qdu.layout.PaddleLayout`). If no line
   matches, fall back to the full page (and count how often). Keep the answer instruction from
   `qdu.pipelines.prompts` identical; add a prompt variant that says the image is a crop.
2. Cost: the crop path pays OCR + layout + its model call (`PATH_USES_OCR["crop"] = True`, layout
   time as `page_s`). B2's cost is text + (crop if escalated), via `Policy.called_before`.
3. Runbook preset `"crop-pilot"` on `benchmarks/pilot_300.csv`.

**Cheap check (gate B):** the 300 pilot questions (~0.5 units): crop alone and crop as the
escalation target. Continue to the full set only if the crop keeps vision's score (within the
rerun floor) at clearly fewer tokens, or B2 reaches always-vision's score at lower cost.

---

## 5. Step 4–5 — Retrain, then the one-time test

4. Retrain the three-way router with every path that passed its gate and S5/S6; choose all
   thresholds / λ on val only.
5. Run `qdu router evaluate --split test` once, then the TAT-DQA unseen-data test (TAT-DQA is in
   the test split only). Write the success-criteria table (CLAUDE.md Phase 5): against
   always-hybrid (the pre-presentation's criterion) and against always-vision (the stricter one),
   in scenarios C, B and A.

---

## 6. Evaluation rules taken from the literature

- **Rerun floor:** re-running the text path with identical settings changed 287 of 12,305
  answers (2.3%; bf16 GPU non-determinism) and moved dataset means by ≤ 0.3 points. State it next
  to every difference; do not call differences within it a gain (Routing Is Least Learnable:
  differences under ~4 pp unresolved in their setting).
- **Random-escalation control** for every cascade (B2): escalate the same share of questions at
  random and compare.
- **Two-condition test before a new path** (Pre-Inference Routing): the cheap option must fail
  often enough, and its failures must be predictable from pre-answer features (AUC well above
  0.5). Report both numbers.
- **Matched question-only control** for every new signal (Beyond the Query): a signal counts only
  if it beats the question-only router on held-out data (`--baseline question-only`).
- **Never tune on test** (TableDART chose λ on test). Thresholds on val only.
- **Cost denominators:** per question, scenarios C (OCR per question), B (OCR shared per page),
  A (tokens); measured, not estimated.

---

## 7. Plain-language reporting (for the slides and thesis)

For every step write, in Turkish for the slides and English for the thesis: what we tried, why
(cite the paper that motivated it), the cheap check and its result, and the decision (continue
or stop). Negative results are reported, not hidden. Emre's slide deck is generated outside the
repo; send him the numbers and the one-line conclusion per step.

---

## 8. Budget and order

| Step | Where | GPU cost (est.) |
|---|---|---|
| 1 C: S5/S6, three-way two-stage router | laptop | 0 |
| 1 C: CLIP timing on the L4 | Colab | ~0.3 units |
| 2 A: vision-low run on full.csv (gate A) | Colab | ~4–8 units |
| 3 B: crop pilot on 300 questions (gate B) | Colab | ~0.5 units |
| 3 B: crop on full.csv (only if gate B passes) | Colab | ~5–8 units |
| 4–5 retrain, test, TAT-DQA | laptop | 0 |

Colab L4 bills ~1.94 compute units per hour (Nov 2026). Do not start a GPU run without Emre's
go-ahead; he approves the Drive access himself.

---

## Appendix — code of the 14 Nov val analyses (reproduce before building on them)

Run from the repo root with `uv run --with sentence-transformers python -I <script>`; the CLIP
script must run first (it writes `clip_pages.npz` into the folder given as argument). These are
scratch analyses, not project code: re-implement properly in `qdu` (section 2.1) rather than
committing them as scripts.

### A1. Learning curve (train fraction by document → text_ok AUC)

```python
import random, statistics as st
from pathlib import Path
from qdu.router import read_labels, load_ocr, router_inputs, build_features, EXTRACTORS
from qdu.layout import load_layout
from qdu.router.ablation import predictability
lab = Path('results/full-qwen3.5-4b/labels.csv')
train = read_labels(lab, splits=['train']); val = read_labels(lab, splits=['val'])
ex = [EXTRACTORS[k]() for k in ('s1', 's2', 's3', 's4')]
feats = build_features(router_inputs(train + val, load_ocr(), load_layout()), ex).features
docs = sorted({r.doc_id for r in train})
print('train', len(train), 'questions on', len(docs), 'docs; val', len(val))
for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
    aucs = []
    for seed in (0, 1, 2):
        rng = random.Random(seed); keep = set(rng.sample(docs, max(1, int(frac * len(docs)))))
        sub = [r for r in train if r.doc_id in keep]
        auc, within, _ = predictability(sub, val, feats, tfidf=True, seed=seed)
        aucs.append((auc['text_ok'], within))
        if frac == 1.0: break
    print(f'{frac:>5.0%} train ({len(sub):>5} q): text_ok AUC {st.mean(a for a,_ in aucs):.3f}  within-dataset {st.mean(w for _,w in aucs):.3f}')
```

### A2. CLIP page embeddings and timing (writes clip_pages.npz; arg: output folder)

```python
import sys
import time, json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdu.data import read_manifest, ImageStore
recs = {}
for r in read_manifest(Path('benchmarks/full.csv')):
    recs.setdefault(r.doc_id, r)
m = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device='cpu')
store = ImageStore()
m.encode([store.get(next(iter(recs.values()))).convert('RGB')])  # warm-up
ids, vecs, secs = [], [], []
for i, (doc, r) in enumerate(recs.items()):
    img = store.get(r).convert('RGB')
    t = time.perf_counter(); v = m.encode([img], normalize_embeddings=True)[0]; secs.append(time.perf_counter() - t)
    ids.append(doc); vecs.append(v.astype(np.float32))
    if i % 500 == 0: print(i, f'{np.median(secs)*1000:.0f} ms/page', flush=True)
np.savez_compressed(sys.argv[1] + '/clip_pages.npz', ids=np.array(ids), vecs=np.stack(vecs), secs=np.array(secs))
print('done', len(ids), f'median {np.median(secs)*1000:.0f} ms, mean {np.mean(secs)*1000:.0f} ms per page')
```

### A3. Richer signals → text_ok AUC (arg: folder with clip_pages.npz)

```python
"""Does the router get better with richer signals? Validation AUC of text_ok (Doğukan's
ablation setup: LightGBM, S1-S4 + question TF-IDF), adding a question embedding and a CLIP
embedding of the page image. PCA is fitted on training pages only."""
import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from qdu.router import read_labels, load_ocr, router_inputs, build_features, EXTRACTORS
from qdu.layout import load_layout
from qdu.router.ablation import predictability

S = Path(sys.argv[1])
lab = Path('results/full-qwen3.5-4b/labels.csv')
train = read_labels(lab, splits=['train']); val = read_labels(lab, splits=['val'])
rows = train + val
ex = [EXTRACTORS[k]() for k in ('s1', 's2', 's3', 's4')]
base = build_features(router_inputs(rows, load_ocr(), load_layout()), ex).features

qm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
QE = qm.encode([r.question for r in rows], batch_size=64, normalize_embeddings=True)
qemb = {r.question_id: {f'qe_{i}': float(v) for i, v in enumerate(QE[j])} for j, r in enumerate(rows)}

z = np.load(S / 'clip_pages.npz')
page = dict(zip(z['ids'].tolist(), z['vecs']))
train_docs = sorted({r.doc_id for r in train})
pca = PCA(n_components=32, random_state=0).fit(np.stack([page[d] for d in train_docs]))
def clip_feats(dims):
    out = {}
    for r in rows:
        v = page[r.doc_id] if dims == 512 else pca.transform(page[r.doc_id][None])[0]
        out[r.question_id] = {f'pg_{i}': float(x) for i, x in enumerate(v)}
    return out
pg512, pg32 = clip_feats(512), clip_feats(32)

# CLIP text-image agreement: does the question "look like" the page?
cm = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device='cpu')
QT = cm.encode([r.question for r in rows], batch_size=64, normalize_embeddings=True)
sim = {r.question_id: {'pg_qsim': float(QT[j] @ page[r.doc_id])} for j, r in enumerate(rows)}

def merge(*fs):
    return {q: {k: v for f in fs for k, v in f[q].items()} for q in base}

configs = [
    ('S1-S4 + TF-IDF (now)', base),
    ('+ question embedding', merge(base, qemb)),
    ('+ page image (CLIP, 32-d PCA)', merge(base, pg32, sim)),
    ('+ page image (CLIP, 512-d)', merge(base, pg512, sim)),
    ('+ question emb. + page image (32-d)', merge(base, qemb, pg32, sim)),
]
for name, f in configs:
    auc, within, _ = predictability(train, val, f, tfidf=True, seed=0)
    print(f'{name:<38} text_ok {auc["text_ok"]:.3f}  within-dataset {within:.3f}  beats {auc["beats_vision"]:.3f}', flush=True)
```

### A4. OCR-free router within the always-vision budget (arg: same folder; for the GPU-time variant replace the CLIP page time by 0.005 s)

```python
"""OCR-free router: S1 + S3 (layout) + TF-IDF [+ question embedding] [+ CLIP page], trained on
train to predict text_ok; send to text when p >= t, else vision. Costs include the router's own
layout and CLIP time per page and its per-question time; no OCR unless text is chosen.
Threshold swept on val (as the project's other routers), CIs vs always-vision."""
import sys, time
import numpy as np
import lightgbm as lgb
from pathlib import Path
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from qdu.router import read_labels, router_inputs, build_features, EXTRACTORS, Policy, evaluate
from qdu.layout import load_layout

S = Path(sys.argv[1])
lab = Path('results/full-qwen3.5-4b/labels.csv')
train = read_labels(lab, splits=['train']); val = read_labels(lab, splits=['val'])
rows = train + val
layout = load_layout()
tab = build_features(router_inputs(rows, None, layout), [EXTRACTORS['s1'](), EXTRACTORS['s3']()])
base = tab.features
qm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
t0 = time.perf_counter(); QE = qm.encode([r.question for r in rows], batch_size=64, normalize_embeddings=True); qe_s = (time.perf_counter() - t0) / len(rows)
qe = {r.question_id: QE[j] for j, r in enumerate(rows)}
z = np.load(S / 'clip_pages.npz'); page = dict(zip(z['ids'].tolist(), z['vecs'])); page_s = dict(zip(z['ids'].tolist(), z['secs']))

def design(rs, use_qe, use_pg, vec, fit):
    names = sorted({k for r in train for k in base[r.question_id]})
    blocks = [sparse.csr_matrix(np.array([[base[r.question_id].get(n, np.nan) for n in names] for r in rs]))]
    q = [r.question for r in rs]
    if fit: vec.fit(q)
    blocks.append(vec.transform(q))
    if use_qe: blocks.append(sparse.csr_matrix(np.stack([qe[r.question_id] for r in rs])))
    if use_pg: blocks.append(sparse.csr_matrix(np.stack([page[r.doc_id] for r in rs])))
    return sparse.hstack(blocks, format='csr')

vision_c = sum(r.cost_c['vision'] for r in val) / len(val)
results = []
for name, use_qe, use_pg in [('OCR-free: S1+S3+TF-IDF', False, False), ('+ question emb.', True, False), ('+ CLIP page', False, True), ('+ question emb. + CLIP page', True, True)]:
    vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True)
    xtr, xva = design(train, use_qe, use_pg, vec, True), design(val, use_qe, use_pg, vec, False)
    y = np.array([r.score['text'] >= r.score['vision'] for r in train], dtype=int)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1).fit(xtr, y)
    p = m.predict_proba(xva)[:, 1]
    def policy(t):
        choice = {r.question_id: ('text' if pi >= t else 'vision') for r, pi in zip(val, p)}
        pg = {r.question_id: layout[r.doc_id].latency_s + (page_s[r.doc_id] if use_pg else 0.0) for r in val}
        qs = {r.question_id: sum(tab.seconds[r.question_id].values()) + (qe_s if use_qe else 0.0) for r in val}
        return Policy(f'{name} t={t:.2f}', choice, page_s=pg, question_s=qs)
    best = None
    for t in np.round(np.arange(0.3, 0.96, 0.025), 3):
        pol = policy(t); sc = sum(r.score[pol.choice[r.question_id]] for r in val) / len(val); c = sum(pol.cost(r, 'C') for r in val) / len(val)
        if c <= vision_c and (best is None or sc > best[1]): best = (t, sc, c)
    print(f'{name:<30} best within vision budget: t={best[0]:.3f} score {best[1]*100:.1f} C {best[2]:.2f} s', flush=True)
    results.append(policy(best[0]))
for r in evaluate(val, results, resamples=1000):
    if r.group == 'ALL' and r.method not in ('always-text', 'always-hybrid'):
        s, c = r.score_vs_baseline, r.cost_c_vs_baseline
        print(f'  {r.method:<40} {r.score*100:5.1f}  C {r.cost_c:.2f} s  B {r.cost_b:.2f} s  tok {r.tokens:6.0f}  ' + (f'score {s.estimate*100:+.1f} [{s.low*100:+.1f}, {s.high*100:+.1f}]  C {c.estimate:+.2f} [{c.low:+.2f}, {c.high:+.2f}]' if s else ''))
```
