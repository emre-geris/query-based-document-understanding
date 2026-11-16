# vision-low: every question with the page at 1/4 of its pixels (Phase 7, direction A)

Colab L4, 15 Nov 2026 (19:54–21:52 UTC, ~2 h, ~4 compute units), commit `45a5de3`, Qwen3.5-4B on
vLLM 0.33 with the full run's flags, runbook preset `vision-low`: `qdu benchmark run --pipelines
vision-low --manifest benchmarks/full.csv`. The vision pipeline with the page downscaled to 25% of
its pixels (aspect kept, bicubic) before the call (`VISION_LOW_FRACTION`); resizing is outside the
measured model latency, as image encoding is. No OCR. 12,305 / 12,305 questions, no failed call;
`qdu benchmark audit` passes and the reports reproduce byte for byte on a laptop. The browser lost
its connection to Colab once; the runtime kept running and nothing was lost.

`labels.csv` here is the label table with vision-low as a fourth path (`qdu router labels
results/full-qwen3.5-4b --extra-run results/full-vision-low-qwen3.5-4b --extra-paths vision-low`,
split scheme v2); the other columns equal `results/full-qwen3.5-4b/labels.csv`.

## Results (all 12,305 questions, primary metric × 100, C = seconds per question incl. OCR)

| | Vision | **Vision-low** | Vision-low − vision | C s vision → low | Tokens vision → low |
|---|---|---|---|---|---|
| DocVQA | 94.3 | **92.9** | −1.5 [−2.0, −0.9] | 1.68 → 0.52 | 3,738 → 1,036 |
| InfographicVQA | 79.2 | 65.2 | −14.0 [−16.1, −12.0] | 1.88 → 0.51 | 3,561 → 1,030 |
| ChartQA | 83.0 | 70.2 | −12.8 [−14.4, −11.3] | 0.29 → 0.23 | 518 → 190 |
| TAT-DQA | 54.7 | 48.3 | −6.5 [−7.7, −5.1] | 1.71 → 0.60 | 3,760 → 1,012 |
| **All** | 83.3 | 76.0 | −7.3 [−7.9, −6.7] | 1.45 → **0.47** | 3,046 → 860 |

The low-resolution page costs a third of the full one; on scanned documents (DocVQA) it loses only
1.5 points, on infographics and charts, where small print and thin marks carry the answer, 13–14.
It is also cheaper than the text path under scenario C (0.47 vs 1.08 s), since text pays OCR.

**Gate A** (oracle over the extended path set clearly cheaper at equal score): the oracle over
text / vision / hybrid / vision-low scores 88.6 at 0.58 s against 87.5 at 1.15 s for the oracle
over the three original paths: half the cost at a higher score. **Passes.**

**Pre-Inference Routing's two conditions** (val, fitted on train): the cheap path fails often
enough — vision-low scores below vision on 11.4% of questions (InfographicVQA 22%, ChartQA 13%,
TAT-DQA 12%, DocVQA 5%); and its failures are partly predictable before answering — LightGBM on
the OCR-free signals S1 + S3 (+ question TF-IDF) separates them with AUC 0.683 (0.593 with S1
alone; mean within dataset 0.592), mostly from the page's size and layout, not its dataset name.

## The four-path router (val, 2,558 questions; `qdu router two-stage --paths
text,vision,hybrid,vision-low`)

A detail head (stage 1, no OCR) decides whether a page the router sends as an image goes at full
or 1/4 resolution. Thresholds chosen on val within the budget; CIs resample documents (TAT-DQA by
report).

| Budget (val C) | Score | C s | vs always-hybrid (82.6, 2.34 s) | vs always-vision (83.1, 1.48 s) |
|---|---|---|---|---|
| always-vision | 83.2 | 1.42 | +0.6 [−0.2, +1.4], −39% | +0.2 [−0.2, +0.5], −0.05 s [−0.09, −0.02] |
| always-vision × 0.9 | 82.9 | 1.28 | +0.3 [−0.5, +1.1], −45% | −0.2 [−0.6, +0.3], −13% |
| **always-vision × 0.8** | **82.6** | **1.14** | **−0.1 [−0.9, +0.8], −51%** | −0.5 [−1.1, +0.1], −23% |
| always-vision × 0.7 | 82.3 | 1.03 | −0.3 [−1.1, +0.6], −56% | −0.7 [−1.3, −0.1], −30% |
| always-hybrid × 0.75 | 83.4 | 1.73 | +0.8 [+0.1, +1.5], −26% | +0.4 [−0.1, +0.8], +0.26 s |

With the budget at always-vision's cost the router is, for the first time, cheaper than
always-vision (by 3%, CI excludes 0) at an equal score. At 80% of that budget it keeps
always-hybrid's score at half its cost, against 29% for the three-path router. The saving comes
from DocVQA (49% of its pages go at low resolution; 93.9 vs hybrid 93.5 at 1.16 vs 2.58 s), while
infographics and charts stay mostly at full resolution (90%, 89%). On TAT-DQA the tighter budget
leaves little for hybrid (54.1 vs hybrid 56.5, −2.3 [−5.0, +0.5]).
