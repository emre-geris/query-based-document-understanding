# Crop path on the 300 pilot questions (Phase 7, direction B, gate B)

Colab L4, 15 Nov 2026, commit `45a5de3`, Qwen3.5-4B on vLLM 0.33 (same flags as the full run),
runbook preset `crop-pilot`: `qdu benchmark run --pipelines crop --manifest
benchmarks/pilot_300.csv --layout results/full-qwen3.5-4b/layout.jsonl.gz --stored-ocr
results/full-qwen3.5-4b/ocr.jsonl.gz`. The session started without Drive (its OAuth window could
not be completed from the browser automation); the first 163 answers were written to the Colab
disk, copied to Drive once it was mounted, and the run resumed there (137 more, no failures).
`qdu benchmark audit` passes.

**What the crop path does** (`qdu.pipelines.crop`): the OCR lines that share the most content
terms with the question (as signal S4), grown to the PP-DocLayout regions that contain them, plus
a 3% margin, at least half the page wide and a quarter high; that part of the full-resolution page
is sent with a crop prompt (same answer instruction). With no matching line, the whole page.
It pays the page's OCR (the labelled run's L4 time) and layout (laptop CPU time).

## Results

Same 300 questions, other paths from `results/full-qwen3.5-4b`; score = primary metric × 100,
C = seconds per question incl. OCR and layout, CIs resample documents (1,000, seed 0).

| Path | Score | C s | Tokens | Score − vision | Tokens − vision |
|---|---|---|---|---|---|
| text | 54.7 | 0.94 | 574 | −30.5 [−36.4, −24.3] | −2,049 |
| **crop** | **74.4** | 1.75 | 1,946 | **−10.8 [−14.7, −7.0]** | −677 [−873, −511] |
| vision | 85.2 | 1.31 | 2,623 | – | – |
| hybrid | 87.4 | 2.08 | 3,142 | +2.2 [−0.0, +4.7] | +520 |

Per dataset (crop / vision): ChartQA 76.0 / 87.0, DocVQA 83.7 / 93.9, InfographicVQA 63.5 / 74.7.
The crop covers on average 74% of the page (median 92%); 108 of 300 questions were sent the whole
page.

**B2, crop as the escalation target of a text-first cascade** (296 questions with a text
confidence from `results/full-text-logprobs-qwen3.5-4b`; the least confident share is escalated;
an escalated question pays text plus the crop's model call, its OCR already paid):

| Escalated | text → crop | text → vision | random → crop |
|---|---|---|---|
| 20% | 63.7 @ 1.07 s | 64.9 @ 1.07 s | 58.3 @ 1.16 s |
| 40% | 69.4 @ 1.32 s | 72.2 @ 1.39 s | 61.3 @ 1.38 s |
| 60% | 71.2 @ 1.59 s | 76.9 @ 1.69 s | 65.5 @ 1.60 s |
| 80% | 74.1 @ 1.84 s | 82.9 @ 1.99 s | 69.4 @ 1.81 s |

Always-vision on the same questions: 85.0 at 1.32 s.

## Decision: gate B fails, direction B stops

- The crop loses 10.8 points against the whole page for 26% fewer tokens, and is slower than
  vision under scenario C because it needs OCR to locate the evidence.
- As an escalation target it is barely cheaper than the whole page and clearly worse; no
  escalation share reaches always-vision. Confidence-guided escalation beats the random control
  by 4–6 points, so the escalation signal works; what fails is the crop.
- The offline check before the run predicted this: on the pilot questions whose answer the OCR
  contains, shrinking the crop to ~40% of the page keeps the answer inside it only ~70% of the
  time. The words a question shares with the page are often not where its answer is (a label vs
  its value, a chart title vs its bars).

The crop path stays in the code (`--pipelines crop`) as a recorded negative result.
