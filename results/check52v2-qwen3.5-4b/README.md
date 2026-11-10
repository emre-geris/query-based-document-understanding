# Check run: 52 questions × 3 paths before the full run

## Purpose

A short Colab L4 run (10 Nov 2026) to verify, before the 12k-question full run, that the
new OCR (PP-OCRv5 mobile), the scores and the cost accounting are correct and traceable. It is
a wiring and accounting check, not a result: 13 questions per dataset are far too few for
accuracy claims.

- Manifest `benchmarks/check_52.csv`: 13 questions per dataset (DocVQA, InfographicVQA,
  ChartQA, TAT-DQA), one per document, all router splits, seed 7.
- Qwen3.5-4B on vLLM 0.33.0, Colab L4, bf16, thinking off, prefix and multimodal caches off,
  `--max-model-len 32768`, image cap 16.8 MP (flags in `run_info.json`).
- OCR: PP-OCRv5 mobile (paddleocr 3.7.0, paddlepaddle 3.2.2, oneDNN on), run once per page
  on the same VM's CPU (Intel Xeon @ 2.20 GHz × 12) before any model call.
- Code: commit `1a25166`, clean tree. Reports regenerated on a laptop with
  `qdu benchmark report` are byte-identical to the ones written on Colab;
  `qdu benchmark audit` re-derives every figure from `predictions.jsonl` and passes.

## Results

Primary metric per dataset (ANLS; ChartQA relaxed accuracy; TAT-DQA F1), seconds per
question under scenario C (OCR paid per question; B equals C here, one question per page):

| | Text | Vision | Hybrid | Oracle |
|---|---|---|---|---|
| Score, all 52 | 0.599 | 0.787 | 0.797 | 0.845 |
| Model s | 0.33 | 1.37 | 1.54 | |
| OCR s | 2.94 | 0 | 2.94 | |
| Total s (C) | 3.27 | 1.37 | 4.48 | 1.52 |
| Input tokens | 696 | 3025 | 3670 | 2636 |

Per dataset: `metrics.csv`, `by_dataset.csv`; latency- and token-tie-broken oracle in
`summary.json` and `cost-tokens/`.

## Findings

1. **TAT-DQA pages were 224 × 224 thumbnails.** The release's page PNGs are all 224 × 224;
   a first check run (`check52-qwen3.5-4b`, not kept) scored TAT-DQA at 0.19 F1 with
   0-38 OCR characters per page. Pages are now rendered from the bundled PDFs at 200 dpi;
   this run scores 0.53-0.59.
2. **PP-OCRv5 needs paddlepaddle 3.2.2.** With 3.3.1 oneDNN crashes on Linux as on Windows,
   and without oneDNN OCR took 21-32 s per DocVQA page instead of ~3 s.
3. **CPU OCR costs more than a vision call.** At 2.9 s (DocVQA) to 4.3 s (TAT-DQA) per page,
   OCR alone exceeds a vision call (0.3-2.0 s). Under scenario C the text path (3.27 s) is
   costlier than vision (1.37 s), so a router can only save cost under scenarios A (tokens)
   and B (OCR shared per document), or with faster OCR; under C its gain is accuracy.
4. **The image cap matters only for the largest pages.** The 15 largest pages
   (`smoke-check52-qwen3.5-4b`, not kept) hit the 16.8 MP cap: ~16.4k image tokens and
   ~11.7 s per vision call, with no errors. In `full.csv` only 1-3% of pages reach the cap.

## Caveats

- 13 questions per dataset: no per-dataset conclusion is reliable.
- OCR time is CPU time on a 12-core Colab VM; it is hardware-specific, like all latency.
