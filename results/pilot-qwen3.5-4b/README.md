# Pilot: Qwen3.5-4B on 300 questions (30 Oct 2026)

**Setup.** 300 questions from `benchmarks/pilot_300.csv` (100 each from DocVQA, InfographicVQA,
ChartQA; router-train split; at most one question per document). Every question was answered
by all three paths with the same model, Qwen3.5-4B (bf16, thinking off, greedy, max 64 tokens),
served by vLLM 0.32 on a Colab **NVIDIA L4** with prefix and multimodal caches **off**.
OCR: Tesseract 5.3 (PSM 3) on the Colab CPU. Images capped at 4 MP. 900 calls, 0 failures.
Code: commit `eb6e1de` (see `run_info.json`).

Files: `predictions.jsonl` (one row per call), `run_info.json` (GPU, server, settings),
reports with oracle ties broken by measured latency (this folder) and by tokens (`cost-tokens/`).
Regenerate with `uv run qdu benchmark report results/pilot-qwen3.5-4b [--cost tokens]`.

## Results

ANLS, then mean cost per question (latency = model + OCR seconds; tokens = input + output).

| Path | DocVQA | InfographicVQA | ChartQA | **All 300** | Latency | Tokens |
|---|---|---|---|---|---|---|
| Always text (OCR → LLM) | 0.605 | 0.311 | 0.205 | 0.374 | 1.39 s | 429 |
| Always vision | 0.949 | 0.731 | 0.843 | 0.841 | **1.22 s** | 2,103 |
| Always hybrid | 0.949 | 0.783 | 0.835 | 0.856 | 2.46 s | 2,478 |
| **Oracle router** (latency ties) | 0.985 | 0.813 | 0.871 | **0.890** | **1.22 s** | 1,632 |
| **Oracle router** (token ties) | 0.985 | 0.813 | 0.871 | **0.890** | 1.32 s | **1,336** |

ChartQA relaxed accuracy (its standard metric): text 0.250, vision 0.870, hybrid 0.870,
oracle 0.880. Exact match / F1 are in `metrics.csv`.

Answers were re-derived from the raw model output after a cleaning fix (`rescore_info.json`):
the hybrid path copied space thousands separators from the OCR text ("140 858" for gold
"140858"), which both metrics scored as wrong. 9 of 900 answers changed; the oracle did not.

## Findings

1. **There is headroom for a router.** A perfect router is +3.4 ANLS over the best single path
   (hybrid) at half its latency, and +4.9 ANLS over always-vision at the same latency (or 37%
   fewer tokens). The gap appears on all three datasets.
2. **The baseline to beat is always-vision, not always-hybrid.** Always-vision is within
   1.5 ANLS of always-hybrid at half the latency, so it nearly meets the presentation's first
   success criterion on its own. A router is only useful if it beats always-vision.
3. **"Text is cheap" depends on how cost is measured.** The model answers from OCR text in
   0.25 s, but Tesseract on a CPU takes 1.15 s per page, so the text path is *slower* than
   vision in wall-clock time while using 5x fewer tokens. Results are reported under both cost
   measures.
4. **Each path wins somewhere.** Oracle labels (latency ties): text 52, vision 235, hybrid 13.
   Text wins mainly on DocVQA (clean typed pages), hybrid on InfographicVQA.
5. **OCR text shapes the answer format.** On ChartQA, vision and hybrid looked different
   (0.843 vs 0.829 ANLS) mostly because hybrid wrote "140 858" as printed in the OCR text;
   after normalising numbers they are equal on relaxed accuracy (0.870), with one real win each.
6. **OCR quality limits the text and hybrid paths.** Text ANLS is low on DocVQA scans and
   infographics; a stronger OCR (PP-OCRv5) may raise both paths and change the label mix.

## Caveats

- 100 questions per dataset; differences of a few points need bootstrap confidence intervals
  (Phase 5) before they are claimed.
- Latency is specific to an L4 GPU and a Colab CPU for OCR; token counts are hardware-free.
- The "oracle" is an upper bound, not a router: it knows each path's score in advance.
