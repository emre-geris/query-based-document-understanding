# Full labelling: Qwen3.5-4B, 12,305 questions × 3 paths (10–11 Nov 2026)

## Purpose

Router labels: every question of the four datasets (`benchmarks/full.csv`, all router splits)
answered by the text, vision and hybrid paths, with each path's cost. This is the label table
that router training and evaluation (Phases 3–5) start from.

- 12,305 questions on 3,586 pages: DocVQA 5,349, InfographicVQA 2,801, ChartQA 2,500,
  TAT-DQA 1,655. 36,915 successful (question, path) pairs, none missing.
- Qwen3.5-4B on vLLM 0.33.0, one Colab **NVIDIA L4**, bf16, thinking off, greedy, max 64
  tokens, prefix and multimodal caches off, `--max-model-len 32768`, image cap 16.8 MP.
- OCR: PP-OCRv5 mobile through RapidOCR 3.10 on ONNX Runtime CUDA (`rapid-mobile-gpu`), on
  the **same GPU** as vLLM, as a single-GPU deployment would run it. OCR runs once per page;
  questions are processed one at a time, so OCR and the model never run at the same moment.
- Code: commit `3ad741c` for the last session (`run_info.json`); every session's commit,
  packages and vLLM flags are in `run_history.jsonl`. Reports regenerated on a laptop with
  `qdu benchmark report` (latency and `--cost tokens`) are byte-identical to the Colab ones;
  `qdu benchmark audit` re-derives every figure from `predictions.jsonl` and passes.

Files: `predictions.jsonl` (one row per call, failed calls included; the latest successful row
per question and path counts), reports with oracle ties broken by latency (this folder) and by
tokens (`cost-tokens/`).

## Results

Primary metric per dataset (ANLS for DocVQA and InfographicVQA, relaxed accuracy for ChartQA,
F1 for TAT-DQA). Seconds per question: C charges each question its page's OCR, B divides it
among the page's questions.

| | Text | Vision | Hybrid | Oracle (latency ties) |
|---|---|---|---|---|
| DocVQA | 0.775 | **0.943** | 0.942 | 0.965 |
| InfographicVQA | 0.480 | **0.792** | 0.789 | 0.852 |
| ChartQA | 0.424 | **0.830** | 0.819 | 0.866 |
| TAT-DQA | 0.554 | 0.547 | **0.570** | 0.636 |
| **All 12,305** | 0.607 | **0.833** | 0.832 | **0.875** |
| Total s, C | 1.08 | 1.45 | 2.32 | 1.15 |
| Total s, B | 0.50 | 1.45 | 1.74 | 0.86 |
| Model s | 0.33 | 1.45 | 1.57 | 0.78 |
| Input tokens | 639 | 3,040 | 3,630 | 1,611 |

OCR time per page (mean): ChartQA 0.23 s, DocVQA 0.77 s, InfographicVQA 0.92 s, TAT-DQA
1.10 s.

Oracle minus always-vision, with 95% CIs from a paired bootstrap that resamples documents
(1,000 resamples, seed 0):

| | Oracle − vision | Hybrid − vision | Text − vision |
|---|---|---|---|
| All (3,586 docs) | +4.2 [+3.9, +4.6] | −0.1 [−0.5, +0.3] | −22.6 [−23.6, −21.5] |
| DocVQA | +2.2 [+1.8, +2.6] | −0.2 [−0.6, +0.2] | −16.8 [−18.2, −15.6] |
| InfographicVQA | +6.0 [+5.0, +7.0] | −0.3 [−1.3, +0.6] | −31.2 [−33.7, −28.7] |
| ChartQA | +3.6 [+2.9, +4.3] | −1.2 [−2.1, −0.2] | −40.6 [−42.8, −38.4] |
| TAT-DQA | +8.9 [+7.7, +10.2] | +2.2 [+0.9, +3.6] | +0.6 [−1.1, +2.4] |

## Findings

1. **The pilot's picture holds at full scale.** Vision and hybrid are tied overall
   (−0.1 points, CI includes 0), and the oracle beats always-vision by +4.2 points
   [+3.9, +4.6] while costing less: 1.15 s against 1.45 s per question under scenario C,
   0.86 s under B, and 1,611 against 3,040 input tokens.
2. **Most of the oracle's gain is cost, not accuracy.** The latency-tie-broken oracle picks
   text for 5,080 questions, vision for 6,942 and hybrid for 283; only 934 questions (7.6%)
   have a single best path (vision 370, text 282, hybrid 282). With GPU OCR, text is the
   cheapest path under C (1.08 s) as well as B, so a router saves most by sending easy
   questions to text.
3. **TAT-DQA behaves differently.** It is the only dataset where hybrid beats vision
   (+2.2 [+0.9, +3.6]) and text is as good as vision; its oracle headroom (+8.9) is the
   largest. Its questions need exact figures from dense financial tables, where OCR text helps.
   TAT-DQA is the unseen-data test set, so the router never trains on it.
4. **Text fails on charts and infographics.** Without the image, text loses 31–41 points on
   InfographicVQA and ChartQA, where layout and graphics carry the answer.
5. **74 answers hit the 64-token limit** (hybrid 38, vision 19, text 17), 0.2% of calls;
   no prediction is empty.

## How the run went

Four Colab sessions (`run_history.jsonl`):

| Session (UTC) | Commit | vLLM memory | What happened |
|---|---|---|---|
| 10 Nov 16:13 | `81bafdd` | 0.80 | Started; restarted 12 minutes later with the OCR GPU-arena fix (`595ade4`) |
| 10 Nov 16:25–22:27 | `595ade4` | 0.80 | DocVQA and the first InfographicVQA pages. OCR's GPU memory grew on large pages: 377 OCR calls (48 pages) failed out of memory, then vLLM itself crashed with CUDA out of memory; 660 calls after it failed to connect |
| 10 Nov 23:14–11 Nov ~00:20 | `595ade4` | 0.65, `--max-num-seqs 16` | vLLM restarted from a terminal script with the flags of `3ad741c`; failed calls retried. Ended when Colab disconnected |
| 11 Nov 08:09–14:04 | `3ad741c` | 0.65, `--max-num-seqs 16` | Runbook from the top: check run passed, OCR pre-pass of the remaining 2,188 pages, the rest of the calls. No failures |

- **All 1,037 failed calls were retried successfully** (connection 660, OCR out of memory 377);
  the failed rows stay in `predictions.jsonl` and are ignored by the reports.
- **The memory change did not change model latency.** Calls on the same datasets before and
  after it have similar medians (InfographicVQA vision 1.02 s at 0.80, 0.91 s at 0.65; text
  0.26 s and 0.30 s). vLLM at 0.65 needs `--max-num-seqs 16`: its default of 256 sequences
  needs more Mamba cache blocks (one per sequence) than fit; calls are sequential anyway.
- **OCR times come from the session that first OCR'd each page**, always on the L4 next to a
  loaded but idle vLLM (`ocr_hardware`). DocVQA and some InfographicVQA pages were OCR'd with
  vLLM at 0.80, the other 2,188 pages at 0.65.
- Model-call throughput was ~34–48 calls per minute (InfographicVQA slowest, ChartQA ~120).

## Caveats

- Latency is hardware-specific (one L4, OCR and model sharing it); token counts are not.
- The oracle has no signal cost; a real router pays for its own OCR and layout (scenario C)
  and will not reach it.
- Questions per document vary (DocVQA ~4.2 per page, ChartQA ~1.7), so CIs resample documents.
