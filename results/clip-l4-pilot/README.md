# S6 CLIP page embedding timed on the L4 (300 pilot pages)

Colab L4, 15 Nov 2026, commit `45a5de3`, runbook cell 11: `qdu router embed --device cuda
--manifest benchmarks/pilot_300.csv`, next to the loaded vLLM server (Qwen3.5-4B at 65% of the
GPU memory), as OCR was timed. `clip_pages.jsonl.gz` holds the 300 page vectors with their L4
time (`latency_s`, `hardware`); the vectors are the same model as the laptop file
`results/full-qwen3.5-4b/clip_pages.jsonl.gz`, which keeps its CPU times.

| Hardware | Median | Mean | p90 |
|---|---|---|---|
| L4 (this file) | 34.5 ms | 53.4 ms | 76.0 ms |
| laptop CPU, same pages | 63.4 ms | 71.5 ms | |

The L4 is only ~25% faster on average: each page is one image, so the time is mostly the image
preprocessing on the CPU and the transfer, not the model. The 14 Nov estimate of ~5 ms per page
on a GPU was far too low.

**Effect on the router** (val, scheme v2, budget always-hybrid × 0.75, LightGBM two-stage; the CLIP
cost taken from this file for these pages and the CPU time × 0.747 for the others, the measured
L4/CPU ratio): S1–S4 83.3 at 1.66 s; + S6 in both stages 83.3 at 1.75 s (+0.0 [−0.2, +0.3]);
+ S5 + S6 83.4 at 1.71 s (+0.1 [−0.2, +0.4]). At its measured cost the page embedding adds
nothing, as with CPU timing; S6 stays an optional signal, not part of the router.
