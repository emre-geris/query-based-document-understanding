# Pilot: Qwen3.5-4B on 300 questions (30 Oct 2026)

**Purpose.** A calibration run before labelling the full datasets: does a router have headroom,
and are the measurement settings right? Several settings were corrected because of it (see
*Changes for the full run*).

**Setup.** 300 questions from `benchmarks/pilot_300.csv` (100 each from DocVQA, InfographicVQA,
ChartQA; router-train split; at most one question per document). Every question was answered
by all three paths with the same model, Qwen3.5-4B (bf16, thinking off, greedy, max 64 tokens),
served by vLLM 0.32 on a Colab **NVIDIA L4** with prefix and multimodal caches **off**.
OCR: Tesseract 5.3 (PSM 3) on the Colab CPU. Images capped at 4 MP. 900 calls, 0 failures.
Code: commit `eb6e1de` for the calls (`run_info.json`); answers re-derived from the raw outputs
and rescored at a later commit (`rescore_info.json`).

Files: `predictions.jsonl` (one row per call), `run_info.json` (GPU, server, settings),
reports with oracle ties broken by measured latency (this folder) and by tokens (`cost-tokens/`).
Regenerate with `uv run qdu benchmark report results/pilot-qwen3.5-4b [--cost tokens]`.

## Results

Each dataset's primary metric: ANLS for DocVQA and InfographicVQA, relaxed accuracy (5%
numeric tolerance) for ChartQA. The oracle picks paths by the same metric. Cost is the mean per
question (latency = model + OCR seconds; tokens = input + output).

| Path | DocVQA | InfographicVQA | ChartQA | **All 300** | Latency | Tokens |
|---|---|---|---|---|---|---|
| Always text (OCR → LLM) | 0.605 | 0.321 | 0.250 | 0.392 | 1.39 s | 429 |
| Always vision | 0.949 | 0.741 | 0.870 | 0.853 | **1.22 s** | 2,103 |
| Always hybrid | 0.949 | 0.793 | 0.870 | 0.871 | 2.46 s | 2,478 |
| **Oracle router** (latency ties) | 0.985 | 0.823 | 0.880 | **0.896** | **1.22 s** | 1,631 |
| **Oracle router** (token ties) | 0.985 | 0.823 | 0.880 | **0.896** | 1.32 s | **1,319** |

ChartQA ANLS, for reference: vision 0.843, hybrid 0.835. ANLS, relaxed accuracy, exact match
and F1 for every dataset are in `metrics.csv`. "All 300" averages the three primary metrics.

Answers were re-derived from the stored raw output twice, without new model calls: space
thousands separators copied from the OCR text ("140 858" for gold "140858"; 9 answers) and
numbered lists ("1. courage / 2. clout / 3. astuteness", of which only the first line was
scored; 3 answers).

## Findings

Brackets are paired bootstrap 95% confidence intervals over questions (10,000 resamples; one
question per document, so questions are independent units).

1. **There is headroom for a router.** A perfect router is +4.3 points over always-vision
   [+2.2, +6.5] at the same latency (or 37% fewer tokens) and +2.6 over always-hybrid
   [+1.0, +4.4] at half its latency. Per dataset the gain over always-vision is +3.6 on DocVQA
   [+0.5, +7.4], +8.3 on InfographicVQA [+3.4, +13.8] and only +1.0 on ChartQA [0.0, +3.0]:
   on charts a router can save cost but hardly gain accuracy.
2. **Vision and hybrid are statistically tied.** Hybrid is +1.7 over vision [-0.6, +4.1], so
   neither is "the best single path"; always-vision costs half as much. A router must
   therefore be shown to beat always-vision, not only always-hybrid.
3. **"Text is cheap" depends on how cost is measured.** The model answers from OCR text in
   0.25 s, but Tesseract on a CPU takes 1.15 s per page, so the text path is *slower* than
   vision in wall-clock time while using 5x fewer tokens. Results are reported under both cost
   measures, and the oracle's labels depend on it: text gets 52 labels with latency ties and
   132 with token ties.
4. **Few questions are won by a single path.** Only one path is best on 20 of 300 questions
   (hybrid 10, vision 7, text 3), plus 4 where text and hybrid tie. 151 are solved equally by
   vision and hybrid, 96 tie across all three paths, and 29 get 0 from every path. The text
   labels are ties that text wins on cost. The router's value therefore lies mostly in
   **saving cost on easy questions**, and partly in picking the one path that works.
5. **Hybrid's InfographicVQA edge may partly come from the 4 MP cap.** 27 of the 100
   infographics were downscaled for the model while OCR read them at full resolution. On those
   pages hybrid leads vision by +11.1 [-3.7, +25.9], on the other 73 by +3.1 [-1.6, +8.4].
   Neither interval excludes zero, so this is a warning, not a result; the full run no longer
   downscales below the model's own limit.
6. **OCR quality limits the text and hybrid paths.** Text scores are low on DocVQA scans and
   infographics; a stronger OCR (PP-OCRv5) may raise both paths and change the label mix.
7. **The all-zero questions are mostly genuine model errors.** A manual look at the 29 found
   counting, multi-step arithmetic and multi-hop infographic questions. Two are artefacts of
   ChartQA's metric, which requires an exact match for text answers: every path answered
   "Facebook Messenger" and "Bridgestone" for gold "Facebook Messenger\*" and "Bridgestone
   (Japan)". We keep the standard metric, with ANLS reported beside it.

## Caveats

- 100 questions per dataset. The overall headroom is significant; several per-dataset
  differences are not (see intervals above).
- "All 300" weights the three datasets equally, mixes ANLS with relaxed accuracy, and the
  sample over-represents rare question types (stratified, at least 5 per type), so it is not
  the natural distribution of any dataset.
- Latency is specific to an L4 GPU and a Colab CPU for OCR (shared with the vLLM process);
  token counts are hardware-free.
- The vLLM flags of this run (caches off, `--max-model-len 16384`) were set by hand and are not
  in `run_info.json`; the server log at the time showed `enable_prefix_caching=False`.
- Whether Qwen3.5 saw these evaluation sets during training is unknown.
- The "oracle" is an upper bound, not a router: it knows each path's score in advance.

## Changes for the full run

| Pilot | Full run |
|---|---|
| Images capped at 4 MP (a quarter of DocVQA/InfographicVQA pages downscaled) | Cap = Qwen3.5's own processor limit (16.8 MP); `--max-model-len 32768` |
| Oracle chose paths by ANLS on every dataset | By each dataset's primary metric (`qdu.evaluation.primary`) |
| List answers scored by their first line | Lists joined into one answer |
| vLLM flags only in the notebook | Recorded in `run_info.json` (`QDU_SERVER_ARGS`) |
| One question per document (manifest) | `qdu benchmark sample --all-questions`; confidence intervals must resample by document |
