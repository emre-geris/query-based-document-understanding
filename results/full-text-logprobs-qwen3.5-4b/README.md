# Text path with log-probabilities: Qwen3.5-4B, 12,305 questions (11 Nov 2026)

## Purpose

A confidence for every text-path answer, for the text-then-vision cascade baseline
(`qdu.router.cascade`): the full labelling run (`results/full-qwen3.5-4b`) did not log token
log-probabilities. This run repeats only the text path on every question of
`benchmarks/full.csv` with `QDU_LOGPROBS=1`; each row's `token_logprobs` holds the
log-probability of every generated token.

- Same setup as the labelled run: Qwen3.5-4B on vLLM 0.33.0, one Colab **NVIDIA L4**, bf16,
  thinking off, greedy, max 64 tokens, caches off, vLLM at 65% GPU memory with
  `--max-num-seqs 16`; OCR read from the same PP-OCRv5 cache, so the prompts are identical.
- Code: commit `da228db` (`run_info.json`); runbook preset `RUN = "text-logprobs"`.
- One session, 11 Nov 2026 17:38–19:07 UTC: 12,305 calls, none failed; the 52-question
  shared-GPU check passed first. Audit passes; reports regenerated on a laptop are
  byte-identical to the Colab ones.

The label table, costs and scores of the project stay those of `results/full-qwen3.5-4b`; this
run only supplies confidences.

## Findings

1. **287 answers (2.3%) differ from the labelled run** (ChartQA 88, TAT-DQA 72,
   InfographicVQA 70, DocVQA 57), with the same prompts and greedy decoding: bf16 GPU
   arithmetic is not bit-reproducible across runs. Mean scores move by at most 0.3 points
   (ChartQA 0.421 vs 0.424, DocVQA 0.774 vs 0.775, InfographicVQA 0.477 vs 0.480, TAT-DQA
   0.554 vs 0.554). `read_confidence` leaves these 287 out, since their confidence belongs to
   another answer; the cascade escalates them to vision.
2. **Asking for log-probabilities did not change latency**: median text model time 0.281 s
   against 0.283 s in the labelled run.
3. **The confidence cascade never reaches always-vision.** On the validation split
   (2,174 questions; always-vision 88.0 points, 1.44 s under C), keeping the text answer when
   the answer probability `exp(sum of token log-probabilities)` is at least t:

   | t | score | C s | B s | tokens | answered by text |
   |---|---|---|---|---|---|
   | 0.5 | 78.4 | 1.43 | 0.91 | 1,469 | 62% |
   | 0.8 | 85.3 | 1.82 | 1.31 | 2,276 | 36% |
   | 0.9 | 86.8 | 1.96 | 1.45 | 2,585 | 27% |
   | 0.99 | 87.8 | 2.27 | 1.75 | 3,240 | 8% |

   At always-vision's C cost it is 9.6 points worse; under B and A it is cheaper than vision
   at t ≤ 0.8 but 2.7+ points worse. Mean-token and least-token probabilities do no better.
   The model is confident about many wrong text answers, so a cascade on its own confidence
   is a weak baseline; the oracle (+3.5 points at 1.08 s on validation) shows the room a
   signal-based router has.
