# Related work and where this project stands

Literature check of 14 Nov 2026 (web search; abstracts and secondary pages, full papers not yet
read). It positions the router against existing work and motivates the final-phase directions
in CLAUDE.md (Phase 7). Re-check details against the papers before citing numbers in the thesis.

## Modality routing for document QA (closest work)

| Work | What it does | Relation to this project |
|---|---|---|
| Pottanigari et al., *Text, Pixels, or Both? Evaluating Input Representations for Multimodal Document QA* (arXiv 2609.22628, 2026) | Fixed prompt/judge, four commercial models; page images most accurate but costlier; one representation alone correct on 19–25% of items; per-question text-vs-image router (TF-IDF + logistic regression), explicitly a proof of concept | Our question-only baseline (`qdu router baseline question-only`). They suggest document features as future work; we have them (S1–S4, layout, CLIP) and measure that they add little within a C budget |
| Rajendran et al., *EcoDoc* (ACL 2025 Industry) | Per-page image-or-text choice from page characteristics and query intent; OCR, layout and captions precomputed at ingestion; up to 2.29× lower latency, 10× lower cost | Our EcoDoc-style baseline; its precomputation is our scenario B. It does not charge per-query signal cost (our scenario C) |
| Dwivedi et al., *SAFE-Cascade* (arXiv 2606.19646, 2026) | ChartQA only: OCR → text model → learned router escalates to a VLM; matches full-VLM accuracy with 26.9% fewer VLM calls, 9.3% lower estimated cost | Our confidence and learned cascades. On all four datasets with measured latency, escalations pay twice and never reach always-vision at its cost |
| Wei et al., *Routing Is Least Learnable Where It Is Most Valuable* (arXiv 2608.06171, NeurIPS 2026) | Web agents choose text, pixels or both; modes are complementary but no routing policy robustly beats the best fixed mode; reruns flip 12–14% of outcomes; cost savings are what holds | Same pattern as our result (no router beats always-vision; 2.3% run-to-run flips): our negative finding is consistent with the literature |
| Kuru et al., *RetrievalRouter* (arXiv 2608.25625, 2026) | Query-only router over five retrieval pipelines with an accuracy–latency λ | Retrieval, not answering; same λ trade-off as our decision rule |

## Adaptive resolution and cropping (directions A and B)

| Work | What it does | Gap it leaves |
|---|---|---|
| *AdaptVision* (Lin et al., CVPR 2026, arXiv 2512.03794); VisionThink | The VLM starts from a quarter-resolution image and requests crops when needed; trained with RL | Changes and retrains the model; no choice of modality (OCR text vs image) |
| *C²-Routing* (Lu & Chen, arXiv 2611.14469, 2026) | Chooses cropping or compression (token cap 256/512) per image and question | No OCR / text path; not document-specific |
| *Task-Aware Resolution Optimization* (EMNLP 2025); arXiv 2608.09176 | Training-free resolution choice from image complexity; a negative result for cost-aware allocation on document-like tasks | Resolution only; the negative result is a warning for direction A |
| ViCrop, FOCUS, HAVC, ZoomEye (training-free); Q-Zoom (trained, documents) | Crop the region the question needs, from attention maps or search | ViCrop's gain is marginal on recent Qwen-VL models; document-specific, training-free cropping from OCR/layout evidence is the least covered case |

## Cost of the router's own signals (direction C)

No document-QA work found that charges a router, per query, for the signals it computes (OCR,
layout). EcoDoc precomputes them; DISCO (ICLR 2026 workshop, arXiv 2603.23511) compares OCR
pipelines and VLMs without per-query signal cost. Our measurement that OCR-dependent signals
cost more than they save under scenario C (one-stage S1–S4 router: 75.3 vs always-vision 88.0
within its budget) is not reported elsewhere as far as this check found.

## What is already distinctive here

1. Router cost includes its own OCR/layout/signal time; three cost scenarios (C/B/A).
2. One model for all paths, every question labelled on every path, document-level bootstrap
   CIs, and measured run-to-run variation.
3. Finding: with full-resolution pages a strong VLM makes hybrid redundant except on dense
   numeric tables (TAT-DQA).

These are rigour, not a new method; Phase 7 adds the method.

## Sources

- https://arxiv.org/abs/2609.22628 (Text, Pixels, or Both?)
- https://aclanthology.org/2025.acl-industry.109.pdf (EcoDoc)
- https://arxiv.org/pdf/2606.19646 (SAFE-Cascade)
- https://arxiv.org/abs/2608.06171 (Routing Is Least Learnable Where It Is Most Valuable)
- https://arxiv.org/pdf/2608.25625 (RetrievalRouter)
- https://arxiv.org/html/2512.03794v2 (AdaptVision, CVPR 2026)
- https://deeplearn.org/arxiv/868841/c$^2$-routing:-complementary-cropping-and-compression-for-efficient-vision-language-inference (C²-Routing; mirror, arXiv 2611.14469)
- https://aclanthology.org/2025.emnlp-main.795.pdf (Task-Aware Resolution Optimization)
- https://arxiv.org/pdf/2608.09176 (cost-aware token allocation, negative on documents)
- https://arxiv.org/html/2604.06912 (Q-Zoom)
- https://arxiv.org/abs/2603.23511 (DISCO)
