# Related work and where this project stands

Literature review of 14 Nov 2026. EcoDoc was read in full from the ACL PDF; the others from
their arXiv HTML full text (parts of long appendices via summaries, marked "(summary)").
Task-Aware Resolution Optimization (EMNLP 2025) could only be read at abstract level. Check
numbers against the papers before quoting them in the thesis.

## 1. Modality routing for document QA

**Pottanigari et al., *Text, Pixels, or Both? Evaluating Input Representations for Multimodal
Document QA*** (arXiv 2609.22628, Sep 2026). GPT-5.4(-mini) and Gemini 3.1 (Pro, Flash-Lite)
on an internal enterprise KB (5,169 q) and MMLongBench-Doc (1,091 q); in-house OCR, 300-DPI page
images. Image beats text on accuracy (KB oracle pages 0.708 vs 0.611; text+image 0.731) at a
latency and cost premium (3.25 / 4.90 / 4.81 s; $0.61 / $0.91 / $1.29 per 100 q).
Router: **binary text vs image** (text+image excluded), label = "image right and text wrong"
(13.2% positives), **TF-IDF of the question only**, logistic regression. Pooled test:
always-text 63.6%, always-image 67.7%, router 66.2% at ~30% lower median latency than always-image.
So it does **not** beat always-image either. No router feature cost, no resolution or cropping,
no paired significance tests. Their own future work: "document features in addition to the
question text", realistic retrieval, and "paired significance testing over the per-question
correctness scores".

**Rajendran et al., *EcoDoc*** (ACL 2025 Industry, pp. 1530–1537). RAG system (ColPali page
embeddings, GPT-4o). Per retrieved page, a rule-based *modality selector* chooses image or text:
a Query-Intent Classifier (question embedding vs 10 LLM-generated "text" and 10 "page image"
example queries, mean similarity), a Page Content Analyzer (layout detector: is there text?),
then OCR and a Text Relevance Scorer (similarity threshold 0.45) when text is chosen. Baseline:
always image. Accuracy judged by manual inspection, only on queries whose retrieved pages are
correct: DocVQA 0.52 / 0.73 (k = 1 / 4, same as baseline), TAT-DQA 0.65 / 0.69 vs 0.66 / 0.70.
Latency 1.81 vs 4.15 s (DocVQA), 3.35 vs 4.51 s (TAT-DQA); cost reported as **LLM API cost
only** (0.24× and 0.10× of baseline); the selector's own layout and OCR time are not broken out.
No text+image option, no resolution choice, no CIs.

**Dwivedi et al., *SAFE-Cascade*** (CIKM 2026 demo, arXiv 2606.19646). ChartQA only (test
N = 375): OCR (Azure) → gpt-5-mini text answer → Random Forest on post-answer features (OCR
length, question–OCR overlap, answer flags, answer-in-OCR) → escalate to a VLM if p ≥ 0.3.
69.1% [64.0, 74.1] vs full-VLM 67.7% [62.7, 72.3] with 73.1% VLM calls; cost −9.3%. Latency
and cost are **estimated** from pricing profiles, not measured. No text+image path, no
resolution. Future work: pre-routing (deciding before the text call).

**Wei et al., *Routing Is Least Learnable Where It Is Most Valuable*** (NeurIPS 2026, arXiv
2608.06171). Web agents, six observation modes (text, pixels, both), three backbones, 7,686
episodes. "No implementable policy we could construct yields a robust improvement over the
trivial fixed policies"; triage AUROC 0.65–0.72; reruns change 12.1–14.3% of outcomes, so
differences under ~4 pp are unresolved. What holds: sending never-solved tasks to the cheapest
mode cuts cost 9.5–30.6% at unchanged success. Recommendations: report rerun floors next to
mode deltas, compare cascades with a random-escalation control, state cost denominators.

**Kuru et al., *RetrievalRouter*** (arXiv 2608.25625, 2026). Query-only router over five
retrieval pipelines with an accuracy–latency λ: retrieval, not answering.

**From the team's October scan** (PDFs in the project folder; first pages and the
pre-presentation's summaries read):

- **Xing et al., *TableDART*** (ICLR 2026, arXiv 2509.14671). Tables only: a 2.59M-parameter MLP
  gate chooses Text-only, Image-only or Fusion per table–query pair from frozen encoders' features;
  Fusion runs both experts plus an LLM agent (three model calls). The only prior three-way router
  found, but not on document pages, no OCR, and λ was picked on test (pre-presentation note).
- **Rajendran, *Pre-Inference Routing for Cost-Efficient Document Field Extraction*** (arXiv
  2608.06607). Chooses a cheap or expensive extractor per document from 13 intrinsic features
  (OCR confidence, image quality, layout) with a random forest; cost −31–33% (receipts), −77%
  (degraded forms) within 0.02 F1. Routing helps only if (1) the cheap model fails often and
  (2) those failures are predictable from visible features; it offers a two-part test to check
  both before deployment. Pre-inference routing beats a confidence cascade because it avoids
  the extra cheap call.
- **Li, Zhang & Ming, *Beyond the Query*** (arXiv 2609.12437). Adding retrieval-time signals to
  a query-only router gives +0.016 on development data but +0.003 on the held-out set (CI
  includes 0): extra signals must be shown to beat a matched query-only control.
- **Li & Gai, *Modality Relevance is not Modality Utility*** (arXiv 2607.05438). On
  MultiModalQA, a question's apparent visual relevance weakly predicts whether the image is
  needed; answering cheaply from text first, then escalating to VLM evidence only where a
  verifier localizes a visual gap (as compact "sidecars"), recovers always-VLM accuracy with far
  fewer visual calls.
- **Ak et al., *An Empirical Study of VLM Pipelines for Long-Document QA*** (Amazon, arXiv
  2609.29933). Uses **Qwen3.5-4B/9B/27B** and Sonnet 4.5; an oracle over pipelines gains ~13
  points over the best single one, "though evidence-type routing recovers almost none of it".
- MARA (ACM MM 2025), UniversalRAG (ACL 2026), Adaptive-RAG (NAACL 2024): adaptive retrieval
  (query-aligned regions, modality/granularity routing, complexity routing); related, not page
  QA routing.

## 2. Adaptive resolution and cropping

**VisionThink** (arXiv 2507.13348) and **AdaptVision** (CVPR 2026, arXiv 2512.03794), both on
Qwen2.5-VL-7B and **RL-trained**: start from a 1/4-pixel image; the model itself asks for a 2×
resize (VisionThink) or a crop (AdaptVision). Without escalation, 1/4 resolution costs DocVQA
only −0.8 points (95.1 → 94.3) but ChartQA −16.9 and OCRBench −12.7. VisionThink asks for
high resolution on 6.5% of DocVQA questions vs 79% of ChartQA (summary). Visual tokens: ~51%
(VisionThink), 33% (AdaptVision). No OCR text path, no external router.

**C²-Routing** (arXiv 2611.14469). A trained router (SmolVLM-500M features, Grounding-DINO
crops, 207k labelled questions) chooses crop, 256/512-token compression or the original input
for frozen Qwen2.5-VL / InternVL3; −36.5% FLOPs, −42.4% visual tokens at 99.2% of the score.
Documents lose most under compression (e.g. Qwen2.5-VL-7B DocVQA 92.8 → 89.4 balanced). No
OCR/text path.

**Not All Visual Tokens Are Equally Safe to Remove** (arXiv 2608.09176). Giving more tokens to
high-consequence questions (amounts, dates, IDs) helps DocVQA (high-tier error 0.300 → 0.133)
and InfographicVQA, but not TextVQA (cost-weighted error 0.174 vs 0.139 uniform) or ChartQA
(CI includes 0).

**Cropping:** ViCrop (ICLR 2025) finds the region from the model's attention/gradients and
*adds* the crop to the full image (more tokens), tested on LLaVA-1.5 and InstructBLIP only.
Q-Zoom (arXiv 2604.06912) trains a gate and a region proposer on Qwen2.5-VL / Qwen3-VL and
reports −53% visual tokens on document/OCR sets. **None localizes the region from OCR boxes or
document layout**, and none is training-free on recent models with token savings.

**Task-Aware Resolution Optimization** (EMNLP 2025): resolution from image complexity and
uncertainty variance, followed by parameter-efficient fine-tuning (abstract only).

## 3. What the pre-presentation promised (Oct 2026) and where it stands

The October pre-presentation (`Sunum_Literatur_ve_Ozgun_Deger`) named three gaps and a focus:
(1) a router choosing text / image / hybrid on document pages; (2) an evidence-location signal
(where the question's words are on the page and how well they were read); (3) a fair comparison
against all fixed paths, with OCR-inclusive cost and an unseen-data test. Success: near
always-hybrid quality at clearly lower cost, and a significant gain from the evidence signal
(AS3) over the question-only router and EcoDoc; otherwise a controlled negative finding.

Status, 14 Nov 2026 (val): cost goal met (routers −36% vs always-hybrid at equal score);
gap 3 met; gap 1 not yet (the best routers choose only text or vision); AS3 negative (S4 adds
+0.001 AUC, CI includes 0), matching Beyond the Query. Phase 7 keeps the promised identity and
turns the evidence location from a routing feature into an action (direction B).

## 4. Gaps this project can fill

1. **Modality × resolution routing outside the model.** Modality routers (Text/Pixels/Both,
   EcoDoc, SAFE-Cascade) send the image at one resolution; resolution methods (VisionThink,
   AdaptVision, C²-Routing) have no OCR text path and retrain the model or a large router. The
   need for resolution is strongly question- and dataset-dependent (DocVQA −0.8 vs ChartQA −16.9
   at 1/4 pixels), which an external router could exploit without touching the VLM.
2. **The router pays for its own signals, per question.** Text/Pixels/Both and SAFE-Cascade do
   not count feature cost; EcoDoc counts LLM API cost only. Our scenario C charges OCR, layout
   and signal time, and shows that an OCR-dependent router loses its savings.
3. **Three-way choice including text+image.** Text/Pixels/Both and SAFE-Cascade exclude it;
   EcoDoc has no such option.
4. **Evidence-guided cropping from OCR and layout, training-free**, as a cheap path and as a
   cheap *escalation*: answer from text first and, if needed, send only the evidence region
   instead of the whole page. Not found in the cropping literature (attention- or training-based);
   it makes the post-hoc escalation of Modality Relevance ≠ Utility affordable for page QA.
5. **Evaluation protocol.** Paired document-level CIs (asked for by Text/Pixels/Both), a
   measured rerun floor (2.3% of text answers; Routing Is Least Learnable asks for it), measured
   (not estimated) latency, three cost denominators.

Consistent with the literature: Text/Pixels/Both's router (−1.5 vs always-image) and the
web-agent study also fail to beat the best fixed representation; our "no router beats
always-vision" is the known difficulty, while cost savings are what holds.

## Sources

- https://arxiv.org/html/2609.22628 (Text, Pixels, or Both?)
- https://aclanthology.org/2025.acl-industry.109.pdf (EcoDoc)
- https://arxiv.org/html/2606.19646v1 (SAFE-Cascade)
- https://arxiv.org/html/2608.06171 (Routing Is Least Learnable Where It Is Most Valuable)
- https://arxiv.org/pdf/2608.25625 (RetrievalRouter)
- https://arxiv.org/abs/2507.13348 (VisionThink)
- https://arxiv.org/html/2512.03794v2 (AdaptVision)
- https://arxiv.org/html/2611.14469 (C²-Routing)
- https://arxiv.org/html/2608.09176 (Not All Visual Tokens Are Equally Safe to Remove)
- https://arxiv.org/html/2502.17422 (ViCrop)
- https://arxiv.org/html/2604.06912 (Q-Zoom)
- https://aclanthology.org/2025.emnlp-main.795/ (Task-Aware Resolution Optimization, abstract)
- https://arxiv.org/abs/2509.14671 (TableDART)
- https://arxiv.org/abs/2608.06607 (Pre-Inference Routing)
- https://arxiv.org/abs/2609.12437 (Beyond the Query)
- https://arxiv.org/abs/2607.05438 (Modality Relevance is not Modality Utility)
- https://arxiv.org/abs/2609.29933 (VLM Pipelines for Long-Document QA)
- https://arxiv.org/abs/2604.16313 (MARA); https://arxiv.org/abs/2504.20734 (UniversalRAG); https://doi.org/10.18653/v1/2024.naacl-long.389 (Adaptive-RAG)
