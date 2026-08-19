# Trace-Based Uncertainty Estimation — Analysis Pipeline (Stage 2–4)

## Purpose and audience

This document specifies the **analysis** that runs on the generation output produced by Stage 1 (`README_generation_pipeline.md`). It is written for two readers:
- a **coding agent** implementing the pipeline, and
- a **methodology supervisor** auditing the design.

Every non-obvious choice below was decided deliberately; the agent must implement exactly these and must not silently change them (see "Decisions the agent must not make alone" at the end).

**Research question:** Can features extracted from a reasoning model's chain-of-thought *trace* (not just its final answer) estimate uncertainty — i.e., predict whether the model's answer is correct — and do they add value over standard final-answer uncertainty methods?

**This run's scope:** MedQA, two reasoning models — DeepSeek-R1-Distill-Llama-8B and Qwen3-4B (both reasoning; NO non-reasoning control in this run, so H3 is NOT tested here). The pipeline must be written **generically over model and dataset** so MMLU-Pro, free-form datasets (e.g. TriviaQA), and non-reasoning controls can be added later by pointing it at new generation files.

---

## Inputs

Stage 1 produced, per (model, dataset):
- `data/generations/{dataset}_{model_short}.jsonl` — one record per question.
- `data/generations/{dataset}_{model_short}_manifest.json` — run config, token IDs, counts.

Each JSONL record contains (see Stage 1 schema): `question_id`, `dataset`, `answerable`, `question`, `options`, `gold_answer`, a `greedy` object, a `samples` array of 10, a `verbalized_confidence` object, a `ptrue` object, and per-generation `logprob_summary` (NOT full per-token logprobs — those were not stored this run), plus `tag_parse_status`, `extracted_choice`, `choice_method`, and `finish_reason`/truncation indicators per generation.

---

## Global conventions (apply everywhere)

### Correctness label
- **Correct = BPE-cleaned `extracted_choice` of the GREEDY generation exactly equals `gold_answer` (letter match).** Use the cleaned text (the `Ġ`/`Ċ` cleaning from Stage 1). Applies to clean answers too — no harm.
- Questions whose greedy `extracted_choice` is `null` (parse fail) cannot be labeled correct/incorrect → excluded from correctness analysis (reported separately).

### Cleanliness / truncation handling — IMPORTANT
- **Primary analysis set = "all-clean": questions where NO generation (greedy + all 10 samples) was truncated** (`finish_reason != "length"` everywhere). This avoids two confounds: (i) trace length is right-censored at the token cap, (ii) repetition correlates with truncation.
- The pipeline must also compute and **report**:
  - count and % of questions in the all-clean set vs total,
  - the **greedy-clean-only** count (greedy not truncated, samples may be), for reference,
  - **accuracy on clean vs truncated questions** (one comparison) — to disclose that the clean set skews toward easier items.
- All headline results (AUROC, ECE, etc.) are computed on the **all-clean set**. Truncated questions are described, not modeled.
- Source of trace features: the **10 samples** (see features). A question needs its 10 samples clean to be in the all-clean set, which is consistent.

### Where each quantity comes from
- **Correctness, P(True), verbalized confidence:** from the **greedy** generation.
- **All trace features (length, hedging, connectors, repetition, divergence):** computed over the **10 sampled traces**, then aggregated to one value per question (mean across samples, except divergence which is inherently across-sample). Rationale: keeps all trace features on the same 10-sample basis; greedy is reserved for the answer/correctness side.
- **Trace region:** features are computed on the **reasoning portion only** (text between `<think>`/`</think>`), not the post-`</think>` answer.
- **Tokenizer for length:** **the model's own tokenizer** (per-model analysis; never compare raw length across models).

### Frozen lexicons (freeze BEFORE computing; do not tune to results)
P-hacking guard: these lists are fixed once and reported in full in an appendix. Do not adjust them after seeing results.

- **Hedging — formal (verbatim from Ulinski & Hirschberg 2019):** the relational + propositional hedge terms (e.g. think, maybe, perhaps, possibly, probably, suppose, seem, likely, unsure, sort of, kind of, roughly, more or less, approximately, etc.). Use the published list verbatim.
- **Hedging — reasoning extension (frozen, declared as our addition):** wait, hmm, actually, reconsider, on second thought, but wait, hold on, let me think again, not sure, not certain, let me double-check, let me verify, re-examine, alternatively, or maybe, then again, scratch that, correction.
- **Causal/assertion connectors (frozen, exploratory, NEUTRAL framing — NOT assumed to mean "certainty"):** therefore, thus, hence, consequently, so, because, since, implies, it follows that, as a result, this means, which means.

---

## Stage 2 — Inspection (analysis type 1)

Goal: let a human eyeball the data before trusting any model. Produce stats, distributions, and example dumps. Per (model, dataset):

### Stats to compute and save (as a table + printed summary)
- N total, N all-clean, N greedy-clean, N parse-fail (greedy `extracted_choice == null`).
- Accuracy (greedy correct rate) on: all-clean, truncated, and overall.
- Truncation breakdown: greedy / per-sample / verb-conf / p_true (counts and %).
- `tag_parse_status` distribution (strict / no_open_tag / no_close_tag / no_tags).
- `choice_method` distribution (which regex matched).
- Verbalized confidence: distribution (histogram), % null.
- P(True): distribution of `p_true_normalized`, % null.

### Distributions to plot (save each as PDF)
- Trace length (mean over samples), split by correct vs incorrect.
- Each lexicon feature (hedging formal, hedging reasoning, connectors), correct vs incorrect.
- Repetition score, correct vs incorrect.
- Trace divergence, correct vs incorrect.
- Answer semantic entropy (letter-entropy), correct vs incorrect.
- P(True) and verbalized confidence, correct vs incorrect.

### Example dumps (for manual inspection)
Print **2 examples from each category**: correct, incorrect, truncated-greedy, parse-fail. For each: the question, gold answer, greedy answer + extracted choice, the greedy reasoning trace (or first/last 1000 chars if huge), and the computed feature values. Save to a readable text/markdown file.

---

## Stage 3 — Feature generation (analysis type 2)

Compute one feature row per question, save to `data/features/{dataset}_{model_short}.parquet` (and CSV mirror). Features are computed on the **10 samples' reasoning traces** and aggregated to one value per question.

### Feature list (all per-question)

**Trace features (our candidate UQ signals):**
1. **trace_length** — token count of reasoning region (model tokenizer), **mean over 10 samples**.
2. **hedging_formal** — count of formal-hedge terms **per token**, mean over samples.
3. **hedging_reasoning** — count of reasoning-extension hedges per token, mean over samples.
4. **hedging_combined** — formal + reasoning hedges per token, mean over samples. (Keep all three: formal, reasoning, combined — we compare which works.)
5. **connector_density** — connector terms per token, mean over samples. (Exploratory, neutral.)
6. **repetition_score** — `rep-5` (primary): fraction of repeated 5-grams within a trace (1 − distinct_5grams/total_5grams), mean over samples. **Also compute rep-3 and rep-4** as a robustness check (report that results hold across n-gram sizes). rep-5 chosen as primary because reasoning-loop pathologies ("wait… wait… reconsider…") manifest as longer verbatim repeats; shorter n risks flagging incidental phrase reuse. Welleck et al. (2020) and Holtzman et al. (2020) report repetition across a range of n, so reporting 3/4/5 matches standard practice.
   - **Text normalization before n-gram extraction:** lowercase the trace and collapse consecutive whitespace to a single space, then tokenize for n-grams. This is standard, low-stakes text-prep.
   - **Known limitation (do NOT try to fix with regex):** traces heavy in math/code (e.g. printed arrays, coordinate sequences, repeated LaTeX) can inflate repetition legitimately as repetitive *text*. Lowercasing does not address this. We do NOT strip math/code (that risks corrupting traces and introducing bias); instead we note it as a limitation and rely on the correlation/robustness checks in Stage 5 to flag if repetition is tracking content type rather than uncertainty.
7. **trace_divergence** — embedding-based dispersion across the 10 traces: embed each full reasoning trace with a long-context sentence-embedding model (NO 512-token NLI limit), compute mean pairwise cosine **distance** among the 10 embeddings. One value per question. (On probation — if it proves uninformative or unstable, we drop it; record it regardless.)

**Baselines (standard final-answer UQ):**
8. **answer_semantic_entropy** — for MCQ (MedQA): Shannon **entropy over the discrete choice distribution** across the 10 samples' `extracted_choice` (letters). (For free-form datasets added later: NLI/bidirectional-entailment clustering over final answers, then cluster entropy — NOT in scope this run.)
9. **p_true** — `p_true_normalized` from greedy.
10. **verbalized_confidence** — `parsed_confidence`/100 from greedy (0–1).

**Metadata columns:** `question_id`, `model`, `dataset`, `correct` (bool), `in_all_clean` (bool), `greedy_truncated` (bool), `n_samples_clean` (int).

### Notes for the agent
- Lexicon matching: case-insensitive, word-boundary aware (don't match "so" inside "also"); multi-word terms matched as phrases. Per-token normalization uses the model-tokenizer token count of the reasoning region.
- Repetition: compute rep-4 per trace; if a trace has < 4 tokens, score = 0.
- Divergence embedding model: a long-context embedder (e.g. a model with ≥2k token context). Record which model + version is used.
- Every feature must be computable independently per question (no cross-question fitting) so there is **no leakage across CV folds**.
- Handle nulls explicitly (e.g. verb-conf null) — record as NaN, never silently impute; the modeling stage decides handling.

---

## Stage 4 — Modeling and comparison (analysis type 3)

Run **per model** (R1-Distill and Qwen3 separately — do NOT pool). Target = `correct` (binary). Use the **all-clean set** only.

### Preprocessing
- **Standardize all continuous features** (z-score) before logistic regression. Fit the scaler **inside each CV fold on training data only**, apply to the fold's test data (no leakage).
- Rows with NaN in a required feature: for single-feature models, drop NaN rows for that feature; for the combined model, decide one consistent policy (recommend: drop rows with any NaN among the modeled features, and report how many) — flag count to the user.

### Cross-validation
- **Stratified k-fold (k=5)**, stratified on `correct` (preserves class balance per fold).
- All features and the scaler are fit only on training folds. AUROC/ECE are computed on held-out folds and aggregated (report mean across folds).
- Fixed random seed; record it.

### Models to fit and compare (per model, per CV)
1. **Each baseline alone** (single-feature "model" = the raw score as the predictor): answer_semantic_entropy, p_true, verbalized_confidence. (For ranking metrics, the raw score is the predictor; no LR needed, but report consistently.)
2. **Each trace feature alone** (single-feature AUROC): trace_length, hedging_formal, hedging_reasoning, hedging_combined, connector_density, repetition_score, trace_divergence.
3. **Our combined trace model:** logistic regression on the trace features (the hedging variant chosen by the alone-comparison — start with combined; also fit a version using formal+reasoning separately).
4. **Full combined model:** logistic regression on trace features + baselines together.

### Feature contribution analysis
- **Single-feature AUROC** (each feature alone) — already in (2)/(1) above.
- **Leave-one-out** on the combined trace model: drop each feature, refit, measure AUROC change vs the full combined trace model. Report the drop per feature.
- (Both together show "what each feature does alone" and "what it adds on top of the rest.")

### Metrics
- **AUROC** — for every model/feature above (it is threshold-free and works on raw scores).
- **ECE** — computed **only on probabilistic outputs**: the LR models (combined trace, full combined), p_true, verbalized_confidence, and the normalized semantic-entropy-as-confidence if expressed as a probability. **NOT on raw trace features** (length etc. are not probabilities). Use a standard binning (e.g. 10 equal-width bins); report n_bins.
- **Risk–coverage curves** — for the LR models and the baselines; plot and also report a summary (e.g. AURC or accuracy at 80% coverage). Save plots as PDF.

### Bootstrap confidence intervals (REQUIRED)
- For each reported **AUROC**, compute a **95% bootstrap CI** by resampling the held-out predictions with replacement (e.g. 1000 resamples). Report point estimate + CI.
- Purpose (for the supervisor): with ~hundreds of clean questions, a single AUROC number cannot show whether an improvement over a baseline is real or sampling noise. Overlapping CIs ⇒ difference may be noise; clearly separated CIs ⇒ the improvement is reliable. This is what makes the "our method beats baseline X" claim defensible rather than just "our number is bigger."
- Where comparing two methods (e.g. combined trace vs p_true), also report the **bootstrap distribution of the AUROC difference** and the fraction of resamples where our method wins (a simple, honest significance statement).

---

## Stage 5 — Additional analyses (recommended, low cost)

These run on the same feature table; include them.

1. **Feature correlation matrix** — among all features (and with `correct`). Reveals redundancy (e.g. does repetition just track trace_length? does connector_density add anything beyond hedging?). Save heatmap PDF.
2. **Repetition vs truncation robustness** — since repetition correlates with hitting the token cap: confirm repetition still predicts correctness **within the all-clean set** (it does by construction, since all-clean excludes truncated). Additionally report repetition's correlation with trace_length to check it isn't just length in disguise.
3. **Calibration plots** (reliability diagrams) for the probabilistic methods — visual companion to ECE. PDF.
4. **Per-class feature summary table** — mean ± std of each feature for correct vs incorrect, with effect size (e.g. standardized mean difference). Gives an at-a-glance "which features differ most between right and wrong answers."
5. **Cross-model consistency (descriptive)** — once both models are processed, a side-by-side table of single-feature AUROCs for R1 vs Qwen3: do the same features work for both? (Descriptive only; not pooled modeling.)

---

## Outputs

- `data/features/{dataset}_{model_short}.parquet` (+ CSV).
- `results/{dataset}_{model_short}/` containing: stats tables (CSV), all figures (**PDF**), modeling results (CSV: per-method AUROC + CI, ECE, AURC), leave-one-out table, correlation heatmap, example dumps (markdown).
- A **Jupyter notebook** that reproduces the inspection + headline tables/plots interactively, AND the same logic available as runnable scripts (`stage2_inspect.py`, `stage3_features.py`, `stage4_model.py`, `stage5_extra.py`). Both notebook and scripts are required.
- `results/run_manifest.json` for the analysis run: input files used, feature list, lexicon versions (or hashes), embedding-model id, CV seed, library versions.

---

## Acceptance criteria

1. Feature table exists with one row per question, all features + metadata columns, NaNs explicit.
2. Inspection stats + distribution PDFs + example dumps produced for each (model, dataset).
3. All headline metrics computed on the **all-clean set**; clean/truncated counts and accuracy gap reported.
4. Per-model (not pooled) modeling: baselines-alone, trace-features-alone, combined trace LR, full combined LR.
5. Single-feature AUROC **and** leave-one-out both reported.
6. AUROC for every method with **95% bootstrap CI**; pairwise win-fraction for our method vs each baseline.
7. ECE only on probabilistic outputs; risk-coverage curves saved as PDF.
8. Additional analyses (correlation matrix, per-class summary, calibration plots) produced.
9. No leakage: scaler + any fitting done inside CV folds only; features computed per-question.
10. Both notebook and scripts run end-to-end; analysis manifest written.

---

## Decisions the agent must NOT make alone (ask first)
- Changing the cleanliness rule (all-clean primary set) or modeling truncated data.
- Pooling models instead of per-model analysis.
- Changing which generation supplies correctness/P(True)/verb-conf (greedy) vs trace features (10 samples).
- Altering or tuning any frozen lexicon after seeing results.
- Imputing NaNs silently, or changing CV to a single split.
- Dropping bootstrap CIs.
- Adding/removing features from the locked list.
- Using a different correctness definition than BPE-cleaned greedy letter match.

## Out of scope this run
- Free-form datasets and NLI-based answer semantic entropy (only letter-entropy needed for MedQA).
- Non-reasoning control models and the H3 (reasoning-specific) test.
- Token-level entropy methods (full per-token logprobs were not stored; would need a `logprob_mode=full` regeneration).
- Abstention / answerability (Phase 2) and the 3-bucket separation analysis.

---

## Implementation checklist

Work top to bottom. Each stage writes to disk and is independently re-runnable.

### Setup
- [ ] Read both generation JSONL files + manifests; confirm record counts match manifests.
- [ ] Confirm schema fields present on a sampled record (greedy, samples×10, verbalized_confidence, ptrue, logprob_summary, tag_parse_status, extracted_choice, finish_reason).
- [ ] Set and record a global random seed; capture library versions for the analysis manifest.
- [ ] Freeze the three lexicons (hedging-formal verbatim, hedging-reasoning, connectors) into a versioned file; hash them into the manifest. Do this BEFORE computing features.

### Stage 2 — Inspection
- [ ] Compute counts: N total, N all-clean, N greedy-clean, N parse-fail.
- [ ] Compute accuracy on all-clean, truncated, overall; report the clean-vs-truncated accuracy gap.
- [ ] Truncation breakdown (greedy / per-sample / verb-conf / p_true).
- [ ] `tag_parse_status` and `choice_method` distributions.
- [ ] Verbalized-confidence and P(True) distributions (+ % null).
- [ ] Distribution plots (correct vs incorrect) for every feature → save PDF.
- [ ] Dump 2 examples each: correct, incorrect, truncated-greedy, parse-fail → markdown.

### Stage 3 — Features
- [ ] Define clean-set membership per question (`in_all_clean`, `greedy_truncated`, `n_samples_clean`).
- [ ] trace_length (mean over samples, model tokenizer, think-region only).
- [ ] hedging_formal / hedging_reasoning / hedging_combined (per token, mean over samples; case-insensitive, word-boundary, phrase-aware).
- [ ] connector_density (per token, mean over samples; neutral framing).
- [ ] repetition: rep-3, rep-4, rep-5 (mean over samples; lowercase + collapse whitespace first; rep-5 primary).
- [ ] trace_divergence (embed each of 10 traces with a long-context embedder; mean pairwise cosine distance; record embedder id/version).
- [ ] answer_semantic_entropy (letter-distribution entropy over 10 samples' extracted_choice — MCQ form).
- [ ] p_true (greedy p_true_normalized), verbalized_confidence (greedy parsed/100).
- [ ] Metadata columns + explicit NaNs (no silent imputation).
- [ ] Save `data/features/{dataset}_{model_short}.parquet` (+ CSV).

### Stage 4 — Modeling (per model, all-clean set only)
- [ ] Stratified 5-fold CV on `correct`; fixed seed.
- [ ] Standardize features inside each fold (fit on train fold only — NO leakage).
- [ ] Baselines alone: answer_semantic_entropy, p_true, verbalized_confidence.
- [ ] Trace features alone (single-feature AUROC): each of the trace features.
- [ ] Combined trace LR (start with hedging_combined; also a formal+reasoning-separate variant).
- [ ] Full combined LR (trace features + baselines).
- [ ] Leave-one-out on the combined trace model.
- [ ] AUROC for every method + 95% bootstrap CI (≥1000 resamples).
- [ ] Pairwise bootstrap: our combined trace vs each baseline — report win-fraction + AUROC-difference distribution.
- [ ] ECE on probabilistic outputs only (LR models, p_true, verbalized_confidence); report n_bins.
- [ ] Risk-coverage curves + summary (AURC or acc@80% coverage) → PDF.

### Stage 5 — Additional analyses
- [ ] Feature correlation matrix (+ with `correct`) → heatmap PDF.
- [ ] Repetition robustness: confirm signal within all-clean; report correlation of repetition with trace_length and across rep-3/4/5.
- [ ] Calibration / reliability diagrams for probabilistic methods → PDF.
- [ ] Per-class feature summary (mean ± std, standardized mean difference) correct vs incorrect.
- [ ] Cross-model consistency table (R1 vs Qwen3 single-feature AUROCs) once both processed.

### Wrap-up
- [ ] Save all tables (CSV) + figures (PDF) under `results/{dataset}_{model_short}/`.
- [ ] Notebook reproduces inspection + headline tables/plots; scripts run end-to-end.
- [ ] Write `results/run_manifest.json` (inputs, feature list, lexicon hashes, embedder id, CV seed, library versions).
- [ ] Verify all 10 acceptance criteria above are met.
