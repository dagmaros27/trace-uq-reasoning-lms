# Step 3 — Frozen trace_LR Feature Set (FINAL)

This document fixes the definitive `trace_LR` model used in every downstream step. After this point, `trace_LR` means *exactly* this 5-feature logistic regression — no refitting with different features, no per-cell tweaks.

## 1. The frozen feature set

**FINAL** — identical on every (model, dataset) cell:

- `trace_length`
- `rep_5`
- `hedging_formal`
- `connector_density`
- `trace_divergence`

## 2. Exclusions (with reasons)

| feature | reason | evidence |
|---|---|---|
| `rep_3` | near-duplicate of `rep_5` (|r| > 0.98) — same construct | Step 2a |
| `rep_4` | near-duplicate of `rep_5` (|r| > 0.98) — same construct | Step 2a |
| `hedging_combined` | near-duplicate of `hedging_formal` (|r| ≈ 0.91–0.995) | Step 2a |
| `hedging_reasoning` | adds ≈ 0 on top of the other five (LOFO median Δ AUROC = +0.0013 on reasoning-MCQ; near-zero or slightly negative across the rest) — the formal/reasoning split is reported descriptively from Step 2b but is not a `trace_LR` input | Step 2d |

## 3. Selection rationale — THEORY + REDUNDANCY, not test-set performance

This set was NOT chosen by ranking features by their AUROC on the evaluation data. The procedure was:

1. **Theory-led starting set.** The hypothesis names six trace features ex ante: a length proxy, a self-repetition score, two hedging variants, a connector density, and an inter-sample divergence. We did not search wider.

2. **Redundancy collapse (Step 2a).** Within the rep-N family and within the hedging family we keep one representative per near-duplicate cluster (|r| > 0.95). The choice of which copy to keep follows the longer-window / union version (`rep_5`, `hedging_formal`'s formal+reasoning union via `hedging_combined` — but on inspection `hedging_formal` carries the same signal as the combined version on every cell, so we keep `hedging_formal` directly and report `hedging_combined`'s near-duplication as evidence).

3. **Drop one near-zero contributor (Step 2d LOFO).** `hedging_reasoning`'s leave-one-out Δ AUROC is ≈ 0 across the board once the other five are present. Removing it costs nothing on average; keeping it adds a parameter the LR has to estimate. Dropped.

Notably: **`trace_length` and `connector_density` are retained despite small LOFO contributions.** They were declared in the hypothesis and we did not want to drop them after seeing the LOFO numbers (that would *be* performance-driven selection). They are in the set because the theory put them there; their actual contribution is reported honestly.

## 4. Findings to carry into the results narrative

(a) **rep_5 is the primary structural signal.** Highest single-feature AUROC across reasoning-MCQ cells (Step 2b median 0.708) and the largest LOFO Δ (Step 2d median +0.0195 on reasoning-MCQ). `trace_length` is correlated with `rep_5` (0.66–0.77 on reasoning cells, Step 2a) and is itself strongly predictive alone (Step 2b median 0.663 on reasoning-MCQ), but adds little **on top of** `rep_5` in the LR (Step 2d median Δ = +0.0007). Both retained; their relative contribution varies by model.

(b) **trace_divergence is task-dependent.** Near-zero LOFO Δ on MCQ (median +0.0161), materially positive on trivia_qa (median +0.0399; up to +0.138 on qwen3-4b-nothink/trivia_qa). Retained in the single set — its MCQ cost is negligible and its free-form benefit is real. The task-dependence is reported, not eliminated by feature selection.


## 5. Flagged for later (not acted on now)

- `connector_density` is weak everywhere (Step 2b single-feature AUROC median 0.546 on reasoning-MCQ; Step 2d median LOFO Δ near zero, with 95 % CIs entirely below 0 on three cells — all on non-reasoning controls or r1-distill on trivia_qa). A reasoning-vs-non-reasoning feature-set split MAY be revisited later, but the headline uses one set on every cell so the comparison stays clean.


## 6. Protocol (locked, propagated to all later steps)

- Stratified 5-fold CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Seed = **42**.

- Standardisation: `StandardScaler` fit on the training rows of each fold; applied to the held-out fold. No leakage.

- Classifier: `LogisticRegression(max_iter=2000, solver='lbfgs')`. Default L2 regularisation, no class weighting.

- Clean+labelled pool per cell: rows with `in_all_clean & correct.notna()` AND no NaN among the 5 features.

- AUROC + 95 % CI via 1000-resample bootstrap of out-of-fold predictions.

- 13 cells this pass (qwq-32b / mmlu_pro added after the resume).


## 7. Sanity check vs Step 2d LOFO (T2.6)

The spec asks: does T3.2's AUROC match T2.6's `auroc_full`?

**Answer up front:** the spec line as written would have them identical, but the LOFO `auroc_full` was fit on **6** features (the frozen set + `hedging_reasoning`). The frozen set drops `hedging_reasoning`, so T3.2 != LOFO `auroc_full` by design. The correct equivalent in T2.6 is `auroc_without` on the row where `feature == 'hedging_reasoning'` — that *is* the same model as the frozen `trace_LR`, and that comparison **does** match to the 4th decimal.


| cell | T3.2 AUROC | LOFO auroc_full (6 feat) | Δ vs full | LOFO auroc_without hr (5 feat) | Δ vs drop_hr |
|---|---|---|---|---|---|
| qwen3-4b / medqa | 0.7767 | 0.7800 | -0.0033 | 0.7767 | +0.0000 |
| qwen3-4b / mmlu_pro | 0.8175 | 0.8168 | +0.0007 | 0.8175 | +0.0000 |
| qwen3-4b / trivia_qa | 0.8266 | 0.8259 | +0.0007 | 0.8266 | +0.0000 |
| r1-distill-llama-8b / medqa | 0.6701 | 0.6673 | +0.0028 | 0.6701 | +0.0000 |
| r1-distill-llama-8b / mmlu_pro | 0.6977 | 0.7036 | -0.0059 | 0.6977 | +0.0000 |
| r1-distill-llama-8b / trivia_qa | 0.7334 | 0.7332 | +0.0002 | 0.7334 | +0.0000 |
| qwq-32b / mmlu_pro | 0.7436 | nan | +nan | nan | +nan |
| qwq-32b / trivia_qa | 0.7587 | 0.7582 | +0.0005 | 0.7587 | +0.0000 |
| qwen3-4b-nothink / medqa | 0.6528 | 0.6502 | +0.0026 | 0.6528 | +0.0000 |
| qwen3-4b-nothink / mmlu_pro | 0.6909 | 0.6895 | +0.0014 | 0.6909 | +0.0000 |
| qwen3-4b-nothink / trivia_qa | 0.7964 | 0.8004 | -0.0040 | 0.7964 | +0.0000 |
| llama-3.1-8b-instruct / medqa | 0.6541 | 0.6584 | -0.0043 | 0.6541 | +0.0000 |
| llama-3.1-8b-instruct / mmlu_pro | 0.6562 | 0.6569 | -0.0007 | 0.6562 | +0.0000 |
| llama-3.1-8b-instruct / trivia_qa | 0.6478 | 0.6462 | +0.0016 | 0.6478 | +0.0000 |

**Protocol verified.** `delta_vs_drop_hr` max absolute deviation = **0.000000** (≤ 1e-4); the frozen `trace_LR` is identical to LOFO's drop-`hedging_reasoning` model, as expected.

---
STOP. Awaiting joint review before Step 4.