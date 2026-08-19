# Feature-importance ranking — frozen trace_LR (5 features)

LOFO (leave-one-feature-out) on the **frozen 5-feature trace_LR**, using the identical CV protocol as Step 3's canonical fit (`StratifiedKFold(5, shuffle=True, random_state=42)`, standardise inside train folds only). Per-cell Δ AUROC = AUROC(full 5-feat trace_LR) − AUROC(LR on the other 4). **Positive Δ = dropping that feature hurts → the feature is contributing on top of the rest.**

This is the right importance signal for an LR with correlated inputs: single-feature AUROC tells you the standalone strength (reported here as context), but LOFO tells you the marginal contribution given the other features are already in the model. The two together explain why a strong single feature can have small LOFO Δ (because a correlated partner is already carrying the signal).

## Ranking (by median LOFO Δ AUROC on reasoning × MCQ cells)

Reasoning × MCQ is the regime where trace_LR beats the SE baseline (Step 8's central evidence), so the ranking is on those 5 cells (qwen3-4b/medqa, qwen3-4b/mmlu_pro, qwen3-4b/trivia_qa wait no — reasoning models × MCQ datasets only:
qwen3-4b/medqa, qwen3-4b/mmlu_pro, r1-distill/medqa, r1-distill/mmlu_pro, qwq-32b/mmlu_pro).

| rank | feature | median Δ (rsn × MCQ) | range (rsn × MCQ) | median Δ (all 14 cells) | n cells CI > 0 | median single-feat AUROC |
|---|---|---|---|---|---|---|
| 1 | `rep_5` | +0.0248 | [-0.0006, +0.0664] | +0.0038 | 4 / 14 | 0.6204 |
| 2 | `hedging_formal` | +0.0152 | [+0.0100, +0.0345] | +0.0109 | 4 / 14 | 0.5973 |
| 3 | `trace_divergence` | +0.0018 | [-0.0005, +0.0060] | +0.0203 | 7 / 14 | 0.6222 |
| 4 | `connector_density` | +0.0006 | [-0.0076, +0.0027] | +0.0006 | 0 / 14 | 0.5459 |
| 5 | `trace_length` | -0.0014 | [-0.0030, +0.0120] | +0.0024 | 1 / 14 | 0.6484 |

`n cells CI > 0` = number of cells where the LOFO Δ's 95 % bootstrap CI is entirely above zero (the feature is *statistically* contributing on that cell).

## Why this ranking — feature by feature

Reasoning below interleaves the LOFO table, the single-feature AUROC table (Step 2b's T2.2), the Cohen's d table (Step 2b's T2.3) and the pairwise correlation matrix (Step 2a). All numbers traceable.

### Rank 1 — `rep_5`

- **Strongest single feature on reasoning × MCQ** (T2.2 median single-feature AUROC 0.708; max 0.781 on qwq-32b/mmlu_pro). The only single feature that touches the 0.78 ceiling on its own.
- **Largest LOFO Δ on reasoning × MCQ** (median ++0.0248). Removing it costs more than removing any other feature on the cells where trace_LR wins.
- **Cohen's d on reasoning × MCQ averages around −0.4 to −0.8** — repetition of 5-grams in the reasoning trace is the cleanest 'this answer is wrong' marker we have.
- Mechanism: a model that has to repeat itself across sample traces is one that doesn't have a confident single answer to lock onto.

### Rank 2 — `hedging_formal`

- Second-largest LOFO Δ on reasoning × MCQ (median ++0.0152). Smaller single-feature AUROC than rep_5 but a comparable marginal contribution — the two are not redundant.
- Cohen's d consistently negative on reasoning × MCQ (the feature is higher on incorrect answers) — hedge density in formal phrasing tracks model uncertainty directly.
- Why `hedging_formal` and not `hedging_combined` or `hedging_reasoning`: Step 2a showed `hedging_combined ≈ hedging_formal` (|r| > 0.95 on the reasoning models — they are near-duplicates); Step 2d LOFO showed `hedging_reasoning` adds ≈ 0 on top of the other features. The formal lexicon is the one carrying the signal.

### Rank 3 — `trace_divergence`

- **Task-dependent**: median LOFO Δ on reasoning × MCQ is only +0.0018, but on trivia_qa it climbs to >+0.04 across the board (qwen3-4b-nothink/trivia_qa is +0.138, the largest LOFO Δ in the entire study).
- This is the inter-sample disagreement signal — by construction it requires multiple samples (no greedy analogue exists).
- Kept in the frozen set because it's near-free on MCQ and load-bearing on free-form; dropping it would penalise the MCQ-vs-free-form unification of the model.

### Rank 4 — `trace_length`

- Strong as a **single** predictor (T2.2 median 0.663 on reasoning × MCQ; +0.75 on qwen3-4b/medqa).
- But LOFO Δ on reasoning × MCQ is essentially zero (median -0.0014). Why: `trace_length` is correlated with `rep_5` at 0.66–0.77 on the reasoning models (Step 2a). At the LR level, `rep_5` is already capturing the discriminative chunk of the length signal.
- Retained anyway: it was in the original hypothesis, and removing it post-hoc on LOFO grounds would be performance-driven feature selection (which we don't do). Its small marginal contribution is documented honestly.

### Rank 5 — `connector_density`

- Weakest single feature (T2.2 median 0.546 on reasoning × MCQ — barely above chance).
- Smallest LOFO Δ on reasoning × MCQ (+0.0006; three cells have CI entirely below 0, meaning the feature is slightly *hurting* there).
- Cohen's d also flips sign across datasets on three of the five models — the feature is the only one in the set that doesn't have a consistent direction.
- Retained for the same reason as `trace_length`: it was in the theory and we don't perform performance-driven trimming. Its weakness is reported.

## Caveat on this ranking

LOFO measures *marginal* contribution given the other features are present. A high-rank feature here is one that carries unique information; a low-rank feature is one whose signal is already covered by something else. **Low rank ≠ useless.** `trace_length` alone is a strong predictor; it just happens to be highly correlated with `rep_5`, so the LR uses one of them and the LOFO Δ on either is small. If we wanted to deploy a 1-feature model, `trace_length` or `rep_5` (whichever is cheaper to compute) would be a perfectly reasonable choice.

## Files

- `feature_importance_lofo.csv` — full per-cell LOFO table (70 rows = 5 features × 14 cells)
- `feature_importance_ranking.csv` — feature-level summary with rank, sorted by reasoning × MCQ median LOFO Δ
- `feature_importance.md` — this document