# Step 2c — Greedy vs Sampled trace_LR

All numbers below come from `T2c.1.csv` and `T2c.2.csv`. `trace_LR_greedy` uses **1 generation** (4 features: trace_length, rep_5, hedging_formal, connector_density). `trace_LR_sampled` uses **~10 generations** (5 features — adds trace_divergence). trace_divergence has no single-trace analogue, so the comparison is intrinsically asymmetric and any free-form gap reported here mixes two effects: (i) one greedy trace vs the average of 10 sampled traces, and (ii) the absent sampling-based divergence feature.

## 1. Median |Δ AUROC| (sampled − greedy)

- Across all 14 cells: **0.0611** AUROC.
- MCQ cells only (n = 9): median |Δ| = **0.0463**, median signed Δ = +0.0463.
- Free-form (trivia_qa) cells (n = 5): median |Δ| = **0.0681**, median signed Δ = +0.0681.

MCQ vs free-form pattern is the expected one — single-pass model competitive on MCQ, larger gap on free-form (where the absent trace_divergence feature was the strongest single feature on trivia_qa per Step 2b).

## 2. Critical cells — qwen3-4b/medqa and qwen3-4b/mmlu_pro

These two cells are where trace_LR beats semantic_entropy (Step 5). If `trace_LR_greedy` holds up here, the cheap single-pass model wins exactly where it matters.

| cell | n | AUROC greedy | AUROC sampled | Δ (sampled − greedy) | paired 95 % CI | % bootstrap sampled wins |
|---|---|---|---|---|---|---|
| qwen3-4b / medqa | 740 | 0.7331 | 0.7767 | +0.0436 | [+0.0118, +0.0724] | 99.9 % |
| qwen3-4b / mmlu_pro | 730 | 0.8065 | 0.8175 | +0.0110 | [-0.0110, +0.0356] | 83.6 % |

## 3. Cost framing

- `trace_LR_greedy`: **1 generation** per question (the greedy answer the model would have produced anyway).
- `trace_LR_sampled` and `semantic_entropy`: **~10 generations** per question (1 greedy + 10 sampled, or just 10 sampled for SE). Both competing methods require sampling; greedy is the deployment-cheap option.

## 4. Recommendation (not finalised)

Applied rule: if MCQ median Δ AUROC ≤ 0.02, recommend greedy as a viable cheap deployment variant; otherwise report it as MCQ-only.

- MCQ median signed Δ = **+0.0463** > 0.02 → the single-pass model loses too much on MCQ to be recommended as a drop-in replacement.
- Free-form (trivia_qa) median signed Δ = **+0.0681**. Even if this exceeds the threshold, recall that this gap mixes the greedy-vs-sampled effect with the missing trace_divergence feature (which alone explained up to +0.138 AUROC on trivia_qa per Step 2d). So the free-form number is an upper bound on the true greedy-vs-sampled cost.

**Pending**: qwq-32b/mmlu_pro is not in this pass; the overall reasoning-MCQ recommendation is held until that cell is refreshed.

## 5. Single-feature AUROC of each greedy feature (T2c.2)

Median / range of single-feature AUROC across 13 cells, computed on the GREEDY trace:

| feature | min | median | max |
|---|---|---|---|
| `trace_length` | 0.5153 | 0.6240 | 0.7556 |
| `rep_5` | 0.5352 | 0.5998 | 0.7663 |
| `hedging_formal` | 0.5030 | 0.5998 | 0.6865 |
| `connector_density` | 0.5030 | 0.5281 | 0.7279 |

---
STOP. Awaiting joint review before Step 6.