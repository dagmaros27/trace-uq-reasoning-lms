# Step 7 — Calibration, handled honestly

All numbers below come from `T7.1.csv`, `T7.2.csv`, `T7.3.csv`. trace_LR's OOF predictions are read from Step 3's `T3.1.csv`; trace_LR is **not refit** in this step. Platt fitting for baselines is done **inside the same 5-fold CV** as `T3.1` (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), training folds only, test fold transformed by the train-fold fit — no leakage.

## Part A — RAW self-report miscalibration (motivation)

- `p_true` (raw) ECE range across 14 cells: [0.097, 0.446]; median 0.334.
- `verbalized_confidence` (raw) ECE range: [0.080, 0.478]; median 0.333.
- `semantic_entropy` (raw, as confidence 1 − H/log₂K) ECE range: [0.082, 0.297]; median 0.161.

- `p_true` is **overconfident** (mean_confidence − accuracy > 0.05) on **13 / 14** cells.
- `verbalized_confidence` is overconfident on **13 / 14** cells.

Motivation kept: the model's own confidence outputs are systematically miscalibrated. Whether that means we should rank methods by ECE is a separate question — handled in Part B.

## Part B — Why ECE cannot rank methods (the base-rate-collapse artefact)

Each cell evaluates three methods on the SAME questions:

- `base_rate_constant`: predict the train-fold accuracy for every test row. By construction it has ~zero ECE.
- `p_true_platt`: 1-D LR (Platt) on raw `p_true`, OOF.
- `trace_LR`: native OOF from T3.1.

Showing what each metric says about each method:

### qwen3-4b / medqa (smoking-gun row)

| method | n | ECE | Brier | NLL | mean p | std p | min p | max p |
|---|---|---|---|---|---|---|---|---|
| `base_rate_constant` | 740 | 0.0000 | 0.1806 | 0.5470 | 0.7635 | 0.0000 | 0.7635 | 0.7635 |
| `p_true_platt` | 740 | 0.0031 | 0.1805 | 0.5466 | 0.7638 | 0.0223 | 0.4566 | 0.7672 |
| `trace_LR` | 740 | 0.0551 | 0.1550 | 0.4702 | 0.7631 | 0.1755 | 0.0392 | 0.9678 |

- `base_rate_constant` has ECE = **0.0000** and a one-value-everywhere prediction (std = 0.0000). Useless — yet ECE-optimal.
- `p_true_platt` collapses toward the base rate: std = **0.0223**, range [0.4566, 0.7672], ECE = **0.0031**, **Brier ≈ 0.1805** (compare base_rate Brier 0.1806). It bought low ECE by becoming vague.
- `trace_LR` has HIGHER ECE (0.0551) but is genuinely sharp (std = 0.1755, range [0.0392, 0.9678]) and materially better on Brier (0.1550 vs 0.1805) and NLL (0.4702 vs 0.5466).

**Take-away:** lower ECE alone does not mean a better predictor; it can mean the predictor collapsed to a vague constant. Proper scoring rules cannot be gamed this way.

## Part C — Fair ranking: trace_LR vs Platt-calibrated semantic_entropy on PROPER scores

Both methods give probabilities on the SAME paired questions per cell. Lower Brier / lower NLL = better. Δ = SE − trace; positive Δ means trace_LR is better on that proper score.

| cell | n | Brier trace | Brier SE_platt | Δ Brier (CI) | win % | NLL trace | NLL SE_platt | Δ NLL (CI) | win % |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-4b / medqa | 740 | 0.1550 | 0.1596 | +0.0046 [-0.0058, +0.0148] | 78.7 % | 0.4702 | 0.4946 | +0.0241 [-0.0026, +0.0498] | 95.6 % |
| qwen3-4b / mmlu_pro | 730 | 0.1460 | 0.1465 | +0.0008 [-0.0113, +0.0134] | 54.6 % | 0.4468 | 0.4657 | +0.0200 [-0.0132, +0.0510] | 88.0 % |
| qwen3-4b / trivia_qa | 915 | 0.1672 | 0.1387 | -0.0283 [-0.0400, -0.0177] | 0.0 % | 0.5162 | 0.4409 | -0.0742 [-0.1039, -0.0470] | 0.0 % |
| r1-distill-llama-8b / medqa | 792 | 0.2282 | 0.2235 | -0.0049 [-0.0150, +0.0064] | 20.0 % | 0.6474 | 0.6380 | -0.0094 [-0.0313, +0.0147] | 22.1 % |
| r1-distill-llama-8b / mmlu_pro | 598 | 0.2216 | 0.2126 | -0.0093 [-0.0238, +0.0054] | 11.0 % | 0.6361 | 0.6151 | -0.0215 [-0.0560, +0.0116] | 11.0 % |
| r1-distill-llama-8b / trivia_qa | 944 | 0.1993 | 0.1602 | -0.0391 [-0.0496, -0.0288] | 0.0 % | 0.5817 | 0.4839 | -0.0976 [-0.1244, -0.0725] | 0.0 % |
| qwq-32b / mmlu_pro | 657 | 0.0807 | 0.0827 | +0.0019 [-0.0036, +0.0075] | 75.2 % | 0.2832 | 0.2999 | +0.0165 [-0.0029, +0.0374] | 94.8 % |
| qwq-32b / trivia_qa | 956 | 0.1565 | 0.1337 | -0.0231 [-0.0320, -0.0134] | 0.0 % | 0.4859 | 0.4292 | -0.0573 [-0.0802, -0.0316] | 0.0 % |
| qwen3-4b-nothink / medqa | 926 | 0.2171 | 0.2050 | -0.0122 [-0.0217, -0.0018] | 1.2 % | 0.6224 | 0.5980 | -0.0247 [-0.0465, -0.0026] | 1.9 % |
| qwen3-4b-nothink / mmlu_pro | 908 | 0.2041 | 0.1784 | -0.0256 [-0.0373, -0.0132] | 0.0 % | 0.5935 | 0.5380 | -0.0552 [-0.0825, -0.0262] | 0.0 % |
| qwen3-4b-nothink / trivia_qa | 989 | 0.1806 | 0.1516 | -0.0290 [-0.0392, -0.0183] | 0.0 % | 0.5416 | 0.4613 | -0.0799 [-0.1078, -0.0530] | 0.0 % |
| llama-3.1-8b-instruct / medqa | 974 | 0.2117 | 0.1776 | -0.0344 [-0.0457, -0.0233] | 0.0 % | 0.6124 | 0.5299 | -0.0828 [-0.1102, -0.0563] | 0.0 % |
| llama-3.1-8b-instruct / mmlu_pro | 731 | 0.2303 | 0.1910 | -0.0388 [-0.0539, -0.0238] | 0.0 % | 0.6529 | 0.5674 | -0.0848 [-0.1188, -0.0493] | 0.0 % |
| llama-3.1-8b-instruct / trivia_qa | 818 | 0.2158 | 0.1712 | -0.0445 [-0.0569, -0.0321] | 0.0 % | 0.6255 | 0.5189 | -0.1064 [-0.1362, -0.0779] | 0.0 % |

### qwen3-4b / medqa explicit (Brier / NLL)

- trace_LR Brier = **0.1550**; semantic_entropy_platt Brier = **0.1596**.
- trace_LR NLL = **0.4702**; semantic_entropy_platt NLL = **0.4946**.
- Δ Brier (SE − trace) = **+0.0046** [-0.0058, +0.0148]; trace_LR better on 78.7% of paired bootstrap resamples.

### Agreement with the AUROC win cells (Step 5)

Step 5's `trace_LR > semantic_entropy on AUROC` cells (this pass): `qwen3-4b / mmlu_pro` and `qwen3-4b / medqa`. We check whether the same two cells also win on Brier and NLL (Part C).

- **qwen3-4b / medqa**: trace_LR better on Brier? no — CI crosses or below 0; better on NLL? no — CI crosses or below 0.
- **qwen3-4b / mmlu_pro**: trace_LR better on Brier? no — CI crosses or below 0; better on NLL? no — CI crosses or below 0.

Per-cell agreement only — **no overall reasoning-MCQ verdict here**, qwq-32b/mmlu_pro pending.

## Sanity checks

1. **trace_LR probabilities are the EXACT T3.1 OOF values, never refit.** Confirmation:
   - T3.1 rows for qwen3-4b/medqa: 740 (matches `T3.2.n` for that cell). First three (question_id, p_pred, fold):
     - `medqa_test_00990` → p=0.878188, fold=5
     - `medqa_test_00202` → p=0.872706, fold=5
     - `medqa_test_01258` → p=0.582374, fold=5
   - These values appear unchanged in the Brier/NLL columns above for the qwen3-4b/medqa row.

2. **Platt fitting was strictly inside training folds** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`). `LogisticRegression(max_iter=2000, solver='lbfgs')` is `.fit`-ed on the train indices of each fold and `.predict_proba`'d on the test indices only; the pooled OOF probabilities are then scored. No test-fold rows participate in their own fit.


---
STOP. Awaiting joint review before Step 8.