# Step 2b — Per-Feature Descriptive Findings

All numbers below come from `T2.2.csv` (AUROC) and `T2.3.csv` (Cohen's d). No feature selection here — Step 3 freezes the set.

Feature set after Step 2a's rep_n drop: `trace_length`, `rep_5`, `hedging_formal`, `hedging_reasoning`, `connector_density`, `trace_divergence`.

## 1. Strongest single predictors (reasoning models, MCQ cells)

Range of single-feature AUROC across reasoning-MCQ cells (qwen3-4b, r1-distill, qwq-32b excl. mmlu_pro [partial]; datasets medqa + mmlu_pro):

| feature | min | median | max |
|---|---|---|---|
| `trace_length` | 0.602 | 0.660 | 0.750 |
| `rep_5` | 0.608 | 0.683 | 0.772 |
| `hedging_formal` | 0.555 | 0.661 | 0.684 |
| `hedging_reasoning` | 0.551 | 0.585 | 0.628 |
| `connector_density` | 0.502 | 0.546 | 0.590 |
| `trace_divergence` | 0.519 | 0.589 | 0.634 |

- Strongest by median AUROC on reasoning-MCQ cells: `rep_5` (median 0.683) and `hedging_formal` (median 0.661).
- Weakest: `connector_density` (median 0.546).

## 2. `hedging_formal` vs `hedging_reasoning` — single-predictor AUROC per model

Evidence for whether the formal/reasoning split adds independent signal beyond `hedging_combined`. (Combined is excluded from this pass — see Step 2a — but the split is on the table.)

| model | dataset | formal | reasoning | gap (|f − r|) |
|---|---|---|---|---|
| qwen3-4b | medqa | 0.684 | 0.628 | 0.056 |
| qwen3-4b | mmlu_pro | 0.669 | 0.603 | 0.065 |
| qwen3-4b | trivia_qa | 0.650 | 0.768 | 0.118 |
| r1-distill-llama-8b | medqa | 0.616 | 0.575 | 0.042 |
| r1-distill-llama-8b | mmlu_pro | 0.555 | 0.551 | 0.004 |
| r1-distill-llama-8b | trivia_qa | 0.607 | 0.622 | 0.014 |
| qwq-32b | mmlu_pro | 0.661 | 0.585 | 0.077 |
| qwq-32b | trivia_qa | 0.663 | 0.724 | 0.061 |
| qwen3-4b-nothink | medqa | 0.554 | 0.540 | 0.014 |
| qwen3-4b-nothink | mmlu_pro | 0.586 | 0.597 | 0.011 |
| qwen3-4b-nothink | trivia_qa | 0.587 | 0.568 | 0.019 |
| llama-3.1-8b-instruct | medqa | 0.549 | 0.533 | 0.016 |
| llama-3.1-8b-instruct | mmlu_pro | 0.570 | 0.513 | 0.057 |
| llama-3.1-8b-instruct | trivia_qa | 0.541 | 0.542 | 0.001 |

## 3. `trace_divergence` — single-feature AUROC across all cells

| model | dataset | AUROC | strength (|AUROC−0.5|) |
|---|---|---|---|
| qwen3-4b | medqa | 0.519 | 0.019 |
| qwen3-4b | mmlu_pro | 0.619 | 0.119 |
| qwen3-4b | trivia_qa | 0.779 | 0.279 |
| r1-distill-llama-8b | medqa | 0.583 | 0.083 |
| r1-distill-llama-8b | mmlu_pro | 0.589 | 0.089 |
| r1-distill-llama-8b | trivia_qa | 0.718 | 0.218 |
| qwq-32b | mmlu_pro | 0.634 | 0.134 |
| qwq-32b | trivia_qa | 0.708 | 0.208 |
| qwen3-4b-nothink | medqa | 0.606 | 0.105 |
| qwen3-4b-nothink | mmlu_pro | 0.643 | 0.143 |
| qwen3-4b-nothink | trivia_qa | 0.794 | 0.294 |
| llama-3.1-8b-instruct | medqa | 0.594 | 0.093 |
| llama-3.1-8b-instruct | mmlu_pro | 0.609 | 0.109 |
| llama-3.1-8b-instruct | trivia_qa | 0.626 | 0.126 |

- Median `trace_divergence` AUROC across all 13 cells: **0.622**. Maximum: 0.794. Weak single predictor everywhere; evidence for excluding it from the headline `trace_LR` in Step 3 (decision deferred).

## 4. Direction of separation — signed Cohen's d (positive = higher on CORRECT)

Median signed d across cells, per feature. Negative = the feature is higher on incorrect answers (i.e. a marker of likely-wrong reasoning).

| feature | min d | median d | max d |
|---|---|---|---|
| `trace_length` | -0.919 | -0.490 | -0.195 |
| `rep_5` | -1.004 | -0.366 | -0.032 |
| `hedging_formal` | -0.622 | -0.318 | -0.038 |
| `hedging_reasoning` | -1.043 | -0.272 | -0.062 |
| `connector_density` | -0.152 | +0.082 | +0.875 |
| `trace_divergence` | -1.100 | -0.386 | -0.066 |

- Median d < 0 (higher on INCORRECT — uncertainty markers): `trace_length`, `rep_5`, `hedging_formal`, `hedging_reasoning`, `trace_divergence`.
- Median d > 0 (higher on CORRECT — confidence markers): `connector_density`.
- Expected pattern (length / repetition / hedging higher on wrong, connectors higher on right) holds where the data shows it; see the table above for the actual signed magnitudes.

## 5. Flags & surprises

Cases where the same feature points in opposite directions (correct vs incorrect) on different datasets within one model:

- **r1-distill-llama-8b / `connector_density`** sign flips across datasets — medqa: +0.125, mmlu_pro: -0.014, trivia_qa: +0.451
- **qwen3-4b-nothink / `connector_density`** sign flips across datasets — medqa: -0.131, mmlu_pro: -0.152, trivia_qa: +0.019
- **llama-3.1-8b-instruct / `connector_density`** sign flips across datasets — medqa: -0.117, mmlu_pro: -0.147, trivia_qa: +0.075

### llama-3.1-8b-instruct watch (Step 2a flag)

Step 2a noted llama's `trace_length × rep_5` correlation flipped sign across datasets. Cohen's d table for those two features on llama:

| dataset | feature | d | mean_correct | mean_incorrect |
|---|---|---|---|---|
| medqa | `trace_length` | -0.393 | 401.206 | 428.107 |
| medqa | `rep_5` | -0.032 | 0.021 | 0.021 |
| mmlu_pro | `trace_length` | -0.278 | 336.431 | 375.277 |
| mmlu_pro | `rep_5` | -0.135 | 0.058 | 0.064 |
| trivia_qa | `trace_length` | -0.195 | 190.530 | 203.858 |
| trivia_qa | `rep_5` | -0.244 | 0.012 | 0.016 |

---
STOP. Awaiting joint review before Step 3 (feature freeze + LOFO).