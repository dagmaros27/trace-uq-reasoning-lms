# Step 2d — LOFO findings

All numbers below come from `T2.6.csv`. Positive Δ AUROC = dropping that feature hurts the full-model AUROC (the feature is contributing ON TOP of the other five). Negative Δ = dropping helps (the feature is actively hurting that cell).

Feature set (6): `trace_length`, `rep_5`, `hedging_formal`, `hedging_reasoning`, `connector_density`, `trace_divergence`.

## 1. Features carrying signal on top of the rest — reasoning + MCQ cells

Δ AUROC range across reasoning-MCQ cells (qwen3-4b + r1-distill on medqa + mmlu_pro; qwq-32b mmlu_pro skipped this pass):

| feature | min Δ | median Δ | max Δ |
|---|---|---|---|
| `trace_length` | -0.0029 | +0.0007 | +0.0132 |
| `rep_5` | +0.0033 | +0.0195 | +0.0650 |
| `hedging_formal` | +0.0084 | +0.0165 | +0.0328 |
| `hedging_reasoning` | -0.0027 | +0.0013 | +0.0059 |
| `connector_density` | -0.0033 | -0.0011 | +0.0019 |
| `trace_divergence` | -0.0003 | +0.0024 | +0.0114 |

- Top contributors by median Δ AUROC on reasoning-MCQ cells: `rep_5` (median Δ = +0.0195) and `hedging_formal` (median Δ = +0.0165).
- Smallest median contribution: `connector_density` (median Δ = -0.0011).

## 2. `trace_divergence` — per-cell Δ AUROC

Expectation (Step 2b): weak single predictor on MCQ, strong on trivia_qa. LOFO answers whether it contributes once the other 5 features are present.

| model | dataset | Δ AUROC | 95 % CI |
|---|---|---|---|
| qwen3-4b | medqa | -0.0003 | [-0.0022, +0.0021] |
| qwen3-4b | mmlu_pro | +0.0028 | [-0.0077, +0.0144] |
| qwen3-4b | trivia_qa | +0.0155 | [+0.0051, +0.0266] |
| r1-distill-llama-8b | medqa | +0.0021 | [-0.0049, +0.0087] |
| r1-distill-llama-8b | mmlu_pro | +0.0114 | [-0.0028, +0.0265] |
| r1-distill-llama-8b | trivia_qa | +0.0399 | [+0.0151, +0.0613] |
| qwq-32b | trivia_qa | +0.0174 | [-0.0007, +0.0367] |
| qwen3-4b-nothink | medqa | +0.0322 | [+0.0079, +0.0542] |
| qwen3-4b-nothink | mmlu_pro | +0.0271 | [+0.0072, +0.0484] |
| qwen3-4b-nothink | trivia_qa | +0.1376 | [+0.1075, +0.1688] |
| llama-3.1-8b-instruct | medqa | +0.0207 | [-0.0022, +0.0435] |
| llama-3.1-8b-instruct | mmlu_pro | +0.0231 | [-0.0011, +0.0492] |
| llama-3.1-8b-instruct | trivia_qa | +0.0569 | [+0.0143, +0.0979] |

- Median `trace_divergence` Δ AUROC on MCQ cells: **+0.0161**.
- Median on trivia_qa cells: **+0.0399**.
- Pattern: trace_divergence's contribution is task-dependent — small (often near zero) on MCQ, materially positive on trivia_qa. Evidence for keeping it in the unified feature set: the cost on MCQ is negligible and the benefit on free-form is real.

## 3. `hedging_formal` vs `hedging_reasoning` Δ AUROC — does the split still pay once everything else is in?

Step 2b's single-predictor table showed `hedging_reasoning` strong on free-form. LOFO asks whether that survives once `hedging_formal`, trace_length, rep_5, connector_density and trace_divergence are already in the model.

| model | dataset | Δ formal | Δ reasoning |
|---|---|---|---|
| qwen3-4b | medqa | +0.0165 | +0.0033 |
| qwen3-4b | mmlu_pro | +0.0084 | -0.0007 |
| qwen3-4b | trivia_qa | +0.0038 | -0.0007 |
| r1-distill-llama-8b | medqa | +0.0165 | -0.0027 |
| r1-distill-llama-8b | mmlu_pro | +0.0328 | +0.0059 |
| r1-distill-llama-8b | trivia_qa | -0.0003 | -0.0003 |
| qwq-32b | trivia_qa | +0.0206 | -0.0005 |
| qwen3-4b-nothink | medqa | +0.0090 | -0.0026 |
| qwen3-4b-nothink | mmlu_pro | +0.0277 | -0.0014 |
| qwen3-4b-nothink | trivia_qa | -0.0008 | +0.0040 |
| llama-3.1-8b-instruct | medqa | +0.0088 | +0.0042 |
| llama-3.1-8b-instruct | mmlu_pro | +0.0278 | +0.0007 |
| llama-3.1-8b-instruct | trivia_qa | -0.0012 | -0.0016 |

## 4. `connector_density` — per-cell Δ AUROC, split reasoning vs non-reasoning

Step 2b found `connector_density` flips sign across datasets on the non-reasoning controls. LOFO says whether it adds anything beyond the other 5.

### Reasoning models

| model | dataset | Δ AUROC | 95 % CI |
|---|---|---|---|
| qwen3-4b | medqa | +0.0019 | [-0.0030, +0.0068] |
| qwen3-4b | mmlu_pro | +0.0012 | [-0.0027, +0.0049] |
| qwen3-4b | trivia_qa | +0.0006 | [-0.0032, +0.0043] |
| r1-distill-llama-8b | medqa | -0.0033 | [-0.0121, +0.0057] |
| r1-distill-llama-8b | mmlu_pro | -0.0033 | [-0.0077, +0.0009] |
| r1-distill-llama-8b | trivia_qa | -0.0019 | [-0.0037, -0.0001] |
| qwq-32b | trivia_qa | +0.0031 | [-0.0051, +0.0108] |

### Non-reasoning controls

| model | dataset | Δ AUROC | 95 % CI |
|---|---|---|---|
| qwen3-4b-nothink | medqa | +0.0014 | [-0.0087, +0.0120] |
| qwen3-4b-nothink | mmlu_pro | +0.0016 | [-0.0058, +0.0094] |
| qwen3-4b-nothink | trivia_qa | -0.0018 | [-0.0036, -0.0001] |
| llama-3.1-8b-instruct | medqa | +0.0085 | [-0.0034, +0.0207] |
| llama-3.1-8b-instruct | mmlu_pro | +0.0114 | [-0.0054, +0.0283] |
| llama-3.1-8b-instruct | trivia_qa | -0.0021 | [-0.0064, +0.0023] |

- Median `connector_density` Δ on reasoning cells: **+0.0006**.
- Median on non-reasoning cells: **+0.0015**.
- Reported only — Step 3 decides whether to drop `connector_density` from the non-reasoning trace LR. Do NOT drop now.

## 5. Negative Δ AUROC — features that ACTIVELY hurt some cell

These rows show cases where the median delta is negative AND the upper bound of the 95 % CI is below 0 (a non-trivial signal that the feature is hurting). Reported, not acted upon.

| model | dataset | feature | Δ AUROC | 95 % CI |
|---|---|---|---|---|
| qwen3-4b-nothink | medqa | `hedging_reasoning` | -0.0026 | [-0.0045, -0.0008] |
| r1-distill-llama-8b | trivia_qa | `connector_density` | -0.0019 | [-0.0037, -0.0001] |
| qwen3-4b-nothink | trivia_qa | `connector_density` | -0.0018 | [-0.0036, -0.0001] |

---
STOP. Awaiting joint review before Step 3 (feature freeze + headline modelling).