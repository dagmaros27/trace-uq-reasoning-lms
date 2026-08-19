# Step 8 — Reasoning vs Non-Reasoning Synthesis

All numbers below come from `T8.1.csv` and `T8.2.csv`, which in turn pull from `T5.1`, `T5.2`, `T6.1`, `T7.3` and `T1.1`. **No new models are fit in this step.** 14 cells.

## Per-group pattern (T8.2)

| model_type | task_type | n_cells | n_cells_trace_beats_se | median Δ AUROC |
|---|---|---|---|---|
| reasoning | mcq | 5 | **3** | +0.0935 |
| reasoning | free_form | 3 | **0** | -0.0486 |
| non_reasoning | mcq | 4 | **0** | -0.0953 |
| non_reasoning | free_form | 2 | **0** | -0.0949 |

*(`trace_beats_se` is CI-based — paired bootstrap 95 % CI strictly above 0.)*

## The reasoning × MCQ trio (the central evidence)

| cell | n | trace_LR | SE | Δ trace − SE | 95 % CI | brier winner | nll winner |
|---|---|---|---|---|---|---|---|
| qwen3-4b / medqa | 740 | 0.7767 | 0.6828 | **+0.0935** | [+0.0511, +0.1334] | tie | tie |
| qwen3-4b / mmlu_pro | 730 | 0.8175 | 0.7225 | **+0.0969** | [+0.0518, +0.1343] | tie | tie |
| qwq-32b / mmlu_pro | 657 | 0.7436 | 0.5957 | **+0.1467** | [+0.0845, +0.2156] | tie | tie |

## r1-distill — the reasoning model that does NOT show the MCQ effect

| cell | n_clean | trace_LR | SE | Δ trace − SE | 95 % CI | trace_beats_se? |
|---|---|---|---|---|---|---|
| r1-distill-llama-8b / medqa | 792 | 0.6701 | 0.6838 | -0.0140 | [-0.0519, +0.0269] | no |
| r1-distill-llama-8b / mmlu_pro | 599 | 0.6977 | 0.7277 | -0.0321 | [-0.0762, +0.0136] | no |
| r1-distill-llama-8b / trivia_qa | 944 | 0.7334 | 0.8341 | -0.1004 | [-0.1279, -0.0751] | no |

**Associated context (T1.1, descriptive only — not a causal claim):**

| dataset | accuracy on clean | n_clean | truncation (greedy) |
|---|---|---|---|
| medqa | 0.521 | 792 | 75 / 1000 (7.5 %) |
| mmlu_pro | 0.496 | 599 | 243 / 1000 (24.3 %) |
| trivia_qa | 0.381 | 944 | 24 / 1000 (2.4 %) |

## Free-form (trivia_qa) — SE wins for every model

| cell | trace_LR | SE | Δ trace − SE | 95 % CI |
|---|---|---|---|---|
| qwen3-4b / trivia_qa | 0.8266 | 0.8714 | -0.0442 | [-0.0652, -0.0234] |
| qwq-32b / trivia_qa | 0.7587 | 0.8074 | -0.0486 | [-0.0797, -0.0175] |
| r1-distill-llama-8b / trivia_qa | 0.7334 | 0.8341 | -0.1004 | [-0.1279, -0.0751] |
| qwen3-4b-nothink / trivia_qa | 0.7964 | 0.8432 | -0.0461 | [-0.0684, -0.0243] |
| llama-3.1-8b-instruct / trivia_qa | 0.6478 | 0.7913 | -0.1437 | [-0.1808, -0.1057] |

## Non-reasoning models — SE wins throughout

| cell | trace_LR | SE | Δ trace − SE | 95 % CI |
|---|---|---|---|---|
| qwen3-4b-nothink / medqa | 0.6528 | 0.6959 | -0.0428 | [-0.0828, -0.0044] |
| qwen3-4b-nothink / mmlu_pro | 0.6909 | 0.7613 | -0.0698 | [-0.1075, -0.0321] |
| qwen3-4b-nothink / trivia_qa | 0.7964 | 0.8432 | -0.0461 | [-0.0684, -0.0243] |
| llama-3.1-8b-instruct / medqa | 0.6541 | 0.7843 | -0.1305 | [-0.1679, -0.0943] |
| llama-3.1-8b-instruct / mmlu_pro | 0.6562 | 0.7783 | -0.1208 | [-0.1664, -0.0771] |
| llama-3.1-8b-instruct / trivia_qa | 0.6478 | 0.7913 | -0.1437 | [-0.1808, -0.1057] |

## Sanity — every number traces back; no recomputation

Spot check (`qwen3-4b / mmlu_pro`):

- `trace_LR_auroc = 0.8175` ← from T5.1 row (method == 'trace_LR')
- `se_auroc = 0.7225` ← from T5.1 row (method == 'semantic_entropy')
- `trace_minus_se_auroc = +0.0969` and `CI = [+0.0518, +0.1343]` ← T5.2 paired bootstrap row
- `brier_winner = 'tie'`, `nll_winner = 'tie'` ← derived from T7.3 deltas + CIs
- `combined_verdict = 'complementary'` ← derived from T6.1 by reapplying the Step-6 rule

---

**Important framing reminder.** This step reports the pattern in the numbers. The sentence that closes the thesis claim — *the trace-feature discrimination signal is a property of reasoning models on multiple-choice tasks* — is the student's to write, supported by this table and F8.1. The script does NOT write that conclusion. The reasoning × MCQ group has **3 of 5 cells with CI-based trace wins**; every other group has 0.

**Pending**: r1-distill / mmlu_pro is included but is a known weakest cell (T1.1 accuracy ≈ 0.50, n_clean = 599 after heavy truncation). Its non-win is associated with these characteristics; the causal interpretation is reserved for the writeup.

STOP. Awaiting joint review.