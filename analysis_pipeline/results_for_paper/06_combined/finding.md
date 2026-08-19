# Step 6 — Combined Model (trace_LR + semantic_entropy)

All numbers from `T6.1.csv`. **`full_LR` includes both inputs by construction, so it is best-or-tied-best almost everywhere. We report it as an upper-bound reference, not as our method.**

## 1. Does adding semantic_entropy buy anything over trace_LR in the win cells?

These are the cells where trace_LR beats semantic_entropy alone (Step 5). If `full_LR ≈ trace_LR` here, then the trace features already capture what SE has to offer on these cells.

| cell | n | AUROC full_LR | AUROC trace_LR | Δ (full − trace) | paired 95 % CI | % bootstrap full wins |
|---|---|---|---|---|---|---|
| qwen3-4b / medqa | 740 | 0.7912 | 0.7767 | +0.0148 | [+0.0021, +0.0288] | 98.7 % |
| qwen3-4b / mmlu_pro | 730 | 0.8324 | 0.8175 | +0.0149 | [+0.0032, +0.0271] | 99.6 % |

- qwen3-4b / mmlu_pro: adding SE moves AUROC by **+0.0149**; CI is entirely above zero ([+0.0032, +0.0271]).
- qwen3-4b / medqa: adding SE moves AUROC by **+0.0148**; CI is entirely above zero ([+0.0021, +0.0288]).

## 2. Where SE is the strong method — is full_LR's gain over trace_LR coming from the SE component?

On the SE-strong cells (Step 5: free-form trivia_qa across all models, non-reasoning controls on MCQ, r1-distill on MCQ), we expect `full_LR − trace_LR` to be large (SE is doing the work) and `full_LR − SE` to be small (trace adds little where SE is already strong).

| cell | full_LR − trace_LR | CI | full_LR − SE | CI |
|---|---|---|---|---|
| qwen3-4b / trivia_qa | +0.0541 | [+0.0378, +0.0726] | +0.0104 | [+0.0012, +0.0200] |
| r1-distill-llama-8b / medqa | +0.0387 | [+0.0142, +0.0619] | +0.0253 | [+0.0052, +0.0458] |
| r1-distill-llama-8b / mmlu_pro | +0.0434 | [+0.0158, +0.0719] | +0.0119 | [-0.0096, +0.0311] |
| r1-distill-llama-8b / trivia_qa | +0.1005 | [+0.0754, +0.1279] | -0.0003 | [-0.0063, +0.0060] |
| qwq-32b / mmlu_pro | +0.0055 | [-0.0067, +0.0204] | +0.1526 | [+0.0943, +0.2161] |
| qwq-32b / trivia_qa | +0.0717 | [+0.0462, +0.0965] | +0.0227 | [+0.0058, +0.0399] |
| qwen3-4b-nothink / medqa | +0.0583 | [+0.0311, +0.0862] | +0.0151 | [-0.0007, +0.0320] |
| qwen3-4b-nothink / mmlu_pro | +0.0916 | [+0.0642, +0.1201] | +0.0226 | [+0.0081, +0.0367] |
| qwen3-4b-nothink / trivia_qa | +0.0479 | [+0.0330, +0.0649] | +0.0019 | [-0.0084, +0.0118] |
| llama-3.1-8b-instruct / medqa | +0.1292 | [+0.0967, +0.1612] | -0.0011 | [-0.0096, +0.0070] |
| llama-3.1-8b-instruct / mmlu_pro | +0.1279 | [+0.0906, +0.1649] | +0.0074 | [-0.0041, +0.0185] |
| llama-3.1-8b-instruct / trivia_qa | +0.1323 | [+0.0929, +0.1709] | -0.0112 | [-0.0181, -0.0038] |

- Median full_LR − trace_LR on SE-strong cells: **+0.0650** (SE is the big lift over trace alone).
- Median full_LR − SE on SE-strong cells: **+0.0112** (trace adds little once SE is in).

## 3. Complementary vs redundant — per cell

Classification rule, applied to each cell:

- `full_LR ≥ max(trace_LR, SE) + 0.005` AND `full_LR − each` CI strictly above 0 → **complementary** (both inputs add unique signal).
- otherwise if `full_LR ≈ max(trace_LR, SE)` (within 0.01) → **redundant** (full_LR ≈ stronger of the two).
- otherwise → **mixed** (full_LR adds modestly).

| model | dataset | full_LR | trace_LR | SE | Δ full−trace | Δ full−SE | verdict |
|---|---|---|---|---|---|---|---|
| qwen3-4b | medqa | 0.791 | 0.777 | 0.683 | +0.0148 | +0.1082 | complementary |
| qwen3-4b | mmlu_pro | 0.832 | 0.818 | 0.723 | +0.0149 | +0.1109 | complementary |
| qwen3-4b | trivia_qa | 0.882 | 0.827 | 0.871 | +0.0541 | +0.0104 | complementary |
| r1-distill-llama-8b | medqa | 0.709 | 0.670 | 0.684 | +0.0387 | +0.0253 | complementary |
| r1-distill-llama-8b | mmlu_pro | 0.740 | 0.698 | 0.728 | +0.0434 | +0.0119 | mixed |
| r1-distill-llama-8b | trivia_qa | 0.834 | 0.733 | 0.834 | +0.1005 | -0.0003 | redundant |
| qwq-32b | mmlu_pro | 0.749 | 0.744 | 0.596 | +0.0055 | +0.1526 | redundant |
| qwq-32b | trivia_qa | 0.830 | 0.759 | 0.807 | +0.0717 | +0.0227 | complementary |
| qwen3-4b-nothink | medqa | 0.711 | 0.653 | 0.696 | +0.0583 | +0.0151 | mixed |
| qwen3-4b-nothink | mmlu_pro | 0.784 | 0.691 | 0.761 | +0.0916 | +0.0226 | complementary |
| qwen3-4b-nothink | trivia_qa | 0.845 | 0.796 | 0.843 | +0.0479 | +0.0019 | redundant |
| llama-3.1-8b-instruct | medqa | 0.783 | 0.654 | 0.784 | +0.1292 | -0.0011 | redundant |
| llama-3.1-8b-instruct | mmlu_pro | 0.785 | 0.656 | 0.778 | +0.1279 | +0.0074 | redundant |
| llama-3.1-8b-instruct | trivia_qa | 0.780 | 0.648 | 0.791 | +0.1323 | -0.0112 | mixed |

- Cells classified complementary: **6 / 14**.
- Cells classified redundant (full ≈ max(trace, SE)): **5 / 14**.
- Mixed: **3 / 14**.

## 4. Framing reminder

`full_LR` is, by construction, at least as informative as either input on its own. It is the upper-bound reference, not our method. Treat any cell where `full_LR > trace_LR` as showing that **SE contains some signal trace_LR misses on that cell** — not that the proposed method is `trace_LR + SE`.

**Pending:** qwq-32b / mmlu_pro is not in this pass.

---
STOP. Awaiting joint review before Step 7.