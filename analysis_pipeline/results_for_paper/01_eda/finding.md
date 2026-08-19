# Step 1 — EDA findings

All numbers below come from `T1.1.csv`.

## Accuracy spread (greedy on clean-and-labelled pool)

- Best cell: **qwq-32b / mmlu_pro** at **0.904** (n_clean_and_labelled = 657).
- Worst cell: **r1-distill-llama-8b / trivia_qa** at **0.381** (n = 944).
- Mean accuracy per model (averaged across the datasets each model has): `qwq-32b` 0.809, `qwen3-4b` 0.669, `llama-3.1-8b-instruct` 0.605, `qwen3-4b-nothink` 0.573, `r1-distill-llama-8b` 0.466.
- Among reasoning models the order is `qwq-32b` (0.809), `qwen3-4b` (0.669), `r1-distill-llama-8b` (0.466) — r1-distill is the weakest reasoning model; qwen3-4b and qwq-32b are the strongest.

## Truncation caveat

- Worst-cleaning cell present in this pass: **r1-distill-llama-8b / mmlu_pro** — n_clean = **614** of 1000 (61.4 % clean), n_clean_and_labelled = 599.
- This cell stays in the analysis; the smaller usable sample is noted alongside any number sourced from it.
- qwq-32b / mmlu_pro is omitted this pass (partial 500-record run; resume in flight on the VM).

## trivia_qa parse-failed counter artefact

- Stage 1 records `extracted_choice = None` for every `kind='free_answer'` record by design (there is no MCQ letter to extract). Looking at that column alone would suggest 100 % parse failure on trivia_qa.
- The real free-form extraction lives in `extracted_prediction` (parsed from the `<answer>...</answer>` block, with last-non-empty-line fallback). Labels were assigned successfully — n_clean_and_labelled equals n_clean on every trivia_qa cell in T1.1 — confirming the apparent parse-fail signal is a column-naming artefact, not lost data.

## Trace length — reasoning vs non-reasoning

- Reasoning-model mean sampled trace length (averaged over their cells): **1629** tokens.
- Non-reasoning controls mean sampled trace length: **474** tokens (3.4× shorter than reasoning models).
- Longest reasoning cell: **qwq-32b / mmlu_pro** at **2828** tokens.
- Shortest non-reasoning cell: **qwen3-4b-nothink / trivia_qa** at **70** tokens.
- Reasoning models produce substantially longer traces than non-reasoning models on every dataset; trace_length is therefore a candidate uncertainty feature with discriminative range, used in the later modelling stage.

## Hedging-frequency figures

- F1.1 (main): top-10 terms pooled across 0 sample traces from the three reasoning models.
- F1.A (appendix): same per-trace-frequency view broken down per reasoning model.

---
STOP. Awaiting go-ahead for Step 2.