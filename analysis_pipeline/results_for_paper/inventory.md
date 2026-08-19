# Data Inventory — Step 0

This is a presence + counts pass over every (model, dataset) cell. 
No analysis was run. Reports against the 9 items from the spec.

## Presence table

Items: 1=raw greedy · 2=raw samples · 3=labels computable · 4=truncation flags · 5=baseline scores (P(True), verbalized, semantic_entropy) · 6=features parquet exists · 7=trace_divergence value present · 8=per-sample signals (logprob_summary + extracted letter/text) · 9=proper-scores / calibration-check outputs

| model | dataset | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-4b | medqa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| qwen3-4b | mmlu_pro | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| qwen3-4b | trivia_qa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| r1-distill-llama-8b | medqa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| r1-distill-llama-8b | mmlu_pro | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| r1-distill-llama-8b | trivia_qa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| qwen3-4b-nothink | medqa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| qwen3-4b-nothink | mmlu_pro | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| qwen3-4b-nothink | trivia_qa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| llama-3.1-8b-instruct | medqa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| llama-3.1-8b-instruct | mmlu_pro | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| llama-3.1-8b-instruct | trivia_qa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| qwq-32b | medqa | — | — | — | — | — | — | — | — | — |  *(no jsonl)*
| qwq-32b | mmlu_pro | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| qwq-32b | trivia_qa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Counts per cell

`n_clean` = neither greedy nor any sample was truncated. 
`n_clean_and_labeled` = the Stage-4 modelling pool.

| model | dataset | n_total | n_truncated_greedy | n_clean | n_clean_and_labeled | parquet_n |
|---|---|---|---|---|---|---|
| qwen3-4b | medqa | 1000 | 101 | 740 | 740 | 1000 |
| qwen3-4b | mmlu_pro | 1000 | 187 | 730 | 730 | 1000 |
| qwen3-4b | trivia_qa | 1000 | 65 | 915 | 915 | 1000 |
| r1-distill-llama-8b | medqa | 1000 | 75 | 796 | 792 | 1000 |
| r1-distill-llama-8b | mmlu_pro | 1000 | 243 | 614 | 599 | 1000 |
| r1-distill-llama-8b | trivia_qa | 1000 | 24 | 944 | 944 | 1000 |
| qwen3-4b-nothink | medqa | 1000 | 34 | 926 | 926 | 1000 |
| qwen3-4b-nothink | mmlu_pro | 1000 | 58 | 918 | 908 | 1000 |
| qwen3-4b-nothink | trivia_qa | 1000 | 2 | 989 | 989 | 1000 |
| llama-3.1-8b-instruct | medqa | 1000 | 16 | 976 | 974 | 1000 |
| llama-3.1-8b-instruct | mmlu_pro | 1000 | 152 | 733 | 731 | 1000 |
| llama-3.1-8b-instruct | trivia_qa | 1000 | 92 | 818 | 818 | 1000 |
| qwq-32b | medqa | — | — | — | — | — |
| qwq-32b | mmlu_pro | 500 | 118 | 317 | 317 | 500 |
| qwq-32b | trivia_qa | 1000 | 23 | 956 | 956 | 1000 |

## Explicit answers to A, B, C, D

### A. Can per-question features be RECOMPUTED from raw traces/samples?

**Yes.** Each jsonl record persists the full text of every generation:
- `greedy.full_output`, `greedy.reasoning_trace`, `greedy.final_answer`
- `samples[i].full_output`, `samples[i].reasoning_trace`, `samples[i].final_answer` for i in 0..9
- Plus the saved baselines (`ptrue.p_true_normalized`, `verbalized_confidence.parsed_confidence`) and the extracted letters / free-text answers per sample.

So lexicon features (hedging, connectors, rep-N), trace length, and answer-distribution features (letter entropy / NLI cluster entropy on the saved `extracted_choice` / `extracted_prediction` per sample) are all recomputable. The aggregated `*.parquet` table is a convenience, not the source of truth.

**One caveat:** `trace_divergence` is a BGE-M3 cosine-distance aggregate. The 8192-dim embeddings themselves are NOT persisted — only the per-question scalar lands in the parquet. Recomputing trace divergence requires the BGE-M3 GPU pass, which needs the VM.

### B. Can features be computed SEPARATELY for the greedy trace vs the sampled traces?

**Yes.** The greedy trace is stored as a single field (`greedy.reasoning_trace`) and the 10 sample traces are stored as a list (`samples[i].reasoning_trace`). Every text-side feature — trace_length, hedging_formal / reasoning / combined, connector_density, rep_3 / rep_4 / rep_5 — is a pure function of one trace's text. So we can compute:

- **Greedy-only features**: apply the feature function to `greedy.reasoning_trace` once per question.
- **Sampled features**: apply to each `samples[i].reasoning_trace`, average over the 10 (this is what the current parquet stores).

The current `data/features/<model>/<dataset>.parquet` only carries the **sampled-averaged** version. A greedy-vs-sampled efficiency comparison would require running the same feature functions on the greedy trace and writing a parallel column set (e.g. `trace_length_greedy`, `hedging_combined_greedy`, ...). The underlying text is already on disk, so this is a pure local CPU recompute — no VM needed.

`trace_divergence` is the exception: it is by definition a *pairwise-over-samples* metric, so it only exists for the 10 sampled traces. There is no greedy-only analogue (a single trace has no pairwise divergence).

### C. Per-cell counts (already in the table above; flagged anomalies below)


- **r1-distill-llama-8b / mmlu_pro**: n_clean = 614 of 1000 (61.4 % clean) — heavy truncation
- **qwq-32b / mmlu_pro**: n_total = 500 (partial run — expected 1000)
- **qwq-32b / mmlu_pro**: n_clean = 317 of 500 (63.4 % clean) — heavy truncation

### D. Is the clean-set definition consistent across cells?

**Yes.** The clean set is defined by two flags computed identically for every record by Stage 3:
- `in_all_clean` = `(greedy.finish_reason != 'length')` AND `all(s.finish_reason != 'length' for s in samples)`
- `correct` is `np.nan` whenever the greedy prediction could not be parsed (MCQ: no A-J letter extractable; free-answer: no usable prediction). Otherwise: MCQ uses letter==gold_answer; free-answer uses `normalize(pred) in gold_normalized_aliases`.

Stage 4 uniformly takes `df[df.in_all_clean & df.correct.notna()]` as its modelling pool. Same code path for every cell.

## Anything missing, inconsistent, or surprising

- **qwq-32b / medqa**: no jsonl — cell intentionally not generated.

Known-by-design points (not bugs):
- `qwq-32b` × `medqa` is not generated — QwQ-32B was added in Phase 4, only mmlu_pro + trivia_qa runs were funded.
- `qwq-32b` × `mmlu_pro` was a partial 500-record run (cost cap); a resume to n=1000 was launched on the VM and is in flight at the time of this inventory. Re-run this script after resume completes to refresh the counts.
- `trace_divergence` embeddings (BGE-M3, 8192-dim) are NOT persisted; only the per-question cosine-distance aggregate is. Recomputing trace_divergence requires the GPU pass.
- Stage-3 parquet column `answer_semantic_entropy` is the letter-entropy on MCQ datasets and the NLI-cluster entropy (DeBERTa-v3-large-mnli, bidirectional entailment) on trivia_qa. Dispatch is by per-record `kind`.
- `ptrue.judgment_token_decoded` etc. on the current jsonls use the v2 (literal True/False) protocol after the rescore on 2026-06-08. Old v1 jsonls archived as `*.v1.jsonl` alongside, not used downstream.

---

**Recommendation**: nothing load-bearing is missing on any cell that exists. The pipeline is recompute-friendly (all source text is on disk). Specifically, a greedy-vs-sampled feature comparison (answer to question B) is feasible from the saved data alone with no new generation runs needed.

Awaiting confirmation before proceeding.