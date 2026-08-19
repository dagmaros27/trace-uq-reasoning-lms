# Trace-Based Uncertainty Estimation — Generation Pipeline

## Purpose

This project studies whether features extracted from a reasoning model's **chain-of-thought trace** (not just its final answer) can estimate uncertainty — i.e., predict whether the model's answer is correct. This README specifies the **data generation pipeline only** (the expensive, run-once stage). Feature extraction and analysis are downstream and run on the saved files; they are out of scope here except where the schema must support them.

**Core design principle:** Generation is run once and saved exhaustively to disk. All feature extraction, baselines, and analysis read from saved files and never require re-running the model. Capture everything that depends on model internals (logprobs) or extra prompts (verbalized confidence, P(True)) during generation, because they cannot be recovered from text later.

---

## Environment

- **Hardware:** single NVIDIA A100 40GB (GCP `a2-highgpu-1g`).
- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (reasoning model, ~16GB BF16, fits 40GB).
- **Inference engine:** vLLM (required — naive HuggingFace `.generate()` is too slow for 10-sample generation at scale).
- **Python:** 3.10+.
- **Key packages:** `vllm`, `transformers`, `torch`, `datasets<=3.6.0`, `pandas`, `numpy`, `tqdm`.

---

## Locked design decisions (do not change without asking)

These were decided deliberately. The agent must implement exactly these:

1. **Per question, generate:**
   - **1 greedy answer** at `temperature=0` — this is "the answer" used for correctness labeling and P(True).
   - **10 sampled answers** at `temperature=0.7, top_p=0.95` — used for uncertainty/disagreement features.
   - **1 verbalized-confidence response** — a separate prompt asking the model to state confidence 0–100 (Tian et al. style) after answering.
   - **1 P(True) scoring pass** — feed the question + the greedy answer back, ask "(A) True (B) False". **REASON-THEN-JUDGE:** let the model produce its reasoning trace, then read the probability of the "True" vs "False" token at the point where it emits the judgment (i.e., after `</think>`). Do NOT force an immediate first-token judgment — the model is trained to reason first, so the True/False probability is read at the end of reasoning. (If this proves too slow at scale, we may switch to a forced-immediate variant later — but default is reason-then-judge.)

2. **Capture logprobs** on the greedy and sampled generations. Set vLLM `logprobs=5` (capture the top-5 token distribution at each position, not just the single sampled token), so token-level entropy remains computable later. Storage is cheap; regeneration is not.

3. **Reasoning trace and final answer must be split** on the model's think tags. DeepSeek-R1-Distill uses `<think>` ... `</think>`. The reasoning portion is between the tags; the final answer is everything after `</think>`. Store both separately AND store the full raw output (so nothing is lost if tag-splitting needs revisiting). **The splitter must handle malformed tags gracefully** — at temperature 0.7 the model sometimes drops the closing `</think>`, duplicates it, or varies the tag. Use a fallback: if strict tag-matching fails, capture reasoning up to the first clear answer/option selection, and log the record as a tag-parse fallback (do not crash).

4. **Datasets (Phase 1):** MedQA and MMLU-Pro, raw versions (multiple-choice). **Target: 200 questions per dataset to start (validate the pipeline and check for signal), then scale to ~1,000+ per dataset for final results — 200 is too small for defensible AUROC once split into correct/incorrect.** Correctness = exact match on the chosen option letter. (Abstention datasets come later — not in scope now, but the schema must not block them: include `answerable` and `dataset` fields.)

5. **Sampling count is 10.** No pilot/main distinction in settings — the "smoke test" uses identical settings to the full run, just fewer questions, so all data is poolable.

6. **DeepSeek-R1-Distill has NO think/no-think toggle.** It always reasons. Do not attempt to disable thinking. The HuggingFace card recommends forcing output to begin with `<think>\n` to ensure the model reasons; implement this.

---

## Output schema

One JSON record per question. Write as JSON Lines (one JSON object per line) to allow appending and resuming.

```json
{
  "question_id": "string — unique, stable id",
  "dataset": "medqa | mmlu_pro",
  "answerable": true,
  "question": "full question text including options for MCQ",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "gold_answer": "A",

  "greedy": {
    "full_output": "raw model output including think tags",
    "reasoning_trace": "text between <think> and </think>",
    "final_answer": "text after </think>",
    "extracted_choice": "A | B | C | D | null",
    "logprobs": [ ... per-token logprob objects as returned by vLLM ... ]
  },

  "samples": [
    {
      "full_output": "...",
      "reasoning_trace": "...",
      "final_answer": "...",
      "extracted_choice": "B | null",
      "logprobs": [ ... ]
    }
    // exactly 10 of these
  ],

  "verbalized_confidence": {
    "prompt_used": "the exact prompt string",
    "raw_response": "...",
    "parsed_confidence": 85
  },

  "ptrue": {
    "prompt_used": "the exact prompt string",
    "p_true_token_prob": 0.0,
    "p_false_token_prob": 0.0,
    "p_true_normalized": 0.0
  },

  "generation_config": {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "greedy_temp": 0.0,
    "sample_temp": 0.7,
    "top_p": 0.95,
    "n_samples": 10,
    "max_tokens": 4096,
    "seed_or_run_id": "..."
  }
}
```

### Schema notes
- `extracted_choice` is the parsed letter from the final answer (after `</think>`), via a robust regex/parser. If parsing fails, set `null` and log it — do not crash.
- `p_true_normalized = p_true / (p_true + p_false)`.
- Store `prompt_used` strings so prompts are auditable and reproducible.
- Keep `logprobs` even if large. If storage becomes a problem, make logprob capture a config flag (default ON), but default to capturing.

---

## Prompt specifications

### Main generation prompt (greedy + samples)
- Use the DeepSeek-R1-Distill chat template.
- Force the response to begin with `<think>\n` (per the model card) so the model reasons.
- For MCQ: present the question and lettered options, instruct the model to reason and then give the final answer as a single letter.

### Verbalized confidence prompt (separate call)
- After obtaining the answer, prompt the model to provide a confidence score from 0 to 100 that its answer is correct.
- Parse the integer from the response; if multiple numbers, take the one explicitly tied to confidence; if unparseable, store raw + `null`.
- Follow Tian et al. ("Just Ask for Calibration") numeric 0–100 style.

### P(True) prompt (separate scoring pass)
- Construct: question + the greedy final answer + "Is the proposed answer correct? (A) True (B) False".
- **REASON-THEN-JUDGE:** allow the model to produce its `<think>` reasoning, then read the probability of the "True"/"A" token vs "False"/"B" token at the judgment position (after `</think>`), NOT at the first generated token.
- Rationale: DeepSeek-R1-Distill is trained to emit `<think>` first on every output. Reading the first token would capture `<think>`/reasoning tokens, not the judgment — so the True/False probability must be read where the model actually emits its verdict.
- Compute and store raw probs + normalized P(True) = p_true / (p_true + p_false).
- NOTE: tokenization of "A"/"True" matters — verify which exact token the model emits (e.g., " A", "A", " True"). The agent must inspect the tokenizer and confirm the correct target token before computing. Log the chosen token strings.

---

## Downstream feature note: trace disagreement vs. answer semantic entropy

This is out of scope for stage 1 (these are computed later, in stage 2, from saved samples) — BUT the generation must save what these need, and the agent should understand the distinction so it stores the right text cleanly.

There are two separate uncertainty signals, and the rule for whether to use NLI clustering is driven by **whether the text is free-form**, not by whether it's a "trace" or an "answer":

1. **Trace disagreement (our feature) — ALWAYS uses NLI clustering.**
   Reasoning traces are long free-form paragraphs. Even on multiple-choice questions, the *reasoning* varies between samples even when the final letter is the same. So trace disagreement is computed by bidirectional entailment (NLI) clustering across the 10 reasoning traces, then entropy over clusters (Kuhn et al. 2023 machinery, applied to the reasoning text).

2. **Answer semantic entropy (baseline) — depends on the dataset:**
   - **Multiple-choice (MedQA, MMLU-Pro):** the final answer is a single letter (A/B/C/D). NLI clustering is unnecessary and redundant. Semantic entropy here = **entropy over the discrete choice distribution** (count how many of the 10 samples chose each letter, compute entropy of that distribution).
   - **Free-form (e.g. TriviaQA, added later):** the final answer is a phrase. Use **NLI / bidirectional entailment clustering** over the final answers, then entropy over clusters (standard semantic entropy).

**Implication for generation (stage 1):** the choice extractor must be pristine for MCQ — `extracted_choice` should be the clean letter only, with no leftover reasoning text, because the discrete-entropy baseline counts these letters directly and any contamination corrupts it. The full free-form `final_answer` and `reasoning_trace` must also be stored intact (the NLI-based computations read these later).

---



```
stage1_generate.py    -> data/generations/{dataset}.jsonl   (slow, GPU)
stage2_features.py    -> data/features/{dataset}.parquet     (CPU, fast, re-runnable)
stage3_labels.py      -> merged into features table          (CPU)
stage4_analyze.py     -> results/                             (CPU)
```

Only **stage1** is in scope for this task. But stage1 must save everything stages 2–4 will need (already reflected in the schema above).

---

## Implementation checklist

### Setup
- [ ] Provision/confirm A100 40GB; verify CUDA + driver.
- [ ] Create venv; install vllm, transformers, torch, datasets<=3.6.0, pandas, numpy, tqdm.
- [ ] Download `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`; confirm it loads in vLLM on 40GB.
- [ ] Confirm the chat template and the `<think>`/`</think>` tag format by generating on 1 example and printing raw output.

### Dataset loading
- [ ] Load MedQA (confirm exact HF path and split; MCQ format).
- [ ] Load MMLU-Pro (confirm exact HF path and split; MCQ format).
- [ ] Normalize both to a common internal format: `question_id`, `question` (with options inlined), `options` dict, `gold_answer` letter, `dataset`, `answerable=true`.
- [ ] Print 3 normalized examples from each to verify gold-answer alignment.

### Generation core
- [ ] Implement greedy generation (temp 0), capturing full output + logprobs.
- [ ] Implement 10-sample generation (temp 0.7, top_p 0.95), capturing full outputs + logprobs. Use vLLM `n=10` or batched sampling for efficiency.
- [ ] Implement think-tag splitter: `reasoning_trace` = between `<think>`/`</think>`; `final_answer` = after `</think>`. Handle missing/empty tags gracefully (log, don't crash).
- [ ] Implement choice extractor: parse the chosen letter from `final_answer`; `null` + log on failure.
- [ ] Force responses to begin with `<think>\n`.
- [ ] Set `max_tokens` generously (e.g., 4096); record truncation rate (responses hitting the cap).

### Verbalized confidence
- [ ] Implement the separate 0–100 confidence prompt + call.
- [ ] Implement robust integer parser; store raw + parsed (`null` on failure).

### P(True)
- [ ] Inspect tokenizer; identify the exact target tokens for "True"/"A" and "False"/"B". Log them.
- [ ] Implement the immediate-judgment P(True) prompt (no re-reasoning).
- [ ] Read token probabilities; compute raw + normalized P(True); store.

### Saving & robustness
- [ ] Write JSON Lines, one record per question, flushing after each (so a crash loses at most one record).
- [ ] Implement resume: on restart, skip `question_id`s already in the output file.
- [ ] Log per-question: success/failure, parse failures, truncation, wall-clock time.
- [ ] Write a `run_manifest.json`: model, configs, dataset versions, start/end time, counts, library versions.

### Smoke test (identical settings, few questions)
- [ ] Run the full pipeline on **5 questions** from each dataset.
- [ ] Manually inspect: trace/answer split correct? choice parsed? confidence parsed? P(True) sane (in [0,1])? logprobs present?
- [ ] Confirm one full JSON record matches the schema exactly.
- [ ] Estimate per-question wall-clock + token counts; extrapolate full-run time/cost.

### Scale up
- [ ] Run on 200 questions per dataset; re-check parse-failure and truncation rates.
- [ ] If rates acceptable (<~5% parse failure), scale to the full target (e.g., 1,500–3,000 per dataset).
- [ ] Monitor GPU memory; if OOM during 10-sample batches, reduce batch size (NOT sample count — keep n=10).

---

## Acceptance criteria

The generation stage is done when:
1. For every processed question there is one complete JSON record matching the schema.
2. `reasoning_trace` and `final_answer` are correctly separated for a manually inspected sample.
3. `extracted_choice` parse-failure rate is low and all failures are logged (not silently dropped).
4. `verbalized_confidence.parsed_confidence` and `ptrue.p_true_normalized` are present and in valid ranges.
5. `logprobs` are present for greedy and sampled generations.
6. The run is resumable and a `run_manifest.json` records all configs and library versions.
7. Truncation rate is reported.

---

## Out of scope (later stages — do not implement now)

- Hedging density (post-processing; word list frozen later).
- Trace length feature (token count, computed from saved text later).
- Trace disagreement and semantic entropy (entailment clustering over saved samples later).
- AUROC / ECE / risk-coverage / classifier (analysis on saved features later).
- Abstention/answerability datasets and the 3-bucket separation analysis (Phase 2).
- Additional models (non-reasoning controls, Qwen3) — added later by re-running stage1 with a different model id.

## Things the agent must NOT decide on its own (ask first)
- Changing sampling counts, temperatures, or which generation is "the answer."
- Changing the P(True) judgment style (must stay immediate, no re-reasoning).
- Dropping logprob capture.
- Swapping datasets or models.
- Any change to the output schema that removes a field.
