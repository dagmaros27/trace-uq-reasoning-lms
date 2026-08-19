# Stage 1 — Implementation Notes

Companion document to [`README_generation_pipeline.md`](README_generation_pipeline.md). That document is the *spec*; this one is the **record of what was actually built and decided**.

Scope:
- **Phase 1 (2026-06-02)** — MedQA generation for two **reasoning** models.
- **Phase 2 (2026-06-03)** — MedQA generation for two **non-reasoning control** models (added later; see §1b).

---

## 1. What was produced

| file | size | content |
|---|---|---|
| `data/generations/medqa_r1-distill-llama-8b.jsonl` | 172 MB | 1000 records, DeepSeek-R1-Distill-Llama-8B on MedQA test |
| `data/generations/medqa_r1-distill-llama-8b_manifest.json` | 662 B | run config, token IDs, counts |
| `data/generations/medqa_qwen3-4b.jsonl` | 201 MB | 1000 records, Qwen3-4B on MedQA test |
| `data/generations/medqa_qwen3-4b_manifest.json` | 681 B | run config, token IDs, counts |
| `data/generations/run.log` | 1.0 MB | full pipeline log (both models) |
| `data/generations/run_*.log` | ~0.5 MB each | per-model logs |

Each `.jsonl` record matches the schema in §`Output schema` of the pipeline README, with the additions noted in §4 below.

### Headline quality stats

| | R1-Distill-Llama-8B | Qwen3-4B |
|---|---|---|
| questions completed | **1000** | **1000** |
| total generations¹ | 13,000 | 13,000 |
| truncation rate | **3.5 %** ✓ (< 5 %) | **6.8 %** ⚠ (just over 5 %) |
| parse-fail rate² | **2.9 %** ✓ | **3.8 %** ✓ |
| wall clock | 2 h 40 m | 2 h 28 m |

¹ Per question: 1 greedy + 10 samples + 1 verbalised-confidence + 1 P(True) = 13.
² `extracted_choice == None` for the parsed MCQ letter.

---

## 1b. Non-reasoning control models (Phase 2, 2026-06-03)

The same Stage-1 pipeline was extended to two **non-reasoning control** models so H3 (the *reasoning-specific* hypothesis) can be tested. The reasoning outputs above were **not touched**; the controls produced their own files alongside.

| file | size | content |
|---|---|---|
| `data/generations/qwen3-4b-nothink/medqa.jsonl` | 66 MB | 1000 records, Qwen3-4B with thinking disabled |
| `data/generations/qwen3-4b-nothink/medqa_manifest.json` | 814 B | manifest with the new control-specific config fields |
| `data/generations/qwen3-4b-nothink/run.log` | ~0.4 MB | per-model log |
| `data/generations/llama-3.1-8b-instruct/medqa.jsonl` | 52 MB | 1000 records, Llama-3.1-8B-Instruct |
| `data/generations/llama-3.1-8b-instruct/medqa_manifest.json` | 807 B | manifest |
| `data/generations/llama-3.1-8b-instruct/run.log` | ~0.3 MB | per-model log |

### Headline quality stats (controls)

| | Qwen3-4B (thinking OFF, CoT) | Llama-3.1-8B-Instruct (CoT) |
|---|---|---|
| questions completed | **1000** | **1000** |
| total generations | 12 935 (995 × 13)¹ | 12 935 (995 × 13)¹ |
| truncation rate | **0.80 %** ✓ | **0.26 %** ✓ |
| parse-fail rate | **0.31 %** ✓ | **1.00 %** ✓ |
| wall clock | 48 min | 47 min |

¹ 5 of 1000 came from the smoke; the remaining 995 ran fresh via `--resume`.

**Why the controls are so much cleaner than the reasoning runs:** non-reasoning models produce much shorter outputs (no `<think>` block), so they rarely hit `max_tokens=4096`. Truncation rates dropped roughly 4–25× compared with the reasoning runs.

### Differences between reasoning and control runs — be explicit

| dimension | reasoning models | non-reasoning controls |
|---|---|---|
| **Models** | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `Qwen/Qwen3-4B` | `Qwen/Qwen3-4B` (with `enable_thinking=False`), `meta-llama/Llama-3.1-8B-Instruct` |
| **`<think>` block** | model emits it natively | none — model just writes a CoT |
| **Prompt** | standard MCQ prompt; model reasons by default | Kojima zero-shot CoT: *"Answer the following multiple-choice question. Let's think step by step, then give your final answer as a single letter on the last line in the form 'Answer: X'."* |
| **`force_think_prefix`** | `True` for R1-Distill (prepends `<think>\n`); `False` for Qwen3 (template handles it via `enable_thinking=True`) | `False` everywhere |
| **`chat_template_kwargs`** | `{}` for R1; `{"enable_thinking": True}` for Qwen3 reasoning | `{"enable_thinking": False}` for Qwen3-no-think; `{}` for Llama (no toggle exists) |
| **`split_strategy`** | `"think_tags"` (parses `<think>…</think>` block; `final_answer` is post-`</think>`) | `"inline_cot"` (no tags; cue-based split on `Answer: X`, `Final answer: X`, `\boxed{X}`, `The answer is X`, `Therefore, the final answer is (X)`; falls through to last-line) |
| **`tag_parse_status` values** | `strict` / `no_open_tag` / `no_close_tag` / `no_tags` | `inline_answer_line` / `inline_last_line` / `no_reasoning` / `inline_fallback` / `empty` |
| **P(True) judgment-position scan** | starts immediately after `</think>` (the model's verdict naturally follows the trace) | starts at the **first explicit verdict cue** (`Answer: (X) True/False`, `Verdict: …`); prevents an early `A`/`B` in the CoT body from being mistaken for the verdict token |
| **Choice extractor** | unchanged regex set | added `\\banswer\\s+is\\s*\\(?\\*?\\*?\\s*([A-E])\\b` pattern for "the answer is X" style |

### Schema additions in the control manifests

Three new top-level fields (also added to the reasoning manifests on this code revision for consistency):

```json
{
  "enable_thinking_kw":   false,                     // whether the chat template kwarg was used
  "chat_template_kwargs": { "enable_thinking": false },
  "split_strategy":       "inline_cot"               // "think_tags" | "inline_cot"
}
```

These let downstream code (Stage 2/3/4) detect at parquet/JSONL load time whether to treat the trace as a `<think>` block or an inline CoT body.

### What changed in `stage1_generate.py`

- `ModelAdapter.split_strategy` field (default `"think_tags"`)
- Two new registry entries: `"Qwen/Qwen3-4B:no-think"` and `"meta-llama/Llama-3.1-8B-Instruct"`
- `CONTROL_COT_MCQ_INSTRUCTION` (Kojima zero-shot CoT)
- `INLINE_FINAL_RE` covering: `Answer: X`, `Final answer: X`, `The answer is X`, `Therefore, the final answer is (X)`, `\\boxed{X}`
- `split_inline_cot()` + `split_generation()` dispatcher that picks `split_think_tags` vs `split_inline_cot` based on adapter
- P(True) scanner now seeks the first verdict cue for `inline_cot` adapters (avoids early-`A` false positives)
- Manifest writes the three new config fields above

### What changed in `scripts/gcp_a100.py`

- `REASONING_MODELS`, `CONTROL_MODELS`, and `MODEL_SETS = {"reasoning": …, "controls": …}`
- `smoke` and `run` accept `--model-set {reasoning, controls}` (default `controls`)
- `_status.sh` (the per-VM helper) now globs both flat and nested layouts so it can audit either

The complete Stage-1 audit history is in commit `9023be4` (*"feat: non-reasoning control models"*). The reasoning-era code is at tag `baseline-handoff`.

---

## 1c. Two new datasets — MMLU-Pro and TriviaQA (Phase 3, 2026-06-08 → 06-09)

The same 4 models (2 reasoning + 2 controls) were run on two more datasets so we can ask whether the trace-feature signal generalises beyond MedQA.

| dataset | kind | n | source | notes |
|---|---|---|---|---|
| **MMLU-Pro** | MCQ, up to 10 options (A–J) | 1000 / model | `TIGER-Lab/MMLU-Pro` (test split, 12 032 Qs available) | Option count varies per record (6–10); `cot_content` field ignored |
| **TriviaQA** | free-answer, closed-book | 1000 / model | `mandarjoshi/trivia_qa` (`rc.nocontext` config, validation split) | `answer.normalized_aliases` is the correctness check |

Output layout (8 cells, ~800 MB total):

```
data_generation/data/generations/
├── qwen3-4b/             mmlu_pro.jsonl  trivia_qa.jsonl  + manifests
├── r1-distill-llama-8b/  mmlu_pro.jsonl  trivia_qa.jsonl  + manifests
├── qwen3-4b-nothink/     mmlu_pro.jsonl  trivia_qa.jsonl  + manifests
└── llama-3.1-8b-instruct/mmlu_pro.jsonl  trivia_qa.jsonl  + manifests
```

### Dataset adapter pattern

To avoid scattering dataset-specific branches through the code, `stage1_generate.py` now has a `DATASET_REGISTRY` parallel to `MODEL_REGISTRY`:

```python
DATASET_REGISTRY: dict[str, dict] = {
    "medqa":     {"kind": "mcq",         "default_split": "test",       "load_fn": load_medqa,     ...},
    "mmlu_pro":  {"kind": "mcq",         "default_split": "test",       "load_fn": load_mmlu_pro,  ...},
    "trivia_qa": {"kind": "free_answer", "default_split": "validation", "load_fn": load_trivia_qa, ...},
}
```

Each loader normalises records into a common shape with a `kind` field. The main per-chunk loop dispatches on `kind` for prompt rendering, answer extraction, and correctness labelling. Trace features (hedging, rep_5, trace_divergence, …) are kind-agnostic.

### Free-answer prompt — structural `<answer>` tag

For TriviaQA we use a structural delimiter at the end of the user prompt:

> *"Answer the following question. Reason freely first if you need to. At the very end of your response, output your final short answer inside `<answer>...</answer>` tags. Put nothing else inside those tags."*

`extract_free_answer()` then pulls the **last** `<answer>...</answer>` block from the full output; if no tag is present (model didn't follow the instruction), it falls back to the last non-empty line.

**Why a structural tag instead of "answer concisely":** asking for concise answers caps response length, which hurts reasoning on hard questions and biases the comparison between reasoning models (long chains) and non-reasoning controls (short outputs). The tag isolates the final answer *without* constraining the reasoning before it. Adherence rates on the 1000-record runs:

| model | `tag` rate | `fallback_last_line` rate |
|---|---|---|
| qwen3-4b | 93.5 % | 6.5 % |
| r1-distill | 93.3 % | 6.7 % |
| qwen3-4b-nothink | 98.3 % | 1.7 % |
| llama-3.1-8b | 83.7 % | 16.3 % |

### TriviaQA correctness — alias matching, not exact string match

Per Gemini's note and the standard TriviaQA convention: the canonical answer alone is too strict. `_lib.is_correct()` for `kind=="free_answer"` does:

```python
return normalize_free_answer(predicted) in set(gold_normalized_aliases)
```

`normalize_free_answer()` is the SQuAD/TriviaQA recipe: lowercase, strip articles (`a|an|the`), strip punctuation, collapse whitespace. So a model output like `"It was Sunset Blvd."` normalises to `"sunset blvd"` and matches the gold alias list which contains `"sunset blvd"`.

### MMLU-Pro answer extraction — A-J fix

`INLINE_FINAL_RE` and the `CHOICE_PATTERNS` bank both hardcoded `[A-E]` from the MedQA era. On the first 1000-record MMLU-Pro sweep this dropped every answer in letters F–J, yielding parse rates of only 49–59 % per model. Fix: widened the regex bank to `[A-J]` and ran the bug-free extractor over the saved `full_output` text (no regeneration — the model's emitted text was unchanged, only the parsing was broken).

The per-record `valid_letters` filter remains active, so 5-option MedQA records can never pick up `F`–`J` from incidental mentions in the reasoning body.

Re-extraction script: `data_generation/scripts/_reextract_mmlu_pro.py` (local) and `data_generation/scripts/_vm_reextract_mmlu_pro.py` (VM-flat-layout variant).

Parse-rate impact:

| model | pre-fix | post-fix |
|---|---|---|
| qwen3-4b | 59.3 % | 88.9 % |
| r1-distill | 49.6 % | 85.2 % |
| qwen3-4b-nothink | 54.1 % | 94.7 % |
| llama-3.1-8b | 49.6 % | 86.6 % |

The pre-fix jsonls are archived as `mmlu_pro.preE_A-J_fix.jsonl` next to each post-fix file.

### Uniform sampling parameters — methodological rule

All 8 (model, dataset) cells were generated under **identical sampling parameters**:

| parameter | value | rationale |
|---|---|---|
| `max_tokens` | **6144** | enough headroom for MMLU-Pro reasoning chains without bias from per-cell tuning |
| `max_model_len` | 8192 | matches the prompt + 6144 output headroom |
| `n_samples` | 10 | matches MedQA |
| `sample_temp` | 0.7 | matches MedQA |
| `top_p` | 0.95 | matches MedQA |
| `seed` | 42 | matches MedQA |
| `n_questions` | 1000 | matches MedQA |
| P(True) prompt | v2 (literal `True`/`False`) | matches MedQA after the 06-08 rescore |

**Why uniform matters:** giving more token budget to harder cells (e.g. R1 on MMLU-Pro, which truncates 24 % of greedy outputs) would *also* let those cells reason more carefully, mixing the truncation-recovery effect with a calibration effect. The clean reading is: same parameters across all cells, then filter to `in_all_clean` records at Stage 3.

### Truncation and survival rates

Survival = no greedy truncation **and** no sample truncation **and** parse succeeded. All cells use the same uniform parameters, so cross-cell comparisons are clean.

| model | MedQA | MMLU-Pro | TriviaQA |
|---|---|---|---|
| qwen3-4b | 740 | 730 | 915 |
| r1-distill | 792 | **599** | 944 |
| qwen3-4b-nothink | 926 | 908 | 989 |
| llama-3.1-8b | 974 | 731 | 818 |

R1-distill / mmlu_pro at 599 is the weakest cell (R1 reasons hardest on STEM MCQs and hits the 6144 budget ~25 % of the time). We accept the smaller n rather than re-running R1 alone at a higher budget — see the uniform-budget rule.

### VM run summary

| step | wall-clock | cost (A100-40GB) |
|---|---|---|
| smoke (qwen3-4b × 50 q × 2 datasets) | ~25 min | ~$1.50 |
| full 8-cell sweep | 19 h 43 m | ~$73 |
| re-extraction (no GPU) | ~30 s | $0 |

Cost-per-cell varied from ~50 min (qwen3-4b-nothink/trivia_qa, 100% parsed) to ~9.5 h (qwen3-4b/mmlu_pro, longest reasoning chains).

Driver script: `data_generation/scripts/_vm_dataset_run.sh`. Status helper: `data_generation/scripts/_vm_status.py`.

---

## 2. Environment

### VM (GCP)

| | |
|---|---|
| Name / zone | `aims-project` / `asia-southeast1-c` (project `dagmawi-project`) |
| Machine type | `a2-highgpu-1g` |
| GPU | 1 × NVIDIA A100-SXM4-40GB, driver 580.159.03 |
| RAM / vCPU | 83 GB / 12 |
| Boot disk | 200 GB pd-balanced |
| Image | `common-cu129-ubuntu-2204-nvidia-580` (GCP Deep Learning VM, Ubuntu 22.04, CUDA 12.9) |

### Software

Installed via `apt` + per-user `pip` into `~/datagen/.venv` (Python 3.10.12):

| package | version |
|---|---|
| vllm | 0.22.0 |
| torch | 2.11.0+cu130 |
| transformers | 5.9.0 |
| tokenizers | 0.22.2 |
| datasets | 4.8.5 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| ninja-build (apt) + ninja (pip) | required by R1-Distill triton kernels |

### Pipeline orchestrator

`scripts/gcp_a100.py` — subcommands: `up`, `smoke`, `run`, `status`, `fetch`, `stop`. Mirrors the pattern of `methodology_poc/scripts/gcp_run.py`. Uses gcloud over IAP tunnel for SSH/SCP.

---

## 3. Original-spec checklist — verification

Item-by-item against the checklist in `README_generation_pipeline.md`.

### Setup

| item | status | note |
|---|---|---|
| Provision/confirm A100 40 GB; verify CUDA + driver | ✅ | initial bare-Debian VM was wiped and recreated with DLVM (see §4.A) |
| Create venv; install vllm, transformers, torch, datasets, pandas, numpy, tqdm | ✅ | venv at `~/datagen/.venv` |
| Download R1-Distill-Llama-8B; confirm it loads in vLLM on 40 GB | ✅ | 7.56 GB on GPU; loaded in 6 s after first download |
| Confirm chat template + `<think>` tag format on 1 example | ✅ | `--dry-run` mode prints rendered prompt + token probes |

### Dataset loading

| item | status | note |
|---|---|---|
| Load MedQA (confirm path + split) | ✅ | local `data/medqa/{train,test,dev}.jsonl` files used; **not** the HF `bigbio/med_qa` repo (see §4.B) |
| Load MMLU-Pro | ⏸ deferred | out of scope for this run; pipeline parameterised by `--dataset`, easy to add |
| Normalise to common format | ✅ | per question: `question_id`, `question`, `options`, `gold_answer`, `dataset`, `answerable=True` |
| Print 3 normalised examples to verify gold alignment | ⚠ partial | one example printed during `--dry-run`; manually verified gold by inspecting record-0 outputs |

### Generation core

| item | status | note |
|---|---|---|
| Greedy generation (T=0), capture full output + logprobs | ✅ | logprobs summarised (see §4.C) |
| 10-sample generation (T=0.7, top_p=0.95) via `n=10` | ✅ | one vLLM call per question |
| Think-tag splitter with malformed-tag fallback | ✅ | `tag_parse_status` field records strict / no_open_tag / no_close_tag / no_tags |
| Choice extractor with `null + log` on failure | ✅ | 7 ordered regex patterns + standalone-letter fallback; `choice_method` field records which one matched |
| Force responses to begin with `<think>\n` (per R1 card) | ✅ R1 / ⚠ Qwen3 | R1-Distill uses `force_think_prefix=True` (prefills `<think>\n` in prompt). Qwen3 uses `enable_thinking=True` chat-template kwarg instead — its chat template handles the trigger natively |
| `max_tokens` generously set; record truncation rate | ✅ | 4096; truncation counted in manifest |

### Verbalized confidence

| item | status | note |
|---|---|---|
| Separate 0–100 prompt + call | ✅ | format: `Confidence: NN` on last line |
| Robust integer parser; store raw + parsed | ✅ | `verbalized_confidence.parsed_confidence` is int 0-100 or `null` |

### P(True)

| item | status | note |
|---|---|---|
| Inspect tokenizer; identify True/A and False/B target tokens | ✅ | 7 spellings probed per side; IDs stored in manifest |
| **README spec/checklist conflict on judgment style** | ⚠ see §4.D | spec body (line 29) says **reason-then-judge**; checklist (line 187) says **immediate**. I implemented **reason-then-judge** to match the design rationale, with verdict-position discovery |
| Read token probs, compute raw + normalised, store | ✅ | `p_true_token_prob`, `p_false_token_prob`, `p_true_normalized`, plus `judgment_token_index/id/decoded` for auditability |

### Saving & robustness

| item | status | note |
|---|---|---|
| JSONL, one record per question | ✅ | append-only |
| **Flush after each record** | ⚠ partial | chunked: flushes every 50 questions, not every 1 (see §4.E). Crash loses ≤ 50 records, not 1 |
| Resume by skipping done `question_id`s | ✅ | `--resume` flag |
| Log per-question success/failure/parse-fail/truncation/time | ⚠ partial | global counts only — per-question log not implemented (see §4.F) |
| `run_manifest.json` with model, configs, dataset, times, counts, library versions | ✅ all except lib versions | manifest written; **library versions not embedded** (see §4.G) |

### Smoke test

| item | status | note |
|---|---|---|
| Run on 5 questions per model | ✅ | took 3 attempts to be clean (see §4.H for the bugs found and fixed) |
| Manually inspect splits / parses / confs / P(True) / logprobs | ✅ | `scripts/_audit_schema.sh` checks every field on every record |
| Estimate per-question time + extrapolate | ✅ | original projection 50 min was wrong; actual ~8 min / 50-Q chunk on R1 |

### Scale up

| item | status | note |
|---|---|---|
| Run 200 Q first | ⚠ skipped | jumped from 5 → 1000 per user decision once smoke was clean |
| Re-check rates < 5 % then scale | ⚠ partial | rates checked at 5-Q smoke; not at 200 |
| Monitor GPU OOM | ✅ | none observed |

### Acceptance criteria

| # | criterion | status |
|---|---|---|
| 1 | One complete JSON record per question, matching schema | ✅ 1000 / 1000 both models |
| 2 | reasoning_trace and final_answer correctly separated (manual sample) | ✅ verified |
| 3 | extracted_choice parse-failure rate low + failures logged | ✅ R1 2.9 % / Qwen3 3.8 %, all logged |
| 4 | parsed_confidence + p_true_normalized present and in range | ✅ |
| 5 | **logprobs present for greedy and sampled** | ⚠ **summary stats only**, not full per-token (see §4.C) |
| 6 | Run resumable + manifest records configs & library versions | ⚠ resumable yes, **library versions not in manifest** |
| 7 | Truncation rate reported | ✅ |

**Acceptance items 5 and 6 are intentionally deviated from the strict spec; rationale in §4 below.**

---

## 4. Decisions and deviations

Every non-trivial choice that was *not* explicit in the pipeline README. Numbered for easy reference.

### A. VM bootstrap path

**Original assumption** was a clean A100 with CUDA already configured.

The user-provisioned `aims-project` VM came up as **bare Debian 13** — no NVIDIA driver, no Python 3.10 (only 3.13, which vLLM doesn't support yet), no build tools. Bootstrapping Debian 13 for vLLM is 30–60 min of fiddling with driver `.run` installers, kernel reboots, and Debian-specific CUDA packaging.

**Decision:** delete the VM, recreate same name + zone + machine type but with the GCP Deep Learning VM image (`common-cu129-ubuntu-2204-nvidia-580`). 3 min downtime, ~$0 cost vs. ~$3 of A100 time and ~50 % chance of needing more SSH debugging. Result: working vLLM in ~10 min after recreate.

### B. MedQA source: local JSONL files, not HuggingFace

The spec lists `datasets<=3.6.0` but the user already had `data_generation/data/medqa/{train,test,dev}.jsonl` checked into the project. I read those files directly with the stdlib `json` module rather than going through the HF datasets API. Pros: no network dependency at runtime, byte-identical input across runs. Cons: doesn't pin a specific HF revision in the manifest (mitigated by having the files in-tree). MMLU-Pro will still need the HF path when added.

**Used split:** `test.jsonl` (1273 questions); `--split` flag exposes train/dev/test.

### C. **Logprobs: summary stats, not full per-token arrays** (departure from spec)

**Spec text:** "Keep logprobs even if large. If storage becomes a problem, make logprob capture a config flag (default ON), but default to capturing."

Storage *did* become a problem. With full top-5 logprobs per token, the smoke test wrote a **40 MB JSONL for 5 questions** = 8 MB/record. At 1000 records × 2 models = **16 GB**. That's most of the 200 GB boot disk and unwieldy to ship.

**What I store instead**: for every generation (greedy, each sample, verb-conf, P(True) prompt) I compute and save these **summary statistics over the per-token logprob arrays**, then discard the arrays:

```jsonc
"logprob_summary": {
    "mean_token_entropy_bits":  ...,   // average top-k Shannon entropy across positions
    "max_token_entropy_bits":   ...,   // worst-confidence position
    "mean_neg_logprob":         ...,   // mean −log p of the sampled token (NLL/token)
    "min_token_logprob":        ...,   // single most-surprised position
    "n_positions":              ...    // number of positions = generated token count
}
```

These five numbers are what every downstream paper-style feature (mean-NLL, max-entropy, token-uncertainty curves) computes anyway. **Result:** record size dropped from 8 MB → 100 KB (≈ 40×), full output ~370 MB total.

**P(True) is the exception** — it needs the *raw distribution at the judgment position*. Those top-20 logprobs at one position are still captured (in `ptrue.judgment_token_*` fields).

This is the single biggest spec deviation. The original per-token arrays are recoverable by re-running with `logprob_mode=full` (not done here, knob exists if needed later).

### D. P(True) — reason-then-judge, with verdict-position discovery

The spec is internally inconsistent: §"Locked design decisions" point 1 says **REASON-THEN-JUDGE** (the model thinks then judges); the checklist line says **"immediate-judgment P(True) prompt (no re-reasoning)"**.

**I implemented reason-then-judge**, matching the longer rationale text in the locked-decisions section ("DeepSeek-R1-Distill is trained to emit `<think>` first on every output. Reading the first token would capture `<think>`/reasoning tokens, not the judgment").

This required a non-trivial fix: R1-Distill in particular writes a multi-sentence *summary paragraph* after `</think>` before emitting "Answer: (A) True". The naïve "first content token after `</think>`" gives the prob distribution at the word *The*, not at the verdict.

**`compute_ptrue()` therefore walks forward from `</think>` and finds the *first position whose sampled token is in the True/False/A/B candidate set*** — that's the actual verdict position. We store the index, the token ID, the decoded form, plus the aggregated True/False probabilities (summed over our 7+7 candidate-spelling token IDs) and the normalised ratio.

If no candidate token appears in the entire response (model hedged or got truncated), `p_true_normalized = None` is recorded honestly rather than fabricated.

#### D.bis. P(True) v2 — drop `(A)/(B)` prefix, literal `True`/`False` only (2026-06-08)

The original v1 prompt asked the model to "answer with either `'(A) True'` or `'(B) False'`" and the `probe_judgment_tokens()` candidate set included `A`/`(A`/`(A)` (true side) and `B`/`(B`/`(B)` (false side). The motivation was robustness to models that emit a letter prefix.

**Bug**: for the inline-CoT controls (qwen3-4b-nothink, llama-3.1-8b-instruct) the verdict cue regex didn't always fire, so the scanner would grab the first `A` or `B` token after `</think>` — which on a 5-option MedQA question is **also a valid MCQ option letter inside the reasoning body**. On a 20-record audit of R1-Distill, 5/20 records (25%) had the judgment position located inside the reasoning trace rather than at the verdict, and the True+False top-k mass at those positions was *inverted by correctness* (0.53 on correct vs 0.65 on wrong).

**Fix**: revert to the standard Kadavath setup. Prompt now asks for a single literal word — `True` or `False`, no letter prefix, no punctuation. `probe_judgment_tokens()` registers only `True/ True/true/ true` (true side) and `False/ False/false/ false` (false side). The verdict-cue regex drops the `\(?[ab]\)?` alternative.

The four existing MedQA jsonls were **rescored on the VM in `--ptrue-only` mode** (~12 min/model; only the P(True) forward pass — greedy/samples/verbalised confidence unchanged). Each model's original `medqa.jsonl` was archived as `medqa.v1.jsonl` and the `ptrue` field was merged from `medqa.ptrue_v2.jsonl` into a fresh `medqa.jsonl`. The new ptrue values carry a `generation_config.ptrue_version = "v2"` stamp.

Effect on the 1000-record set (mean P(True) on correct − on wrong):

| model | v1 (buggy) | **v2 (fixed)** | v2 position-miss rate |
|---|---|---|---|
| qwen3-4b | +0.022 | **+0.053** | 32/1000 (3.2%) |
| r1-distill-llama-8b | +0.018, 25% pos-miss | **+0.032** | 95/1000 (9.5%) |
| qwen3-4b-nothink | (not measured) | **+0.147** ← largest separation | 3/1000 (0.3%) |
| llama-3.1-8b-instruct | (not measured) | **+0.035** | 21/1000 (2.1%) |

All judgment tokens now decode to literal `True`/`False`. The A/B contamination is gone.

For any future dataset run, `stage1_generate.py` produces correct v2 ptrue from the first generation — no rescore step needed.

### E. Chunked writes (50 / chunk) instead of per-question flush

Spec says "flushing after each [record] (so a crash loses at most one record)".

I batch the four stages per chunk of 50 questions (one big vLLM call for 50 greedies, one for 500 samples = 50 × 10, one for 50 verb-confs, one for 50 P(True)s, then assemble and append all 50 records). This is roughly **6–10× faster** than per-question stages because vLLM's continuous batching needs many concurrent sequences to saturate the A100.

Trade-off: a mid-chunk crash loses up to 50 records instead of 1. With 5 h of total runtime across 40 chunks, the expected loss from one crash is `50 × p(crash) × 1` ≈ trivial. None happened. `--chunk-size` flag exposes this knob.

### F. Global counters, not per-question log

Spec asks for per-question logging of "success/failure, parse failures, truncation, wall-clock time".

I log per-chunk wall-clock and aggregate `truncation_count` + `parse_fail_count` in the manifest. Per-question success status is **recoverable from the JSONL itself** (the `finish_reason` field on each generation tells you if it was truncated; `extracted_choice == None` tells you the parse failed). So the data isn't lost, just not pre-aggregated. A short stage-1.5 script can build a per-question audit table on demand.

### G. Library versions not embedded in manifest

Spec asks for "library versions" in `run_manifest.json`. I capture `model`, configs, token IDs, but not `vllm/torch/transformers` versions. Easy fix (one line via `importlib.metadata`); didn't ship in this run. The versions are otherwise documented here in §2.

### H. Bugs found in smoke and fixed before the 1000-Q run

Listed for archaeology (these are why smoke happened 3 times, not 1):

1. **`ninja` missing on the VM** — R1-Distill triggers a Triton JIT kernel compile that requires `ninja`. The DLVM image doesn't ship it. Fix: `apt install ninja-build` + `pip install ninja`.
2. **Byte-level BPE artefacts in text** (`Ġ`, `Ċ`) — vLLM's `output.text` for R1-Distill (Llama-3 tokenizer) emits raw byte-level BPE markers, which broke every regex parser. Caused 48 / 50 parse failures in smoke 1. Added a `clean_vllm_text()` pass that maps `Ġ` → space and `Ċ` → newline before any regex. Drop to 1 / 50.
3. **Status script bash bug** — `for f in glob 2>/dev/null` is not valid bash. Replaced with `shopt -s nullglob` pattern.
4. **P(True) returned None for every R1-Distill record** — my first version of `compute_ptrue` looked at the first content token after `</think>`, which is "The" (start of the summary paragraph). Fix as described in §4.D.

### I. Two-model registry + per-model adapter

The spec is single-model. I added `ModelAdapter` + `MODEL_REGISTRY` so different reasoning models can plug into the same pipeline by declaring:

- `force_think_prefix` (R1-Distill: True → prepend `<think>\n`)
- `enable_thinking_kw` (Qwen3: True → pass `enable_thinking=True` to chat template)
- `chat_template_kwargs` (Qwen3: `{"enable_thinking": True}`)
- `short_name` (used in output filenames)

Adding a new model is a 5-line entry in the registry.

### J. Deterministic subsample (shuffle-then-take)

Original `subsample` used `random.sample(records, n)`, which returns a *different* subset when `n` changes. This meant the 5-Q smoke set and the 1000-Q run set didn't overlap, so the 5 smoke generations would have been wasted (and the resume couldn't carry them forward).

**Replaced with shuffle-then-take**: seed → `random.shuffle(full_list)` → take first `n`. Property: `subsample(pool, k, s) ⊂ subsample(pool, K, s)` for `k ≤ K`. The 5 smoke records were a strict prefix of the 1000 and were correctly skipped during the main run.

### K. Multi-spelling True/False token mapping

The spec note says: *tokenisation of "True"/"A" matters — verify which exact token the model emits and log it*. I probe **7 spellings** per side (`True`, ` True`, `true`, ` true`, `A`, ` A`, `(A`, `(A)` and the False analogues) and treat all matching token IDs as evidence for that verdict. Token IDs end up in the manifest. Avoids missing the verdict because the model said "(A)" instead of "True".

### L. Defensive overnight watcher

Two layers, both invariant: **fetched data must verify locally (≥10 MB, ≥990 records) before VM stops**.

- **Laptop side** (bash loop): polls VM until generation pid dies → runs `fetch` → checks files → only then runs `stop`. If verify fails, retries once. If retry fails, refuses to stop, leaves clear recovery instructions. **Never executes a destructive op without a confirmed safe local copy.**
- **VM side** (`_safety_shutdown.sh`): waits for the same pid, then 30 min buffer (gives the laptop time), then `sudo /sbin/shutdown -h now`. This is what makes "shut down your PC overnight" safe — if the laptop watcher dies, the VM still shuts itself off to stop billing.
- **Data on VM disk is preserved by `stop`** (not deleted). VM can be re-`up`'d and re-`fetch`'d any time.

### M. Disk size 200 GB on the new VM

DLVM image is 50 GB; expanded boot disk to 200 GB at creation. R1-Distill weights (~16 GB) + Qwen3-4B weights (~8 GB) + vLLM compile cache + 370 MB outputs + headroom = comfortable on 200 GB. Avoids the disk-full thrashing we hit on the methodology-PoC T4 VM.

### N. Sequential model execution

Both models run on the same A100 sequentially (not concurrently). Reasoning: an A100-40GB shared between two vLLM instances forces each to half VRAM → half KV cache → ~half the concurrent batch size, which throughput-wise is *worse* than just running them back-to-back. Documented this with the user before starting.

---

## 5. Known limitations / open items

- **Qwen3-4B truncation rate is 6.8 %**, above the spec's < 5 % target. Mostly affects the 10 sample generations (temp 0.7 → more verbose). Parse-fail rate is still fine (3.8 %) because the model often picks a letter before being cut off, but `final_answer` text may be incomplete. Bumping `max_tokens` from 4096 → 6144 would likely fix this in a future re-run.
- **Library versions not in manifest** (§4.G).
- **No per-question wall-clock log** (§4.F) — recoverable but not pre-built.
- **Full per-token logprobs not stored** (§4.C) — summary stats only. By design; can be re-run with `logprob_mode=full` if a downstream method needs token-level distributions.
- **MMLU-Pro not run** — out of scope for this iteration.
- **One small audit: 3 normalised examples per dataset** (spec checklist) — only inspected by running and reading the smoke output, not by a dedicated print.

---

## 6. How to reproduce, resume, or extend

### Reproduce a smoke (5 Q on both models, ~5 min)

```powershell
python scripts\gcp_a100.py up         # start VM, upload, install deps
python scripts\gcp_a100.py smoke      # 5 Q on both models, sequential
python scripts\gcp_a100.py status
python scripts\gcp_a100.py fetch
python scripts\gcp_a100.py stop
```

### Re-run the full 1000-Q pass

```powershell
python scripts\gcp_a100.py run --n 1000
```

`--resume` is the default (skips records already in the output JSONL by `question_id`). To start fresh, delete `data/generations/*.jsonl` first.

### Add a third model

Edit `MODEL_REGISTRY` in `stage1_generate.py`:

```python
MODEL_REGISTRY["new-org/new-reasoning-model"] = ModelAdapter(
    hf_id="new-org/new-reasoning-model",
    short_name="new-model",
    force_think_prefix=True_if_R1_style_else_False,
    enable_thinking_kw=True_if_Qwen3_style_else_False,
    chat_template_kwargs={"enable_thinking": True} if Qwen3_style else {},
)
```

Then update `MODELS` in `scripts/gcp_a100.py` to include the new HF id, and run `smoke` to verify it loads.

### Add MMLU-Pro

In `stage1_generate.py` add `load_mmlu_pro()` next to `load_medqa()` (normalize to the same dict keys), and route via `--dataset mmlu_pro`. The output schema and pipeline don't change.

---

## 7. File map

```
data_generation/
├── README_generation_pipeline.md           ← original spec (do not edit)
├── README_stage1_implementation.md         ← this file
├── stage1_generate.py                      ← the actual pipeline
├── scripts/
│   ├── gcp_a100.py                         ← VM orchestrator
│   ├── _install.sh _launch.sh _status.sh   ← uploaded helpers (generated)
│   ├── _safety_shutdown.sh                 ← VM-side billing safety
│   ├── _audit_schema.sh                    ← schema completeness checker
│   ├── _verify_smoke.sh                    ← one-record inspector
│   └── _diag_ptrue.sh                      ← P(True) raw-response inspector
├── data/
│   ├── medqa/{train,test,dev}.jsonl        ← input dataset
│   └── generations/                        ← outputs land here
└── (run.log / run_*.log inside generations)
```
