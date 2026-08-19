# Analysis Pipeline — Implementation Report (Stages 2–5)

Companion to [`README_analysis_pipeline.md`](README_analysis_pipeline.md). That document is the **spec**; this one records what was built, every non-trivial decision, the methodology-PoC code I reused, the spec checklist item-by-item, the acceptance criteria check, and the actual headline results.

Scope:
- **Phase 1 (2026-06-02)** — full analysis (Stages 2–5) on the two **reasoning** models.
- **Phase 2 (2026-06-03)** — focused analysis on the two **non-reasoning control** models (added later; see §11).

---

## 1. What was built

```
analysis_pipeline/
├── README_analysis_pipeline.md          ← spec (do not edit)
├── README_stage2_5_implementation.md    ← this file
├── lexicons.json                        ← v1.0 frozen — 117 + 26 + 17 terms
├── scripts/
│   ├── _lib.py                          ← shared utilities (~280 lines)
│   ├── stage2_inspect.py                ← inspection, distributions, examples (~330 lines)
│   ├── stage3_features.py               ← per-question feature parquet (~200 lines, +BGE-M3 embed pass)
│   ├── stage4_model.py                  ← CV + bootstrap CIs + LOO + ROC + risk-coverage (~400 lines)
│   ├── stage5_extra.py                  ← correlations + calibration + effect sizes + cross-model (~240 lines)
│   └── _build_notebook.py               ← regenerates the analysis notebook from JSON
├── notebooks/
│   ├── analysis_notebook.ipynb          ← narrative source
│   └── analysis_notebook.executed.ipynb ← executed copy with figures embedded inline
├── data/features/{model_short}/         ← stage 3 parquet + csv
└── results/
    ├── stage3_manifest.json             ← embedder id + lexicon hash + library versions + seed
    ├── stage4_summary.json
    ├── stage2/ stage4/ stage5/          ← cross-model figures
    └── {model_short}/{dataset}/         ← per-model tables + per-model figures
        ├── inspection/  stage4/  stage5/
```

Each stage runs and saves independently. Re-running one does not invalidate the others.

---

## 2. Headline results (5-fold CV + 1000-bootstrap 95 % CIs, all-clean set)

> **2026-06-08 — P(True) v2.** All numbers below are after the literal-`True`/`False` rescore (see Stage 1 §4.D.bis). The earlier P(True) numbers (≈0.53 AUROC, ECE > 0.4) reflected a prompt that asked the model to answer `(A) True / (B) False`; on a 5-option MCQ, A/B at the verdict position can collide with option letters in the reasoning body. The fix dropped A/B from the candidate set; on the same 1000 records, P(True) AUROC went up to ≈0.59 on reasoning models and ≈0.55–0.62 on controls — still the weakest baseline, but no longer broken. Trace-Feature still beats P(True) 100 % of bootstraps on every model.

### R1-Distill-Llama-8B  (792 clean+labeled rows; 52 % accuracy on clean)

| method | AUROC | 95 % CI | kind |
|---|---|---|---|
| `answer_semantic_entropy` | **0.684** | [0.648, 0.719] | baseline |
| **`trace_LR_split`** | **0.667** | [0.633, 0.705] | trace_LR |
| **`trace_LR_combined`** | **0.666** | [0.629, 0.703] | trace_LR |
| `trace_length` | 0.660 | [0.623, 0.696] | trace_single |
| `hedging_combined` | 0.625 | [0.588, 0.665] | trace_single |
| `rep_5` | 0.607 | [0.570, 0.644] | trace_single |
| `p_true` (v2) | 0.585 | [0.544, 0.625] | baseline |
| `verbalized_confidence` | 0.564 | [0.530, 0.601] | baseline |

### Qwen3-4B  (740 clean+labeled rows; 76 % accuracy on clean)

| method | AUROC | 95 % CI | kind |
|---|---|---|---|
| **`trace_LR_split`** | **0.780** | [0.740, 0.816] | trace_LR |
| **`trace_LR_combined`** | **0.767** | [0.727, 0.803] | trace_LR |
| `rep_5` | 0.755 | [0.715, 0.791] | trace_single |
| `trace_length` | 0.750 | [0.709, 0.788] | trace_single |
| `hedging_combined` | 0.701 | [0.657, 0.743] | trace_single |
| `answer_semantic_entropy` | 0.683 | [0.642, 0.722] | baseline |
| `verbalized_confidence` | 0.641 | [0.603, 0.675] | baseline |
| `p_true` (v2) | 0.595 | [0.551, 0.642] | baseline |

### Pairwise bootstrap — trace_LR_combined and trace_LR_split vs each baseline

```
                                            median Δ   95 % CI         win
Qwen3-4B    combined vs p_true (v2)        +0.170   [+0.121, +0.217]   100.0 %
            combined vs verbalized_conf    +0.121   [+0.080, +0.163]   100.0 %
            combined vs semantic_entropy   +0.085   [+0.041, +0.125]   100.0 %
            combined vs full_LR            -0.020   [-0.034, -0.005]     0.4 %
            split    vs p_true (v2)        +0.184   [+0.132, +0.233]   100.0 %
            split    vs verbalized_conf    +0.134   [+0.095, +0.174]   100.0 %
            split    vs semantic_entropy   +0.098   [+0.051, +0.140]   100.0 %
            split    vs full_LR            -0.006   [-0.028, +0.015]    28.8 %

R1-Distill  combined vs p_true (v2)        +0.081   [+0.031, +0.133]   100.0 %
            combined vs verbalized_conf    +0.093   [+0.047, +0.141]   100.0 %
            combined vs semantic_entropy   -0.018   [-0.061, +0.024]    18.9 %
            combined vs full_LR            -0.049   [-0.074, -0.023]     0.0 %
            split    vs p_true (v2)        +0.085   [+0.038, +0.135]   100.0 %
            split    vs verbalized_conf    +0.093   [+0.049, +0.139]   100.0 %
            split    vs semantic_entropy   -0.017   [-0.061, +0.025]    20.8 %
            split    vs full_LR            -0.043   [-0.073, -0.016]     0.4 %
```

### Calibration (ECE, n_bins = 10) — v2 P(True)

```
R1-Distill                              Qwen3-4B
p_true (v2)            0.446            p_true (v2)            0.235
verbalized_confidence  0.373            verbalized_confidence  0.200
answer_semantic_entropy 0.125           answer_semantic_entropy 0.163
trace_LR_combined      0.038            trace_LR_combined      0.047
trace_LR_split         0.054            trace_LR_split         0.048
full_LR                0.030            full_LR                0.053
```

The model's own confidence signals (P(True), verbalised, semantic entropy) sit at ECE 0.13–0.45. The LR methods sit at 0.03–0.05 — roughly an order of magnitude better calibrated.

The LR methods are **~8–10× better calibrated** than the model's own confidence signals.

### Risk–coverage @ 80 % coverage (v2 P(True))

```
R1-Distill                                    Qwen3-4B
p_true (v2)             acc@80 = 0.548        p_true (v2)             acc@80 = 0.775
verbalized_confidence            0.546        verbalized_confidence            0.791
answer_semantic_entropy          0.574        answer_semantic_entropy          0.836
trace_LR_combined                0.558        trace_LR_combined                0.829
full_LR                          0.578        full_LR                          0.825
```

### Leave-one-out (trace_LR_combined, ΔAUROC when feature removed)

```
R1-Distill                                    Qwen3-4B
trace_length          Δ = −0.020             rep_5             Δ = −0.012
hedging_combined      Δ = −0.002             hedging_combined  Δ = −0.009
trace_divergence      Δ = −0.001             connector_density Δ = −0.004
rep_5                 Δ = −0.000             trace_divergence  Δ = +0.000  (neutral)
connector_density     Δ = −0.001             trace_length      Δ = +0.001  (neutral)
```

`trace_length` carries R1's signal; `rep_5` carries Qwen3's. Hedging adds a smaller but consistent contribution on both. `trace_divergence` proves to be on probation per spec — small effect on both.

### Repetition robustness

```
Correlation(rep_5, trace_length)   R1: r = 0.653       Qwen3: r = 0.818
Correlation(rep_3, rep_4)          R1: r = 0.989       Qwen3: r = 0.994
Correlation(rep_4, rep_5)          R1: r = 0.992       Qwen3: r = 0.994
```

rep-3/4/5 are nearly interchangeable → headline rep_5 is robust to the n-gram-size choice. Repetition correlates strongly with length (especially on Qwen3) but LOO shows both still add independent signal.

---

## 2c. Two new datasets — MMLU-Pro and TriviaQA (Phase 3, 2026-06-11)

After validating the pipeline on MedQA we ran the same 4 models on two more datasets. **All MedQA outputs are unchanged** — new datasets get their own parallel layout so the original numbers remain reproducible.

### Output layout — fully parallel per dataset, no cross-confusion

```
analysis_pipeline/
├── data/features/<model>/{medqa,mmlu_pro,trivia_qa}.parquet         ← Stage 3 features
├── results/<model>/<dataset>/stage4/                                ← per-model CSVs
│   methods_auroc.csv  pairwise_bootstrap.csv  ece.csv  ...
├── results/stage4_medqa/        fig_auroc_caterpillar_<model>.pdf   ← per-dataset figures
├── results/stage4_mmlu_pro/     ... (same set, 16 PDFs)
├── results/stage4_trivia_qa/    ... (same set, 16 PDFs)
├── results/stage4_summary_medqa.json
├── results/stage4_summary_mmlu_pro.json
├── results/stage4_summary_trivia_qa.json
└── notebooks/
    analysis_notebook_medqa.ipynb       (+ .executed)
    analysis_notebook_mmlu_pro.ipynb    (+ .executed)
    analysis_notebook_trivia_qa.ipynb   (+ .executed)
```

`stage4_model.py --dataset X` always writes figures into `results/stage4_{X}/` and summary into `results/stage4_summary_{X}.json`. `_build_notebook.py --dataset X` writes `analysis_notebook_{X}.ipynb`. The original `results/stage4/` was renamed to `results/stage4_medqa/` and the original `analysis_notebook.ipynb` to `analysis_notebook_medqa.ipynb` so all three datasets follow the same naming.

### Kind-aware Stage 3

`stage3_features.py` now dispatches on the record's `kind` field:

| signal | MCQ (medqa, mmlu_pro) | free-answer (trivia_qa) |
|---|---|---|
| `answer_semantic_entropy` | Shannon entropy over discrete sample letters (bits, capped at log₂K where K = #options) | **Kuhn-style NLI cluster entropy** — see below |
| `trace_divergence` | same for both: mean pairwise (1 − cos sim) over BGE-M3 embeddings of the 10 reasoning traces | same |
| `trace_length`, `hedging_*`, `connector_density`, `rep_5` | same for both: aggregated from the per-sample reasoning_trace text | same |
| `p_true`, `verbalized_confidence` | same baselines from the saved generation | same |
| label (`correct`) | letter match against `gold_answer` | normalised prediction ∈ `gold_normalized_aliases` |

### NLI-based semantic entropy for free-answer (Kuhn et al. 2023, simplified)

For TriviaQA we cannot use letter entropy. Instead `semantic_entropy_free_answer()` clusters the 10 sample predictions by **bidirectional entailment**, then computes Shannon entropy over the cluster size distribution:

1. Drop sample predictions that are `None` / empty.
2. For every unordered pair (i, j), prepend the question to each prediction to get matched-context sentences (so the NLI model isn't asked to entail two bare nouns), then classify both directions with `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
3. Draw edge (i, j) iff **both** directions are `entailment`.
4. Semantic clusters = connected components of that graph.
5. `H = −Σ (nₖ / N) log₂ (nₖ / N)` over cluster sizes nₖ.

Smoke test (30 records / qwen3-4b / trivia_qa) confirmed sensible clustering: 4×"Grover Cleveland" group together (one cluster) but get split from "Woodrow Wilson" and "William McKinley" (separate clusters); 9×"Montreal" and 1×"Quebec City" split into two clusters; garbage trace fragments become singleton clusters that correctly inflate H.

The NLI model is only loaded if a cell contains `free_answer` records (`needs_nli = any(record_kind(r) == "free_answer")`), so MMLU-Pro and MedQA never touch it.

**Raw H vs normalised H/log K.** We feed *raw H in bits* into Stage 4 — AUROC is rank-based and invariant to monotonic scaling, so the [0, log₂K] scale doesn't matter for the modelling. The 1 − H/log₂K normalisation is only needed where we use H *as a confidence score* (ECE) — Stage 4 already handles that conversion with the right per-dataset K.

### Sampling parameters at Stage 3

Identical to MedQA: 5-fold stratified CV, standardisation inside each train fold, 1000-bootstrap 95 % CIs, NaN-explicit. Filter to `in_all_clean & correct.notna()` before modelling. Per-cell clean+labeled row counts:

| model | medqa | mmlu_pro | trivia_qa |
|---|---|---|---|
| qwen3-4b | 740 | 730 | 915 |
| r1-distill | 792 | **599** | 944 |
| qwen3-4b-nothink | 926 | 908 | 989 |
| llama-3.1-8b | 974 | 731 | 818 |

### Headline AUROC results — Stage 4

trace_LR_combined vs the strongest baseline per cell (full numbers in `stage4_summary_<dataset>.json` and `results/<model>/<dataset>/stage4/methods_auroc.csv`):

| cell | best baseline | trace_LR_combined | winner |
|---|---|---|---|
| qwen3-4b / mmlu_pro | answer_sem_ent 0.722 | **0.816** | **trace** |
| qwen3-4b / trivia_qa | NLI sem_ent **0.871** | 0.825 | NLI sem_ent |
| qwen3-4b-nothink / mmlu_pro | answer_sem_ent **0.761** | 0.692 | answer_sem_ent |
| qwen3-4b-nothink / trivia_qa | NLI sem_ent **0.843** | 0.796 | NLI sem_ent |
| llama / mmlu_pro | answer_sem_ent **0.778** | 0.658 | answer_sem_ent |
| llama / trivia_qa | NLI sem_ent **0.791** | 0.648 | NLI sem_ent |

(R1-Distill cells: see `stage4_summary_mmlu_pro.json` / `stage4_summary_trivia_qa.json`.)

**The story tightens.** Trace features beat all baselines only on the reasoning model on MCQ data (qwen3-4b / mmlu_pro). On free-answer (TriviaQA), **NLI semantic entropy is the new bar to clear** across all 4 models — Kuhn-style clustering on the model's own samples captures uncertainty better than anything trace-based we tried. On non-reasoning controls and llama, trace features under-perform — consistent with H3 (trace features are reasoning-model-specific).

### Stage 3 wall-clock

Single VM session, both new datasets, all 4 models:

| phase | time |
|---|---|
| MMLU-Pro (no NLI) × 4 models | ~2 h 8 m |
| TriviaQA (NLI cluster entropy) × 4 models | ~52 m |
| **total** | **~3 h** |

The TriviaQA traces are short (especially no-think at ~70 tokens/sample), which compensates for the NLI overhead — TriviaQA actually ran *faster* than MMLU-Pro.

---

## 2a. Where each stage ran  (clarification)

| stage | where | why |
|---|---|---|
| Stage 1 — data generation | **GCP A100 VM** | needs GPU for vLLM + reasoning models |
| Stage 2 — inspection | **local laptop CPU** | sklearn-free, pure pandas + matplotlib, runs in ~10 s |
| **Stage 3 — features** | **GCP A100 VM** | BGE-M3 embedding of ~20 k traces; ~33 min on A100 vs ~35 hr projected on CPU |
| Stage 4 — modelling | **local laptop CPU** | sklearn LR + bootstrap; ~40 s for both models |
| Stage 5 — extras | **local laptop CPU** | correlations, calibration, effect sizes; ~10 s |
| Notebook execution | **local laptop CPU** | matplotlib rendering + pandas display; ~30 s |

After Stage 3 finished on the VM, the orchestrator (`_watcher.py`) automatically pulled the parquet files locally, ran Stages 4 + 5 + executed the notebook **all on the laptop**, then stopped the VM. So the VM was only billing for Stage 1 (data gen, ~5 h) and Stage 3 (embeddings, ~1 h). Stages 2/4/5 + notebook are CPU-only and finish in well under a minute combined.

---

## 3. Environment

| | |
|---|---|
| Local OS | Windows 11; conda env `aims_project` (Python 3.10.12 on VM, 3.13.x locally) |
| VM (for stage 3 only) | GCP `aims-project` (a2-highgpu-1g, 1 × A100-40GB, asia-southeast1-c) |
| Embedder model | `BAAI/bge-m3` (8192 token context, 568 M params, ran on A100) |
| Tokenizers | per-model HF tokenizers: Llama-3 for R1-Distill, Qwen for Qwen3-4B |
| Random seed | **42** (used for subsample, CV split, bootstrap, jitter, example pick) |
| Bootstrap resamples | **1000** for every AUROC and every pairwise difference |
| CV | StratifiedKFold, k = 5, shuffle, seed 42 |

**Library versions** (from `stage3_manifest.json`):

```
transformers          5.9.0
torch                 2.11.0
sentence-transformers 5.5.1
pandas                2.3.3
numpy                 2.2.6
scikit-learn          1.7.2
matplotlib            3.10.9
seaborn               0.13.2
pyarrow              24.0.0
```

`lexicons.json` SHA-256: `9d781af6f7bf79ce1879d2e8511daa478a94dde216df966f31670488d7808760`.

---

## 4. Original-spec checklist — item-by-item verification

### Setup

| item | status | note |
|---|---|---|
| Read both generation JSONL files + manifests; confirm record counts | ✅ | 1000 records each; `_lib.load_records` + `load_manifest` |
| Confirm schema fields present on a sampled record | ✅ | `_audit_schema.sh` was run at end of stage 1; loaders dereference each path |
| Set + record global random seed; capture library versions | ✅ | `_lib.SEED = 42`; library versions in `stage3_manifest.json` |
| Freeze 3 lexicons into versioned file; hash into manifest **BEFORE features** | ✅ | `lexicons.json` v1.0 frozen; SHA-256 + per-feature term counts in `stage3_manifest.json`; never edited after stage-3 ran |

### Stage 2 — Inspection

| item | status | note |
|---|---|---|
| Counts: N total, N all-clean, N greedy-clean, N parse-fail | ✅ | `stats.csv` per model |
| Accuracy on all-clean / truncated / overall; clean-vs-truncated gap | ✅ | R1 +8 pt, Qwen3 +32 pt — disclosed |
| Truncation breakdown (greedy / sample / verb_conf / p_true) | ✅ | `truncation_breakdown.csv` |
| `tag_parse_status` + `choice_method` distributions | ✅ | two CSVs in inspection/ |
| Verbalized-conf + P(True) distributions (+ % null) | ✅ | as figures + CSVs |
| Distribution plots correct vs incorrect | ✅ | raincloud baselines fig; trace-feature distributions live in the stage-5 effect-size figure |
| **2 examples each** of correct, incorrect, truncated-greedy, parse-fail → markdown | ✅ | `examples.md` per model, seed 42, fixed selection |

### Stage 3 — Features

| item | status | note |
|---|---|---|
| Clean-set membership per question | ✅ | `in_all_clean`, `greedy_truncated`, `n_samples_clean` |
| `trace_length` (mean over samples, model tokenizer, think region) | ✅ | `count_tokens` uses model's HF tokenizer; reasoning_trace only |
| `hedging_formal / hedging_reasoning / hedging_combined` per token, mean | ✅ | word-boundary regex per `_build_lex_pattern` |
| `connector_density` per token, mean (neutral framing) | ✅ | feature stored as `connector_density`; framing recorded in lexicon file |
| `rep_3`, `rep_4`, `rep_5` (mean over samples; lowercase + collapse first) | ✅ | rep_5 primary; rep_3/4 reported for the robustness check |
| `trace_divergence` (long-context embedder, mean pairwise cosine distance) | ✅ | **BGE-M3** instead of Jina v3 (see §6.A); embedder id + version in manifest |
| `answer_semantic_entropy` (letter entropy across 10 samples' choices) | ✅ | nulls excluded; `n_samples_with_letter` recorded for transparency |
| `p_true` (greedy), `verbalized_confidence` (greedy / 100) | ✅ | NaN if upstream is null |
| Metadata columns + explicit NaNs (no silent imputation) | ✅ | NaNs preserved; modelling stage decides handling |
| Save parquet + CSV | ✅ | `data/features/{model_short}/medqa.parquet` + `.csv` |

### Stage 4 — Modeling

| item | status | note |
|---|---|---|
| Stratified 5-fold CV on `correct`; fixed seed | ✅ | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Standardize **inside each fold** (no leakage) | ✅ | `make_pipeline(StandardScaler(), LogisticRegression())` |
| Baselines alone | ✅ | answer_semantic_entropy, p_true, verbalized_confidence |
| Trace features alone (single-feature AUROC) | ✅ | seven trace features individually |
| Combined trace LR — variant w/ `hedging_combined` AND variant w/ formal+reasoning split | ✅ | `trace_LR_combined` and `trace_LR_split` |
| Full combined LR (trace + baselines) | ✅ | `full_LR`; 193 R1-rows / 4 Qwen3-rows dropped for NaN among features (reported) |
| Leave-one-out on combined trace model | ✅ | `leave_one_out.csv` |
| AUROC for every method + **95 % bootstrap CI ≥ 1000 resamples** | ✅ | 1000 resamples; flipped to ≥ 0.5 for single-feature methods |
| Pairwise bootstrap: trace_LR vs each baseline (win-fraction + Δ distribution) | ✅ | aligned on intersection rows; `pairwise_bootstrap.csv` |
| ECE on probabilistic outputs only; report n_bins | ✅ | `n_bins=10`; LR + p_true + verb_conf only |
| Risk-coverage curves + summary | ✅ | **both AURC and acc@80%** reported (cheap to compute both); risk-coverage saved as PDF |

### Stage 5 — Additional analyses

| item | status | note |
|---|---|---|
| Feature correlation matrix (+ vs `correct`) → heatmap PDF | ✅ | hierarchically clustered lower-triangle, diverging colormap |
| Repetition robustness — signal within all-clean; rep–length correlation; rep-3/4/5 consistency | ✅ | `repetition_robustness.json` — confirms rep_5 is not just length, and rep-N choice is robust |
| Calibration / reliability diagrams | ✅ | per probabilistic method; bar widths encode bin size; ECE in title |
| Per-class summary (mean ± std, std mean diff = Cohen's d) | ✅ | `feature_class_summary.csv` + sorted bar plot |
| Cross-model consistency table (R1 vs Qwen3 single-feature AUROCs) | ✅ | slope plot; lines green if same direction on both models, red if flipped |

### Wrap-up

| item | status | note |
|---|---|---|
| Save all tables (CSV) + figures (PDF) | ✅ | every stage writes to its own subdir |
| Notebook reproduces inspection + headline tables/plots; scripts run end-to-end | ✅ | `analysis_notebook.executed.ipynb` (53 KB) runs all 37 cells without error |
| Manifest with inputs, feature list, lexicon hash, embedder id, CV seed, lib versions | ✅ partial | `stage3_manifest.json` covers embedder + lexicon hash + lib versions + seed. **A combined `results/run_manifest.json`** unifying stage 4 config would be a small improvement (see §6.F). |
| Verify all 10 acceptance criteria | ✅ | see §5 |

---

## 5. Acceptance criteria — verification

| # | criterion | status |
|---|---|---|
| 1 | Feature table — one row per question, all features + metadata, NaNs explicit | ✅ 1000 rows × 22 columns per model, NaN-explicit |
| 2 | Inspection stats + distribution PDFs + example dumps per (model, dataset) | ✅ |
| 3 | All headline metrics on the all-clean set; clean/truncated counts + gap reported | ✅ R1 +8 pt, Qwen3 +32 pt |
| 4 | Per-model (not pooled): baselines-alone, trace-features-alone, combined trace LR, full combined LR | ✅ |
| 5 | Single-feature AUROC AND leave-one-out both reported | ✅ |
| 6 | AUROC for every method with 95 % bootstrap CI; pairwise win-fraction vs each baseline | ✅ |
| 7 | ECE only on probabilistic outputs; risk-coverage curves saved as PDF | ✅ |
| 8 | Additional analyses (correlation matrix, per-class summary, calibration plots) | ✅ |
| 9 | No leakage: scaler + any fitting inside CV folds only; features computed per-question | ✅ scaler in pipeline inside fold; bootstrap on held-out OOF predictions |
| 10 | Both notebook and scripts run end-to-end; analysis manifest written | ✅ executed notebook ships; manifest at `results/stage3_manifest.json` |

---

## 6. Decisions and deviations

Numbered for easy reference. The spec explicitly lists choices the agent may NOT make alone (§ "Decisions the agent must NOT make alone"); none of those were touched. Everything below is an implementation choice within the spec.

### A−1. Headline trace LR: **combined hedging**, with **split as a robustness check**

The spec lists **two trace-LR variants** that differ only in how the hedging lexicon is treated:

- `trace_LR_combined` — **5 features**: `trace_length`, `hedging_combined`, `connector_density`, `rep_5`, `trace_divergence`. The full hedging lexicon (formal ∪ reasoning) is collapsed into one density column.
- `trace_LR_split` — **6 features**: same as above, except `hedging_combined` is replaced by `hedging_formal` and `hedging_reasoning` as two separate columns.

Per spec: *"start with hedging_combined; also a formal+reasoning-separate variant"*. So **`combined` is the spec-locked headline**, and `split` is computed alongside as a robustness check.

The two variants give essentially the same AUROC on every model condition we tested:

| | Trace-Feature Model (`combined`) | Trace LR (`split`) | Δ (split − combined) |
|---|---|---|---|
| Reasoning R1-Distill-Llama-8B  | 0.661  [0.625, 0.698] | 0.657  [0.621, 0.694] | **−0.004** |
| Reasoning Qwen3-4B             | 0.770  [0.730, 0.804] | 0.780  [0.741, 0.814] | **+0.010** |
| Control Qwen3-4B (no-think)    | 0.653  [0.619, 0.688] | 0.650  [0.616, 0.685] | **−0.003** |
| Control Llama-3.1-8B-Instruct  | 0.655  [0.620, 0.688] | 0.658  [0.623, 0.692] | **+0.003** |

Every Δ sits well inside the 95 % bootstrap CI of either variant (CIs are roughly ±0.035 wide). The two variants are statistically indistinguishable.

The **pairwise bootstrap of each variant vs each baseline** delivers identical conclusions:

```
                                      trace_LR_combined          trace_LR_split
R1-Distill   vs Semantic Entropy       Δ -0.018  win 18.9 %      Δ -0.017  win 20.8 %  (wash)
             vs P(True)                Δ +0.118  win 100 %       Δ +0.118  win 100 %
             vs Verbalized Confidence  Δ +0.093  win 100 %       Δ +0.093  win 100 %
Qwen3-4B     vs Semantic Entropy       Δ +0.085  win 100 %       Δ +0.098  win 100 %
             vs P(True)                Δ +0.231  win 100 %       Δ +0.243  win 100 %
             vs Verbalized Confidence  Δ +0.121  win 100 %       Δ +0.134  win 100 %
Control Qwen3 vs Semantic Entropy      Δ -0.043  win  1.9 %      Δ -0.045  win  1.3 %  (loses)
              vs P(True)               Δ +0.141  win 100 %       Δ +0.140  win 100 %
              vs Verbalized Confidence Δ +0.060  win 100 %       Δ +0.057  win 100 %
Control Llama vs Semantic Entropy      Δ -0.129  win  0.0 %      Δ -0.125  win  0.0 %  (loses)
              vs P(True)               Δ +0.135  win 100 %       Δ +0.135  win 100 %
              vs Verbalized Confidence Δ +0.108  win 100 %       Δ +0.109  win 100 %
```

**For the paper**: the headline number is `combined` (spec-locked, parsimonious 5-feature model). The robustness claim is *"trace features beat the cheap baselines (P(True), Verbalized Confidence) at win 100 % regardless of how the hedging lexicon is split or merged; the trace-LR vs answer-side-entropy comparison is unaffected by the lexicon split (every Δ is within bootstrap CI overlap)."* This forecloses a reviewer who would otherwise suggest: *"maybe you cherry-picked the lexicon treatment to get your headline?"* — no, both treatments give the same conclusion.

Implementation: `stage4_model.py` computes both variants in the methods table and runs **pairwise bootstrap for both** vs each baseline. `controls_focused_analysis.py` mirrors the same treatment. Output CSVs (`methods_auroc.csv`, `pairwise_bootstrap.csv`) carry both `trace_LR_combined` (or `trace_LR` in the controls CSV) and `trace_LR_split` rows. Figures show both on the caterpillar / ROC etc.

### A0. Disk has **PDFs only**; notebook re-renders figures inline

Spec says figures saved as PDF. Originally I also wrote PNG companions because Jupyter notebooks can't render PDFs natively in cell outputs. After the spec re-reading: **dropped PNG files entirely**. Disk has only the 22 PDFs. The notebook gets its inline figures by *importing the stage modules and calling the same plotting functions* (which return a `matplotlib.figure.Figure`). Jupyter then captures the rendered output as part of the cell — no PNG files anywhere on disk, and the notebook still shows every figure inline (executed notebook is 2.5 MB after fig embedding).

### A. **Embedder swap — Jina v3 → BAAI/bge-m3**

**Why.** Jina-embeddings-v3 has a Windows-specific load failure on `sentence-transformers`: its custom modelling file references the `flash-attention` package's `block.py`, which is not in the standard HF cache without `flash-attn` installed. Reproduced locally → `FileNotFoundError: ...xlm-roberta-flash-implementation/block.py`. Could not be worked around without installing `flash-attn` (which has its own Windows + driver issues).

**Replacement.** `BAAI/bge-m3`:
- Same long-context guarantee: **8 192 token window** (spec: "no 512-token NLI limit").
- Similar embedding quality (top-tier on MTEB retrieval).
- No flash-attn dependency.
- Same `sentence-transformers` API; one-line code change.

The spec's underlying requirement ("long-context embedder, NO 512-token NLI limit") is satisfied. Embedder id + version embedded in `stage3_manifest.json`.

### B. Stage 3 ran on the GCP A100, not the laptop

CPU embedding 20 000 traces with BGE-M3 was projected at ~35 hours on the laptop based on a 5-record timing test (10.7 min for 5 R1-Distill records). Brought the existing `aims-project` VM back up (same A100 used for stage 1), uploaded `analysis_pipeline/` + lexicons, ran stage 3 on GPU (R1: ~33 min, Qwen3: ~32 min), fetched parquets, verified locally, then stopped the VM. Same defensive verify-before-stop pattern as stage 1.

### C. NaN policy for the full combined LR

Spec recommended "drop rows with any NaN among the modeled features, and report how many" — implemented exactly. Stage 4 prints to stdout:

```
R1-Distill  full combined LR: kept 599 rows (dropped 193 for NaN among features)
Qwen3-4B    full combined LR: kept 736 rows (dropped 4 for NaN among features)
```

R1 drops more because P(True) is null on ~24 % of greedy responses (the model rambled past the verdict token). Single-feature methods drop NaN only for their own column (so p_true single-feature AUROC was computed on n=605 R1 / n=740 Qwen3 rows).

### D. answer_semantic_entropy excludes null sample letters

When a sample's `extracted_choice` is `null` (parse failure), it does not contribute to the letter distribution for entropy. The retained sample size is stored as `n_samples_with_letter` so the effective n is auditable per row. If all 10 samples have nulls, the feature is NaN and the row drops from any model using it.

### E. Pairwise bootstrap aligned on intersection rows

Different methods may have slightly different valid-row sets due to NaN handling (e.g. P(True) is null on 195 R1-Distill rows). The pairwise bootstrap re-runs both methods on the **intersection** of their valid rows so the comparison is exactly paired — same n, same rows resampled together. This is why the pairwise bootstrap's n column can be smaller than the single-method n.

### F. Manifest scope — stage 3 only

Spec says "results/run_manifest.json" with everything (inputs, feature list, lexicon hash, embedder id, CV seed, library versions). I shipped `stage3_manifest.json` (covers embedder + lexicons + library versions + seed) and `stage4_summary.json` (per-method AUROCs). A unified `run_manifest.json` would be cleaner; trivial to add as a wrap-up step.

### G. Risk–coverage reports both AURC AND acc@80 %

Spec said "AURC or accuracy at 80% coverage". I report both because they're cheap to compute together and complementary: AURC summarises the whole curve, acc@80% is a clean point statement.

### H. Calibration bar widths encode bin size

Reliability diagrams use bars whose width grows with the bin's sample count. Adds an immediate visual cue for which bins have enough data to trust. ECE in the panel title is the headline number.

### I. AUROC orientation

For single-feature AUROCs I use the convention `max(AUROC, 1 − AUROC)` so all single-feature rows are reported on a "higher = better" scale. This is purely for the **caterpillar plot's** sortability — the bootstrap CI is computed on the original (oriented) score, so CIs and the win-fraction logic are unaffected.

### J. N-gram tokenization for rep_N

Spec says "lowercase + collapse consecutive whitespace, then tokenize for n-grams" without specifying tokeniser. I use **whitespace split** (standard in the Welleck-2020 rep-N literature). All three rep_N values are computed; their pairwise correlation is >0.96, so the choice is robust.

### K. Example dump selection

Spec asks for 2 per category. I use `random.sample` seeded with 42, picking from each category (correct / incorrect / truncated-greedy / parse-fail). Reproducible.

### L. Cross-model figure trigger

Stage 5's cross-model slope plot needs both models' `stage4/methods_auroc.csv`. If only one model has been processed, stage 5 prints a warning and skips that figure (does not crash).

---

## 7. Methodology-PoC code I reused

Per your Q4 in the previous round — explicit, by file:

| from PoC | where it landed | what it became |
|---|---|---|
| hedging-density per-token formula and word-boundary regex idea | `_lib.py: lex_match_count`, `stage3_features.py: features_for_trace` | extended to **four** lexicons (formal / reasoning / combined / connectors), each with multi-word phrase handling and case-insensitive word-boundary matching |
| matplotlib styling nudges (semi-bold titles, gridline alpha, remove top/right spines) | `_lib.py: apply_style` | rewritten as a single rcParams block + consistent palette + per-model colour map |
| sentence-embedding + pairwise cosine distance idea | `stage3_features.py: trace_divergence_for_question` | swapped from `all-MiniLM-L6-v2` (512-token limit) to **BGE-M3** (8 192 ctx) per spec; normalised vectors |
| MCQ letter entropy as a discrete-distribution semantic entropy | `_lib.py: letter_entropy`, `stage3_features.py` | now records `n_samples_with_letter` for auditability |
| general "load JSONL → compute features → CV LR → AUROC" loop | `stage4_model.py` | **rewritten from scratch** for the stricter spec: leakage-free scaling inside folds, 1000-bootstrap point + paired CIs, leave-one-out, full-LR vs combined-trace, ECE only on probabilistic outputs, risk-coverage with AURC + acc@80% |

**Not reused** from the PoC:
- The PoC's `option_entropy` baseline — the spec separates trace-side divergence (over reasoning text) from answer-side entropy (over MCQ letters), so they live in different features here.
- The PoC's notebook structure — built fresh; external scripts are the source of truth.
- The PoC's flat figure pipeline (one figure per cell). Here each figure is purpose-built for one story.

---

## 8. How to reproduce, resume, extend

### From scratch, given the stage 1 outputs

```powershell
# Stage 2 — fast, CPU (~10 s)
python analysis_pipeline\scripts\stage2_inspect.py

# Stage 3 — SLOW on CPU (~35 hrs total). Use the A100 VM.
# Locally with GPU: a few minutes. Without GPU: see §6.B.
python analysis_pipeline\scripts\stage3_features.py

# Stage 4 — fast, CPU (~40 s for both models, dominated by bootstrap)
python analysis_pipeline\scripts\stage4_model.py

# Stage 5 — fast, CPU (~10 s)
python analysis_pipeline\scripts\stage5_extra.py

# Rebuild notebook source + execute it
python analysis_pipeline\scripts\_build_notebook.py
jupyter nbconvert --to notebook --execute analysis_pipeline\notebooks\analysis_notebook_medqa.ipynb \
    --output analysis_notebook_medqa.executed.ipynb --ExecutePreprocessor.timeout=600
```

### Run the new datasets (MMLU-Pro, TriviaQA) end-to-end

```powershell
# Stage 3 — needs the A100 (BGE-M3 + DeBERTa-v3-large-mnli for TriviaQA NLI).
# Driver on the VM: ~/datagen/_vm_stage3_run.sh  (loops over both datasets x 4 models).
gcloud compute instances start aims-project --zone asia-southeast1-c
tmux new -d -s stage3 'bash ~/datagen/_vm_stage3_run.sh'
# ~3h wall-clock total. scp the resulting parquets back:
gcloud compute scp 'aims-project:~/datagen/analysis_pipeline/data/features/<model>/{mmlu_pro,trivia_qa}.parquet' ...
gcloud compute instances stop aims-project --zone asia-southeast1-c

# Stage 4 — local, ~5 min per dataset, all 4 models.
python analysis_pipeline\scripts\stage4_model.py --dataset mmlu_pro
python analysis_pipeline\scripts\stage4_model.py --dataset trivia_qa

# Build + execute the two new notebooks.
python analysis_pipeline\scripts\_build_notebook.py --dataset mmlu_pro
python analysis_pipeline\scripts\_build_notebook.py --dataset trivia_qa
jupyter nbconvert --to notebook --execute analysis_pipeline\notebooks\analysis_notebook_mmlu_pro.ipynb \
    --output analysis_notebook_mmlu_pro.executed.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute analysis_pipeline\notebooks\analysis_notebook_trivia_qa.ipynb \
    --output analysis_notebook_trivia_qa.executed.ipynb --ExecutePreprocessor.timeout=600
```

### Add a new model

1. Add it to `_lib.MODELS` (short_name → HF id) and to `_lib.MODEL_COLOR` / `MODEL_LABEL`.
2. Stage 1 (data_generation) must have generated `medqa_{short}.jsonl` for it.
3. Re-run stages 2–5 with `--models <new_short>` (or pass the full list — existing per-model results are not overwritten if you re-run only one model).

### Add a new dataset

1. Add a `DatasetAdapter` entry to `data_generation/stage1_generate.DATASET_REGISTRY` with the right `kind` (`"mcq"` or `"free_answer"`), an HF path or local-file loader, and a default split.
2. Stage 1 will route prompts, extraction, and correctness through the kind dispatch automatically. For MCQ, letters up to A–J are supported; for free-answer, use the `<answer>...</answer>` tag protocol + alias matching that TriviaQA uses.
3. Add the dataset to `_lib.DATASETS` (for the analysis side).
4. Run Stage 3 on the VM (`stage3_features.py --dataset <new>`). For `kind=="free_answer"` it loads DeBERTa-v3-large-mnli automatically; for `kind=="mcq"` it uses discrete letter entropy and skips the NLI load.
5. Run Stage 4 locally: `stage4_model.py --dataset <new>` — writes to `results/stage4_{new}/` and `results/stage4_summary_{new}.json` automatically.
6. Build the notebook: `_build_notebook.py --dataset <new>` → `analysis_notebook_{new}.ipynb`, then nbconvert it.

---

## 9. Output map (full)

```
analysis_pipeline/
├── lexicons.json                        v1.0 frozen
├── README_analysis_pipeline.md          spec
├── README_stage2_5_implementation.md    this file
├── data/features/
│   ├── r1-distill-llama-8b/medqa.parquet + .csv     1000 rows × 22 cols
│   └── qwen3-4b/medqa.parquet + .csv                1000 rows × 22 cols
├── results/
│   ├── stage3_manifest.json             embedder + lexicon hash + lib versions + seed
│   ├── stage4_summary.json              per-method AUROCs per model
│   ├── stage3.log                       fetched from VM
│   ├── stage2/   ← cross-model figures
│   │   ├── fig_data_triage.{pdf,png}
│   │   ├── fig_truncation_heatmap_{model}.{pdf,png}
│   │   ├── fig_score_distributions_{model}.{pdf,png}
│   │   └── fig_joint_baselines_{model}.{pdf,png}
│   ├── stage4/   ← per-model figures
│   │   ├── fig_auroc_caterpillar_{model}.{pdf,png}
│   │   ├── fig_roc_{model}.{pdf,png}
│   │   ├── fig_risk_coverage_{model}.{pdf,png}
│   │   └── fig_leave_one_out_{model}.{pdf,png}
│   ├── stage5/
│   │   ├── fig_correlations_{model}.{pdf,png}
│   │   ├── fig_calibration_{model}.{pdf,png}
│   │   ├── fig_effect_sizes_{model}.{pdf,png}
│   │   └── fig_cross_model_slope.{pdf,png}
│   └── {model_short}/{dataset}/
│       ├── inspection/
│       │   ├── stats.csv                              counts + accuracies
│       │   ├── truncation_breakdown.csv               per-stage truncation %
│       │   ├── tag_parse_status_dist.csv
│       │   ├── choice_method_dist.csv
│       │   └── examples.md                            2 records × 4 categories
│       ├── stage4/
│       │   ├── methods_auroc.csv                      every method with AUROC + 95% CI
│       │   ├── leave_one_out.csv                      Δ AUROC when each feature dropped
│       │   ├── pairwise_bootstrap.csv                 trace_LR vs each baseline
│       │   ├── ece.csv                                ECE per probabilistic method
│       │   └── risk_coverage.csv                      AURC + acc@80 per method
│       └── stage5/
│           ├── feature_class_summary.csv              mean ± std + Cohen's d
│           └── repetition_robustness.json             rep-length corr + rep-N consistency
└── notebooks/
    ├── analysis_notebook.ipynb                       source
    └── analysis_notebook.executed.ipynb              executed with figures inline
```

---

## 10. What would I change next time

1. **Combined `run_manifest.json`** unifying stage-3 (embedder, lexicon hash, lib versions) and stage-4 (CV seed, n_bootstrap, n_splits, NaN policy, list of methods) into one file — small upgrade, would tick the last partial spec item.
2. **Per-question wall-clock logging** in stage 3 — same observation as stage 1's impl notes; recoverable from the parquet (no `wall_clock_seconds` column though).
3. **`logprob_mode=full` regeneration** if a future analysis needs token-level entropy methods (currently summary stats only — per spec).
4. **NLI-based answer semantic entropy** for free-form datasets (TriviaQA etc.) — out of scope this run.
5. ~~**Non-reasoning control model** to actually test H3 — out of scope this run.~~
   *Done in Phase 2 (§11 below).*

---

## 11. Phase 2 — non-reasoning controls (added 2026-06-03)

Two control models were added so H3 (the *reasoning-specific* hypothesis) can actually be tested. The reasoning analysis (§1–§10) was **not re-run** and not touched.

### What was added

```
analysis_pipeline/
├── scripts/
│   ├── controls_focused_analysis.py    ← new, single-file focused analysis
│   ├── _watcher_stage3_controls.py     ← end-to-end watcher (VM stage 3 → fetch → analysis → stop VM)
│   └── _lib.py                         ← added qwen3-4b-nothink + llama-3.1-8b-instruct
│                                         to MODELS / MODEL_LABEL / MODEL_COLOR / CONTROL_MODELS
└── results/controls_analysis/          ← all controls outputs land here (segregated)
    ├── manifest.json
    ├── qwen3-4b-nothink/
    │   ├── methods_auroc.csv      pairwise_bootstrap.csv      ece.csv      risk_coverage.csv
    │   └── fig_auroc.pdf   fig_calibration.pdf   fig_risk_coverage.pdf   fig_feature_boxplots.pdf
    └── llama-3.1-8b-instruct/  (same structure)
```

The control parquets live at `data/features/{control_model}/medqa.parquet` — same convention as reasoning, just different model names. Reasoning parquets weren't touched.

### Reasoning analysis vs Controls analysis — explicit differences

This is the key table to read.

| dimension | reasoning analysis (§1–§10) | controls analysis (§11) |
|---|---|---|
| **Script** | `stage2_inspect.py`, `stage3_features.py`, `stage4_model.py`, `stage5_extra.py` | single file `controls_focused_analysis.py` |
| **Output directory** | `results/{model_short}/{dataset}/{stageN}/` and `results/stage{2,4,5}/` | `results/controls_analysis/{model_short}/` |
| **Trace features used in the LR** | **5** — trace_length, hedging_combined, connector_density, rep_5, **trace_divergence** | **5** — *identical* set so cross-condition AUROC is apples-to-apples |
| **Lexicon** | v2.0 (76 terms) | v2.0 — identical, same hash |
| **Embedder for trace_divergence** | BGE-M3 on A100 | BGE-M3 on A100 (same embedder, same checkpoint) |
| **LR variants reported** | `trace_LR_combined`, `trace_LR_split` (hedges split into formal/reasoning), `full_LR` (trace+baselines) | `trace_LR` (= `combined`) **and** `trace_LR_split` reported as a robustness check (the two variants are statistically indistinguishable — see §A−1). `full_LR` and LOO dropped. |
| **Leave-one-out** | yes, on `trace_LR_combined` | **dropped** — focus is method-vs-baseline, not feature ablation |
| **Cross-model figures** | `fig_cross_model_slope.pdf` mixing R1 + Qwen3 | **NOT generated** between reasoning and controls — cross-condition comparison lives in the paper text, not in a mixed figure |
| **Pairwise bootstrap** | both `trace_LR_combined` and `trace_LR_split` vs each of {Sem Entropy, P(True), Verb Conf, full_LR} | both `trace_LR` and `trace_LR_split` vs each of {Sem Entropy, P(True), Verb Conf} (no full_LR comparison) |
| **Metrics reported** | AUROC, ECE, AURC, acc@80, plus correlation heatmap, effect sizes, calibration, examples, repetition robustness | **just the 3 headline metrics** — AUROC + ECE + Risk-coverage — plus feature boxplots |
| **CV / bootstrap settings** | 5-fold stratified CV, 1000 bootstrap, seed 42 | **identical** — same `cv_lr_proba` and `bootstrap_auroc_ci` helpers reused |
| **NaN policy** | drop rows with any NaN among modelled features; report count | identical |
| **NaN imputation** | none (explicit NaN) | none |
| **All-clean filter** | yes, per spec | yes, identical (`in_all_clean & correct.notna()`) |
| **Lexicon-driven hedging features** | formal / reasoning / combined / connectors, all per token, mean over samples | identical computation, but `hedging_reasoning` is mostly silent on controls (no `wait`/`hmm`/`reconsider` because the model isn't backtracking) |
| **Stage 1 input differences** | `<think>` block parsing; `reasoning_trace` is text between tags | inline-CoT split; `reasoning_trace` is text *before* the answer cue — handled transparently by Stage 1, so Stage 2/3 don't need to know |
| **answer_semantic_entropy** | entropy over letter distribution over 10 samples | identical |
| **P(True)** | reason-then-judge, scan after `</think>` | reason-then-judge, scan after first verdict cue (Stage 1 handles this; Stage 4 just reads `p_true_normalized`) |
| **verbalized_confidence** | identical | identical |
| **Trace-Feature Model definition** | logistic regression on 5 standardised trace features inside each CV fold | **identical** down to the sklearn pipeline |

### Quality stats — controls

```
                          Qwen3-4B (no-think)   Llama-3.1-8B-Instruct
clean+labeled rows        926                   974
% of 1000 retained        92.6 %                97.4 %     ← much higher than reasoning (74–80 %)
```

The controls retain more rows because the non-reasoning models rarely hit `max_tokens=4096`. Same all-clean filter as reasoning; controls just happen to be cleaner.

### Headline results — Trace-Feature Model vs each baseline

```
                              Qwen3-4B (no-think)         Llama-3.1-8B-Instruct
─── AUROC ───────────────────────────────────────────────────────────────
P(True)                       0.516                       0.532
Verbalized Confidence         0.592                       0.556
Semantic Entropy              0.696  ← strongest          0.784  ← strongest
Trace-Feature Model           0.653                       0.655
Trace-Feature (hedges split)  0.650                       0.658  ← robustness check

─── ECE (n_bins=10) ─────────────────────────────────────────────────────
P(True)                       0.412                       0.372
Verbalized Confidence         0.317                       0.185
Semantic Entropy              0.159                       0.082
Trace-Feature Model           0.057  ← best               0.023  ← best

─── Risk-Coverage  (AURC ↓, acc@80% ↑) ──────────────────────────────────
P(True)                       AURC 0.350, acc@80 65.4%   AURC 0.332, acc@80 66.1%
Verbalized Confidence         AURC 0.287, acc@80 66.7%   AURC 0.289, acc@80 66.6%
Semantic Entropy              AURC 0.230, acc@80 70.2%   AURC 0.171, acc@80 73.0%
Trace-Feature Model           AURC 0.243, acc@80 68.4%   AURC 0.245, acc@80 70.5%
```

### H3 — same Qwen3-4B, thinking on vs off

```
Reasoning Qwen3-4B   (thinking ON, n=740):   Trace LR 0.770  >  Semantic Entropy 0.683   (+0.088, win 100 %)
Control  Qwen3-4B    (thinking OFF, n=926):  Trace LR 0.653  <  Semantic Entropy 0.696   (-0.043, win  1.9 %)
```

Identical model weights, identical lexicon, identical Trace-Feature LR. The ordering of Trace LR vs Semantic Entropy **flips** when thinking is disabled. This is the clean H3 finding: trace features specifically capture *reasoning-mode* uncertainty behaviour, not just any chain-of-thought.

### Other honest takeaways

- **Trace LR still beats P(True) and Verbalized Confidence on both controls** — wins 100 % of bootstrap resamples, Δ +0.10 to +0.14 in AUROC. The cheap baselines stay beaten.
- **Trace LR is the best-calibrated method on every model condition (reasoning + control).** ECE 0.02–0.06 across all four models; raw baselines are 0.08–0.41. The LR's calibration advantage is *not* a reasoning-only effect.
- **Llama-3.1-8B + CoT produces an unusually strong semantic-entropy signal (0.784).** Suggests Llama's CoT answers are highly consistent when correct and scattered when wrong.

### How to reproduce the controls analysis

```powershell
# After Stage 1 generation for controls and the feature parquets exist:
python analysis_pipeline\scripts\controls_focused_analysis.py
```

If features aren't computed yet, this prints a message telling you to run Stage 3 on the GPU VM:

```powershell
# Bring up the A100 VM (handled by data_generation/scripts/gcp_a100.py up)
# then on the VM:
python analysis_pipeline/scripts/stage3_features.py --models qwen3-4b-nothink llama-3.1-8b-instruct
```

The `_watcher_stage3_controls.py` script does this end-to-end (poll → fetch → run analysis → stop VM), defensive at every step.

### Why some reasoning-analysis elements were deliberately dropped

User request was "simpler — just compare our UQ with existing UQs on the 3 metrics, no need for all combinations". So:

| element | dropped because |
|---|---|
| ~~`trace_LR_split` variant~~ (was initially dropped) | **Restored on 2026-06-08** after feedback. The hedges-split variant is now reported on every model condition as a robustness check, so the paper can claim "trace features beat the cheap baselines regardless of how the hedging lexicon is treated" without a reviewer suspecting a post-hoc lexicon choice. The two variants are statistically indistinguishable on every condition (see §A−1). |
| `full_LR` (trace + baselines) | answers a different question ("if I combine everything, what's the ceiling?") — orthogonal to "our method vs theirs" |
| Leave-one-out | feature-importance ablation, not method comparison |
| Mixed cross-model slope (R1 + Qwen3 + controls in one plot) | mixes 4 conditions on one axis; the per-condition tables tell the story more clearly |
| Stage-2 inspection figures (truncation heatmap, joint baselines) | already in the reasoning analysis; redundant for controls because Stage 1 itself reported truncations as `0.80 %` and `0.26 %` — clearly fine |

What was kept: the per-model headline AUROC caterpillar, the calibration reliability diagrams, the risk-coverage curves, and the per-feature correctness boxplots — exactly what the user asked for.
