# Trace-Based Uncertainty Estimation for Reasoning Language Models

Code and data for the AIMS essay *"Trace-Based Uncertainty Estimation for
Reasoning Language Models"* (Dagmawi Misker Gedamu, 2026).

**TL;DR** — Reasoning LMs (QwQ, R1, Qwen3-think) are overconfident on wrong
answers, but their *reasoning traces* leak the doubt. A 5-feature logistic
regression on simple trace statistics (`trace_length`, `rep_5`,
`hedging_formal`, `connector_density`, `trace_divergence`) beats semantic
entropy — the strongest black-box baseline — on RL-tuned reasoning models
doing multiple-choice QA (Δ AUROC up to **+0.147**, paired-bootstrap 95% CIs
strictly above zero), and beats P(True) and verbalised confidence everywhere.

See [PIPELINE.md](PIPELINE.md) for a narrative walkthrough of the full study.

---

## Repository layout

```
trace-uq-reasoning-lms/
├── README.md                  ← you are here (reproduction guide)
├── PIPELINE.md                ← what the study is and how the pieces fit
├── requirements.txt
├── data_generation/
│   ├── stage1_generate.py     ← vLLM generation script (GPU)
│   ├── README_generation_pipeline.md   ← full generation spec
│   └── manifests/             ← per-run provenance (model, params, counts)
├── analysis_pipeline/
│   ├── lexicons.json          ← frozen hedging/connector lexicons (versioned)
│   ├── data/features/         ← ★ extracted per-question feature tables
│   │   └── {model}/{dataset}.parquet
│   ├── results_for_paper/     ← reference outputs (tables + figures in the essay)
│   └── scripts/
│       ├── _lib.py            ← shared: paths, metrics, bootstrap, ECE, lexicons
│       ├── stage3_features.py ← raw generations → feature tables (GPU)
│       ├── stage4_model.py    ← per-cell modelling (legacy stage)
│       ├── _paper_step*.py    ← ★ the paper analysis chain (steps 1–8)
│       └── _paper_aux_*.py    ← auxiliary tables (feature importance, greedy vs baselines)
└── docs/                      ← original design specs + implementation notes
```

★ = the two things most users need.

---

## Quick start — reproduce the paper analysis (no GPU, ~15 minutes)

The extracted feature tables are included, so the entire statistical analysis
reproduces on a laptop:

```bash
git clone <this-repo>
cd trace-uq-reasoning-lms
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt

cd analysis_pipeline/scripts
python _paper_step1_eda.py          # T1.x  dataset/EDA tables
python _paper_step2a_corr.py        # feature correlation + redundancy
python _paper_step2b_perfeat.py     # per-feature AUROC + Cohen's d
python _paper_step2d_lofo.py        # leave-one-feature-out
python _paper_step3_freeze.py       # ★ canonical frozen trace_LR fit (OOF predictions)
python _paper_step5_baselines.py    # ★ trace_LR vs baselines + paired bootstrap
python _paper_step6_combined.py     # full_LR upper bound
python _paper_step7_calibration.py  # ECE / Brier / NLL (Platt-scaled, in-fold)
python _paper_step8_synthesis.py    # ★ headline synthesis table
python _paper_step2c_greedy.py      # greedy vs sampled trace_LR (needs step 3 first)
python _paper_aux_feature_importance.py
python _paper_aux_greedy_vs_baselines.py
```

Each script writes CSV tables and PDF figures into
`analysis_pipeline/results_for_paper/<section>/`. The committed contents of
that directory are the reference outputs — after a re-run, your files should
match them (all randomness is seeded; see *Determinism* below).

**Note on `_paper_step2c_greedy.py`:** it re-tokenises greedy traces with each
model's own tokenizer. Llama-3.1's tokenizer is gated on Hugging Face — set
`HF_TOKEN` in your environment (any account with Llama-3.1 access) or the
script will skip those cells.

### Order matters

`_paper_step3_freeze.py` must run before steps 5, 6, 7, 2c, and the aux
scripts (they consume its out-of-fold predictions, `T3.1.csv`). Everything
else is independent.

---

## Full reproduction from scratch (GPU, ~4–6 days of compute)

Stage 1 regenerates every model output; stage 3 re-extracts features.

### Stage 1 — generation (A100-class GPU)

Per question: 1 greedy pass (T=0), 10 sampled passes (T=0.7, top_p=0.95),
1 verbalised-confidence probe, 1 P(True) probe. 1000 questions per
(model, dataset) cell.

```bash
pip install vllm datasets transformers
cd data_generation
python stage1_generate.py --model Qwen/Qwen3-4B          --dataset medqa     --n-questions 1000
python stage1_generate.py --model Qwen/Qwen3-4B          --dataset mmlu_pro  --n-questions 1000
# ... repeat per (model, dataset); see manifests/ for the exact 14 cells + params
```

Models (HF ids): `Qwen/Qwen3-4B` (thinking on and off),
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `Qwen/QwQ-32B` (needs 2×A100,
tensor-parallel 2), `meta-llama/Llama-3.1-8B-Instruct` (gated — needs
`HF_TOKEN`). Datasets: MedQA, MMLU-Pro, TriviaQA. The cell
(QwQ-32B, MedQA) was not generated (cost); 14 cells total.

`data_generation/manifests/` contains the exact manifest of every original
run (sampling params, question counts, seeds, library versions) for
verification. The full generation spec is
`data_generation/README_generation_pipeline.md`.

### Stage 3 — feature extraction (GPU recommended)

```bash
pip install sentence-transformers torch transformers
cd analysis_pipeline/scripts
python stage3_features.py --model qwen3-4b --dataset medqa
# ... repeat per cell
```

Embeds every trace with **BAAI/bge-m3** (8192-token context) for
`trace_divergence`, computes lexical/structural features against the frozen
`lexicons.json`, and computes NLI-cluster semantic entropy (DeBERTa-v3-MNLI)
for TriviaQA. Output goes to `analysis_pipeline/data/features/` — the same
tables that ship with this repo.

Raw generations are ~3 GB and are not tracked in git. Regenerate them via
stage 1, or contact the author for the archive.

---

## Determinism

Every stochastic step is seeded (`seed=42` throughout): StratifiedKFold
splits, bootstrap resampling (`np.random.RandomState(42)`), and generation
sampling seeds are recorded in the manifests. Re-running the analysis chain
on the shipped feature tables reproduces the paper numbers exactly.

## Key results to expect

| cell | Δ AUROC (trace_LR − semantic entropy) | 95% CI | bootstrap wins |
|---|---|---|---|
| qwen3-4b / MedQA | +0.094 | [+0.051, +0.133] | 100% |
| qwen3-4b / MMLU-Pro | +0.097 | [+0.052, +0.134] | 100% |
| qwq-32b / MMLU-Pro | **+0.147** | [+0.085, +0.216] | 100% |

trace_LR wins on 3 of the 5 reasoning × MCQ cells and 0 of the 9 cells
elsewhere — the effect is specific to RL-tuned reasoning models on
multiple-choice tasks. `analysis_pipeline/results_for_paper/08_.../T8.1.csv`
is the full synthesis table.

## Citation

```bibtex
@mastersthesis{gedamu2026trace,
  title  = {Trace-Based Uncertainty Estimation for Reasoning Language Models},
  author = {Gedamu, Dagmawi Misker},
  school = {African Institute for Mathematical Sciences (AIMS) South Africa},
  year   = {2026}
}
```
