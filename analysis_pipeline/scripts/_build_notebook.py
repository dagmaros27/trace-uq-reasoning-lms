"""Generate the analysis notebook. The notebook is a narrative — it loads the
saved CSVs / parquets (not re-running heavy stuff) and **re-renders figures
inline by calling the same plotting functions used by the stage scripts**.

Disk has only PDFs (per spec). The notebook output cells embed rendered
PNGs that Jupyter creates automatically when matplotlib figures are returned
from a code cell — no PNG files on disk.

Usage:
    python _build_notebook.py                          # MedQA (default)
    python _build_notebook.py --dataset mmlu_pro
    python _build_notebook.py --dataset trivia_qa
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--dataset", default="medqa",
                 choices=["medqa", "mmlu_pro", "trivia_qa"])
_args = _ap.parse_args()
DATASET = _args.dataset

# Per-dataset narrative bits.
_DATASET_LABEL = {
    "medqa":     "MedQA",
    "mmlu_pro":  "MMLU-Pro",
    "trivia_qa": "TriviaQA",
}
_DATASET_KIND  = {
    "medqa":     "MCQ (5 options)",
    "mmlu_pro":  "MCQ (up to 10 options)",
    "trivia_qa": "free-answer (closed-book, alias-based scoring)",
}
_SEMANTIC_ENTROPY_NOTE = {
    "medqa":     "discrete letter entropy across the 10 samples (Shannon, bits).",
    "mmlu_pro":  "discrete letter entropy across the 10 samples (Shannon, bits).",
    "trivia_qa": "Kuhn-style NLI cluster entropy: bidirectional entailment with "
                 "`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, "
                 "connected components = semantic clusters, Shannon entropy over cluster sizes (bits).",
}

cells = []

def md(src):   cells.append({"cell_type": "markdown", "metadata": {}, "source": [src]})
def code(src): cells.append({"cell_type": "code", "execution_count": None,
                              "metadata": {}, "outputs": [], "source": [src]})

# ─── Header ──────────────────────────────────────────────────────────────────
md(rf"""# Trace-Based Uncertainty Estimation — Analysis Notebook

**Dataset.** {_DATASET_LABEL[DATASET]} ({_DATASET_KIND[DATASET]}).

**Research question.** Can features extracted from a reasoning model's chain-of-thought *trace*
estimate uncertainty better than standard final-answer baselines (P(True), verbalized confidence,
answer semantic entropy)?

**This run.** 4 models: 2 reasoning (`DeepSeek-R1-Distill-Llama-8B`, `Qwen3-4B` with thinking)
plus 2 non-reasoning controls (`Qwen3-4B` with thinking off, `Llama-3.1-8B-Instruct`).

**`answer_semantic_entropy`.** {_SEMANTIC_ENTROPY_NOTE[DATASET]}

**Scope of this notebook.** Narrative + figures rendered live. The heavy work (features, modeling,
extras) is already done by the stage scripts (`stage2_inspect.py`, `stage3_features.py`,
`stage4_model.py`, `stage5_extra.py`); the notebook only re-loads the saved tables/parquets and
re-runs the **plotting** code so matplotlib displays each figure inline. PDFs of the same figures
are saved on disk under `analysis_pipeline/results/`.

**Reading order.**
1. Data overview (clean set, truncation, accuracy)
2. Baseline confidence sanity check
3. Features at a glance (correlations, effect sizes)
4. Headline modelling result — AUROC caterpillar with bootstrap CIs
5. Leave-one-out feature importance
6. ROC + risk–coverage
7. Calibration
8. Cross-model consistency
9. Pairwise bootstrap: real gain or noise?
10. Take-aways""")

# ─── Setup ───────────────────────────────────────────────────────────────────
md("## 1. Setup")
code(r"""import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

sys.path.insert(0, str(Path.cwd().parent / "scripts"))
import _lib as L
import stage2_inspect as S2
import stage4_model as S4
import stage5_extra as S5

L.apply_style()
MODELS  = list(L.MODELS.keys())
DATASET = """ + json.dumps(DATASET) + r"""
print("Models      :", MODELS)
print("Dataset     :", DATASET)
print("Project root:", L.PROJECT)""")

code(r"""# Preload everything the figures need
records = {m: L.load_records(m, DATASET) for m in MODELS}
feats   = {m: pd.read_parquet(L.FEATURES_DIR / m / f"{DATASET}.parquet") for m in MODELS}
print({m: f"{len(records[m])} records, {len(feats[m])} feature rows" for m in MODELS})""")

code(r"""def show_csv(rel, head=None, fmt=None):
    p = L.RESULTS_DIR / rel
    if not p.exists():
        print('Missing:', p); return
    df = pd.read_csv(p)
    if head: df = df.head(head)
    if fmt:
        for c, f in fmt.items():
            if c in df.columns: df[c] = df[c].map(f)
    display(df)""")

# ─── 2. Data overview ───────────────────────────────────────────────────────
md(r"""## 2. Data overview — where each question lands

Before any modeling we sort the 1 000 questions per model into:

* **all-clean** — no truncation anywhere (greedy + 10 samples + verb_conf + P(True))
* **truncated** — at least one generation hit the 4 096-token cap
* **parse-fail** — greedy answer letter could not be extracted

All headline metrics use the **all-clean set only** (per spec). The clean/truncated accuracy gap is
reported because clean questions skew towards easier items.""")

code(r"""# Cross-model triage flow
S2.fig_data_triage(records, None);""")

md("### Per-model stats")
code(r"""for m in MODELS:
    display(Markdown(f"#### {L.MODEL_LABEL[m]}"))
    show_csv(f"{m}/{DATASET}/inspection/stats.csv",
             fmt={"acc_overall": "{:.3f}".format, "acc_all_clean": "{:.3f}".format,
                  "acc_truncated": "{:.3f}".format, "pct_all_clean": "{:.1f}%".format,
                  "pct_truncated_any": "{:.1f}%".format})""")

md("### Truncation breakdown by generation stage")
code(r"""for m in MODELS:
    display(Markdown(f"#### {L.MODEL_LABEL[m]}"))
    show_csv(f"{m}/{DATASET}/inspection/truncation_breakdown.csv",
             fmt={"pct": "{:.2f}%".format})""")

md(r"""### Per-question truncation map
Each row = one question that had at least one truncated generation, sorted by total truncations.
Red = truncated. The side panel shows how many truncations the question had.""")
code(r"""for m in MODELS:
    S2.fig_truncation_heatmap(records[m], m, None);""")

# ─── 3. Baseline sanity check ───────────────────────────────────────────────
md(r"""## 3. Baseline confidence — visual sanity check

Are P(True) and verbalized confidence doing what we'd expect — higher on correct answers than
incorrect? Raincloud plots (half-violin + jittered strip + box) below.""")
code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    S2.fig_score_distributions(records[m], m);""")

md("### Are the two baselines redundant? Joint distribution + marginals")
code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    S2.fig_joint_baselines(records[m], m);""")

# ─── 4. Features ────────────────────────────────────────────────────────────
md(r"""## 4. Features at a glance — what's redundant, what differs by correctness

### Correlation matrix (clustered, lower-triangle)
Highly correlated features (|r| > 0.5) indicate redundancy — e.g. does `rep_5` just track
`trace_length`? Bright cells annotated with the value.""")
code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    df_clean = feats[m][feats[m]['in_all_clean'] & feats[m]['correct'].notna()].copy()
    df_clean['correct'] = df_clean['correct'].astype(int)
    S5.fig_correlation_matrix(df_clean, m);""")

md(r"""### Effect sizes (Cohen's *d*)
Per feature, how much do correct vs incorrect answers differ on average?
Positive = feature *higher* in correct answers (confidence signal).
Negative = feature *higher* in incorrect answers (uncertainty signal).""")
code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    df_clean = feats[m][feats[m]['in_all_clean'] & feats[m]['correct'].notna()].copy()
    df_clean['correct'] = df_clean['correct'].astype(int)
    summary = S5.class_summary_table(df_clean, m)
    S5.fig_effect_sizes(summary, m);""")

md(r"""### Feature distributions — correct vs incorrect (boxplots)
Same effect sizes, now as actual distributions. Each panel: correct (green) vs incorrect (red)
for one feature. Visually richer than the Cohen's-*d* bar above.""")
code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    df_clean = feats[m][feats[m]['in_all_clean'] & feats[m]['correct'].notna()].copy()
    df_clean['correct'] = df_clean['correct'].astype(int)
    S5.fig_feature_boxplots(df_clean, m);""")

md(r"""### Top-10 hedging phrases — which terms actually fire
For each model, count every occurrence of every term in the v2.0 hedging lexicon
(relational formal-hedges ∪ reasoning-extension hedges = 76 terms) across all samples in the
all-clean set. The bars below show the 10 phrases that contribute most.""")
code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    S5.fig_top_hedges(records[m], m);""")

# ─── 5. Headline AUROC ──────────────────────────────────────────────────────
md(r"""## 5. Headline modelling result — predicting greedy correctness

Every method below is evaluated with **5-fold stratified CV** + **1 000-bootstrap 95 % CIs** on
the all-clean set. Per-model (no pooling). Standardisation fitted inside each train fold (no
leakage).

**Method legend**
* `baseline` — final-answer-only (P(True), verbalized confidence, answer semantic entropy)
* `trace_single` — one trace feature on its own
* `trace_LR` — logistic regression on multiple trace features
* `full_LR` — trace features + baselines combined""")

code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    p = L.RESULTS_DIR / m / DATASET / "stage4" / "methods_auroc.csv"
    if p.exists():
        df = pd.read_csv(p)
        S4.fig_auroc_caterpillar(df, m);""")

md("### Underlying AUROC table (sorted descending)")
code(r"""for m in MODELS:
    display(Markdown(f"#### {L.MODEL_LABEL[m]}"))
    p = L.RESULTS_DIR / m / DATASET / "stage4" / "methods_auroc.csv"
    if p.exists():
        df = pd.read_csv(p).sort_values("auroc", ascending=False)
        df["AUROC [95% CI]"] = df.apply(lambda r: f"{r['auroc']:.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", axis=1)
        display(df[["method", "kind", "n", "AUROC [95% CI]"]])""")

# ─── 6. Leave-one-out ────────────────────────────────────────────────────────
md(r"""## 6. Leave-one-out — which trace feature carries the signal?

For the primary `trace_LR_combined` model, drop one feature at a time and re-fit. Dumbbells below
show AUROC without each feature (grey dot) compared to the full-feature AUROC (coloured dot,
dashed line). Bigger drop ⇒ more uniquely informative feature.""")
code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    loo_df = pd.read_csv(L.RESULTS_DIR / m / DATASET / "stage4" / "leave_one_out.csv")
    methods_df = pd.read_csv(L.RESULTS_DIR / m / DATASET / "stage4" / "methods_auroc.csv")
    auc_full = float(methods_df[methods_df['method'] == 'trace_LR_combined']['auroc'].iloc[0])
    S4.fig_loo(loo_df.to_dict(orient="records"), auc_full, m);""")

# ─── 7. ROC + risk-coverage ─────────────────────────────────────────────────
md(r"""## 7. ROC curves & risk–coverage

ROC: how each headline method trades FPR for TPR. Risk-coverage: if the model can abstain on its
k % least-confident questions, what error rate remains on the rest?""")
code(r"""# These two figures need the CV out-of-fold scores, which we recompute on the fly.
# This is fast (a few seconds per model).
from sklearn.model_selection import StratifiedKFold

def _scores_for(model_short):
    df = feats[model_short]
    df = df[df['in_all_clean'] & df['correct'].notna()].copy().reset_index(drop=True)
    df['correct'] = df['correct'].astype(int)
    scores, labels = {}, {}
    for f in S4.BASELINES + S4.TRACE_FEATURES:
        s, lab = S4.single_feature_score(df, f)
        if len(s): scores[f] = s; labels[f] = lab
    # combined trace LR (5-fold CV oof)
    for name, ff in S4.COMBINED_TRACE_VARIANTS.items():
        sub = df.dropna(subset=ff + ['correct'])
        X = sub[ff].values; y = sub['correct'].astype(int).values
        scores[name] = S4.cv_lr_proba(X, y); labels[name] = y
    # full LR
    full = S4.COMBINED_TRACE_VARIANTS['trace_LR_combined'] + S4.BASELINES
    sub = df.dropna(subset=full + ['correct'])
    X = sub[full].values; y = sub['correct'].astype(int).values
    scores['full_LR'] = S4.cv_lr_proba(X, y); labels['full_LR'] = y
    # risk-coverage curves dict
    rc = {n: L.risk_coverage_curve(scores[n], labels[n]) for n in
          ['p_true', 'verbalized_confidence', 'answer_semantic_entropy',
           'trace_LR_combined', 'full_LR'] if n in scores}
    return scores, labels, rc

cached_scores = {m: _scores_for(m) for m in MODELS}
print('precomputed CV scores for ROC and risk-coverage plots')""")

code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]} — ROC"))
    sc, lab, rc = cached_scores[m]
    S4.fig_roc_curves(sc, lab, m);""")

code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]} — Risk vs coverage"))
    sc, lab, rc = cached_scores[m]
    S4.fig_risk_coverage(rc, m);""")

# ─── 8. Calibration ─────────────────────────────────────────────────────────
md(r"""## 8. Calibration (reliability diagrams)

Diagonal dashed line = perfect calibration. **Bar widths encode bin size** so you can see which
bins have enough data to trust. ECE in each panel title — lower is better.""")
code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    df_clean = feats[m][feats[m]['in_all_clean'] & feats[m]['correct'].notna()].copy()
    df_clean['correct'] = df_clean['correct'].astype(int)
    S5.fig_calibration(df_clean, m);""")

# ─── 9. Cross-model ─────────────────────────────────────────────────────────
md(r"""## 9. Cross-model consistency

If a feature only works on one model, that's worth knowing. Slope plot: each method's AUROC for
R1-Distill on the left, Qwen3-4B on the right. Green line = same direction on both models. Red
line = flips — works on one but not the other.""")
code(r"""summaries = {}
for m in MODELS:
    p = L.RESULTS_DIR / m / DATASET / "stage4" / "methods_auroc.csv"
    if p.exists(): summaries[m] = pd.read_csv(p)
if len(summaries) >= 2:
    S5.fig_cross_model_slope(summaries);""")

# ─── 10. Pairwise bootstrap ─────────────────────────────────────────────────
md(r"""## 10. Is the gain real, or noise?

Pairwise bootstrap of *combined trace LR* − *each baseline* tells us the **win fraction** (P our
method ≥ baseline on bootstrap resamples). If the 95 % CI of the difference excludes 0, the gain
is reliable beyond sampling noise.""")
code(r"""for m in MODELS:
    display(Markdown(f"### {L.MODEL_LABEL[m]}"))
    p = L.RESULTS_DIR / m / DATASET / "stage4" / "pairwise_bootstrap.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["Δ AUROC [95% CI]"] = df.apply(lambda r: f"{r['diff_median']:+.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", axis=1)
        df["win"] = df["win_fraction"].map(lambda x: f"{x:.1%}")
        display(df[["ours", "vs", "Δ AUROC [95% CI]", "win", "n"]])""")

# ─── 11. Take-aways ─────────────────────────────────────────────────────────
md(r"""## 11. Take-aways (edit before sharing)

1. **Best single trace feature** — top `trace_single` row.
2. **Does combining trace features help over the strongest baseline?** Compare `trace_LR_combined`
   to the strongest `baseline` row. The pairwise-bootstrap CI for the difference says whether the
   gap is real.
3. **Does adding baselines on top of trace features help?** Compare `full_LR` to `trace_LR_combined`.
4. **Does the picture hold across both models?** Cross-model slope.

> *Intentionally left for you to fill in once you've read the figures.*

---

## Appendix — re-running parts

```bash
python analysis_pipeline/scripts/stage2_inspect.py
python analysis_pipeline/scripts/stage3_features.py     # SLOW; needs GPU
python analysis_pipeline/scripts/stage4_model.py
python analysis_pipeline/scripts/stage5_extra.py
```

Each stage writes **PDFs only** to `analysis_pipeline/results/{stage}/`. Lexicon hash + embedder id
+ library versions + seed are in `results/stage3_manifest.json`.""")

# ─── Write notebook ──────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "cells": cells,
}
_filename = f"analysis_notebook_{DATASET}.ipynb"
out = Path(f"D:/new_final_project/analysis_pipeline/notebooks/{_filename}")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Wrote {out}  ({len(cells)} cells)")
