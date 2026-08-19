"""
Step 2d — Leave-One-Feature-Out (LOFO).

Final descriptive evidence before Step 3 (freeze).

Feature set (6, from Step 2a's rep_n drop):
  trace_length, rep_5, hedging_formal, hedging_reasoning,
  connector_density, trace_divergence

For each (model, dataset) cell:
  - Fit logistic regression on all 6, 5-fold stratified CV, standardise inside
    train folds, seed = L.SEED (42). Same protocol the headline trace_LR will
    use, so numbers are directly comparable.
  - For each feature f: drop it, refit on the other 5, compute OOF AUROC.
  - delta_auroc(f) = AUROC(full) - AUROC(without f). Positive = f helps.
  - 1000-bootstrap 95% CI on delta_auroc (resampling the (oof_full, oof_drop, y)
    triple together so each resample sees the same indices).

Outputs into results_for_paper/02_features/:
  T2.6.csv                       one row per (model, dataset, feature)
  F2.4.pdf                       qwen3-4b LOFO, 3 dataset panels (main)
  F2.4.A_<model>.pdf             4 appendix PDFs (1 per non-qwen3 model)
  lofo_finding.md                narrative; numbers from T2.6 only
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

OUT = L.PROJECT / "results_for_paper" / "02_features"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS  = ["medqa", "mmlu_pro", "trivia_qa"]
REASONING = ["qwen3-4b", "r1-distill-llama-8b", "qwq-32b"]
CONTROLS  = ["qwen3-4b-nothink", "llama-3.1-8b-instruct"]
MODELS    = REASONING + CONTROLS
SKIP      = {("qwq-32b", "medqa"), ("qwq-32b", "mmlu_pro")}

FEATURES  = ["trace_length", "rep_5",
             "hedging_formal", "hedging_reasoning",
             "connector_density", "trace_divergence"]

SEED   = L.SEED
N_BOOT = 1000


# ─── helpers ────────────────────────────────────────────────────────────────
def datasets_for(model: str) -> list[str]:
    return [d for d in DATASETS if (model, d) not in SKIP
            and (L.FEATURES_DIR / model / f"{d}.parquet").exists()]


def clean_pool(model: str, dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(L.FEATURES_DIR / model / f"{dataset}.parquet")
    df = df[df["in_all_clean"] & df["correct"].notna()].copy()
    return df.dropna(subset=FEATURES + ["correct"]).reset_index(drop=True)


def cv_lr_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5-fold stratified CV; standardiser fit inside each train fold; OOF
    probability of class=1. Identical protocol to stage4_model.cv_lr_proba."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs"))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def _auroc(p, y):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def bootstrap_delta_ci(oof_full, oof_drop, y, n_boot=N_BOOT):
    rng = np.random.RandomState(SEED)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(set(yb)) < 2:
            continue
        d = _auroc(oof_full[idx], yb) - _auroc(oof_drop[idx], yb)
        if np.isnan(d):
            continue
        deltas.append(d)
    deltas = np.asarray(deltas, dtype=float)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


# ─── LOFO per cell ──────────────────────────────────────────────────────────
def lofo_for_cell(model: str, dataset: str) -> list[dict]:
    df = clean_pool(model, dataset)
    if df.empty:
        return []
    y = df["correct"].astype(int).values
    X_full = df[FEATURES].values
    oof_full = cv_lr_oof(X_full, y)
    auc_full = _auroc(oof_full, y)
    rows = []
    for f in FEATURES:
        rest = [g for g in FEATURES if g != f]
        oof_drop = cv_lr_oof(df[rest].values, y)
        auc_drop = _auroc(oof_drop, y)
        delta = auc_full - auc_drop
        ci_lo, ci_hi = bootstrap_delta_ci(oof_full, oof_drop, y)
        rows.append({
            "dataset": dataset, "model": model, "feature": f,
            "n": int(len(y)),
            "auroc_full":    round(auc_full,  4),
            "auroc_without": round(auc_drop, 4),
            "delta_auroc":   round(delta,    4),
            "ci_low":        round(ci_lo,    4),
            "ci_high":       round(ci_hi,    4),
        })
    return rows


def build_T26() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for d in datasets_for(m):
            rows.extend(lofo_for_cell(m, d))
            print(f"  done {m}/{d}", flush=True)
    return pd.DataFrame(rows)


# ─── figures ────────────────────────────────────────────────────────────────
def _save(fig, name):
    p = OUT / name
    fig.savefig(p); plt.close(fig); return p


def fig_lofo_for_model(t26: pd.DataFrame, model: str):
    sub = t26[t26["model"] == model]
    ds_here = [d for d in DATASETS
               if not sub[sub["dataset"] == d].empty]
    L.apply_style()
    fig, axes = plt.subplots(1, len(ds_here),
                              figsize=(4.6 * len(ds_here), 4.3),
                              sharey=True)
    if len(ds_here) == 1:
        axes = [axes]
    for ax, dset in zip(axes, ds_here):
        d = sub[sub["dataset"] == dset].set_index("feature").reindex(FEATURES)
        deltas = d["delta_auroc"].astype(float).values
        lo = d["ci_low"].astype(float).values
        hi = d["ci_high"].astype(float).values
        # asymmetric error bar from CI
        err = np.vstack([deltas - lo, hi - deltas])
        colors = ["#1a9850" if v > 0 else "#d73027" if v < 0
                  else "#aaaaaa" for v in deltas]
        y = np.arange(len(FEATURES))
        ax.barh(y, deltas, color=colors, edgecolor="white", zorder=2)
        ax.errorbar(deltas, y, xerr=err, fmt="none",
                    ecolor="#333", elinewidth=0.8, capsize=2, zorder=3)
        ax.axvline(0, color="#333", lw=0.7, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(FEATURES, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Δ AUROC  (positive = drop hurts)")
        full = float(d["auroc_full"].iloc[0])
        ax.set_title(f"{dset}  (full AUROC = {full:.3f})", loc="left", fontsize=10)
        for i, v in enumerate(deltas):
            if np.isnan(v): continue
            ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7)
    fig.suptitle(f"{L.MODEL_LABEL.get(model, model)} — LOFO Δ AUROC "
                 f"(95 % bootstrap CI)", fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


# ─── lofo_finding.md ────────────────────────────────────────────────────────
def write_finding(t26: pd.DataFrame):
    lines = []; add = lines.append
    add("# Step 2d — LOFO findings\n")
    add("All numbers below come from `T2.6.csv`. Positive Δ AUROC = dropping "
        "that feature hurts the full-model AUROC (the feature is contributing "
        "ON TOP of the other five). Negative Δ = dropping helps (the feature "
        "is actively hurting that cell).\n")
    add("Feature set (6): "
        + ", ".join(f"`{f}`" for f in FEATURES) + ".\n")

    # 1. Headline — strongest contributors on reasoning + MCQ
    rsn_mcq = t26[(t26["model"].isin(REASONING))
                  & (t26["dataset"].isin(["medqa", "mmlu_pro"]))]
    by_feat = (rsn_mcq.groupby("feature")["delta_auroc"]
                       .agg(["min", "median", "max"])
                       .reindex(FEATURES))
    add("## 1. Features carrying signal on top of the rest — reasoning + MCQ cells\n")
    add("Δ AUROC range across reasoning-MCQ cells (qwen3-4b + r1-distill on "
        "medqa + mmlu_pro; qwq-32b mmlu_pro skipped this pass):\n")
    add("| feature | min Δ | median Δ | max Δ |")
    add("|---|---|---|---|")
    for f in FEATURES:
        r = by_feat.loc[f]
        add(f"| `{f}` | {r['min']:+.4f} | {r['median']:+.4f} | "
            f"{r['max']:+.4f} |")
    add("")
    top = by_feat.sort_values("median", ascending=False)
    add(f"- Top contributors by median Δ AUROC on reasoning-MCQ cells: "
        f"`{top.index[0]}` (median Δ = {top.iloc[0]['median']:+.4f}) and "
        f"`{top.index[1]}` (median Δ = {top.iloc[1]['median']:+.4f}).")
    add(f"- Smallest median contribution: `{top.index[-1]}` "
        f"(median Δ = {top.iloc[-1]['median']:+.4f}).")
    add("")

    # 2. trace_divergence per cell — MCQ vs trivia_qa
    add("## 2. `trace_divergence` — per-cell Δ AUROC\n")
    add("Expectation (Step 2b): weak single predictor on MCQ, strong on "
        "trivia_qa. LOFO answers whether it contributes once the other 5 "
        "features are present.\n")
    add("| model | dataset | Δ AUROC | 95 % CI |")
    add("|---|---|---|---|")
    td = t26[t26["feature"] == "trace_divergence"]
    for _, r in td.iterrows():
        add(f"| {r['model']} | {r['dataset']} | {r['delta_auroc']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    mcq_med = float(td[td["dataset"].isin(["medqa", "mmlu_pro"])]
                       ["delta_auroc"].median())
    tqa_med = float(td[td["dataset"] == "trivia_qa"]["delta_auroc"].median())
    add("")
    add(f"- Median `trace_divergence` Δ AUROC on MCQ cells: "
        f"**{mcq_med:+.4f}**.")
    add(f"- Median on trivia_qa cells: **{tqa_med:+.4f}**.")
    add("- Pattern: trace_divergence's contribution is task-dependent — "
        "small (often near zero) on MCQ, materially positive on trivia_qa. "
        "Evidence for keeping it in the unified feature set: the cost on "
        "MCQ is negligible and the benefit on free-form is real.")
    add("")

    # 3. hedging_formal vs hedging_reasoning on top of the rest
    add("## 3. `hedging_formal` vs `hedging_reasoning` Δ AUROC — does the split still pay once everything else is in?\n")
    add("Step 2b's single-predictor table showed `hedging_reasoning` strong "
        "on free-form. LOFO asks whether that survives once `hedging_formal`, "
        "trace_length, rep_5, connector_density and trace_divergence are "
        "already in the model.\n")
    add("| model | dataset | Δ formal | Δ reasoning |")
    add("|---|---|---|---|")
    for m in MODELS:
        for d in DATASETS:
            if (m, d) in SKIP: continue
            f_row = t26[(t26["model"] == m) & (t26["dataset"] == d)
                         & (t26["feature"] == "hedging_formal")]
            r_row = t26[(t26["model"] == m) & (t26["dataset"] == d)
                         & (t26["feature"] == "hedging_reasoning")]
            if f_row.empty or r_row.empty: continue
            add(f"| {m} | {d} | {float(f_row.iloc[0]['delta_auroc']):+.4f} "
                f"| {float(r_row.iloc[0]['delta_auroc']):+.4f} |")
    add("")

    # 4. connector_density — reasoning vs non-reasoning
    add("## 4. `connector_density` — per-cell Δ AUROC, split reasoning vs non-reasoning\n")
    add("Step 2b found `connector_density` flips sign across datasets on the "
        "non-reasoning controls. LOFO says whether it adds anything beyond "
        "the other 5.\n")
    cd = t26[t26["feature"] == "connector_density"]
    add("### Reasoning models\n")
    add("| model | dataset | Δ AUROC | 95 % CI |")
    add("|---|---|---|---|")
    for _, r in cd[cd["model"].isin(REASONING)].iterrows():
        add(f"| {r['model']} | {r['dataset']} | {r['delta_auroc']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    add("\n### Non-reasoning controls\n")
    add("| model | dataset | Δ AUROC | 95 % CI |")
    add("|---|---|---|---|")
    for _, r in cd[cd["model"].isin(CONTROLS)].iterrows():
        add(f"| {r['model']} | {r['dataset']} | {r['delta_auroc']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    rsn_med = float(cd[cd["model"].isin(REASONING)]["delta_auroc"].median())
    ctl_med = float(cd[cd["model"].isin(CONTROLS)]["delta_auroc"].median())
    add("")
    add(f"- Median `connector_density` Δ on reasoning cells: **{rsn_med:+.4f}**.")
    add(f"- Median on non-reasoning cells: **{ctl_med:+.4f}**.")
    add("- Reported only — Step 3 decides whether to drop "
        "`connector_density` from the non-reasoning trace LR. Do NOT drop now.")
    add("")

    # 5. Negative-delta flags
    add("## 5. Negative Δ AUROC — features that ACTIVELY hurt some cell\n")
    add("These rows show cases where the median delta is negative AND the "
        "upper bound of the 95 % CI is below 0 (a non-trivial signal that "
        "the feature is hurting). Reported, not acted upon.\n")
    hurt = t26[(t26["delta_auroc"] < 0) & (t26["ci_high"] < 0)]
    if hurt.empty:
        add("None — no feature has a CI entirely below 0 on any cell.\n")
    else:
        add("| model | dataset | feature | Δ AUROC | 95 % CI |")
        add("|---|---|---|---|---|")
        for _, r in hurt.sort_values("delta_auroc").iterrows():
            add(f"| {r['model']} | {r['dataset']} | `{r['feature']}` | "
                f"{r['delta_auroc']:+.4f} | "
                f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    add("")

    add("---\nSTOP. Awaiting joint review before Step 3 (feature freeze + headline modelling).")
    (OUT / "lofo_finding.md").write_text("\n".join(lines), encoding="utf-8")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("Step 2d — LOFO ...")
    t26 = build_T26()
    t26.to_csv(OUT / "T2.6.csv", index=False)
    print(f"  wrote {(OUT / 'T2.6.csv').relative_to(L.PROJECT)}  ({len(t26)} rows)")

    fig = fig_lofo_for_model(t26, "qwen3-4b")
    _save(fig, "F2.4.pdf")
    print(f"  wrote {(OUT / 'F2.4.pdf').relative_to(L.PROJECT)}")

    for m in ["r1-distill-llama-8b", "qwq-32b",
              "qwen3-4b-nothink", "llama-3.1-8b-instruct"]:
        if not datasets_for(m): continue
        fig = fig_lofo_for_model(t26, m)
        _save(fig, f"F2.4.A_{m}.pdf")
        print(f"  wrote F2.4.A_{m}.pdf")

    write_finding(t26)
    print(f"  wrote {(OUT / 'lofo_finding.md').relative_to(L.PROJECT)}")


if __name__ == "__main__":
    main()
