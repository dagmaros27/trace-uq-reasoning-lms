"""
Step 2b — Per-feature descriptive characterisation.

DESCRIPTIVE only. No feature selection here.

Feature set (6, after Step 2a's rep_n drop):
  trace_length, rep_5, hedging_formal, hedging_reasoning,
  connector_density, trace_divergence

Outputs into results_for_paper/02_features/:
  T2.2.csv                      single-feature AUROC + 95% bootstrap CI per cell
  T2.3.csv                      signed Cohen's d per cell (positive = higher on
                                CORRECT)
  F2.2.pdf                      qwen3-4b Cohen's d, 3 dataset panels (main)
  F2.3.pdf                      qwen3-4b / medqa boxplots, 6 features (main)
  F2.2.A_<model>.pdf            4 appendix PDFs (1 per non-qwen3 model)
  F2.3.A_<model>_<dataset>.pdf  13 appendix PDFs (1 per cell)
  perfeature_finding.md         narrative, numbers from CSVs only
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
SKIP      = {("qwq-32b", "medqa")}   # mmlu_pro added after the n=1000 resume

# Feature set frozen for THIS pass per Step 2a (rep_n + hedging_combined dropped)
FEATURES  = ["trace_length", "rep_5",
             "hedging_formal", "hedging_reasoning",
             "connector_density", "trace_divergence"]

N_BOOT = 1000
SEED   = L.SEED


# ─── helpers ────────────────────────────────────────────────────────────────
def datasets_for(model: str) -> list[str]:
    return [d for d in DATASETS if (model, d) not in SKIP
            and (L.FEATURES_DIR / model / f"{d}.parquet").exists()]


def clean_pool(model: str, dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(L.FEATURES_DIR / model / f"{dataset}.parquet")
    return df[df["in_all_clean"] & df["correct"].notna()].copy()


def all_cells():
    for m in MODELS:
        for d in datasets_for(m):
            yield m, d


# ─── single-feature AUROC + bootstrap CI ────────────────────────────────────
def _auc_with_orient(score: np.ndarray, label: np.ndarray):
    """Return AUROC oriented so it is >= 0.5. Returns auc, sign (+1 if raw
    higher => more correct, -1 if inverted)."""
    try:
        auc_raw = float(roc_auc_score(label, score))
    except ValueError:
        return float("nan"), 0
    if auc_raw >= 0.5:
        return auc_raw, +1
    return 1.0 - auc_raw, -1


def auroc_ci(score: np.ndarray, label: np.ndarray):
    """Bootstrap CI for the orientation-adjusted AUROC."""
    mask = ~np.isnan(score)
    score, label = score[mask], label[mask]
    if len(score) == 0 or len(set(label)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": int(len(score))}
    auc, sign = _auc_with_orient(score, label)
    rng = np.random.RandomState(SEED)
    aucs = []
    n = len(score)
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, size=n)
        s = score[idx]; l = label[idx]
        if len(set(l)) < 2:
            continue
        a, _ = _auc_with_orient(s, l)
        aucs.append(a)
    aucs = np.asarray(aucs, dtype=float)
    return {"auroc": auc,
            "ci_low":  float(np.quantile(aucs, 0.025)),
            "ci_high": float(np.quantile(aucs, 0.975)),
            "n": int(n)}


# ─── Cohen's d (signed; positive = higher on CORRECT) ───────────────────────
def cohens_d(score: np.ndarray, label: np.ndarray):
    mask = ~np.isnan(score)
    score, label = score[mask], label[mask]
    c = score[label == 1]; i = score[label == 0]
    n_c, n_i = int(len(c)), int(len(i))
    if n_c < 2 or n_i < 2:
        return {"cohens_d": float("nan"),
                "mean_correct": float(c.mean()) if n_c else float("nan"),
                "mean_incorrect": float(i.mean()) if n_i else float("nan"),
                "n_correct": n_c, "n_incorrect": n_i}
    s_c = float(c.std(ddof=1)); s_i = float(i.std(ddof=1))
    s_pool_num = (n_c - 1) * s_c ** 2 + (n_i - 1) * s_i ** 2
    s_pool = float(np.sqrt(s_pool_num / (n_c + n_i - 2)))
    d = (float(c.mean()) - float(i.mean())) / s_pool if s_pool > 0 else float("nan")
    return {"cohens_d": d,
            "mean_correct": float(c.mean()),
            "mean_incorrect": float(i.mean()),
            "n_correct": n_c, "n_incorrect": n_i}


# ─── build T2.2 + T2.3 ─────────────────────────────────────────────────────
def build_tables():
    rows_auc, rows_d = [], []
    for m, d in all_cells():
        df = clean_pool(m, d)
        y = df["correct"].values.astype(int)
        for f in FEATURES:
            sub = df.dropna(subset=[f])
            ys = sub["correct"].values.astype(int)
            xs = sub[f].values.astype(float)
            r = auroc_ci(xs, ys)
            rows_auc.append({
                "dataset": d, "model": m, "feature": f,
                "n": r["n"],
                "auroc": round(r["auroc"], 4) if r["auroc"] == r["auroc"] else None,
                "ci_low": round(r["ci_low"], 4) if r["ci_low"] == r["ci_low"] else None,
                "ci_high": round(r["ci_high"], 4) if r["ci_high"] == r["ci_high"] else None,
                "auroc_strength": (round(abs(r["auroc"] - 0.5), 4)
                                    if r["auroc"] == r["auroc"] else None),
            })
            cd = cohens_d(xs, ys)
            rows_d.append({
                "dataset": d, "model": m, "feature": f,
                "cohens_d": (round(cd["cohens_d"], 4)
                              if cd["cohens_d"] == cd["cohens_d"] else None),
                "mean_correct": round(cd["mean_correct"], 6),
                "mean_incorrect": round(cd["mean_incorrect"], 6),
                "n_correct": cd["n_correct"],
                "n_incorrect": cd["n_incorrect"],
            })
    t22 = pd.DataFrame(rows_auc)
    t23 = pd.DataFrame(rows_d)
    return t22, t23


# ─── figure helpers ─────────────────────────────────────────────────────────
def _save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_cohens_d_for_model(t23: pd.DataFrame, model: str, title_pre: str):
    """One row of dataset panels, signed-bar Cohen's d."""
    sub = t23[t23["model"] == model]
    datasets_present = [d for d in DATASETS
                        if not sub[sub["dataset"] == d].empty]
    L.apply_style()
    fig, axes = plt.subplots(1, len(datasets_present),
                              figsize=(4.6 * len(datasets_present), 4.2),
                              sharey=True)
    if len(datasets_present) == 1:
        axes = [axes]
    for ax, dset in zip(axes, datasets_present):
        d = sub[sub["dataset"] == dset].set_index("feature")
        ds = d.reindex(FEATURES)
        vals = ds["cohens_d"].astype(float).values
        colors = ["#1a9850" if v > 0 else "#d73027" if v < 0
                  else "#aaaaaa" for v in vals]
        y = np.arange(len(FEATURES))
        ax.barh(y, vals, color=colors, edgecolor="white")
        ax.axvline(0, color="#333", lw=0.7)
        ax.set_yticks(y); ax.set_yticklabels(FEATURES, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Cohen's d  (+ higher on CORRECT)")
        ax.set_title(f"{dset}", loc="left", fontsize=10)
        for i, v in enumerate(vals):
            if np.isnan(v): continue
            ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.suptitle(f"{title_pre} — signed Cohen's d (positive = higher on CORRECT)",
                 fontsize=10, y=1.02, x=0.5)
    fig.tight_layout()
    return fig


def fig_boxplots_for_cell(model: str, dataset: str):
    """6-panel small-multiple boxplot: each feature, correct vs incorrect."""
    df = clean_pool(model, dataset)
    L.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))
    axes = axes.reshape(-1)
    for ax, f in zip(axes, FEATURES):
        sub = df.dropna(subset=[f]).copy()
        if sub.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center"); ax.axis("off"); continue
        correct  = sub.loc[sub["correct"] == 1, f].values
        incorrect = sub.loc[sub["correct"] == 0, f].values
        bp = ax.boxplot([correct, incorrect],
                        tick_labels=[f"correct (n={len(correct)})",
                                     f"incorrect (n={len(incorrect)})"],
                        patch_artist=True, widths=0.55, showfliers=False)
        for patch, c in zip(bp["boxes"], ["#1a9850", "#d73027"]):
            patch.set_facecolor(c); patch.set_alpha(0.5)
            patch.set_edgecolor("#222")
        for med in bp["medians"]:
            med.set_color("#222"); med.set_linewidth(1.4)
        ax.set_title(f, loc="left", fontsize=10)
    fig.suptitle(f"{L.MODEL_LABEL.get(model, model)}  /  {dataset}  — "
                 f"feature distributions split by greedy correctness",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    return fig


# ─── perfeature_finding.md ──────────────────────────────────────────────────
def write_finding(t22: pd.DataFrame, t23: pd.DataFrame):
    lines = []; add = lines.append
    add("# Step 2b — Per-Feature Descriptive Findings\n")
    add("All numbers below come from `T2.2.csv` (AUROC) and `T2.3.csv` "
        "(Cohen's d). No feature selection here — Step 3 freezes the set.\n")
    add("Feature set after Step 2a's rep_n drop: "
        + ", ".join(f"`{f}`" for f in FEATURES) + ".\n")

    rsn_mcq_mask = (t22["model"].isin(REASONING)) & (t22["dataset"].isin(
        ["medqa", "mmlu_pro"]))

    # 1. strongest single predictors (reasoning + MCQ)
    add("## 1. Strongest single predictors (reasoning models, MCQ cells)\n")
    rsn = t22[rsn_mcq_mask].copy()
    by_feat = (rsn.groupby("feature")["auroc"]
                  .agg(["min", "median", "max"])
                  .reindex(FEATURES))
    add("Range of single-feature AUROC across reasoning-MCQ cells "
        "(qwen3-4b, r1-distill, qwq-32b excl. mmlu_pro [partial]; "
        "datasets medqa + mmlu_pro):\n")
    add("| feature | min | median | max |")
    add("|---|---|---|---|")
    for f in FEATURES:
        r = by_feat.loc[f]
        add(f"| `{f}` | {r['min']:.3f} | {r['median']:.3f} | {r['max']:.3f} |")
    top = by_feat.sort_values("median", ascending=False)
    add("")
    add(f"- Strongest by median AUROC on reasoning-MCQ cells: "
        f"`{top.index[0]}` (median {top.iloc[0]['median']:.3f}) and "
        f"`{top.index[1]}` (median {top.iloc[1]['median']:.3f}).")
    add(f"- Weakest: `{top.index[-1]}` (median {top.iloc[-1]['median']:.3f}).")
    add("")

    # 2. hedging_formal vs hedging_reasoning per model
    add("## 2. `hedging_formal` vs `hedging_reasoning` — single-predictor AUROC per model\n")
    add("Evidence for whether the formal/reasoning split adds independent "
        "signal beyond `hedging_combined`. (Combined is excluded from this "
        "pass — see Step 2a — but the split is on the table.)\n")
    add("| model | dataset | formal | reasoning | gap (|f − r|) |")
    add("|---|---|---|---|---|")
    for m in MODELS:
        for d in DATASETS:
            if (m, d) in SKIP: continue
            row_f = t22[(t22["model"] == m) & (t22["dataset"] == d) &
                        (t22["feature"] == "hedging_formal")]
            row_r = t22[(t22["model"] == m) & (t22["dataset"] == d) &
                        (t22["feature"] == "hedging_reasoning")]
            if row_f.empty or row_r.empty: continue
            a_f = float(row_f.iloc[0]["auroc"])
            a_r = float(row_r.iloc[0]["auroc"])
            add(f"| {m} | {d} | {a_f:.3f} | {a_r:.3f} | {abs(a_f - a_r):.3f} |")
    add("")

    # 3. trace_divergence weakness
    add("## 3. `trace_divergence` — single-feature AUROC across all cells\n")
    add("| model | dataset | AUROC | strength (|AUROC−0.5|) |")
    add("|---|---|---|---|")
    td = t22[t22["feature"] == "trace_divergence"]
    for _, r in td.iterrows():
        add(f"| {r['model']} | {r['dataset']} | {r['auroc']:.3f} | "
            f"{r['auroc_strength']:.3f} |")
    add("")
    med = float(td["auroc"].median()); mx  = float(td["auroc"].max())
    add(f"- Median `trace_divergence` AUROC across all 13 cells: **{med:.3f}**. "
        f"Maximum: {mx:.3f}. Weak single predictor everywhere; evidence for "
        "excluding it from the headline `trace_LR` in Step 3 (decision deferred).")
    add("")

    # 4. Cohen's d direction patterns
    add("## 4. Direction of separation — signed Cohen's d (positive = higher on CORRECT)\n")
    add("Median signed d across cells, per feature. Negative = the feature is "
        "higher on incorrect answers (i.e. a marker of likely-wrong reasoning).\n")
    by_feat_d = (t23.groupby("feature")["cohens_d"]
                    .agg(["min", "median", "max"])
                    .reindex(FEATURES))
    add("| feature | min d | median d | max d |")
    add("|---|---|---|---|")
    for f in FEATURES:
        r = by_feat_d.loc[f]
        add(f"| `{f}` | {r['min']:+.3f} | {r['median']:+.3f} | "
            f"{r['max']:+.3f} |")
    add("")
    neg = [f for f in FEATURES
           if (by_feat_d.loc[f, "median"] < 0)]
    pos = [f for f in FEATURES
           if (by_feat_d.loc[f, "median"] > 0)]
    add(f"- Median d < 0 (higher on INCORRECT — uncertainty markers): "
        + ", ".join(f"`{f}`" for f in neg) + ".")
    add(f"- Median d > 0 (higher on CORRECT — confidence markers): "
        + ", ".join(f"`{f}`" for f in pos) + ".")
    add("- Expected pattern (length / repetition / hedging higher on wrong, "
        "connectors higher on right) holds where the data shows it; see the "
        "table above for the actual signed magnitudes.")
    add("")

    # 5. Surprises / sign-flip risks
    add("## 5. Flags & surprises\n")
    flags = []
    # Per-model sign flips: a feature with d in one cell, opposite in another
    for m in MODELS:
        for f in FEATURES:
            ds = t23[(t23["model"] == m) & (t23["feature"] == f)]
            if ds.empty: continue
            signs = set(np.sign(ds["cohens_d"].dropna().values))
            signs.discard(0.0)
            if len(signs) > 1:
                per = ", ".join(f"{r['dataset']}: {r['cohens_d']:+.3f}"
                                for _, r in ds.iterrows())
                flags.append(f"- **{m} / `{f}`** sign flips across datasets — {per}")
    if flags:
        add("Cases where the same feature points in opposite directions "
            "(correct vs incorrect) on different datasets within one model:\n")
        for x in flags: add(x)
    else:
        add("No within-model sign flips on Cohen's d.")
    add("")

    # llama specifically — the Step 2a flag
    llama_flip = t23[(t23["model"] == "llama-3.1-8b-instruct")
                     & (t23["feature"].isin(["trace_length", "rep_5"]))]
    add("### llama-3.1-8b-instruct watch (Step 2a flag)\n")
    add("Step 2a noted llama's `trace_length × rep_5` correlation flipped "
        "sign across datasets. Cohen's d table for those two features on "
        "llama:\n")
    add("| dataset | feature | d | mean_correct | mean_incorrect |")
    add("|---|---|---|---|---|")
    for _, r in llama_flip.iterrows():
        add(f"| {r['dataset']} | `{r['feature']}` | "
            f"{r['cohens_d']:+.3f} | {r['mean_correct']:.3f} | "
            f"{r['mean_incorrect']:.3f} |")
    add("")
    add("---\nSTOP. Awaiting joint review before Step 3 (feature freeze + LOFO).")
    (OUT / "perfeature_finding.md").write_text("\n".join(lines), encoding="utf-8")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("Step 2b — building T2.2 + T2.3 ...")
    t22, t23 = build_tables()
    t22.to_csv(OUT / "T2.2.csv", index=False)
    t23.to_csv(OUT / "T2.3.csv", index=False)
    print(f"  wrote {(OUT / 'T2.2.csv').relative_to(L.PROJECT)}  ({len(t22)} rows)")
    print(f"  wrote {(OUT / 'T2.3.csv').relative_to(L.PROJECT)}  ({len(t23)} rows)")

    # Main-text figures
    print("Main-text figures ...")
    fig = fig_cohens_d_for_model(t23, "qwen3-4b", "qwen3-4b")
    _save(fig, "F2.2.pdf")
    print(f"  wrote {(OUT / 'F2.2.pdf').relative_to(L.PROJECT)}")

    fig = fig_boxplots_for_cell("qwen3-4b", "medqa")
    _save(fig, "F2.3.pdf")
    print(f"  wrote {(OUT / 'F2.3.pdf').relative_to(L.PROJECT)}")

    # Appendix F2.2.A — Cohen's d per non-qwen3 model
    print("Appendix F2.2.A (Cohen's d per model) ...")
    for m in ["r1-distill-llama-8b", "qwq-32b",
              "qwen3-4b-nothink", "llama-3.1-8b-instruct"]:
        if not datasets_for(m): continue
        fig = fig_cohens_d_for_model(t23, m, m)
        _save(fig, f"F2.2.A_{m}.pdf")
        print(f"  wrote F2.2.A_{m}.pdf")

    # Appendix F2.3.A — boxplots per cell
    print("Appendix F2.3.A (boxplots per cell) ...")
    for m, d in all_cells():
        fig = fig_boxplots_for_cell(m, d)
        _save(fig, f"F2.3.A_{m}_{d}.pdf")
        print(f"  wrote F2.3.A_{m}_{d}.pdf")

    write_finding(t22, t23)
    print(f"  wrote {(OUT / 'perfeature_finding.md').relative_to(L.PROJECT)}")


if __name__ == "__main__":
    main()
