"""
Stage 5 — Additional analyses (low-cost extras).

Per spec:
  • Feature correlation matrix (+ vs correct) → heatmap PDF.
  • Repetition robustness — confirm signal in all-clean; correlation of repetition with trace_length;
    rep-3/4/5 consistency.
  • Calibration plots (reliability diagrams) for probabilistic methods.
  • Per-class feature summary table (mean ± std, standardized mean difference).
  • Cross-model consistency — side-by-side AUROCs.

Outputs per model in results/{model_short}/{dataset}/stage5/, and a cross-model
results/stage5_crossmodel.* at the top.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _lib as L

ANALYSIS_COLS = [
    "trace_length", "hedging_formal", "hedging_reasoning", "hedging_combined",
    "connector_density", "rep_3", "rep_4", "rep_5", "trace_divergence",
    "answer_semantic_entropy", "p_true", "verbalized_confidence",
]


def load_features(model_short: str, dataset: str) -> pd.DataFrame:
    p = L.FEATURES_DIR / model_short / f"{dataset}.parquet"
    df = pd.read_parquet(p)
    df = df[df["in_all_clean"] & df["correct"].notna()].copy().reset_index(drop=True)
    df["correct"] = df["correct"].astype(int)
    return df


# ─── Correlation matrix (clustered, lower-triangle, diverging) ────────────────
def fig_correlation_matrix(df: pd.DataFrame, model_short: str):
    """Hierarchically clustered lower-triangle heatmap of |Pearson|. Diagonal hides."""
    L.apply_style()
    cols = [c for c in ANALYSIS_COLS if c in df.columns] + ["correct"]
    sub = df[cols].dropna()
    if sub.empty:
        return
    M = sub.corr().values
    # Cluster on absolute correlation distance
    dist = 1.0 - np.abs(M)
    np.fill_diagonal(dist, 0)
    Z = linkage(dist[np.triu_indices_from(dist, k=1)], method="average")
    order = leaves_list(Z)
    M = M[np.ix_(order, order)]
    labels = [L.label_for(cols[i]) for i in order]

    # Mask the upper triangle
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)
    M_lo = np.where(mask, np.nan, M)

    fig, ax = plt.subplots(figsize=(9, 8.2))
    im = ax.imshow(M_lo, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10.5)
    ax.set_yticklabels(labels, fontsize=10.5)
    # Annotate non-trivial cells (slightly larger text + better contrast)
    for i in range(len(labels)):
        for j in range(i + 1):
            v = M[i, j]
            if abs(v) > 0.20:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(v) > 0.55 else "#222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, shrink=0.85)
    cbar.set_label("Pearson r")
    ax.set_title(f"{L.MODEL_LABEL[model_short]} — feature correlations (clustered)",
                 loc="left", pad=12)
    plt.tight_layout()
    return fig


# ─── Calibration / reliability diagrams ──────────────────────────────────────
def cv_lr_proba(X, y, n_splits=5, seed=L.SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, solver="lbfgs"))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def fig_calibration(df: pd.DataFrame, model_short: str):
    """Reliability diagrams for the probabilistic methods, side by side.
       Per spec: includes 'normalized semantic-entropy-as-confidence'
       computed as 1 - H / log2(N_options). N_options=5 for MedQA."""
    import math as _math
    L.apply_style()
    methods = []
    # p_true and verbalized_confidence as confidences directly
    if "p_true" in df.columns:
        v = df.dropna(subset=["p_true"])
        methods.append(("P(True)",   np.clip(v["p_true"].values, 0, 1), v["correct"].values))
    if "verbalized_confidence" in df.columns:
        v = df.dropna(subset=["verbalized_confidence"])
        methods.append(("verbalized", np.clip(v["verbalized_confidence"].values, 0, 1), v["correct"].values))
    # answer_semantic_entropy → confidence (1 − H/log2(5))
    if "answer_semantic_entropy" in df.columns:
        H_MAX = _math.log2(5)
        v = df.dropna(subset=["answer_semantic_entropy"])
        conf = np.clip(1.0 - v["answer_semantic_entropy"].values / H_MAX, 0, 1)
        methods.append(("sem-entropy", conf, v["correct"].values))
    # Combined trace LR (5-fold CV oof probas)
    feats = ["trace_length", "hedging_combined", "connector_density", "rep_5", "trace_divergence"]
    feats = [f for f in feats if f in df.columns]
    sub = df.dropna(subset=feats + ["correct"])
    if not sub.empty:
        oof = cv_lr_proba(sub[feats].values, sub["correct"].astype(int).values)
        methods.append(("trace LR", oof, sub["correct"].astype(int).values))

    if not methods:
        return
    cols = len(methods)
    fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 4.6), sharey=True)
    if cols == 1: axes = [axes]

    palette = {"P(True)":     L.PALETTE["baseline"],
               "verbalized":  "#f16913",
               "sem-entropy": "#8c510a",
               "trace LR":    L.PALETTE["highlight"]}

    for ax, (name, probs, labs) in zip(axes, methods):
        ece_info = L.expected_calibration_error(probs, labs, n_bins=10)
        centers = ece_info["bin_centers"]
        acc = ece_info["bin_acc"]
        conf = ece_info["bin_conf"]
        counts = ece_info["bin_count"]
        ax.plot([0, 1], [0, 1], color="#999", lw=1, linestyle="--")
        # Bar widths proportional to bin frequency (perceptually encodes coverage)
        widths = 0.07 + 0.06 * (counts / max(counts.max(), 1))
        for c, a, w, n in zip(centers, acc, widths, counts):
            if np.isnan(a): continue
            ax.bar(c, a, width=w, color=palette[name], alpha=0.55, edgecolor="white", linewidth=0.5)
            ax.text(c, a + 0.02, f"n={n}", ha="center", fontsize=7, color="#666")
        # Line through bin (conf vs accuracy)
        valid = ~np.isnan(acc)
        ax.plot(conf[valid], acc[valid], color=palette[name], lw=2, marker="o")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted confidence")
        ax.set_title(f"{name}  ECE={ece_info['ece']:.3f}", loc="left")
    axes[0].set_ylabel("empirical accuracy")
    fig.suptitle(f"{L.MODEL_LABEL[model_short]} — calibration (reliability diagrams)",
                 fontsize=12, fontweight="bold", y=1.02)
    return fig


# ─── Per-class feature summary (effect sizes) ────────────────────────────────
def class_summary_table(df: pd.DataFrame, model_short: str) -> pd.DataFrame:
    """For each feature: mean ± std for correct & incorrect, and standardized mean difference (Cohen's d)."""
    rows = []
    for f in ANALYSIS_COLS:
        if f not in df.columns: continue
        sub = df[[f, "correct"]].dropna()
        c = sub[sub["correct"] == 1][f].values
        i = sub[sub["correct"] == 0][f].values
        if len(c) < 2 or len(i) < 2:
            continue
        m_c, m_i = float(np.mean(c)), float(np.mean(i))
        s_c, s_i = float(np.std(c, ddof=1)), float(np.std(i, ddof=1))
        sp = np.sqrt(((len(c) - 1) * s_c ** 2 + (len(i) - 1) * s_i ** 2) /
                     (len(c) + len(i) - 2)) if (len(c) + len(i) - 2) > 0 else np.nan
        d = (m_c - m_i) / sp if sp and sp > 0 else float("nan")
        rows.append({
            "feature": f,
            "mean_correct": m_c, "std_correct": s_c, "n_correct": len(c),
            "mean_incorrect": m_i, "std_incorrect": s_i, "n_incorrect": len(i),
            "cohens_d": d,
        })
    return pd.DataFrame(rows).sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)


def fig_effect_sizes(summary: pd.DataFrame, model_short: str):
    """Cohen's d bar chart, sorted by |d|, color-coded by direction."""
    L.apply_style()
    df = summary.dropna(subset=["cohens_d"]).copy().sort_values("cohens_d")
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(df) + 1.5))
    colors = ["#d73027" if d < 0 else "#1a9850" for d in df["cohens_d"]]
    display_labels = [L.label_for(f) for f in df["feature"]]
    ax.barh(display_labels, df["cohens_d"], color=colors, alpha=0.85, edgecolor="white", height=0.7)
    ax.axvline(0, color="#666", lw=1.0)
    max_abs = max(abs(df["cohens_d"].max()), abs(df["cohens_d"].min()))
    pad = max_abs * 0.04
    for i, d in enumerate(df["cohens_d"]):
        ax.text(d + (pad if d >= 0 else -pad), i, f"{d:+.2f}",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=10, color="#222", fontweight="semibold")
    # Add a bit of room either side so labels don't clip
    ax.set_xlim(df["cohens_d"].min() - max_abs * 0.18,
                df["cohens_d"].max() + max_abs * 0.18)
    ax.set_xlabel("Cohen's d  (positive ⇒ feature higher in correct answers)")
    ax.set_title(f"{L.MODEL_LABEL[model_short]} — feature effect sizes "
                 "(correct vs incorrect, all-clean set)", loc="left", pad=12)
    plt.tight_layout()
    return fig


# ─── Repetition robustness ────────────────────────────────────────────────────
def repetition_robustness(df: pd.DataFrame, model_short: str) -> dict:
    """Correlations between rep-3/4/5 and trace_length; show that rep is not 'length in disguise'."""
    out = {}
    rep_cols = [c for c in ["rep_3", "rep_4", "rep_5"] if c in df.columns]
    for r in rep_cols:
        sub = df[[r, "trace_length", "correct"]].dropna()
        if len(sub) < 5:
            continue
        out[f"corr({r},trace_length)"] = float(sub[r].corr(sub["trace_length"]))
    # rep-N consistency: correlations among themselves
    if len(rep_cols) >= 2:
        sub = df[rep_cols].dropna()
        cm = sub.corr()
        out["rep_consistency"] = cm.round(3).to_dict()
    return out


# ─── NEW: Per-feature correctness boxplots ──────────────────────────────────
def fig_feature_boxplots(df: pd.DataFrame, model_short: str):
    """One panel per feature; boxes for correct vs incorrect side-by-side.
       Mirrors the methodology-PoC layout but with proper spacing."""
    L.apply_style()
    feats = [c for c in [
        "trace_length", "hedging_combined", "connector_density", "rep_5",
        "trace_divergence", "answer_semantic_entropy", "p_true", "verbalized_confidence",
    ] if c in df.columns]
    feats = [(c, L.label_for(c)) for c in feats]
    n = len(feats)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (col, label) in zip(axes, feats):
        sub = df[[col, "correct"]].dropna()
        c = sub[sub["correct"] == 1][col].values
        i = sub[sub["correct"] == 0][col].values
        bp = ax.boxplot([c, i],
                        labels=["correct", "incorrect"],
                        patch_artist=True,
                        widths=0.55,
                        showfliers=True,
                        flierprops=dict(marker=".", markersize=4,
                                        markerfacecolor="#888", markeredgecolor="none"),
                        medianprops=dict(color="black", linewidth=1.6),
                        whiskerprops=dict(color="#222"),
                        capprops=dict(color="#222"))
        for patch, fc in zip(bp["boxes"], [L.PALETTE["correct"], L.PALETTE["incorrect"]]):
            patch.set_facecolor(fc); patch.set_alpha(0.55); patch.set_edgecolor("#222")
        ax.set_title(label, fontsize=10.5, pad=8)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.4)

    # Hide unused axes
    for ax in axes[len(feats):]:
        ax.axis("off")

    fig.suptitle(f"{L.MODEL_LABEL[model_short]} — feature distributions, correct vs incorrect",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─── NEW: Top-10 hedging phrases ─────────────────────────────────────────────
from collections import Counter as _C


def _count_hedging_terms(records: list[dict], model_short: str) -> _C:
    """Sweep every sample's reasoning trace, count matches of each hedging term."""
    terms = L.all_lexicon_terms()["hedging_combined"]
    counter = _C()
    for r in records:
        if not L.is_all_clean(r): continue
        for s in r["samples"]:
            text = (s.get("reasoning_trace") or "").lower()
            if not text: continue
            for t in terms:
                # word-boundary aware
                import re as _re
                pat = _re.escape(t.lower())
                if " " in t:
                    n = len(_re.findall(pat, text))
                else:
                    n = len(_re.findall(rf"\b{pat}\b", text))
                if n:
                    counter[t] += n
    return counter


def fig_top_hedges(records: list[dict], model_short: str, top_n: int = 10):
    """Horizontal bar chart of the top-N hedging phrases by total occurrences."""
    L.apply_style()
    c = _count_hedging_terms(records, model_short)
    if not c:
        return None
    top = c.most_common(top_n)
    labels = [t for t, _ in top][::-1]
    counts = [n for _, n in top][::-1]
    total  = sum(c.values())

    fig, ax = plt.subplots(figsize=(8, 0.55 * top_n + 1.4))
    bars = ax.barh(labels, counts, color=L.PALETTE["highlight"], alpha=0.85,
                   edgecolor="white", height=0.72)
    for b, n in zip(bars, counts):
        ax.text(n + max(counts) * 0.012, b.get_y() + b.get_height() / 2,
                f"{n:,}  ({100*n/total:.1f}%)",
                va="center", fontsize=9, color="#222")
    ax.set_xlabel("occurrences across all samples (all-clean set)")
    ax.set_xlim(0, max(counts) * 1.18)
    ax.set_title(f"{L.MODEL_LABEL[model_short]} — top {top_n} hedging phrases  "
                 f"(total matches: {total:,})", loc="left", pad=10)
    ax.grid(axis="x", alpha=0.3)
    return fig


# ─── Cross-model AUROC slope plot ─────────────────────────────────────────────
def fig_cross_model_slope(model_summaries: dict[str, pd.DataFrame]):
    """Slope plot: each method's AUROC for R1 ↔ AUROC for Qwen3.
       Highlights features that work for BOTH models vs one only."""
    L.apply_style()
    models = list(model_summaries.keys())
    if len(models) != 2:
        return
    a_df, b_df = model_summaries[models[0]], model_summaries[models[1]]
    merged = a_df[["method", "auroc"]].merge(b_df[["method", "auroc"]],
                                              on="method", suffixes=(f"_{models[0]}", f"_{models[1]}"))
    merged = merged.sort_values(f"auroc_{models[0]}", ascending=False)

    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(merged) + 1.5))
    for _, row in merged.iterrows():
        a, b = row[f"auroc_{models[0]}"], row[f"auroc_{models[1]}"]
        c1, c2 = L.MODEL_COLOR[models[0]], L.MODEL_COLOR[models[1]]
        # Line color by direction: monotone improvement = green; opposite = red
        agree = (a > 0.5 and b > 0.5) or (a < 0.5 and b < 0.5)
        line_color = "#1a9850" if agree else "#d73027"
        ax.plot([0, 1], [a, b], color=line_color, lw=1.6, alpha=0.6)
        ax.scatter([0], [a], color=c1, s=80, zorder=4, edgecolor="white", linewidth=1)
        ax.scatter([1], [b], color=c2, s=80, zorder=4, edgecolor="white", linewidth=1)
        # Label on the right
        ax.text(1.05, b, L.label_for(row["method"]), va="center", fontsize=9, color="#222")
    ax.axhline(0.5, color="#bbb", lw=0.8, linestyle="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels([L.MODEL_LABEL[m] for m in models], fontsize=10)
    ax.set_ylabel("AUROC")
    ax.set_xlim(-0.05, 2.1)
    ax.set_title("Cross-model AUROC — which features generalize?", loc="left", pad=10)
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_for_model(model_short: str, dataset: str) -> dict:
    print(f"\n=== Stage 5 — extras for {model_short}/{dataset} ===")
    df = load_features(model_short, dataset)
    out_dir = L.RESULTS_DIR / model_short / dataset / "stage5"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in [
        (f"fig_correlations_{model_short}",  fig_correlation_matrix(df, model_short)),
        (f"fig_calibration_{model_short}",   fig_calibration(df, model_short)),
    ]:
        if fig is not None:
            L.save_fig(fig, name, subdir="stage5")
            plt.close(fig)
    summary = class_summary_table(df, model_short)
    summary.to_csv(out_dir / "feature_class_summary.csv", index=False)
    # NEW figs: feature boxplots + top hedging phrases
    records = L.load_records(model_short, dataset)
    df_clean_records = df  # already filtered to all-clean+labeled in load_features
    for name, fig in [
        (f"fig_effect_sizes_{model_short}",    fig_effect_sizes(summary, model_short)),
        (f"fig_feature_boxplots_{model_short}", fig_feature_boxplots(df_clean_records, model_short)),
        (f"fig_top_hedges_{model_short}",       fig_top_hedges(records, model_short)),
    ]:
        if fig is not None:
            L.save_fig(fig, name, subdir="stage5")
            plt.close(fig)
    rep = repetition_robustness(df, model_short)
    (out_dir / "repetition_robustness.json").write_text(json.dumps(rep, indent=2))
    print(f"  wrote correlations, calibration, effect-sizes, repetition robustness")
    return {"model": model_short, "summary": summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(L.MODELS.keys()))
    ap.add_argument("--dataset", default="medqa")
    args = ap.parse_args()
    L.set_seeds()
    L.apply_style()

    # Per-model
    for m in args.models:
        run_for_model(m, args.dataset)

    # Cross-model AUROC slope (needs stage 4's methods_auroc.csv)
    summaries = {}
    for m in args.models:
        p = L.RESULTS_DIR / m / args.dataset / "stage4" / "methods_auroc.csv"
        if p.exists():
            summaries[m] = pd.read_csv(p)
    if len(summaries) >= 2:
        fig_cm = fig_cross_model_slope(summaries)
        if fig_cm is not None:
            L.save_fig(fig_cm, "fig_cross_model_slope", subdir="stage5")
            plt.close(fig_cm)
        print("  wrote fig_cross_model_slope.pdf")
    else:
        print("  (need stage 4 outputs for both models for cross-model figure)")


if __name__ == "__main__":
    main()
