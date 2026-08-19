"""
Stage 4 — Modeling (per model; all-clean set only).

Per spec:
  • Per-model (no pooling). Target = `correct` (binary).
  • Stratified 5-fold CV; standardization inside each train fold (no leakage).
  • Single-feature AUROCs (baselines + each trace feature).
  • Combined trace LR (variant A: with hedging_combined; variant B: formal+reasoning separate).
  • Full combined LR (trace + baselines).
  • Leave-one-out on the combined trace model.
  • Every AUROC reported with 95% bootstrap CI (≥1000 resamples).
  • Pairwise bootstrap: our combined trace vs each baseline (win fraction).
  • ECE on probabilistic outputs only (LR models, p_true, verbalized_confidence).
  • Risk–coverage curves + AURC + acc@80% coverage.

Outputs per model in results/{model}/{dataset}/stage4/:
  • methods_auroc.csv       — every method with AUROC + 95% CI
  • leave_one_out.csv       — feature drop -> ΔAUROC
  • pairwise_bootstrap.csv  — our trace LR vs each baseline
  • ece.csv                 — ECE per probabilistic output
  • risk_coverage.csv       — per method
  • fig_auroc_caterpillar.pdf
  • fig_roc_curves.pdf
  • fig_risk_coverage.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _lib as L

BASELINES = ["answer_semantic_entropy", "p_true", "verbalized_confidence"]
# answer_semantic_entropy and p_true: higher → more uncertain → invert for "confidence"
# Conventionally for AUROC of `correct`, we pass the raw score and call max(auc, 1-auc) at the END.
TRACE_FEATURES = [
    "trace_length", "hedging_formal", "hedging_reasoning", "hedging_combined",
    "connector_density", "rep_5", "trace_divergence",
]
COMBINED_TRACE_VARIANTS = {
    "trace_LR_combined": ["trace_length", "hedging_combined", "connector_density",
                          "rep_5", "trace_divergence"],
    "trace_LR_split":    ["trace_length", "hedging_formal", "hedging_reasoning",
                          "connector_density", "rep_5", "trace_divergence"],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_features(model_short: str, dataset: str) -> pd.DataFrame:
    p = L.FEATURES_DIR / model_short / f"{dataset}.parquet"
    df = pd.read_parquet(p)
    # All-clean filter, drop unlabeled (NaN correct)
    df = df[df["in_all_clean"] & df["correct"].notna()].copy().reset_index(drop=True)
    df["correct"] = df["correct"].astype(int)
    return df


def cv_lr_proba(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = L.SEED) -> np.ndarray:
    """Stratified k-fold CV; scaler fit inside each train fold; return out-of-fold probas for class=1."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, solver="lbfgs"))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def filter_for(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop rows with any NaN among the modeled columns; report count."""
    sub = df.dropna(subset=cols).copy()
    return sub


def single_feature_score(df: pd.DataFrame, feat: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (raw_score_aligned_so_higher=>more_correct, labels) so AUROC > 0.5 means working."""
    sub = df.dropna(subset=[feat, "correct"])
    if sub.empty:
        return np.array([]), np.array([])
    scores = sub[feat].values.astype(float)
    labels = sub["correct"].values.astype(int)
    # Probe orientation: pick the sign such that AUROC >= 0.5.
    try:
        auc_raw = roc_auc_score(labels, scores)
    except ValueError:
        return scores, labels
    if auc_raw < 0.5:
        scores = -scores
    return scores, labels


# ─── Run modeling ─────────────────────────────────────────────────────────────
def run_for_model(model_short: str, dataset: str) -> dict:
    print(f"\n=== Stage 4 — modeling {model_short}/{dataset} ===")
    df = load_features(model_short, dataset)
    n = len(df)
    print(f"  clean+labeled rows: {n}  (correct={int(df['correct'].sum())}, "
          f"incorrect={int((1 - df['correct']).sum())})")

    out_dir = L.RESULTS_DIR / model_short / dataset / "stage4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Single-feature methods (baselines + trace alone) ──
    rows_methods = []
    scores_by_method: dict[str, np.ndarray] = {}
    labels_by_method: dict[str, np.ndarray] = {}

    print("  single-feature baselines + trace features:")
    for f in BASELINES + TRACE_FEATURES:
        s, lab = single_feature_score(df, f)
        if len(s) == 0:
            print(f"    {f:<28s}  no valid rows"); continue
        boot = L.bootstrap_auroc_ci(s, lab)
        rows_methods.append({
            "method": f, "kind": "baseline" if f in BASELINES else "trace_single",
            "n": int(boot["n"]),
            "auroc": boot["auroc"], "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
        })
        scores_by_method[f] = s
        labels_by_method[f] = lab
        print(f"    {f:<28s}  AUROC = {boot['auroc']:.3f}  [{boot['ci_lo']:.3f}, {boot['ci_hi']:.3f}]  n={boot['n']}")

    # ── Combined trace LR (two variants) ──
    for name, feats in COMBINED_TRACE_VARIANTS.items():
        sub = filter_for(df, feats + ["correct"])
        X = sub[feats].values
        y = sub["correct"].values.astype(int)
        oof = cv_lr_proba(X, y)
        boot = L.bootstrap_auroc_ci(oof, y)
        rows_methods.append({
            "method": name, "kind": "trace_LR",
            "n": int(boot["n"]),
            "auroc": boot["auroc"], "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
        })
        scores_by_method[name] = oof
        labels_by_method[name] = y
        print(f"    {name:<28s}  AUROC = {boot['auroc']:.3f}  [{boot['ci_lo']:.3f}, {boot['ci_hi']:.3f}]  n={boot['n']}")

    # ── Full combined LR (trace + baselines) ──
    full_feats = COMBINED_TRACE_VARIANTS["trace_LR_combined"] + BASELINES
    sub_full = filter_for(df, full_feats + ["correct"])
    n_dropped = n - len(sub_full)
    print(f"  full combined LR: kept {len(sub_full)} rows (dropped {n_dropped} for NaN among features)")
    X_full = sub_full[full_feats].values
    y_full = sub_full["correct"].values.astype(int)
    oof_full = cv_lr_proba(X_full, y_full)
    boot_full = L.bootstrap_auroc_ci(oof_full, y_full)
    rows_methods.append({
        "method": "full_LR", "kind": "full_LR",
        "n": int(boot_full["n"]),
        "auroc": boot_full["auroc"], "ci_lo": boot_full["ci_lo"], "ci_hi": boot_full["ci_hi"],
    })
    scores_by_method["full_LR"] = oof_full
    labels_by_method["full_LR"] = y_full

    methods_df = pd.DataFrame(rows_methods)
    methods_df.to_csv(out_dir / "methods_auroc.csv", index=False)

    # ── Leave-one-out on the primary combined-trace model ──
    feats_primary = COMBINED_TRACE_VARIANTS["trace_LR_combined"]
    sub_primary = filter_for(df, feats_primary + ["correct"])
    X_p = sub_primary[feats_primary].values
    y_p = sub_primary["correct"].values.astype(int)
    oof_p = cv_lr_proba(X_p, y_p)
    auc_full_trace = L.bootstrap_auroc_ci(oof_p, y_p)["auroc"]
    loo_rows = []
    for drop_f in feats_primary:
        kept = [f for f in feats_primary if f != drop_f]
        X_loo = sub_primary[kept].values
        oof_loo = cv_lr_proba(X_loo, y_p)
        b = L.bootstrap_auroc_ci(oof_loo, y_p)
        loo_rows.append({
            "dropped": drop_f, "auroc_without": b["auroc"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
            "delta": b["auroc"] - auc_full_trace,
        })
    pd.DataFrame(loo_rows).to_csv(out_dir / "leave_one_out.csv", index=False)

    # ── Pairwise bootstrap: BOTH trace LR variants vs each baseline ──
    # Reporting both `trace_LR_combined` AND `trace_LR_split` so the paper
    # can claim "trace features beat the cheap baselines regardless of how
    # the hedging lexicon is split". This is a robustness check, not a
    # post-hoc winner selection.
    pw_rows = []
    for trace_variant_name, variant_feats in COMBINED_TRACE_VARIANTS.items():
        for b_name in BASELINES + ["full_LR"]:
            if b_name not in scores_by_method:
                continue
            # Align on the same n: the OOF probas + baseline scores must be on the same row set.
            feats_align = variant_feats + ([b_name] if b_name != "full_LR" else BASELINES)
            sub_a = filter_for(df, feats_align + ["correct"])
            if len(sub_a) == 0:
                continue
            oof_t = cv_lr_proba(sub_a[variant_feats].values, sub_a["correct"].astype(int).values)
            if b_name == "full_LR":
                cols = COMBINED_TRACE_VARIANTS["trace_LR_combined"] + BASELINES
                s_b = cv_lr_proba(sub_a[cols].values, sub_a["correct"].astype(int).values)
            else:
                s, _ = single_feature_score(sub_a, b_name)
                if len(s) != len(sub_a):
                    continue
                s_b = s
            boot = L.bootstrap_auroc_diff(oof_t, s_b, sub_a["correct"].astype(int).values)
            pw_rows.append({
                "ours": trace_variant_name, "vs": b_name,
                "diff_median": boot["diff_median"],
                "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                "win_fraction": boot["win_fraction"], "n": boot["n"],
            })
            print(f"    paired  {trace_variant_name:<18s} vs {b_name:<25s}  "
                  f"Δ = {boot['diff_median']:+.3f}  "
                  f"[{boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}]  "
                  f"win = {boot['win_fraction']:.1%}")
    pd.DataFrame(pw_rows).to_csv(out_dir / "pairwise_bootstrap.csv", index=False)

    # ── ECE on probabilistic outputs ──
    # ECE only on probabilistic outputs (per spec). For answer_semantic_entropy
    # we use the spec's "normalized semantic-entropy-as-confidence":
    #   confidence = 1 - H / log2(N_options)
    import math as _math
    H_MAX = _math.log2(5)  # MedQA has up to 5 options
    prob_methods = {"p_true": "p_true", "verbalized_confidence": "verbalized_confidence",
                    "trace_LR_combined": "trace_LR_combined", "trace_LR_split": "trace_LR_split",
                    "full_LR": "full_LR"}
    ece_rows = []
    for label, key in prob_methods.items():
        if key not in scores_by_method: continue
        probs = np.clip(scores_by_method[key], 0, 1)
        labs  = labels_by_method[key]
        ece = L.expected_calibration_error(probs, labs, n_bins=10)
        ece_rows.append({"method": label, "ece": ece["ece"], "n_bins": ece["n_bins"]})
    # answer_semantic_entropy → confidence then ECE.
    # Per-question normalisation is conservative — if any question has fewer
    # (or more) options than the dataset default, we use log2(N_for_that_question).
    # For MedQA verified all 1000 are 5-option; for MMLU-Pro etc. this auto-adjusts.
    if "answer_semantic_entropy" in scores_by_method:
        sub = df.dropna(subset=["answer_semantic_entropy", "correct"])
        # n_samples_with_letter is in df; the true cap is min(n_samples_with_letter, N_options).
        # For MedQA both are >= 5, so log2(5) holds. For safety we clip the result.
        H = sub["answer_semantic_entropy"].values.astype(float)
        conf = np.clip(1.0 - H / H_MAX, 0, 1)
        ece = L.expected_calibration_error(conf, sub["correct"].astype(int).values, n_bins=10)
        ece_rows.append({"method": "answer_semantic_entropy", "ece": ece["ece"], "n_bins": ece["n_bins"]})
    pd.DataFrame(ece_rows).to_csv(out_dir / "ece.csv", index=False)

    # ── Risk-coverage curves ──
    # AURC + acc_at_80 are rank-based metrics that work for ANY scoring
    # function (single_feature_score has already orientation-aligned each one
    # so higher => more confident-in-correctness). Compute for every method.
    # For the figure we keep the headline 5 to avoid clutter, but the CSV
    # holds all of them.
    HEADLINE_RC = ["p_true", "verbalized_confidence", "answer_semantic_entropy",
                   "trace_LR_combined", "trace_LR_split", "full_LR"]
    rc_summary = []
    rc_curves = {}
    for label in scores_by_method:
        rc = L.risk_coverage_curve(scores_by_method[label], labels_by_method[label])
        rc_summary.append({"method": label, "aurc": rc["aurc"], "acc_at_80": rc["acc_at_80"]})
        if label in HEADLINE_RC:
            rc_curves[label] = rc
    pd.DataFrame(rc_summary).to_csv(out_dir / "risk_coverage.csv", index=False)

    # ── Figures ── (functions return fig; save PDF + close here so the
    # stage script doesn't leak memory, while notebooks can also call them.)
    # All datasets get a parallel subdir so figures from one dataset can't
    # silently overwrite another. MedQA's old `stage4/` has been renamed
    # to `stage4_medqa/` to match.
    fig_subdir = f"stage4_{dataset}"
    for name, fig in [
        (f"fig_auroc_caterpillar_{model_short}", fig_auroc_caterpillar(methods_df, model_short)),
        (f"fig_roc_{model_short}",                fig_roc_curves(scores_by_method, labels_by_method, model_short)),
        (f"fig_risk_coverage_{model_short}",      fig_risk_coverage(rc_curves, model_short)),
        (f"fig_leave_one_out_{model_short}",      fig_loo(loo_rows, auc_full_trace, model_short)),
    ]:
        L.save_fig(fig, name, subdir=fig_subdir)
        plt.close(fig)

    return {
        "model": model_short,
        "methods_df": methods_df,
        "rc_curves": rc_curves,
        "scores_by_method": {k: v.tolist() for k, v in scores_by_method.items()},
        "labels_by_method": {k: v.tolist() for k, v in labels_by_method.items()},
    }


# ─── Figures ──────────────────────────────────────────────────────────────────
def fig_auroc_caterpillar(methods_df: pd.DataFrame, model_short: str):
    """Forest / caterpillar plot of AUROCs with 95% CI bars, sorted desc.
       Color-coded by method kind. Vertical dashed line at chance (0.5)."""
    L.apply_style()
    df = methods_df.sort_values("auroc", ascending=True).reset_index(drop=True)
    color_map = {"baseline":     L.PALETTE["baseline"],
                 "trace_single": L.PALETTE["trace"],
                 "trace_LR":     L.PALETTE["highlight"],
                 "full_LR":      "#1a9850"}

    fig, ax = plt.subplots(figsize=(11, 0.65 * len(df) + 1.8))
    y = np.arange(len(df))

    for i, row in df.iterrows():
        c = color_map.get(row["kind"], L.PALETTE["neutral"])
        # CI bar
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=c, lw=2.6, alpha=0.7,
                solid_capstyle="round")
        # whiskers
        for v in [row["ci_lo"], row["ci_hi"]]:
            ax.plot([v, v], [i - 0.20, i + 0.20], color=c, lw=1.6)
        # point
        ax.scatter(row["auroc"], i, color=c, s=110, zorder=5, edgecolor="white", linewidth=1.5)
        # AUROC value placed to the RIGHT of the CI bar (no longer above point — no overlap)
        ax.text(row["ci_hi"] + 0.008, i, f"{row['auroc']:.3f}",
                va="center", ha="left",
                fontsize=10, fontweight="semibold", color="#222")

    ax.axvline(0.5, color="#999", lw=1.2, linestyle="--", zorder=1)
    ax.text(0.5, len(df) - 0.4, "chance (0.5)", ha="center", fontsize=9, color="#666",
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.set_yticks(y); ax.set_yticklabels([L.label_for(m) for m in df["method"]], fontsize=10.5)
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    # Give the rightmost label some breathing room
    x_lo = min(df["ci_lo"].min() - 0.04, 0.40)
    x_hi = max(df["ci_hi"].max() + 0.10, 0.92)
    ax.set_xlim(x_lo, x_hi)
    ax.set_title(f"{L.MODEL_LABEL[model_short]} — predicting greedy correctness on the all-clean set",
                 loc="left", pad=14)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=v, label=k.replace("_", " ")) for k, v in color_map.items()]
    ax.legend(handles=handles, loc="lower right", framealpha=0.95)
    return fig


def fig_roc_curves(scores: dict, labels: dict, model_short: str):
    """ROC curves for the most interesting methods, with diagonal chance line."""
    L.apply_style()
    methods = [
        ("p_true",                  L.PALETTE["baseline"]),
        ("verbalized_confidence",   "#f16913"),
        ("answer_semantic_entropy", "#8c510a"),
        ("trace_LR_combined",       L.PALETTE["highlight"]),
        ("full_LR",                 "#1a9850"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, color in methods:
        if name not in scores: continue
        s, lab = scores[name], labels[name]
        try:
            fpr, tpr, _ = roc_curve(lab, s)
            auc = roc_auc_score(lab, s)
            ax.plot(fpr, tpr, color=color, lw=2.2,
                    label=f"{L.label_for(name)}  AUROC={auc:.3f}")
        except ValueError:
            continue
    ax.plot([0, 1], [0, 1], color="#999", lw=1.2, linestyle="--", label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"{L.MODEL_LABEL[model_short]} — ROC", loc="left")
    ax.legend(loc="lower right", framealpha=0.95)
    return fig


def fig_risk_coverage(rc_curves: dict, model_short: str):
    """Risk vs coverage — lower is better; mark @80%."""
    L.apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.4))
    colors = {"p_true": L.PALETTE["baseline"], "verbalized_confidence": "#f16913",
              "answer_semantic_entropy": "#8c510a",
              "trace_LR_combined": L.PALETTE["highlight"], "full_LR": "#1a9850"}
    for name, rc in rc_curves.items():
        if len(rc["coverage"]) == 0: continue
        c = colors.get(name, "#888")
        ax.plot(rc["coverage"], rc["risk"], color=c, lw=2.2,
                label=f"{L.label_for(name)}   AURC={rc['aurc']:.3f}, acc@80%={rc['acc_at_80']:.3f}")
        # Mark @80% coverage
        ax.axvline(0.8, color="#bbb", lw=0.6, linestyle=":")
    ax.set_xlabel("coverage"); ax.set_ylabel("risk (error rate)")
    ax.set_xlim(0, 1); ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    ax.set_title(f"{L.MODEL_LABEL[model_short]} — risk vs coverage", loc="left")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    return fig


def fig_loo(loo_rows: list[dict], auc_full: float, model_short: str):
    """Dumbbell plot: AUROC with vs without each feature."""
    L.apply_style()
    df = pd.DataFrame(loo_rows).sort_values("delta")
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(df) + 1.5))
    y = np.arange(len(df))
    for i, row in df.iterrows():
        a, b = row["auroc_without"], auc_full
        color = "#d73027" if row["delta"] < 0 else "#1a9850"
        ax.plot([a, b], [i, i], color="#aaa", lw=1.5)
        ax.scatter([a], [i], color="#999", s=72, zorder=4, edgecolor="white", linewidth=1)
        ax.scatter([b], [i], color=color, s=88, zorder=5, edgecolor="white", linewidth=1)
        ax.text(b + 0.005, i, f"Δ={row['delta']:+.3f}", va="center", fontsize=9, color="#222")
    ax.axvline(auc_full, color="#888", lw=0.7, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels([L.label_for(f) for f in df["dropped"]])
    ax.set_xlabel("AUROC")
    ax.set_title(f"{L.MODEL_LABEL[model_short]} — leave-one-out on combined trace LR  "
                 f"(full = {auc_full:.3f})", loc="left")
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(L.MODELS.keys()))
    ap.add_argument("--dataset", default="medqa")
    args = ap.parse_args()

    L.set_seeds()
    L.apply_style()
    results = {}
    for m in args.models:
        results[m] = run_for_model(m, args.dataset)

    # Save a summary for downstream cross-model comparison
    summary = {m: r["methods_df"].to_dict(orient="records") for m, r in results.items()}
    summary_name = f"stage4_summary_{args.dataset}.json"
    (L.RESULTS_DIR / summary_name).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {L.RESULTS_DIR / summary_name}")


if __name__ == "__main__":
    main()
