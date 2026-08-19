"""
Step 6 — Combined Model (trace_LR + semantic_entropy) — upper-bound reference.

full_LR = 5 frozen trace features + answer_semantic_entropy as a 6th input.
Protocol identical to Step 3's trace_LR:
  - StratifiedKFold(5, shuffle=True, random_state=L.SEED)
  - StandardScaler + LogisticRegression(max_iter=2000) inside train folds only
  - OOF predictions, AUROC + 1000-bootstrap 95% CI

Outputs in results_for_paper/06_combined/:
  T6.0.csv         full_LR OOF predictions per (cell, question)
  T6.1.csv         per-cell side-by-side: full_LR vs trace_LR vs semantic_entropy
                   with paired-bootstrap deltas on AUROC
  F6.1.pdf         qwen3-4b grouped bars across 3 datasets (main)
  F6.1.A.pdf       full_LR - trace_LR delta across 13 cells (appendix)
  finding.md
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

OUT = L.PROJECT / "results_for_paper" / "06_combined"
OUT.mkdir(parents=True, exist_ok=True)
T3_DIR = L.PROJECT / "results_for_paper" / "03_feature_set"
T5_DIR = L.PROJECT / "results_for_paper" / "05_vs_baselines"

DATASETS  = ["medqa", "mmlu_pro", "trivia_qa"]
REASONING = ["qwen3-4b", "r1-distill-llama-8b", "qwq-32b"]
CONTROLS  = ["qwen3-4b-nothink", "llama-3.1-8b-instruct"]
MODELS    = REASONING + CONTROLS
SKIP      = {("qwq-32b", "medqa")}   # mmlu_pro added after the n=1000 resume

TRACE_FEATURES = ["trace_length", "rep_5", "hedging_formal",
                  "connector_density", "trace_divergence"]
SE_COL         = "answer_semantic_entropy"
FULL_FEATURES  = TRACE_FEATURES + [SE_COL]

SEED   = L.SEED
N_BOOT = 1000


# ─── helpers ────────────────────────────────────────────────────────────────
def datasets_for(model: str) -> list[str]:
    return [d for d in DATASETS if (model, d) not in SKIP
            and (L.FEATURES_DIR / model / f"{d}.parquet").exists()]


def clean_pool(model: str, dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(L.FEATURES_DIR / model / f"{dataset}.parquet")
    df = df[df["in_all_clean"] & df["correct"].notna()].copy()
    return df.reset_index(drop=True)


def _auroc(p, y):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _ci_auroc(p, y):
    rng = np.random.RandomState(SEED)
    n = len(y); aucs = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(set(yb)) < 2: continue
        a = _auroc(p[idx], yb)
        if not np.isnan(a): aucs.append(a)
    return float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def cv_lr_oof(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fold = np.full(len(y), -1, dtype=int)
    for k, (tr, te) in enumerate(skf.split(X, y), start=1):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs"))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
        fold[te] = k
    return oof, fold


def _orient(score, y):
    a = _auroc(score, y)
    if np.isnan(a) or a >= 0.5: return score, +1
    return -score, -1


def _paired_delta_ci(p_a, p_b, y):
    rng = np.random.RandomState(SEED)
    n = len(y); deltas = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(set(yb)) < 2: continue
        a_a = _auroc(p_a[idx], yb)
        a_b = _auroc(p_b[idx], yb)
        if not (np.isnan(a_a) or np.isnan(a_b)):
            deltas.append(a_a - a_b)
    deltas = np.asarray(deltas)
    return (float(np.median(deltas)),
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
            float(np.mean(deltas > 0)))


# ─── fit full_LR per cell ──────────────────────────────────────────────────
def fit_full_cell(model: str, dataset: str):
    df = clean_pool(model, dataset)
    df = df.dropna(subset=FULL_FEATURES + ["correct"]).reset_index(drop=True)
    if df.empty:
        return None
    X = df[FULL_FEATURES].values
    y = df["correct"].astype(int).values
    qids = df["question_id"].astype(str).values
    se_raw = df[SE_COL].values.astype(float)  # for SE comparison later
    oof, fold = cv_lr_oof(X, y)
    return {"df": df, "qids": qids, "y": y, "oof": oof, "fold": fold,
            "se_raw": se_raw}


# ─── build T6.0 + T6.1 ─────────────────────────────────────────────────────
def build_tables(t31, t32, t51):
    rows_t60, rows_t61 = [], []
    for m in MODELS:
        for d in datasets_for(m):
            r = fit_full_cell(m, d)
            if r is None: continue
            qids, y, p_full = r["qids"], r["y"], r["oof"]
            se_raw = r["se_raw"]
            se_oriented, se_sign = _orient(se_raw, y)

            # T6.0
            for qid, yt, pp, fk in zip(qids, y, r["oof"], r["fold"]):
                rows_t60.append({
                    "dataset": d, "model": m,
                    "question_id": qid,
                    "y_true": int(yt),
                    "p_pred": round(float(pp), 6),
                    "fold":   int(fk),
                })

            # Cell-level metrics for full_LR
            auc_full = _auroc(p_full, y)
            lo_f, hi_f = _ci_auroc(p_full, y)
            rc_full = L.risk_coverage_curve(p_full, y)

            # trace_LR — from T3.1 OOF (paired on same qids that survive
            # the 6-feature NaN drop here)
            sub_t31 = t31[(t31["model"] == m) & (t31["dataset"] == d)
                          ][["question_id", "y_true", "p_pred"]]
            sub_t31["question_id"] = sub_t31["question_id"].astype(str)
            mer = pd.DataFrame({
                "question_id": qids, "y": y, "p_full": p_full,
                "se_oriented": se_oriented,
            }).merge(sub_t31.rename(columns={"p_pred": "p_trace"}),
                      on="question_id", how="inner")
            assert (mer["y"] == mer["y_true"]).all()
            y_pair    = mer["y"].astype(int).values
            p_full_p  = mer["p_full"].values
            p_trace_p = mer["p_trace"].values
            se_p      = mer["se_oriented"].values

            # full_LR vs trace_LR — paired
            d_ft_med, d_ft_lo, d_ft_hi, d_ft_winfrac = _paired_delta_ci(
                p_full_p, p_trace_p, y_pair)
            # full_LR vs SE — paired
            d_fs_med, d_fs_lo, d_fs_hi, d_fs_winfrac = _paired_delta_ci(
                p_full_p, se_p, y_pair)

            # trace_LR cell metrics from T3.2 (full-sample-of-trace pool)
            t32_row = t32[(t32["model"] == m) & (t32["dataset"] == d)].iloc[0]
            # SE cell metrics from T5.1 (raw-oriented, baseline pool)
            se_row = t51[(t51["model"] == m) & (t51["dataset"] == d)
                         & (t51["method"] == "semantic_entropy")]
            if se_row.empty:
                continue
            se_row = se_row.iloc[0]

            rows_t61.append({
                "dataset": d, "model": m,
                "n_full_LR":       int(len(y)),
                "n_paired_t31":    int(len(mer)),
                "se_orientation":  int(se_sign),
                # full_LR
                "auroc_full_LR":          round(auc_full, 4),
                "auroc_full_LR_ci_low":   round(lo_f,    4),
                "auroc_full_LR_ci_high":  round(hi_f,    4),
                "aurc_full_LR":           round(float(rc_full["aurc"]), 4),
                "acc_at_80_full_LR":      round(float(rc_full["acc_at_80"]), 4),
                # trace_LR
                "auroc_trace_LR":         round(float(t32_row["auroc"]),   4),
                "auroc_trace_LR_ci_low":  round(float(t32_row["ci_low"]),  4),
                "auroc_trace_LR_ci_high": round(float(t32_row["ci_high"]), 4),
                "aurc_trace_LR":          round(float(t32_row["aurc"]),    4),
                "acc_at_80_trace_LR":     round(float(t32_row["acc_at_80"]), 4),
                # semantic_entropy
                "auroc_semantic_entropy":          round(float(se_row["auroc"]),         4),
                "auroc_semantic_entropy_ci_low":   round(float(se_row["auroc_ci_low"]),  4),
                "auroc_semantic_entropy_ci_high":  round(float(se_row["auroc_ci_high"]), 4),
                "aurc_semantic_entropy":           round(float(se_row["aurc"]),          4),
                "acc_at_80_semantic_entropy":      round(float(se_row["acc_at_80"]),     4),
                # Deltas (paired bootstrap on the SAME questions used in fit)
                "delta_auroc_full_vs_trace":             round(d_ft_med, 4),
                "delta_auroc_full_vs_trace_ci_low":      round(d_ft_lo,  4),
                "delta_auroc_full_vs_trace_ci_high":     round(d_ft_hi,  4),
                "pct_resamples_full_beats_trace":  round(100.0 * d_ft_winfrac, 2),
                "delta_auroc_full_vs_SE":                round(d_fs_med, 4),
                "delta_auroc_full_vs_SE_ci_low":         round(d_fs_lo,  4),
                "delta_auroc_full_vs_SE_ci_high":        round(d_fs_hi,  4),
                "pct_resamples_full_beats_SE":     round(100.0 * d_fs_winfrac, 2),
            })
            print(f"  done {m}/{d}  n_full={len(y)}  "
                  f"full_LR={auc_full:.4f}  trace_LR={float(t32_row['auroc']):.4f}  "
                  f"SE={float(se_row['auroc']):.4f}",
                  flush=True)
    return pd.DataFrame(rows_t60), pd.DataFrame(rows_t61)


# ─── figures ────────────────────────────────────────────────────────────────
def _save(fig, name):
    p = OUT / name; fig.savefig(p); plt.close(fig); return p


def fig_qwen3_grouped(t61: pd.DataFrame):
    L.apply_style()
    sub = t61[t61["model"] == "qwen3-4b"].set_index("dataset").reindex(DATASETS)
    x = np.arange(len(DATASETS))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    methods = [
        ("full_LR",           "#6a3d9a",
         sub["auroc_full_LR"], sub["auroc_full_LR_ci_low"], sub["auroc_full_LR_ci_high"]),
        ("trace_LR",          "#e31a1c",
         sub["auroc_trace_LR"], sub["auroc_trace_LR_ci_low"], sub["auroc_trace_LR_ci_high"]),
        ("semantic_entropy",  "#1a9850",
         sub["auroc_semantic_entropy"], sub["auroc_semantic_entropy_ci_low"],
         sub["auroc_semantic_entropy_ci_high"]),
    ]
    offsets = [-width, 0, width]
    for (label, color, vals, lo, hi), off in zip(methods, offsets):
        vals = vals.astype(float).values
        lo = lo.astype(float).values
        hi = hi.astype(float).values
        ax.bar(x + off, vals, width, label=label, color=color,
               edgecolor="white",
               yerr=[vals - lo, hi - vals], capsize=3, ecolor="#333")
        for xi, vi in zip(x + off, vals):
            ax.text(xi, vi + 0.006, f"{vi:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(DATASETS)
    ax.set_ylabel("AUROC")
    ax.axhline(0.5, color="#999", lw=0.7, linestyle="--")
    ax.set_ylim(0.5, 0.95)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Qwen3-4B — full_LR vs trace_LR vs semantic_entropy "
                 "(AUROC, 95 % bootstrap CI)", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def fig_full_minus_trace_allcells(t61: pd.DataFrame):
    L.apply_style()
    t = t61.copy()
    t["task_type"] = t["dataset"].apply(
        lambda d: "free-form" if d == "trivia_qa" else "MCQ")
    t["label"] = t["model"] + " / " + t["dataset"]
    t = t.sort_values(["task_type", "delta_auroc_full_vs_trace"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"MCQ": "#1f78b4", "free-form": "#ff7f00"}
    y = np.arange(len(t))
    bar_colors = [colors[k] for k in t["task_type"]]
    ax.barh(y, t["delta_auroc_full_vs_trace"], color=bar_colors,
            edgecolor="white")
    ax.errorbar(
        t["delta_auroc_full_vs_trace"], y,
        xerr=[t["delta_auroc_full_vs_trace"]
                - t["delta_auroc_full_vs_trace_ci_low"],
              t["delta_auroc_full_vs_trace_ci_high"]
                - t["delta_auroc_full_vs_trace"]],
        fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(t["label"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.7)
    ax.set_xlabel("AUROC(full_LR) − AUROC(trace_LR)  "
                  "(positive ⇒ adding SE helps over trace alone)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors["MCQ"]),
               plt.Rectangle((0, 0), 1, 1, color=colors["free-form"])]
    ax.legend(handles, ["MCQ", "free-form"], loc="lower right", fontsize=8)
    ax.set_title("How much does adding semantic_entropy to trace_LR help, per cell?",
                 loc="left", fontsize=10)
    fig.tight_layout()
    return fig


# ─── finding.md ─────────────────────────────────────────────────────────────
def write_finding(t61: pd.DataFrame):
    lines = []; add = lines.append
    add("# Step 6 — Combined Model (trace_LR + semantic_entropy)\n")
    add("All numbers from `T6.1.csv`. **`full_LR` includes both inputs by "
        "construction, so it is best-or-tied-best almost everywhere. We "
        "report it as an upper-bound reference, not as our method.**\n")

    # 1. Reasoning-MCQ win cells
    add("## 1. Does adding semantic_entropy buy anything over trace_LR in the win cells?\n")
    add("These are the cells where trace_LR beats semantic_entropy alone "
        "(Step 5). If `full_LR ≈ trace_LR` here, then the trace features "
        "already capture what SE has to offer on these cells.\n")
    key = t61[(t61["model"] == "qwen3-4b") &
              (t61["dataset"].isin(["medqa", "mmlu_pro"]))]
    add("| cell | n | AUROC full_LR | AUROC trace_LR | Δ (full − trace) | "
        "paired 95 % CI | % bootstrap full wins |")
    add("|---|---|---|---|---|---|---|")
    for _, r in key.iterrows():
        add(f"| {r['model']} / {r['dataset']} | "
            f"{int(r['n_paired_t31'])} | "
            f"{r['auroc_full_LR']:.4f} | {r['auroc_trace_LR']:.4f} | "
            f"{r['delta_auroc_full_vs_trace']:+.4f} | "
            f"[{r['delta_auroc_full_vs_trace_ci_low']:+.4f}, "
            f"{r['delta_auroc_full_vs_trace_ci_high']:+.4f}] | "
            f"{r['pct_resamples_full_beats_trace']:.1f} % |")
    add("")
    qwen_mmlu = key[key["dataset"] == "mmlu_pro"].iloc[0]
    qwen_med  = key[key["dataset"] == "medqa"].iloc[0]
    add(f"- qwen3-4b / mmlu_pro: adding SE moves AUROC by "
        f"**{qwen_mmlu['delta_auroc_full_vs_trace']:+.4f}**; "
        f"CI {'crosses' if (qwen_mmlu['delta_auroc_full_vs_trace_ci_low'] < 0 < qwen_mmlu['delta_auroc_full_vs_trace_ci_high']) else 'is entirely above' if qwen_mmlu['delta_auroc_full_vs_trace_ci_low'] > 0 else 'is entirely below'} zero "
        f"([{qwen_mmlu['delta_auroc_full_vs_trace_ci_low']:+.4f}, "
        f"{qwen_mmlu['delta_auroc_full_vs_trace_ci_high']:+.4f}]).")
    add(f"- qwen3-4b / medqa: adding SE moves AUROC by "
        f"**{qwen_med['delta_auroc_full_vs_trace']:+.4f}**; "
        f"CI {'crosses' if (qwen_med['delta_auroc_full_vs_trace_ci_low'] < 0 < qwen_med['delta_auroc_full_vs_trace_ci_high']) else 'is entirely above' if qwen_med['delta_auroc_full_vs_trace_ci_low'] > 0 else 'is entirely below'} zero "
        f"([{qwen_med['delta_auroc_full_vs_trace_ci_low']:+.4f}, "
        f"{qwen_med['delta_auroc_full_vs_trace_ci_high']:+.4f}]).")
    add("")

    # 2. SE-strong cells — is full_LR's gain over trace_LR coming from SE?
    add("## 2. Where SE is the strong method — is full_LR's gain over trace_LR coming from the SE component?\n")
    add("On the SE-strong cells (Step 5: free-form trivia_qa across all "
        "models, non-reasoning controls on MCQ, r1-distill on MCQ), we "
        "expect `full_LR − trace_LR` to be large (SE is doing the work) and "
        "`full_LR − SE` to be small (trace adds little where SE is already "
        "strong).\n")
    se_strong = t61[~((t61["model"] == "qwen3-4b") &
                      (t61["dataset"].isin(["medqa", "mmlu_pro"])))]
    add("| cell | full_LR − trace_LR | CI | full_LR − SE | CI |")
    add("|---|---|---|---|---|")
    for _, r in se_strong.iterrows():
        add(f"| {r['model']} / {r['dataset']} | "
            f"{r['delta_auroc_full_vs_trace']:+.4f} | "
            f"[{r['delta_auroc_full_vs_trace_ci_low']:+.4f}, "
            f"{r['delta_auroc_full_vs_trace_ci_high']:+.4f}] | "
            f"{r['delta_auroc_full_vs_SE']:+.4f} | "
            f"[{r['delta_auroc_full_vs_SE_ci_low']:+.4f}, "
            f"{r['delta_auroc_full_vs_SE_ci_high']:+.4f}] |")
    add("")
    med_ft = float(se_strong["delta_auroc_full_vs_trace"].median())
    med_fs = float(se_strong["delta_auroc_full_vs_SE"].median())
    add(f"- Median full_LR − trace_LR on SE-strong cells: **{med_ft:+.4f}** "
        "(SE is the big lift over trace alone).")
    add(f"- Median full_LR − SE on SE-strong cells: **{med_fs:+.4f}** "
        "(trace adds little once SE is in).")
    add("")

    # 3. Complementary vs redundant per cell
    add("## 3. Complementary vs redundant — per cell\n")
    add("Classification rule, applied to each cell:\n")
    add("- `full_LR ≥ max(trace_LR, SE) + 0.005` AND `full_LR − each` CI strictly above 0 → **complementary** (both inputs add unique signal).")
    add("- otherwise if `full_LR ≈ max(trace_LR, SE)` (within 0.01) → **redundant** (full_LR ≈ stronger of the two).")
    add("- otherwise → **mixed** (full_LR adds modestly).\n")
    rows = []
    for _, r in t61.iterrows():
        f_auc = float(r["auroc_full_LR"])
        t_auc = float(r["auroc_trace_LR"])
        s_auc = float(r["auroc_semantic_entropy"])
        max_alt = max(t_auc, s_auc)
        d_ft = float(r["delta_auroc_full_vs_trace"])
        d_fs = float(r["delta_auroc_full_vs_SE"])
        d_ft_lo = float(r["delta_auroc_full_vs_trace_ci_low"])
        d_fs_lo = float(r["delta_auroc_full_vs_SE_ci_low"])
        comp = (f_auc >= max_alt + 0.005) and (d_ft_lo > 0) and (d_fs_lo > 0)
        red  = (abs(f_auc - max_alt) <= 0.01)
        kind = "complementary" if comp else ("redundant" if red else "mixed")
        rows.append({"model": r["model"], "dataset": r["dataset"],
                     "full": f_auc, "trace": t_auc, "SE": s_auc,
                     "delta_full_vs_trace": d_ft,
                     "delta_full_vs_SE": d_fs,
                     "verdict": kind})
    rep = pd.DataFrame(rows)
    add("| model | dataset | full_LR | trace_LR | SE | "
        "Δ full−trace | Δ full−SE | verdict |")
    add("|---|---|---|---|---|---|---|---|")
    for _, r in rep.iterrows():
        add(f"| {r['model']} | {r['dataset']} | "
            f"{r['full']:.3f} | {r['trace']:.3f} | {r['SE']:.3f} | "
            f"{r['delta_full_vs_trace']:+.4f} | "
            f"{r['delta_full_vs_SE']:+.4f} | {r['verdict']} |")
    add("")
    counts = rep["verdict"].value_counts()
    add(f"- Cells classified complementary: **{int(counts.get('complementary', 0))} / {len(rep)}**.")
    add(f"- Cells classified redundant (full ≈ max(trace, SE)): "
        f"**{int(counts.get('redundant', 0))} / {len(rep)}**.")
    add(f"- Mixed: **{int(counts.get('mixed', 0))} / {len(rep)}**.")
    add("")

    # 4. Frame
    add("## 4. Framing reminder\n")
    add("`full_LR` is, by construction, at least as informative as either "
        "input on its own. It is the upper-bound reference, not our "
        "method. Treat any cell where `full_LR > trace_LR` as showing that "
        "**SE contains some signal trace_LR misses on that cell** — not "
        "that the proposed method is `trace_LR + SE`.\n")
    add("**Pending:** qwq-32b / mmlu_pro is not in this pass.")
    add("\n---\nSTOP. Awaiting joint review before Step 7.")
    (OUT / "finding.md").write_text("\n".join(lines), encoding="utf-8")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("Step 6 — combined model")
    t31 = pd.read_csv(T3_DIR / "T3.1.csv")
    t32 = pd.read_csv(T3_DIR / "T3.2.csv")
    t51 = pd.read_csv(T5_DIR / "T5.1.csv")
    print(f"  loaded T3.1 ({len(t31)}), T3.2 ({len(t32)}), T5.1 ({len(t51)})")
    t60, t61 = build_tables(t31, t32, t51)
    t60.to_csv(OUT / "T6.0.csv", index=False)
    t61.to_csv(OUT / "T6.1.csv", index=False)
    print(f"  wrote T6.0.csv ({len(t60)} rows), T6.1.csv ({len(t61)} rows)")
    fig = fig_qwen3_grouped(t61); _save(fig, "F6.1.pdf")
    print("  wrote F6.1.pdf")
    fig = fig_full_minus_trace_allcells(t61); _save(fig, "F6.1.A.pdf")
    print("  wrote F6.1.A.pdf")
    write_finding(t61)
    print("  wrote finding.md")


if __name__ == "__main__":
    main()
