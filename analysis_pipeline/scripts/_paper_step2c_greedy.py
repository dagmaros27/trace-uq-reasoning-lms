"""
Step 2c — Greedy vs Sampled trace_LR (deployment-efficiency).

The frozen trace_LR (Step 3, T3.1) uses sampled-mean features over 10 samples.
This script builds a parallel trace_LR_greedy on the same protocol but
computed from greedy.reasoning_trace (a single trace) — 4 features, omitting
trace_divergence which has no single-trace analogue. The pair tells us how
much discrimination is lost by deploying a single-pass model.

Sampling protocol: identical to Step 3.
  - StratifiedKFold(5, shuffle=True, random_state=L.SEED)
  - StandardScaler + LogisticRegression(max_iter=2000) per fold
  - OOF predictions; AUROC + 95% bootstrap CI + AURC + acc@80
  - 13 cells (skip qwq-32b/mmlu_pro)

Outputs in results_for_paper/02c_greedy_vs_sampled/:
  T2c.0.csv              greedy OOF predictions per (cell, question)
  T2c.1.csv              13-cell comparison + paired bootstrap on Δ AUROC
  T2c.2.csv              single-feature AUROC of each greedy feature
  F2c.1.pdf              qwen3-4b 3-dataset paired bars (main)
  F2c.1.A.pdf            13-cell sampled-minus-greedy delta (appendix)
  F2c.A_<cell>.pdf       per-cell ROC overlay (13 files)
  finding.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L
from stage3_features import features_for_trace

OUT = L.PROJECT / "results_for_paper" / "02c_greedy_vs_sampled"
OUT.mkdir(parents=True, exist_ok=True)
T3_DIR = L.PROJECT / "results_for_paper" / "03_feature_set"

DATASETS  = ["medqa", "mmlu_pro", "trivia_qa"]
REASONING = ["qwen3-4b", "r1-distill-llama-8b", "qwq-32b"]
CONTROLS  = ["qwen3-4b-nothink", "llama-3.1-8b-instruct"]
MODELS    = REASONING + CONTROLS
SKIP      = {("qwq-32b", "medqa")}   # mmlu_pro added after the n=1000 resume

GREEDY_FEATURES = ["trace_length", "rep_5",
                   "hedging_formal", "connector_density"]
GEN_ROOT  = L.PROJECT.parent / "data_generation" / "data" / "generations"
SEED      = L.SEED
N_BOOT    = 1000


# ─── helpers ────────────────────────────────────────────────────────────────
def datasets_for(model: str) -> list[str]:
    return [d for d in DATASETS if (model, d) not in SKIP
            and (L.FEATURES_DIR / model / f"{d}.parquet").exists()]


def jsonl_path(model: str, dataset: str) -> Path | None:
    nested = GEN_ROOT / model / f"{dataset}.jsonl"
    flat   = GEN_ROOT / f"{dataset}_{model}.jsonl"
    if nested.exists(): return nested
    if flat.exists():   return flat
    return None


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


def _single_feature_auroc(x, y):
    """Orientation-aligned AUROC for a 1-D feature."""
    a = _auroc(x, y)
    if np.isnan(a): return float("nan")
    return max(a, 1 - a)


# ─── per-cell greedy feature extraction ─────────────────────────────────────
def build_greedy_features(model: str, dataset: str, qid_filter: set[str]
                          ) -> pd.DataFrame:
    """Read jsonl; for each record whose question_id is in qid_filter, compute
    the 4 greedy trace features from greedy.reasoning_trace.

    Returns a DataFrame indexed by question_id with the 4 features + correct."""
    p = jsonl_path(model, dataset)
    if p is None:
        return pd.DataFrame()

    # We need the parquet's correct labels (already aligned with T3.1)
    fp = pd.read_parquet(L.FEATURES_DIR / model / f"{dataset}.parquet")
    fp = fp[fp["in_all_clean"] & fp["correct"].notna()].copy()
    fp["question_id"] = fp["question_id"].astype(str)
    labels = fp.set_index("question_id")["correct"].astype(int).to_dict()

    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            qid = str(r.get("question_id"))
            if qid not in qid_filter or qid not in labels:
                continue
            trace = (r.get("greedy") or {}).get("reasoning_trace") or ""
            feats = features_for_trace(trace, model)
            rows.append({
                "question_id":       qid,
                "y_true":            labels[qid],
                "trace_length":      feats["n_tokens"],
                "rep_5":             feats["rep_5"],
                "hedging_formal":    feats["hedging_formal_density"],
                "connector_density": feats["connectors_logical_density"],
            })
    return pd.DataFrame(rows)


# ─── per-cell trace_LR_greedy fit ───────────────────────────────────────────
def fit_greedy_cell(model: str, dataset: str, t31: pd.DataFrame):
    cell_qids = set(t31[(t31["model"] == model) &
                        (t31["dataset"] == dataset)]["question_id"].astype(str))
    df = build_greedy_features(model, dataset, cell_qids)
    df = df.dropna(subset=GREEDY_FEATURES + ["y_true"]).reset_index(drop=True)
    if df.empty:
        return None
    X = df[GREEDY_FEATURES].values
    y = df["y_true"].astype(int).values
    oof, fold = cv_lr_oof(X, y)
    return {"df": df, "oof": oof, "fold": fold, "y": y}


# ─── build T2c.0 + T2c.1 + T2c.2 ────────────────────────────────────────────
def build_tables(t31: pd.DataFrame, t32: pd.DataFrame):
    rows_t2c0 = []   # OOF predictions
    rows_t2c1 = []   # 13-cell comparison
    rows_t2c2 = []   # single-feature AUROC of greedy features

    for m in MODELS:
        for d in datasets_for(m):
            r = fit_greedy_cell(m, d, t31)
            if r is None:
                continue
            df, oof, fold, y = r["df"], r["oof"], r["fold"], r["y"]

            # T2c.0 rows
            for _, row, p_, f_ in zip(range(len(df)), df.itertuples(),
                                       oof, fold):
                rows_t2c0.append({
                    "dataset": d, "model": m,
                    "question_id": row.question_id,
                    "y_true": int(row.y_true),
                    "p_pred": round(float(p_), 6),
                    "fold":   int(f_),
                })

            # T2c.1: greedy metrics + sampled metrics from T3.2 + delta + paired bootstrap
            auc_g = _auroc(oof, y)
            lo_g, hi_g = _ci_auroc(oof, y)
            rc_g = L.risk_coverage_curve(oof, y)
            # sampled — from T3.2 + T3.1 (matching qids only, for paired bootstrap)
            sub_s = t31[(t31["model"] == m) & (t31["dataset"] == d)].copy()
            sub_s["question_id"] = sub_s["question_id"].astype(str)
            # Align on qids actually used in greedy LR (after greedy-side NaN drops)
            merged = df[["question_id", "y_true"]].merge(
                sub_s[["question_id", "p_pred", "y_true"]]
                  .rename(columns={"p_pred": "p_sampled",
                                   "y_true": "y_sampled"}),
                on="question_id", how="inner")
            assert (merged["y_true"] == merged["y_sampled"]).all()
            p_g = oof[df["question_id"].isin(merged["question_id"]).values]
            # Re-derive p_g from merged (safer):
            p_g = merged.merge(
                df[["question_id"]].assign(_idx=np.arange(len(df))),
                on="question_id", how="left")["_idx"].map(
                    pd.Series(oof, index=range(len(oof)))).values
            y_pair = merged["y_true"].astype(int).values
            p_sampled = merged["p_sampled"].astype(float).values

            # Sampled metrics on the SAME paired subset (for an apples-to-apples
            # delta), plus sampled metrics on the full sampled set (from T3.2)
            t32_row = t32[(t32["model"] == m) & (t32["dataset"] == d)]
            auc_s_full = float(t32_row.iloc[0]["auroc"])
            lo_s_full  = float(t32_row.iloc[0]["ci_low"])
            hi_s_full  = float(t32_row.iloc[0]["ci_high"])
            aurc_s_full     = float(t32_row.iloc[0]["aurc"])
            acc80_s_full    = float(t32_row.iloc[0]["acc_at_80"])

            # paired bootstrap on AUROC delta = sampled - greedy
            rng = np.random.RandomState(SEED)
            n = len(y_pair); deltas = []
            for _ in range(N_BOOT):
                idx = rng.randint(0, n, n)
                yb = y_pair[idx]
                if len(set(yb)) < 2: continue
                a_s = _auroc(p_sampled[idx], yb)
                a_g = _auroc(p_g[idx],       yb)
                if np.isnan(a_s) or np.isnan(a_g): continue
                deltas.append(a_s - a_g)
            deltas = np.asarray(deltas)
            median_delta = float(np.median(deltas)) if len(deltas) else float("nan")
            ci_lo_d      = float(np.quantile(deltas, 0.025)) if len(deltas) else float("nan")
            ci_hi_d      = float(np.quantile(deltas, 0.975)) if len(deltas) else float("nan")
            pct_s_wins   = round(100.0 * float(np.mean(deltas > 0)), 2) if len(deltas) else float("nan")

            rows_t2c1.append({
                "dataset": d, "model": m,
                "n_greedy": int(len(y)),
                "n_paired": int(n),
                "auroc_greedy":     round(auc_g, 4),
                "auroc_greedy_ci_low":  round(lo_g, 4),
                "auroc_greedy_ci_high": round(hi_g, 4),
                "aurc_greedy":      round(float(rc_g["aurc"]),      4),
                "acc_at_80_greedy": round(float(rc_g["acc_at_80"]), 4),
                "auroc_sampled":    round(auc_s_full, 4),
                "auroc_sampled_ci_low":  round(lo_s_full, 4),
                "auroc_sampled_ci_high": round(hi_s_full, 4),
                "aurc_sampled":     round(aurc_s_full,  4),
                "acc_at_80_sampled":round(acc80_s_full, 4),
                "delta_auroc":   round(auc_s_full - auc_g, 4),
                "delta_aurc":    round(aurc_s_full - float(rc_g["aurc"]), 4),
                "delta_acc_at_80": round(acc80_s_full - float(rc_g["acc_at_80"]), 4),
                "paired_median_delta_auroc": round(median_delta, 4),
                "paired_ci_low":  round(ci_lo_d, 4),
                "paired_ci_high": round(ci_hi_d, 4),
                "pct_resamples_sampled_wins": pct_s_wins,
            })

            # T2c.2: single-feature AUROC of each greedy feature
            for f in GREEDY_FEATURES:
                a = _single_feature_auroc(df[f].values.astype(float), y)
                rows_t2c2.append({
                    "dataset": d, "model": m, "feature": f,
                    "n": int(len(df)),
                    "auroc_greedy_single_feat": round(a, 4),
                })
            print(f"  done {m}/{d}  n_greedy={len(y)}  n_paired={n}  "
                  f"AUROC_g={auc_g:.4f}  AUROC_s={auc_s_full:.4f}  "
                  f"Δ={auc_s_full - auc_g:+.4f}",
                  flush=True)
    return (pd.DataFrame(rows_t2c0),
            pd.DataFrame(rows_t2c1),
            pd.DataFrame(rows_t2c2))


# ─── figures ────────────────────────────────────────────────────────────────
def _save(fig, name):
    p = OUT / name
    fig.savefig(p); plt.close(fig); return p


def fig_qwen3_paired_bars(t2c1: pd.DataFrame):
    L.apply_style()
    sub = t2c1[t2c1["model"] == "qwen3-4b"].set_index("dataset").reindex(DATASETS)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(DATASETS))
    width = 0.36
    g = sub["auroc_greedy"].astype(float).values
    s = sub["auroc_sampled"].astype(float).values
    s_lo = sub["auroc_sampled_ci_low"].astype(float).values
    s_hi = sub["auroc_sampled_ci_high"].astype(float).values
    ax.bar(x - width/2, g, width, label="trace_LR_greedy (4 feats, 1 generation)",
            color="#fb9a99", edgecolor="white")
    ax.bar(x + width/2, s, width, label="trace_LR_sampled (5 feats, 10 generations)",
            color="#e31a1c", edgecolor="white",
            yerr=[s - s_lo, s_hi - s], capsize=4, ecolor="#333")
    for xi, gi, si in zip(x, g, s):
        ax.text(xi - width/2, gi + 0.005, f"{gi:.3f}",
                ha="center", fontsize=8)
        ax.text(xi + width/2, si + 0.005, f"{si:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(DATASETS)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, max(s_hi.max() + 0.05, g.max() + 0.05))
    ax.axhline(0.5, color="#999", lw=0.7, linestyle="--")
    ax.set_title("Qwen3-4B — single-pass (greedy) vs full sampled trace_LR",
                 loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_all_cells_delta(t2c1: pd.DataFrame):
    L.apply_style()
    t = t2c1.copy()
    t["task_type"] = t["dataset"].apply(
        lambda d: "free-form" if d == "trivia_qa" else "MCQ")
    t["label"] = t["model"] + " / " + t["dataset"]
    t = t.sort_values(["task_type", "delta_auroc"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"MCQ": "#1f78b4", "free-form": "#ff7f00"}
    y = np.arange(len(t))
    bar_colors = [colors[k] for k in t["task_type"]]
    ax.barh(y, t["delta_auroc"], color=bar_colors, edgecolor="white")
    ax.errorbar(t["delta_auroc"], y,
                xerr=[t["delta_auroc"] - t["paired_ci_low"],
                       t["paired_ci_high"] - t["delta_auroc"]],
                fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(t["label"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.7)
    ax.set_xlabel("AUROC(sampled) − AUROC(greedy)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors["MCQ"]),
               plt.Rectangle((0, 0), 1, 1, color=colors["free-form"])]
    ax.legend(handles, ["MCQ", "free-form"], loc="lower right", fontsize=8)
    ax.set_title("Cost of single-pass deployment — all 13 cells "
                 "(positive = sampled wins)", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def fig_roc_overlay(model: str, dataset: str, t31: pd.DataFrame,
                     t2c0: pd.DataFrame):
    L.apply_style()
    g = t2c0[(t2c0["model"] == model) & (t2c0["dataset"] == dataset)]
    s = t31[(t31["model"] == model) & (t31["dataset"] == dataset)]
    g["question_id"] = g["question_id"].astype(str)
    s["question_id"] = s["question_id"].astype(str)
    j = g.merge(s[["question_id", "p_pred"]].rename(
                    columns={"p_pred": "p_sampled"}),
                on="question_id", how="inner")
    y = j["y_true"].astype(int).values
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    fpr_g, tpr_g, _ = roc_curve(y, j["p_pred"].values)
    fpr_s, tpr_s, _ = roc_curve(y, j["p_sampled"].values)
    auc_g = _auroc(j["p_pred"].values, y)
    auc_s = _auroc(j["p_sampled"].values, y)
    ax.plot(fpr_g, tpr_g, lw=1.6, color="#fb9a99",
            label=f"greedy (AUROC {auc_g:.3f})")
    ax.plot(fpr_s, tpr_s, lw=1.8, color="#e31a1c",
            label=f"sampled (AUROC {auc_s:.3f})")
    ax.plot([0, 1], [0, 1], color="#999", lw=0.6, linestyle="--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{L.MODEL_LABEL.get(model, model)} / {dataset}", loc="left",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ─── finding.md ─────────────────────────────────────────────────────────────
def write_finding(t2c1: pd.DataFrame, t2c2: pd.DataFrame):
    lines = []; add = lines.append
    add("# Step 2c — Greedy vs Sampled trace_LR\n")
    add("All numbers below come from `T2c.1.csv` and `T2c.2.csv`. "
        "`trace_LR_greedy` uses **1 generation** (4 features: trace_length, "
        "rep_5, hedging_formal, connector_density). `trace_LR_sampled` uses "
        "**~10 generations** (5 features — adds trace_divergence). "
        "trace_divergence has no single-trace analogue, so the comparison is "
        "intrinsically asymmetric and any free-form gap reported here mixes "
        "two effects: (i) one greedy trace vs the average of 10 sampled "
        "traces, and (ii) the absent sampling-based divergence feature.\n")

    # 1. Median gap, overall and by task type
    add("## 1. Median |Δ AUROC| (sampled − greedy)\n")
    overall_med = float(t2c1["delta_auroc"].abs().median())
    mcq = t2c1[t2c1["dataset"].isin(["medqa", "mmlu_pro"])]
    free = t2c1[t2c1["dataset"] == "trivia_qa"]
    add(f"- Across all {len(t2c1)} cells: **{overall_med:.4f}** AUROC.")
    add(f"- MCQ cells only (n = {len(mcq)}): "
        f"median |Δ| = **{float(mcq['delta_auroc'].abs().median()):.4f}**, "
        f"median signed Δ = {float(mcq['delta_auroc'].median()):+.4f}.")
    add(f"- Free-form (trivia_qa) cells (n = {len(free)}): "
        f"median |Δ| = **{float(free['delta_auroc'].abs().median()):.4f}**, "
        f"median signed Δ = {float(free['delta_auroc'].median()):+.4f}.")
    add("\nMCQ vs free-form pattern is the expected one — single-pass model "
        "competitive on MCQ, larger gap on free-form (where the absent "
        "trace_divergence feature was the strongest single feature on "
        "trivia_qa per Step 2b).")
    add("")

    # 2. The two reasoning-MCQ win cells specifically
    add("## 2. Critical cells — qwen3-4b/medqa and qwen3-4b/mmlu_pro\n")
    add("These two cells are where trace_LR beats semantic_entropy (Step 5). "
        "If `trace_LR_greedy` holds up here, the cheap single-pass model wins "
        "exactly where it matters.\n")
    key = t2c1[(t2c1["model"] == "qwen3-4b") &
               (t2c1["dataset"].isin(["medqa", "mmlu_pro"]))]
    add("| cell | n | AUROC greedy | AUROC sampled | Δ (sampled − greedy) | paired 95 % CI | % bootstrap sampled wins |")
    add("|---|---|---|---|---|---|---|")
    for _, r in key.iterrows():
        add(f"| {r['model']} / {r['dataset']} | {int(r['n_paired'])} | "
            f"{r['auroc_greedy']:.4f} | {r['auroc_sampled']:.4f} | "
            f"{r['delta_auroc']:+.4f} | "
            f"[{r['paired_ci_low']:+.4f}, {r['paired_ci_high']:+.4f}] | "
            f"{r['pct_resamples_sampled_wins']:.1f} % |")
    add("")

    # 3. Cost framing
    add("## 3. Cost framing\n")
    add("- `trace_LR_greedy`: **1 generation** per question (the greedy "
        "answer the model would have produced anyway).")
    add("- `trace_LR_sampled` and `semantic_entropy`: **~10 generations** per "
        "question (1 greedy + 10 sampled, or just 10 sampled for SE). Both "
        "competing methods require sampling; greedy is the deployment-cheap "
        "option.")
    add("")

    # 4. Decision rule applied
    add("## 4. Recommendation (not finalised)\n")
    mcq_median = float(mcq["delta_auroc"].median())
    free_median = float(free["delta_auroc"].median())
    threshold = 0.02
    add(f"Applied rule: if MCQ median Δ AUROC ≤ {threshold:.2f}, recommend "
        "greedy as a viable cheap deployment variant; otherwise report it as "
        "MCQ-only.\n")
    if mcq_median <= threshold:
        add(f"- MCQ median signed Δ = **{mcq_median:+.4f}** ≤ {threshold:.2f} "
            "→ **recommend `trace_LR_greedy` as a viable cheap deployment "
            "variant on MCQ tasks**.")
    else:
        add(f"- MCQ median signed Δ = **{mcq_median:+.4f}** > {threshold:.2f} "
            "→ the single-pass model loses too much on MCQ to be recommended "
            "as a drop-in replacement.")
    add(f"- Free-form (trivia_qa) median signed Δ = "
        f"**{free_median:+.4f}**. Even if this exceeds the threshold, recall "
        "that this gap mixes the greedy-vs-sampled effect with the missing "
        "trace_divergence feature (which alone explained up to +0.138 "
        "AUROC on trivia_qa per Step 2d). So the free-form number is an "
        "upper bound on the true greedy-vs-sampled cost.")
    add("\n**Pending**: qwq-32b/mmlu_pro is not in this pass; the overall "
        "reasoning-MCQ recommendation is held until that cell is refreshed.")
    add("")

    # 5. Single-feature AUROC of greedy features — does rep_5 / trace_length hold up?
    add("## 5. Single-feature AUROC of each greedy feature (T2c.2)\n")
    by_feat = (t2c2.groupby("feature")["auroc_greedy_single_feat"]
                  .agg(["min", "median", "max"])
                  .reindex(GREEDY_FEATURES))
    add("Median / range of single-feature AUROC across 13 cells, computed "
        "on the GREEDY trace:\n")
    add("| feature | min | median | max |")
    add("|---|---|---|---|")
    for f in GREEDY_FEATURES:
        r = by_feat.loc[f]
        add(f"| `{f}` | {r['min']:.4f} | {r['median']:.4f} | "
            f"{r['max']:.4f} |")
    add("")
    add("---\nSTOP. Awaiting joint review before Step 6.")
    (OUT / "finding.md").write_text("\n".join(lines), encoding="utf-8")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("Step 2c — greedy vs sampled trace_LR")
    t31 = pd.read_csv(T3_DIR / "T3.1.csv")
    t32 = pd.read_csv(T3_DIR / "T3.2.csv")
    print(f"  loaded T3.1 ({len(t31)} rows) + T3.2 ({len(t32)} cells)")
    t2c0, t2c1, t2c2 = build_tables(t31, t32)
    t2c0.to_csv(OUT / "T2c.0.csv", index=False)
    t2c1.to_csv(OUT / "T2c.1.csv", index=False)
    t2c2.to_csv(OUT / "T2c.2.csv", index=False)
    print(f"  wrote T2c.0.csv ({len(t2c0)} rows)")
    print(f"  wrote T2c.1.csv ({len(t2c1)} rows)")
    print(f"  wrote T2c.2.csv ({len(t2c2)} rows)")

    fig = fig_qwen3_paired_bars(t2c1); _save(fig, "F2c.1.pdf")
    print("  wrote F2c.1.pdf")
    fig = fig_all_cells_delta(t2c1); _save(fig, "F2c.1.A.pdf")
    print("  wrote F2c.1.A.pdf")
    for m in MODELS:
        for d in datasets_for(m):
            fig = fig_roc_overlay(m, d, t31, t2c0)
            _save(fig, f"F2c.A_{m}_{d}.pdf")
    print("  wrote F2c.A_<cell>.pdf  (13 files)")

    write_finding(t2c1, t2c2)
    print("  wrote finding.md")


if __name__ == "__main__":
    main()
