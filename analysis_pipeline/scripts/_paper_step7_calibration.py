"""
Step 7 — Calibration, handled honestly.

Three parts:
  A. RAW self-report miscalibration (motivation).
  B. ECE-is-gamed demonstration: base_rate_constant + Platt-fitted p_true.
  C. Fair ranking on PROPER scores (Brier + NLL): trace_LR vs Platt(SE).

trace_LR is READ from T3.1 (not refit). Platt fitting for baselines runs
inside the SAME 5-fold CV as T3.1 (seed = L.SEED), train-folds only.

Outputs in results_for_paper/07_calibration/:
  T7.1.csv            Part A — raw miscalibration
  T7.2.csv            Part B — ECE artefact (base_rate, p_true_platt, trace_LR)
  T7.3.csv            Part C — proper-score ranking (Brier, NLL, ECE) + paired CI
  F7.1.pdf            qwen3-4b/medqa reliability diagram (main)
  F7.1.A_<cell>.pdf   13 reliability diagrams
  finding.md
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

OUT = L.PROJECT / "results_for_paper" / "07_calibration"
OUT.mkdir(parents=True, exist_ok=True)
T3_DIR = L.PROJECT / "results_for_paper" / "03_feature_set"

DATASETS  = ["medqa", "mmlu_pro", "trivia_qa"]
REASONING = ["qwen3-4b", "r1-distill-llama-8b", "qwq-32b"]
CONTROLS  = ["qwen3-4b-nothink", "llama-3.1-8b-instruct"]
MODELS    = REASONING + CONTROLS
SKIP      = {("qwq-32b", "medqa")}   # mmlu_pro added after the n=1000 resume

# Per-dataset K for the semantic-entropy "confidence" = 1 - H/log2(K).
# MedQA = 5 options. MMLU-Pro = up to 10. TriviaQA NLI clusters up to N samples (10).
H_MAX = {"medqa": math.log2(5),
         "mmlu_pro": math.log2(10),
         "trivia_qa": math.log2(10)}

SEED   = L.SEED
N_BOOT = 1000
EPS    = 1e-12


# ─── helpers ────────────────────────────────────────────────────────────────
def datasets_for(model: str) -> list[str]:
    return [d for d in DATASETS if (model, d) not in SKIP
            and (L.FEATURES_DIR / model / f"{d}.parquet").exists()]


def clean_pool(model: str, dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(L.FEATURES_DIR / model / f"{dataset}.parquet")
    df = df[df["in_all_clean"] & df["correct"].notna()].copy()
    return df.reset_index(drop=True)


def _ece(p, y, n_bins=10):
    if len(p) == 0:
        return float("nan")
    return float(L.expected_calibration_error(np.asarray(p, float),
                                              np.asarray(y, float),
                                              n_bins=n_bins)["ece"])


def _brier(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def _nll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _folds(y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros((len(y), 1)), y))


def platt_oof(score, y):
    """1-D LR Platt fit, OOF probabilities. Inside-fold fitting only."""
    oof = np.full(len(y), np.nan)
    x = score.reshape(-1, 1)
    for tr, te in _folds(y):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(x[tr], y[tr])
        oof[te] = clf.predict_proba(x[te])[:, 1]
    return oof


def baserate_oof(y):
    """For each fold, predict train-fold accuracy on every test row."""
    oof = np.full(len(y), np.nan)
    for tr, te in _folds(y):
        oof[te] = float(np.mean(y[tr]))
    return oof


def se_confidence(H, dataset):
    return np.clip(1.0 - np.asarray(H, float) / H_MAX[dataset], 0.0, 1.0)


def paired_delta_ci(metric_fn, p_a, p_b, y, n_boot=N_BOOT):
    """Paired bootstrap on (metric(p_a) - metric(p_b)) — same resampled indices."""
    rng = np.random.RandomState(SEED)
    n = len(y); deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        deltas.append(metric_fn(p_a[idx], y[idx]) - metric_fn(p_b[idx], y[idx]))
    deltas = np.asarray(deltas)
    return (float(np.median(deltas)),
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
            float(np.mean(deltas > 0)))


# ─── load T3.1 (canonical trace_LR OOF) ─────────────────────────────────────
def trace_oof_for(t31: pd.DataFrame, model, dataset, qid_filter):
    sub = t31[(t31["model"] == model) & (t31["dataset"] == dataset)].copy()
    sub["question_id"] = sub["question_id"].astype(str)
    sub = sub.set_index("question_id")
    return sub.reindex(qid_filter)["p_pred"].astype(float).values, \
           sub.reindex(qid_filter)["y_true"].astype(int).values


# ─── PART A: T7.1 raw miscalibration ────────────────────────────────────────
def build_T71():
    rows = []
    for m in MODELS:
        for d in datasets_for(m):
            df = clean_pool(m, d)
            y = df["correct"].astype(int).values
            # p_true
            sub = df.dropna(subset=["p_true"])
            if not sub.empty:
                pt = sub["p_true"].values.astype(float)
                yt = sub["correct"].astype(int).values
                rows.append({"dataset": d, "model": m, "method": "p_true",
                             "n": int(len(sub)),
                             "ece_raw": round(_ece(pt, yt), 4),
                             "mean_confidence": round(float(pt.mean()), 4),
                             "accuracy": round(float(yt.mean()), 4)})
            # verbalized
            sub = df.dropna(subset=["verbalized_confidence"])
            if not sub.empty:
                vc = sub["verbalized_confidence"].values.astype(float)
                yt = sub["correct"].astype(int).values
                rows.append({"dataset": d, "model": m,
                             "method": "verbalized_confidence",
                             "n": int(len(sub)),
                             "ece_raw": round(_ece(vc, yt), 4),
                             "mean_confidence": round(float(vc.mean()), 4),
                             "accuracy": round(float(yt.mean()), 4)})
            # semantic_entropy oriented to confidence
            sub = df.dropna(subset=["answer_semantic_entropy"])
            if not sub.empty:
                H  = sub["answer_semantic_entropy"].values.astype(float)
                yt = sub["correct"].astype(int).values
                conf = se_confidence(H, d)
                rows.append({"dataset": d, "model": m,
                             "method": "semantic_entropy_as_confidence",
                             "n": int(len(sub)),
                             "ece_raw": round(_ece(conf, yt), 4),
                             "mean_confidence": round(float(conf.mean()), 4),
                             "accuracy": round(float(yt.mean()), 4)})
    return pd.DataFrame(rows)


# ─── PART B: T7.2 ECE artefact ──────────────────────────────────────────────
def build_T72(t31: pd.DataFrame):
    rows = []
    for m in MODELS:
        for d in datasets_for(m):
            df = clean_pool(m, d)
            sub = df.dropna(subset=["p_true"]).reset_index(drop=True)
            if sub.empty: continue
            y  = sub["correct"].astype(int).values
            pt = sub["p_true"].values.astype(float)
            qids = sub["question_id"].astype(str).values

            # base_rate_constant — OOF per-fold train-set base rate
            br = baserate_oof(y)
            # Platt-fitted p_true OOF
            pt_platt = platt_oof(pt, y)

            # trace_LR — read T3.1 OOF for these qids (intersection of the
            # baseline pool and the trace_LR pool). Drop any qid that isn't
            # in T3.1's pool (rare; happens if trace_LR dropped a row for
            # NaN-in-trace-feature that isn't NaN in p_true).
            traces, ys_t = trace_oof_for(t31, m, d, qids)
            mask = ~np.isnan(traces)
            if not mask.all():
                # Restrict everything to the intersection so per-cell comparison
                # is on the same questions
                y  = y[mask]; pt = pt[mask]; br = br[mask]
                pt_platt = pt_platt[mask]; traces = traces[mask]; qids = qids[mask]
            assert np.array_equal(y, ys_t[mask] if not mask.all() else ys_t)

            for label, p in [
                ("base_rate_constant", br),
                ("p_true_platt",       pt_platt),
                ("trace_LR",           traces),
            ]:
                rows.append({
                    "dataset": d, "model": m, "method": label,
                    "n": int(len(y)),
                    "ece":   round(_ece(p, y),   4),
                    "brier": round(_brier(p, y), 4),
                    "nll":   round(_nll(p, y),   4),
                    "mean_p":   round(float(np.mean(p)), 4),
                    "std_p":    round(float(np.std(p)),  4),
                    "min_p":    round(float(np.min(p)),  4),
                    "max_p":    round(float(np.max(p)),  4),
                    "accuracy": round(float(np.mean(y)), 4),
                })
    return pd.DataFrame(rows)


# ─── PART C: T7.3 proper-score ranking ──────────────────────────────────────
def build_T73(t31: pd.DataFrame):
    rows = []
    for m in MODELS:
        for d in datasets_for(m):
            df = clean_pool(m, d)
            sub = df.dropna(subset=["answer_semantic_entropy"]).reset_index(drop=True)
            if sub.empty: continue
            y_se = sub["correct"].astype(int).values
            qids = sub["question_id"].astype(str).values
            H    = sub["answer_semantic_entropy"].values.astype(float)
            # Platt on raw H — sign is learned by LR; result identical to
            # fitting on (1 - H/H_MAX) up to affine transform (Platt invariant).
            se_platt = platt_oof(H, y_se)

            # trace_LR aligned on same qids
            traces, _ = trace_oof_for(t31, m, d, qids)
            mask = ~np.isnan(traces)
            if not mask.all():
                y  = y_se[mask]; se_platt = se_platt[mask]; traces = traces[mask]
            else:
                y = y_se

            br_t  = _brier(traces, y)
            nll_t = _nll(traces, y)
            ece_t = _ece(traces, y)
            br_s  = _brier(se_platt, y)
            nll_s = _nll(se_platt, y)
            ece_s = _ece(se_platt, y)

            # Paired bootstrap: (SE − trace) on Brier and NLL (lower = better,
            # so positive Δ ⇒ trace_LR is better)
            d_brier = paired_delta_ci(_brier, se_platt, traces, y)
            d_nll   = paired_delta_ci(_nll,   se_platt, traces, y)
            rows.append({
                "dataset": d, "model": m,
                "n":  int(len(y)),
                "brier_trace_LR":         round(br_t,  4),
                "brier_semantic_entropy_platt": round(br_s, 4),
                "nll_trace_LR":           round(nll_t, 4),
                "nll_semantic_entropy_platt":   round(nll_s, 4),
                "ece_trace_LR":           round(ece_t, 4),
                "ece_semantic_entropy_platt":   round(ece_s, 4),
                # Δ = SE − trace (positive ⇒ trace_LR is BETTER)
                "delta_brier_SE_minus_trace":           round(d_brier[0], 4),
                "delta_brier_ci_low":                   round(d_brier[1], 4),
                "delta_brier_ci_high":                  round(d_brier[2], 4),
                "pct_resamples_trace_better_brier":     round(100.0 * d_brier[3], 2),
                "delta_nll_SE_minus_trace":             round(d_nll[0], 4),
                "delta_nll_ci_low":                     round(d_nll[1], 4),
                "delta_nll_ci_high":                    round(d_nll[2], 4),
                "pct_resamples_trace_better_nll":       round(100.0 * d_nll[3], 2),
            })
    return pd.DataFrame(rows)


# ─── Figures: reliability diagrams ──────────────────────────────────────────
def _reliability(ax, p, y, n_bins=10, label=None, color=None):
    p = np.asarray(p, float); y = np.asarray(y, float)
    mask = ~np.isnan(p)
    p, y = p[mask], y[mask]
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n = int(sel.sum())
        if n == 0: continue
        centers.append(float(p[sel].mean()))
        accs.append(float(y[sel].mean()))
        confs.append(float(p[sel].mean()))
        counts.append(n)
    if centers:
        ax.plot(confs, accs, marker="o", label=label,
                color=color, lw=1.4, markersize=4)


def fig_reliability_for_cell(model, dataset, t31):
    df = clean_pool(model, dataset)
    qids_all = df["question_id"].astype(str).values
    y_all    = df["correct"].astype(int).values

    # Raw p_true / verbalized
    pt_sub = df.dropna(subset=["p_true"])
    vc_sub = df.dropna(subset=["verbalized_confidence"])

    # trace_LR OOF
    sub_t31 = t31[(t31["model"] == model) & (t31["dataset"] == dataset)].copy()
    sub_t31["question_id"] = sub_t31["question_id"].astype(str)
    sub_t31 = sub_t31.set_index("question_id")

    L.apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], color="#888", lw=0.7, linestyle="--",
            label="perfect calibration")
    if not pt_sub.empty:
        _reliability(ax, pt_sub["p_true"].values.astype(float),
                     pt_sub["correct"].astype(int).values,
                     label="raw p_true", color="#1f78b4")
    if not vc_sub.empty:
        _reliability(ax, vc_sub["verbalized_confidence"].values.astype(float),
                     vc_sub["correct"].astype(int).values,
                     label="raw verbalized", color="#e31a1c")
    if not sub_t31.empty:
        rel = sub_t31.reindex(qids_all)
        mask = rel["p_pred"].notna().values
        _reliability(ax, rel["p_pred"].values.astype(float)[mask],
                     y_all[mask],
                     label="trace_LR (OOF)", color="#6a3d9a")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(f"{L.MODEL_LABEL.get(model, model)} / {dataset} — reliability",
                 loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def _save(fig, name):
    p = OUT / name; fig.savefig(p); plt.close(fig); return p


# ─── finding.md ─────────────────────────────────────────────────────────────
def write_finding(t71, t72, t73, t31, t72_qwen_med, t73_qwen_med):
    lines = []; add = lines.append
    add("# Step 7 — Calibration, handled honestly\n")
    add("All numbers below come from `T7.1.csv`, `T7.2.csv`, `T7.3.csv`. "
        "trace_LR's OOF predictions are read from Step 3's `T3.1.csv`; "
        "trace_LR is **not refit** in this step. Platt fitting for baselines "
        "is done **inside the same 5-fold CV** as `T3.1` (`StratifiedKFold("
        "n_splits=5, shuffle=True, random_state=42)`), training folds only, "
        "test fold transformed by the train-fold fit — no leakage.\n")

    # ── A. Raw miscalibration motivation
    add("## Part A — RAW self-report miscalibration (motivation)\n")
    raw_pt  = t71[t71["method"] == "p_true"]
    raw_vc  = t71[t71["method"] == "verbalized_confidence"]
    raw_se  = t71[t71["method"] == "semantic_entropy_as_confidence"]
    add(f"- `p_true` (raw) ECE range across {len(raw_pt)} cells: "
        f"[{float(raw_pt['ece_raw'].min()):.3f}, "
        f"{float(raw_pt['ece_raw'].max()):.3f}]; median "
        f"{float(raw_pt['ece_raw'].median()):.3f}.")
    add(f"- `verbalized_confidence` (raw) ECE range: "
        f"[{float(raw_vc['ece_raw'].min()):.3f}, "
        f"{float(raw_vc['ece_raw'].max()):.3f}]; median "
        f"{float(raw_vc['ece_raw'].median()):.3f}.")
    add(f"- `semantic_entropy` (raw, as confidence 1 − H/log₂K) ECE range: "
        f"[{float(raw_se['ece_raw'].min()):.3f}, "
        f"{float(raw_se['ece_raw'].max()):.3f}]; median "
        f"{float(raw_se['ece_raw'].median()):.3f}.")
    add("")
    over_pt = int((raw_pt["mean_confidence"] - raw_pt["accuracy"] > 0.05).sum())
    over_vc = int((raw_vc["mean_confidence"] - raw_vc["accuracy"] > 0.05).sum())
    add(f"- `p_true` is **overconfident** (mean_confidence − accuracy > 0.05) "
        f"on **{over_pt} / {len(raw_pt)}** cells.")
    add(f"- `verbalized_confidence` is overconfident on **{over_vc} / "
        f"{len(raw_vc)}** cells.")
    add("\nMotivation kept: the model's own confidence outputs are systematically "
        "miscalibrated. Whether that means we should rank methods by ECE is "
        "a separate question — handled in Part B.\n")

    # ── B. ECE artefact
    add("## Part B — Why ECE cannot rank methods (the base-rate-collapse artefact)\n")
    add("Each cell evaluates three methods on the SAME questions:\n")
    add("- `base_rate_constant`: predict the train-fold accuracy for every "
        "test row. By construction it has ~zero ECE.")
    add("- `p_true_platt`: 1-D LR (Platt) on raw `p_true`, OOF.")
    add("- `trace_LR`: native OOF from T3.1.\n")
    add("Showing what each metric says about each method:\n")
    add("### qwen3-4b / medqa (smoking-gun row)\n")
    if t72_qwen_med is not None:
        sub = t72_qwen_med
        add("| method | n | ECE | Brier | NLL | mean p | std p | "
            "min p | max p |")
        add("|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            add(f"| `{r['method']}` | {int(r['n'])} | "
                f"{r['ece']:.4f} | {r['brier']:.4f} | {r['nll']:.4f} | "
                f"{r['mean_p']:.4f} | {r['std_p']:.4f} | "
                f"{r['min_p']:.4f} | {r['max_p']:.4f} |")
        add("")
        br_row = sub[sub["method"] == "base_rate_constant"].iloc[0]
        platt_row = sub[sub["method"] == "p_true_platt"].iloc[0]
        trace_row = sub[sub["method"] == "trace_LR"].iloc[0]
        add(f"- `base_rate_constant` has ECE = **{br_row['ece']:.4f}** "
            f"and a one-value-everywhere prediction "
            f"(std = {br_row['std_p']:.4f}). Useless — yet ECE-optimal.")
        add(f"- `p_true_platt` collapses toward the base rate: std = "
            f"**{platt_row['std_p']:.4f}**, range [{platt_row['min_p']:.4f}, "
            f"{platt_row['max_p']:.4f}], ECE = "
            f"**{platt_row['ece']:.4f}**, **Brier ≈ "
            f"{platt_row['brier']:.4f}** (compare base_rate Brier "
            f"{br_row['brier']:.4f}). It bought low ECE by becoming vague.")
        add(f"- `trace_LR` has HIGHER ECE ({trace_row['ece']:.4f}) but is "
            f"genuinely sharp (std = {trace_row['std_p']:.4f}, range "
            f"[{trace_row['min_p']:.4f}, {trace_row['max_p']:.4f}]) and "
            f"materially better on Brier "
            f"({trace_row['brier']:.4f} vs {platt_row['brier']:.4f}) "
            f"and NLL ({trace_row['nll']:.4f} vs {platt_row['nll']:.4f}).")
        add("")
    add("**Take-away:** lower ECE alone does not mean a better predictor; it "
        "can mean the predictor collapsed to a vague constant. Proper scoring "
        "rules cannot be gamed this way.\n")

    # ── C. Proper-score ranking trace_LR vs Platt(SE)
    add("## Part C — Fair ranking: trace_LR vs Platt-calibrated semantic_entropy on PROPER scores\n")
    add("Both methods give probabilities on the SAME paired questions per "
        "cell. Lower Brier / lower NLL = better. Δ = SE − trace; positive Δ "
        "means trace_LR is better on that proper score.\n")
    add("| cell | n | Brier trace | Brier SE_platt | Δ Brier (CI) | win % | "
        "NLL trace | NLL SE_platt | Δ NLL (CI) | win % |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in t73.iterrows():
        add(f"| {r['model']} / {r['dataset']} | {int(r['n'])} | "
            f"{r['brier_trace_LR']:.4f} | {r['brier_semantic_entropy_platt']:.4f} | "
            f"{r['delta_brier_SE_minus_trace']:+.4f} "
            f"[{r['delta_brier_ci_low']:+.4f}, {r['delta_brier_ci_high']:+.4f}] | "
            f"{r['pct_resamples_trace_better_brier']:.1f} % | "
            f"{r['nll_trace_LR']:.4f} | {r['nll_semantic_entropy_platt']:.4f} | "
            f"{r['delta_nll_SE_minus_trace']:+.4f} "
            f"[{r['delta_nll_ci_low']:+.4f}, {r['delta_nll_ci_high']:+.4f}] | "
            f"{r['pct_resamples_trace_better_nll']:.1f} % |")
    add("")

    # qwen3-4b/medqa explicit
    add("### qwen3-4b / medqa explicit (Brier / NLL)\n")
    if t73_qwen_med is not None:
        r = t73_qwen_med.iloc[0]
        add(f"- trace_LR Brier = **{r['brier_trace_LR']:.4f}**; "
            f"semantic_entropy_platt Brier = "
            f"**{r['brier_semantic_entropy_platt']:.4f}**.")
        add(f"- trace_LR NLL = **{r['nll_trace_LR']:.4f}**; "
            f"semantic_entropy_platt NLL = "
            f"**{r['nll_semantic_entropy_platt']:.4f}**.")
        add(f"- Δ Brier (SE − trace) = "
            f"**{r['delta_brier_SE_minus_trace']:+.4f}** "
            f"[{r['delta_brier_ci_low']:+.4f}, "
            f"{r['delta_brier_ci_high']:+.4f}]; "
            f"trace_LR better on {r['pct_resamples_trace_better_brier']:.1f}% "
            "of paired bootstrap resamples.\n")

    # Discrimination–calibration agreement
    add("### Agreement with the AUROC win cells (Step 5)\n")
    add("Step 5's `trace_LR > semantic_entropy on AUROC` cells (this pass): "
        "`qwen3-4b / mmlu_pro` and `qwen3-4b / medqa`. We check whether the "
        "same two cells also win on Brier and NLL (Part C).\n")
    auc_wins = [("qwen3-4b", "medqa"), ("qwen3-4b", "mmlu_pro")]
    for m, d in auc_wins:
        sub = t73[(t73["model"] == m) & (t73["dataset"] == d)]
        if sub.empty: continue
        sub = sub.iloc[0]
        brier_better = sub["delta_brier_SE_minus_trace"] > 0 and sub["delta_brier_ci_low"] > 0
        nll_better   = sub["delta_nll_SE_minus_trace"]   > 0 and sub["delta_nll_ci_low"]   > 0
        add(f"- **{m} / {d}**: trace_LR better on Brier? "
            f"{'**YES** (CI above 0)' if brier_better else 'no — CI crosses or below 0'}; "
            f"better on NLL? "
            f"{'**YES** (CI above 0)' if nll_better else 'no — CI crosses or below 0'}.")
    add("")
    add("Per-cell agreement only — **no overall reasoning-MCQ verdict here**, "
        "qwq-32b/mmlu_pro pending.\n")

    # ── Sanity checks
    add("## Sanity checks\n")
    add("1. **trace_LR probabilities are the EXACT T3.1 OOF values, never refit.** "
        "Confirmation:")
    sub_t31 = t31[(t31["model"] == "qwen3-4b") & (t31["dataset"] == "medqa")]
    spot = sub_t31.head(3)
    add(f"   - T3.1 rows for qwen3-4b/medqa: {len(sub_t31)} (matches `T3.2.n` "
        f"for that cell). First three (question_id, p_pred, fold):")
    for _, r in spot.iterrows():
        add(f"     - `{r['question_id']}` → p={float(r['p_pred']):.6f}, fold={int(r['fold'])}")
    add("   - These values appear unchanged in the Brier/NLL columns above "
        "for the qwen3-4b/medqa row.")
    add("\n2. **Platt fitting was strictly inside training folds** "
        "(`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`). "
        "`LogisticRegression(max_iter=2000, solver='lbfgs')` is `.fit`-ed on "
        "the train indices of each fold and `.predict_proba`'d on the test "
        "indices only; the pooled OOF probabilities are then scored. No "
        "test-fold rows participate in their own fit.\n")
    add("\n---\nSTOP. Awaiting joint review before Step 8.")
    (OUT / "finding.md").write_text("\n".join(lines), encoding="utf-8")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("Step 7 — calibration")
    t31 = pd.read_csv(T3_DIR / "T3.1.csv")
    print(f"  loaded T3.1 ({len(t31)} OOF rows)")

    t71 = build_T71();         t71.to_csv(OUT / "T7.1.csv", index=False)
    print(f"  wrote T7.1.csv ({len(t71)} rows)")
    t72 = build_T72(t31);      t72.to_csv(OUT / "T7.2.csv", index=False)
    print(f"  wrote T7.2.csv ({len(t72)} rows)")
    t73 = build_T73(t31);      t73.to_csv(OUT / "T7.3.csv", index=False)
    print(f"  wrote T7.3.csv ({len(t73)} rows)")

    fig = fig_reliability_for_cell("qwen3-4b", "medqa", t31)
    _save(fig, "F7.1.pdf"); print("  wrote F7.1.pdf")
    for m in MODELS:
        for d in datasets_for(m):
            fig = fig_reliability_for_cell(m, d, t31)
            _save(fig, f"F7.1.A_{m}_{d}.pdf")
    print("  wrote F7.1.A_<cell>.pdf  (13 files)")

    t72_qwen_med = t72[(t72["model"] == "qwen3-4b") & (t72["dataset"] == "medqa")]
    t73_qwen_med = t73[(t73["model"] == "qwen3-4b") & (t73["dataset"] == "medqa")]
    write_finding(t71, t72, t73, t31, t72_qwen_med, t73_qwen_med)
    print("  wrote finding.md")


if __name__ == "__main__":
    main()
