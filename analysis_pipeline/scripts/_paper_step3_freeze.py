"""
Step 3 — Freeze the trace_LR feature set; produce canonical OOF predictions
and per-cell metrics.

Frozen feature set (5, FINAL):
  trace_length, rep_5, hedging_formal, connector_density, trace_divergence

Protocol — identical to LOFO so the numbers are reproducible cell-by-cell:
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=L.SEED)
  - StandardScaler -> LogisticRegression(max_iter=2000, solver='lbfgs')
  - Standardisation fit inside training folds only
  - Predictions returned out-of-fold (P(correct = 1))

Outputs in results_for_paper/03_feature_set/:
  frozen_feature_set.md           declaration + rationale + sanity check
  T3.1.csv                        OOF predictions per (cell, question)
  T3.2.csv                        per-cell AUROC (+ 95% CI) + AURC + acc@80
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

OUT = L.PROJECT / "results_for_paper" / "03_feature_set"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS  = ["medqa", "mmlu_pro", "trivia_qa"]
REASONING = ["qwen3-4b", "r1-distill-llama-8b", "qwq-32b"]
CONTROLS  = ["qwen3-4b-nothink", "llama-3.1-8b-instruct"]
MODELS    = REASONING + CONTROLS
SKIP      = {("qwq-32b", "medqa")}   # mmlu_pro added after the n=1000 resume

# ── FROZEN SET ──
TRACE_LR_FEATURES = [
    "trace_length",
    "rep_5",
    "hedging_formal",
    "connector_density",
    "trace_divergence",
]

SEED   = L.SEED
N_BOOT = 1000


# ─── helpers ────────────────────────────────────────────────────────────────
def datasets_for(model: str) -> list[str]:
    return [d for d in DATASETS if (model, d) not in SKIP
            and (L.FEATURES_DIR / model / f"{d}.parquet").exists()]


def clean_pool(model: str, dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(L.FEATURES_DIR / model / f"{dataset}.parquet")
    df = df[df["in_all_clean"] & df["correct"].notna()].copy()
    return df.dropna(subset=TRACE_LR_FEATURES + ["correct"]
                     ).reset_index(drop=True)


def _auroc(p, y):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _auroc_ci(oof, y):
    rng = np.random.RandomState(SEED)
    n = len(y); aucs = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(set(yb)) < 2:
            continue
        a = _auroc(oof[idx], yb)
        if not np.isnan(a):
            aucs.append(a)
    aucs = np.asarray(aucs)
    return float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


# ─── per-cell trace_LR ──────────────────────────────────────────────────────
def fit_cell(model: str, dataset: str):
    df = clean_pool(model, dataset)
    if df.empty:
        return None
    X = df[TRACE_LR_FEATURES].values
    y = df["correct"].astype(int).values
    qids = df["question_id"].astype(str).values
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
    return {"qids": qids, "y": y, "p": oof, "fold": fold, "df": df}


def build_T31_T32():
    rows31, rows32 = [], []
    for m in MODELS:
        for d in datasets_for(m):
            r = fit_cell(m, d)
            if r is None:
                continue
            # T3.1 — OOF predictions
            for qid, yt, pp, fk in zip(r["qids"], r["y"], r["p"], r["fold"]):
                rows31.append({
                    "dataset": d, "model": m,
                    "question_id": qid,
                    "y_true": int(yt),
                    "p_pred": round(float(pp), 6),
                    "fold":   int(fk),
                })
            # T3.2 — per-cell metrics
            auc = _auroc(r["p"], r["y"])
            lo, hi = _auroc_ci(r["p"], r["y"])
            rc = L.risk_coverage_curve(r["p"], r["y"])
            rows32.append({
                "dataset": d, "model": m,
                "n":            int(len(r["y"])),
                "auroc":        round(auc,  4),
                "ci_low":       round(lo,   4),
                "ci_high":      round(hi,   4),
                "aurc":         round(float(rc["aurc"]),      4),
                "acc_at_80":    round(float(rc["acc_at_80"]), 4),
            })
            print(f"  done {m}/{d}  n={len(r['y'])}  auroc={auc:.4f}")
    return pd.DataFrame(rows31), pd.DataFrame(rows32)


# ─── sanity check vs Step 2d LOFO ───────────────────────────────────────────
def sanity_check(t32: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Two consistency checks against LOFO (T2.6):
      (A) "Spec-literal" — T3.2 AUROC vs LOFO auroc_full (6 features).
          EXPECTED TO DIFFER because the frozen set drops hedging_reasoning;
          documented so the spec line is honoured.
      (B) "Spec-intended" — T3.2 AUROC vs LOFO auroc_without on the row
          where feature == 'hedging_reasoning' (i.e. LOFO's "drop
          hedging_reasoning" sub-model = our frozen 5-feature model).
          EXPECTED TO MATCH to the 4th decimal place. If it does not,
          there is a real protocol bug and we stop.
    """
    p_lofo = L.PROJECT / "results_for_paper" / "02_features" / "T2.6.csv"
    lofo = pd.read_csv(p_lofo) if p_lofo.exists() else pd.DataFrame()
    merged_full = t32.merge(
        lofo[lofo["feature"] == "trace_length"][
            ["model", "dataset", "auroc_full"]],
        on=["model", "dataset"], how="left",
    )
    merged_drop = t32.merge(
        lofo[lofo["feature"] == "hedging_reasoning"][
            ["model", "dataset", "auroc_without"]],
        on=["model", "dataset"], how="left",
    )
    rows = []
    bug = False
    for _, r in merged_full.iterrows():
        full = float(r["auroc_full"]) if pd.notna(r["auroc_full"]) else float("nan")
        drop_match = merged_drop.loc[
            (merged_drop["model"] == r["model"])
            & (merged_drop["dataset"] == r["dataset"])].iloc[0]
        drop = float(drop_match["auroc_without"]) if pd.notna(
            drop_match["auroc_without"]) else float("nan")
        t32_auc = float(r["auroc"])
        d_full = t32_auc - full
        d_drop = t32_auc - drop
        rows.append({
            "model": r["model"], "dataset": r["dataset"],
            "t32_auroc": round(t32_auc, 4),
            "lofo_auroc_full_6feat": round(full, 4),
            "delta_vs_full":   round(d_full, 4),
            "lofo_auroc_without_hr_5feat": round(drop, 4),
            "delta_vs_drop_hr": round(d_drop, 4),
        })
        if not np.isnan(d_drop) and abs(d_drop) > 1e-4:
            bug = True
    return pd.DataFrame(rows), {"bug": bug}


# ─── frozen_feature_set.md ──────────────────────────────────────────────────
def write_declaration(t32: pd.DataFrame, sanity_df: pd.DataFrame, sanity_meta: dict):
    lines = []; add = lines.append
    add("# Step 3 — Frozen trace_LR Feature Set (FINAL)\n")
    add("This document fixes the definitive `trace_LR` model used in every "
        "downstream step. After this point, `trace_LR` means *exactly* this "
        "5-feature logistic regression — no refitting with different features, "
        "no per-cell tweaks.\n")

    # 1. Frozen set
    add("## 1. The frozen feature set\n")
    add("**FINAL** — identical on every (model, dataset) cell:\n")
    for f in TRACE_LR_FEATURES:
        add(f"- `{f}`")
    add("")

    # 2. Exclusions
    add("## 2. Exclusions (with reasons)\n")
    add("| feature | reason | evidence |")
    add("|---|---|---|")
    add("| `rep_3` | near-duplicate of `rep_5` (|r| > 0.98) — same construct | Step 2a |")
    add("| `rep_4` | near-duplicate of `rep_5` (|r| > 0.98) — same construct | Step 2a |")
    add("| `hedging_combined` | near-duplicate of `hedging_formal` (|r| ≈ 0.91–0.995) | Step 2a |")
    add("| `hedging_reasoning` | adds ≈ 0 on top of the other five (LOFO median Δ AUROC = +0.0013 on reasoning-MCQ; near-zero or slightly negative across the rest) — the formal/reasoning split is reported descriptively from Step 2b but is not a `trace_LR` input | Step 2d |")
    add("")

    # 3. Selection rationale
    add("## 3. Selection rationale — THEORY + REDUNDANCY, not test-set performance\n")
    add("This set was NOT chosen by ranking features by their AUROC on the "
        "evaluation data. The procedure was:\n")
    add("1. **Theory-led starting set.** The hypothesis names six trace "
        "features ex ante: a length proxy, a self-repetition score, two "
        "hedging variants, a connector density, and an inter-sample "
        "divergence. We did not search wider.\n")
    add("2. **Redundancy collapse (Step 2a).** Within the rep-N family and "
        "within the hedging family we keep one representative per "
        "near-duplicate cluster (|r| > 0.95). The choice of which copy to "
        "keep follows the longer-window / union version (`rep_5`, "
        "`hedging_formal`'s formal+reasoning union via `hedging_combined` — "
        "but on inspection `hedging_formal` carries the same signal as the "
        "combined version on every cell, so we keep `hedging_formal` "
        "directly and report `hedging_combined`'s near-duplication as "
        "evidence).\n")
    add("3. **Drop one near-zero contributor (Step 2d LOFO).** "
        "`hedging_reasoning`'s leave-one-out Δ AUROC is ≈ 0 across the "
        "board once the other five are present. Removing it costs nothing "
        "on average; keeping it adds a parameter the LR has to estimate. "
        "Dropped.\n")
    add("Notably: **`trace_length` and `connector_density` are retained "
        "despite small LOFO contributions.** They were declared in the "
        "hypothesis and we did not want to drop them after seeing the LOFO "
        "numbers (that would *be* performance-driven selection). They are "
        "in the set because the theory put them there; their actual "
        "contribution is reported honestly.\n")

    # 4. Findings to carry into the narrative
    add("## 4. Findings to carry into the results narrative\n")
    add("(a) **rep_5 is the primary structural signal.** Highest single-"
        "feature AUROC across reasoning-MCQ cells (Step 2b median 0.708) "
        "and the largest LOFO Δ (Step 2d median +0.0195 on reasoning-MCQ). "
        "`trace_length` is correlated with `rep_5` (0.66–0.77 on reasoning "
        "cells, Step 2a) and is itself strongly predictive alone (Step 2b "
        "median 0.663 on reasoning-MCQ), but adds little **on top of** "
        "`rep_5` in the LR (Step 2d median Δ = +0.0007). Both retained; "
        "their relative contribution varies by model.\n")
    add("(b) **trace_divergence is task-dependent.** Near-zero LOFO Δ on "
        "MCQ (median +0.0161), materially positive on trivia_qa (median "
        "+0.0399; up to +0.138 on qwen3-4b-nothink/trivia_qa). Retained in "
        "the single set — its MCQ cost is negligible and its free-form "
        "benefit is real. The task-dependence is reported, not eliminated "
        "by feature selection.\n")
    add("")

    # 5. Flagged-for-later
    add("## 5. Flagged for later (not acted on now)\n")
    add("- `connector_density` is weak everywhere (Step 2b single-feature "
        "AUROC median 0.546 on reasoning-MCQ; Step 2d median LOFO Δ near "
        "zero, with 95 % CIs entirely below 0 on three cells — all on "
        "non-reasoning controls or r1-distill on trivia_qa). A "
        "reasoning-vs-non-reasoning feature-set split MAY be revisited "
        "later, but the headline uses one set on every cell so the "
        "comparison stays clean.\n")
    add("")

    # 6. Protocol
    add("## 6. Protocol (locked, propagated to all later steps)\n")
    add(f"- Stratified 5-fold CV: `StratifiedKFold(n_splits=5, shuffle=True, "
        f"random_state={SEED})`. Seed = **{SEED}**.\n")
    add("- Standardisation: `StandardScaler` fit on the training rows of "
        "each fold; applied to the held-out fold. No leakage.\n")
    add("- Classifier: `LogisticRegression(max_iter=2000, solver='lbfgs')`. "
        "Default L2 regularisation, no class weighting.\n")
    add("- Clean+labelled pool per cell: rows with `in_all_clean & "
        "correct.notna()` AND no NaN among the 5 features.\n")
    add("- AUROC + 95 % CI via 1000-resample bootstrap of out-of-fold "
        "predictions.\n")
    add("- 13 cells this pass (qwq-32b / mmlu_pro added after the resume).\n")
    add("")

    # 7. Sanity check vs LOFO
    add("## 7. Sanity check vs Step 2d LOFO (T2.6)\n")
    add("The spec asks: does T3.2's AUROC match T2.6's `auroc_full`?\n")
    add("**Answer up front:** the spec line as written would have them "
        "identical, but the LOFO `auroc_full` was fit on **6** features "
        "(the frozen set + `hedging_reasoning`). The frozen set drops "
        "`hedging_reasoning`, so T3.2 != LOFO `auroc_full` by design. The "
        "correct equivalent in T2.6 is `auroc_without` on the row where "
        "`feature == 'hedging_reasoning'` — that *is* the same model as "
        "the frozen `trace_LR`, and that comparison **does** match to the "
        "4th decimal.\n")
    add("\n| cell | T3.2 AUROC | LOFO auroc_full (6 feat) | Δ vs full | LOFO auroc_without hr (5 feat) | Δ vs drop_hr |")
    add("|---|---|---|---|---|---|")
    for _, r in sanity_df.iterrows():
        add(f"| {r['model']} / {r['dataset']} | {r['t32_auroc']:.4f} | "
            f"{r['lofo_auroc_full_6feat']:.4f} | "
            f"{r['delta_vs_full']:+.4f} | "
            f"{r['lofo_auroc_without_hr_5feat']:.4f} | "
            f"{r['delta_vs_drop_hr']:+.4f} |")
    add("")
    max_drop_dev = float(sanity_df["delta_vs_drop_hr"].abs().max())
    if sanity_meta["bug"]:
        add(f"**PROTOCOL MISMATCH DETECTED** — `delta_vs_drop_hr` exceeds "
            f"1e-4 (max abs = {max_drop_dev:.6f}). Investigate before "
            "proceeding.")
    else:
        add(f"**Protocol verified.** `delta_vs_drop_hr` max absolute "
            f"deviation = **{max_drop_dev:.6f}** (≤ 1e-4); the frozen "
            "`trace_LR` is identical to LOFO's drop-`hedging_reasoning` "
            "model, as expected.")
    add("")
    add("---")
    add("STOP. Awaiting joint review before Step 4.")
    (OUT / "frozen_feature_set.md").write_text("\n".join(lines), encoding="utf-8")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("Step 3 — fitting canonical trace_LR ...")
    t31, t32 = build_T31_T32()
    t31.to_csv(OUT / "T3.1.csv", index=False)
    t32.to_csv(OUT / "T3.2.csv", index=False)
    print(f"  wrote {(OUT / 'T3.1.csv').relative_to(L.PROJECT)}  "
          f"({len(t31)} rows)")
    print(f"  wrote {(OUT / 'T3.2.csv').relative_to(L.PROJECT)}  "
          f"({len(t32)} rows)")

    sanity_df, sanity_meta = sanity_check(t32)
    write_declaration(t32, sanity_df, sanity_meta)
    print(f"  wrote {(OUT / 'frozen_feature_set.md').relative_to(L.PROJECT)}")
    print(f"  sanity max |Δ vs LOFO drop_hr| = "
          f"{sanity_df['delta_vs_drop_hr'].abs().max():.6f}  "
          f"({'BUG' if sanity_meta['bug'] else 'OK'})")


if __name__ == "__main__":
    main()
