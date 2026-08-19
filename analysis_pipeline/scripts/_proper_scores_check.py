"""
Follow-up to _calibration_check.py: ECE alone is gameable (a base-rate
constant predictor scores ECE ~ 0 while being useless), and it rewards the
weakly-discriminative baselines for collapsing toward the base rate under
Platt scaling. The rigorous comparison uses a PROPER SCORING RULE that
decomposes into calibration + sharpness and therefore cannot be gamed:
Brier score and NLL (log loss). This script computes ECE, Brier, and NLL on
a common footing for every method, plus a base-rate-constant reference row
that demonstrates the ECE artifact.

Common footing (all out-of-fold, same 5-fold stratified CV, seed 42):
  - baselines: Platt-calibrated OOF probabilities (1-D logistic on the score)
  - trace_LR_combined / full_LR: their native LR OOF probabilities,
    recomputed with the SAME seed as Stage 4 (deterministic -> identical)
  - base_rate_constant: predicts the train-fold accuracy for every test row
    (the degenerate ECE-minimiser; included to expose the artifact)

Outputs (new files only, nothing existing touched):
  results/calibration_check/<dataset>_proper_scores.{csv,json}
  results/calibration_check/summary_proper_scores.{csv,json}

Lower is better for all three: ECE, Brier, NLL.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

BASELINES = ["answer_semantic_entropy", "p_true", "verbalized_confidence"]
TRACE_COMBINED = ["trace_length", "hedging_combined", "connector_density",
                  "rep_5", "trace_divergence"]
FULL_FEATS = TRACE_COMBINED + BASELINES
H_MAX = math.log2(5)
SEED, N_SPLITS, N_BINS = L.SEED, 5, 10
EPS = 1e-12

OUT = L.RESULTS_DIR / "calibration_check"
OUT.mkdir(parents=True, exist_ok=True)


def _models_for(dataset: str) -> list[str]:
    return [m for m in L.MODELS.keys()
            if (L.FEATURES_DIR / m / f"{dataset}.parquet").exists()]


def _load_clean(model: str, dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(L.FEATURES_DIR / model / f"{dataset}.parquet")
    df = df[df["in_all_clean"] & df["correct"].notna()].copy().reset_index(drop=True)
    df["correct"] = df["correct"].astype(int)
    return df


def _folds(y: np.ndarray):
    return list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=SEED).split(np.zeros((len(y), 1)), y))


def _ece(p, y):
    return L.expected_calibration_error(p, y, n_bins=N_BINS)["ece"]


def _brier(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def _nll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auroc(p, y):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _platt_oof(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    x = score.reshape(-1, 1)
    for tr, te in _folds(y):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(x[tr], y[tr])
        oof[te] = clf.predict_proba(x[te])[:, 1]
    return oof


def _lr_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    oof = np.full(len(y), np.nan)
    for tr, te in _folds(y):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, solver="lbfgs"))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def _baserate_oof(y: np.ndarray) -> np.ndarray:
    """Degenerate ECE-minimiser: every test row gets the TRAIN fold accuracy."""
    oof = np.full(len(y), np.nan)
    for tr, te in _folds(y):
        oof[te] = y[tr].mean()
    return oof


def _score_input(sub: pd.DataFrame, method: str) -> np.ndarray:
    if method == "answer_semantic_entropy":
        return 1.0 - sub["answer_semantic_entropy"].values.astype(float) / H_MAX
    return sub[method].values.astype(float)


def cell(model: str, dataset: str) -> list[dict]:
    df = _load_clean(model, dataset)
    rows = []

    def add(method, role, p, y, n):
        rows.append({
            "dataset": dataset, "model": model, "model_label": L.MODEL_LABEL[model],
            "method": method, "role": role, "n": int(n),
            "auroc": _auroc(p, y), "ece": _ece(p, y),
            "brier": _brier(p, y), "nll": _nll(p, y),
        })

    # baselines -> Platt-calibrated OOF
    for b in BASELINES:
        sub = df.dropna(subset=[b, "correct"])
        if sub.empty or sub["correct"].nunique() < 2:
            continue
        y = sub["correct"].values.astype(int)
        p = _platt_oof(_score_input(sub, b), y)
        add(b, "baseline_platt", p, y, len(sub))

    # trace_LR_combined -> native LR OOF (same seed as Stage 4)
    sub_t = df.dropna(subset=TRACE_COMBINED + ["correct"])
    if not sub_t.empty:
        y = sub_t["correct"].values.astype(int)
        p = _lr_oof(sub_t[TRACE_COMBINED].values, y)
        add("trace_LR_combined", "fitted_lr", p, y, len(sub_t))
        # base-rate reference uses the same row set as trace_LR for comparability
        add("base_rate_constant", "reference_degenerate",
            _baserate_oof(y), y, len(sub_t))

    # full_LR -> native LR OOF
    sub_f = df.dropna(subset=FULL_FEATS + ["correct"])
    if not sub_f.empty:
        y = sub_f["correct"].values.astype(int)
        p = _lr_oof(sub_f[FULL_FEATS].values, y)
        add("full_LR", "fitted_lr", p, y, len(sub_f))

    return rows


def _json_safe(df):
    out = []
    for rec in df.to_dict(orient="records"):
        d = {}
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v): d[k] = None
            elif isinstance(v, np.integer): d[k] = int(v)
            elif isinstance(v, np.floating): d[k] = float(v)
            elif isinstance(v, np.bool_): d[k] = bool(v)
            else: d[k] = v
        out.append(d)
    return out


def main():
    summary = []
    for ds in L.DATASETS:
        rows = []
        for m in _models_for(ds):
            rows.extend(cell(m, ds))
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d.to_csv(OUT / f"{ds}_proper_scores.csv", index=False)
        (OUT / f"{ds}_proper_scores.json").write_text(
            json.dumps(_json_safe(d), indent=2), encoding="utf-8")
        print(f"  wrote {(OUT / f'{ds}_proper_scores.csv').relative_to(L.PROJECT)}  ({len(d)} rows)")

        # best method per cell by each metric (exclude the degenerate reference)
        for m in d["model"].unique():
            c = d[(d["model"] == m) & (d["role"] != "reference_degenerate")]
            tl = c[c["method"] == "trace_LR_combined"]
            if tl.empty:
                continue
            tlr = tl.iloc[0]
            best_brier = c.loc[c["brier"].idxmin()]
            best_nll   = c.loc[c["nll"].idxmin()]
            best_ece   = c.loc[c["ece"].idxmin()]
            summary.append({
                "dataset": ds, "model": m, "model_label": L.MODEL_LABEL[m],
                "trace_LR_brier": float(tlr["brier"]),
                "best_brier_method": best_brier["method"],
                "best_brier_value": float(best_brier["brier"]),
                "trace_LR_best_by_brier": bool(best_brier["method"] == "trace_LR_combined"),
                "trace_LR_nll": float(tlr["nll"]),
                "best_nll_method": best_nll["method"],
                "best_nll_value": float(best_nll["nll"]),
                "trace_LR_best_by_nll": bool(best_nll["method"] == "trace_LR_combined"),
                "best_ece_method": best_ece["method"],
                "best_ece_value": float(best_ece["ece"]),
                "trace_LR_best_by_ece": bool(best_ece["method"] == "trace_LR_combined"),
            })

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "summary_proper_scores.csv", index=False)
    (OUT / "summary_proper_scores.json").write_text(
        json.dumps(_json_safe(sdf), indent=2), encoding="utf-8")

    print("\n===== best method per cell (baselines Platt-calibrated; lower=better) =====")
    print(f"{'dataset':10s} {'model':22s} {'BRIER best':<24s} {'NLL best':<24s} {'ECE best':<24s}")
    for r in summary:
        print(f"{r['dataset']:10s} {r['model']:22s} "
              f"{r['best_brier_method']:<24s} {r['best_nll_method']:<24s} {r['best_ece_method']:<24s}")
    nb = sum(r["trace_LR_best_by_brier"] for r in summary)
    nn = sum(r["trace_LR_best_by_nll"] for r in summary)
    ne = sum(r["trace_LR_best_by_ece"] for r in summary)
    tot = len(summary)
    print(f"\ntrace_LR_combined is the single best by  Brier: {nb}/{tot}   "
          f"NLL: {nn}/{tot}   ECE: {ne}/{tot}")
    print(f"Outputs -> {OUT.relative_to(L.PROJECT)}/")


if __name__ == "__main__":
    main()
