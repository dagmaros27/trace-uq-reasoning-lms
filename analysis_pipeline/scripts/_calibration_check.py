"""
ADDITIVE, NON-DESTRUCTIVE calibration check.

Question: does trace_LR_combined still have the best ECE once the baselines
are ALSO calibrated with a fitted 1-D map (so the comparison is fitted-vs-
fitted, not fitted-vs-raw)?

For each (model, dataset) cell, for each raw baseline
  {answer_semantic_entropy (confidence form 1 - H/log2(5)), p_true,
   verbalized_confidence}:
  - fit a 1-D calibrator INSIDE the same 5-fold stratified CV that Stage 4
    used (StratifiedKFold(5, shuffle=True, random_state=L.SEED)), fitting on
    train folds only and transforming the held-out fold (no leakage);
  - Platt scaling (LogisticRegression on the single score) = PRIMARY;
  - Isotonic regression = secondary check;
  - recompute ECE on the calibrated out-of-fold predictions with the SAME
    ECE implementation + n_bins=10.

trace_LR_combined and full_LR are NOT refit; their existing ECE (read straight
from the cell's ece.csv) is carried into all three ECE columns.

Outputs (new files only):
  results/calibration_check/<dataset>_calibrated_ece.{csv,json}
  results/calibration_check/summary_trace_lr_still_best.{csv,json}
  results/calibration_check/README.md

Nothing existing is modified.
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
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

BASELINES   = ["answer_semantic_entropy", "p_true", "verbalized_confidence"]
FITTED_REF  = ["trace_LR_combined", "full_LR"]   # carried forward, not refit
H_MAX       = math.log2(5)                        # same as Stage 4's ECE
N_SPLITS    = 5
SEED        = L.SEED
N_BINS      = 10

OUT = L.RESULTS_DIR / "calibration_check"
OUT.mkdir(parents=True, exist_ok=True)


def _models_for(dataset: str) -> list[str]:
    return [m for m in L.MODELS.keys()
            if (L.FEATURES_DIR / m / f"{dataset}.parquet").exists()]


def _existing_ece(model: str, dataset: str) -> dict[str, float]:
    p = L.RESULTS_DIR / model / dataset / "stage4" / "ece.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return {row["method"]: float(row["ece"]) for _, row in d.iterrows()}


def _confidence_input(sub: pd.DataFrame, method: str) -> np.ndarray:
    """The 1-D score we feed the calibrator. Oriented so higher => more
    confident-in-correct. For answer_semantic_entropy we use exactly the
    confidence form Stage 4 reports ECE on: 1 - H/log2(5). (Platt scaling is
    invariant to the log-K constant; isotonic only needs the monotone
    orientation, which this provides.)"""
    if method == "answer_semantic_entropy":
        H = sub["answer_semantic_entropy"].values.astype(float)
        return 1.0 - H / H_MAX
    return sub[method].values.astype(float)


def _cv_calibrate(x: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    """Out-of-fold calibrated probabilities. `kind` in {"platt","isotonic"}.
    Same fold scheme as Stage 4: StratifiedKFold(5, shuffle, SEED)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    xx = x.reshape(-1, 1)
    for tr, te in skf.split(xx, y):
        if kind == "platt":
            clf = LogisticRegression(max_iter=2000, solver="lbfgs")
            clf.fit(xx[tr], y[tr])
            oof[te] = clf.predict_proba(xx[te])[:, 1]
        else:  # isotonic
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x[tr], y[tr])
            oof[te] = iso.predict(x[te])
    return oof


def _ece(probs: np.ndarray, labels: np.ndarray) -> float:
    return L.expected_calibration_error(probs, labels, n_bins=N_BINS)["ece"]


def calibrate_cell(model: str, dataset: str) -> list[dict]:
    df = L.load_features_for(model, dataset) if hasattr(L, "load_features_for") else None
    if df is None:
        # mirror Stage 4's load_features exactly
        p = L.FEATURES_DIR / model / f"{dataset}.parquet"
        df = pd.read_parquet(p)
        df = df[df["in_all_clean"] & df["correct"].notna()].copy().reset_index(drop=True)
        df["correct"] = df["correct"].astype(int)

    existing = _existing_ece(model, dataset)
    rows = []

    for b in BASELINES:
        sub = df.dropna(subset=[b, "correct"]).copy()
        if sub.empty or sub["correct"].nunique() < 2:
            rows.append({"dataset": dataset, "model": model,
                         "model_label": L.MODEL_LABEL[model], "method": b,
                         "method_role": "baseline_calibrated",
                         "n": int(len(sub)),
                         "ece_raw": existing.get(b, float("nan")),
                         "ece_platt": float("nan"),
                         "ece_isotonic": float("nan")})
            continue
        x = _confidence_input(sub, b)
        y = sub["correct"].values.astype(int)
        platt = _cv_calibrate(x, y, "platt")
        iso   = _cv_calibrate(x, y, "isotonic")
        rows.append({
            "dataset": dataset, "model": model,
            "model_label": L.MODEL_LABEL[model], "method": b,
            "method_role": "baseline_calibrated",
            "n": int(len(sub)),
            "ece_raw":      existing.get(b, float("nan")),
            "ece_platt":    _ece(platt, y),
            "ece_isotonic": _ece(iso, y),
        })

    # Fitted references: carry existing ECE into all 3 columns (not refit).
    for f in FITTED_REF:
        if f in existing:
            rows.append({
                "dataset": dataset, "model": model,
                "model_label": L.MODEL_LABEL[model], "method": f,
                "method_role": "fitted_reference",
                "n": None,
                "ece_raw":      existing[f],
                "ece_platt":    existing[f],
                "ece_isotonic": existing[f],
            })
    return rows


def main():
    all_rows = []
    summary = []
    for ds in L.DATASETS:
        ds_rows = []
        for m in _models_for(ds):
            ds_rows.extend(calibrate_cell(m, ds))
        if not ds_rows:
            print(f"  {ds}: no cells")
            continue
        df_ds = pd.DataFrame(ds_rows)
        # write per-dataset
        csv = OUT / f"{ds}_calibrated_ece.csv"
        js  = OUT / f"{ds}_calibrated_ece.json"
        df_ds.to_csv(csv, index=False)
        js.write_text(json.dumps(_json_safe(df_ds), indent=2), encoding="utf-8")
        print(f"  wrote {csv.relative_to(L.PROJECT)}  ({len(df_ds)} rows)")
        all_rows.extend(ds_rows)

        # summary per (model, dataset)
        for m in df_ds["model"].unique():
            cell = df_ds[df_ds["model"] == m]
            tl = cell[cell["method"] == "trace_LR_combined"]
            if tl.empty:
                continue
            trace_ece = float(tl.iloc[0]["ece_raw"])
            base = cell[cell["method_role"] == "baseline_calibrated"]
            # primary = Platt; also track isotonic
            best_base_platt = base.loc[base["ece_platt"].idxmin()] if base["ece_platt"].notna().any() else None
            best_base_iso   = base.loc[base["ece_isotonic"].idxmin()] if base["ece_isotonic"].notna().any() else None
            row = {
                "dataset": ds, "model": m, "model_label": L.MODEL_LABEL[m],
                "trace_LR_ece": trace_ece,
            }
            if best_base_platt is not None:
                row["best_calibrated_baseline_platt"] = best_base_platt["method"]
                row["best_baseline_ece_platt"] = float(best_base_platt["ece_platt"])
                row["trace_LR_still_best_after_calibration"] = bool(trace_ece < float(best_base_platt["ece_platt"]))
                row["margin_platt"] = float(best_base_platt["ece_platt"]) - trace_ece
            if best_base_iso is not None:
                row["best_calibrated_baseline_isotonic"] = best_base_iso["method"]
                row["best_baseline_ece_isotonic"] = float(best_base_iso["ece_isotonic"])
                row["trace_LR_still_best_isotonic"] = bool(trace_ece < float(best_base_iso["ece_isotonic"]))
            summary.append(row)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "summary_trace_lr_still_best.csv", index=False)
    (OUT / "summary_trace_lr_still_best.json").write_text(
        json.dumps(_json_safe(sdf), indent=2), encoding="utf-8")
    _write_readme()

    # console summary
    print("\n================ SUMMARY (primary = Platt) ================")
    print(f"{'dataset':10s} {'model':22s} {'traceLR':>8s} {'bestBase':>9s} "
          f"{'baseline':<22s} {'stillBest':>9s}")
    for r in summary:
        print(f"{r['dataset']:10s} {r['model']:22s} "
              f"{r['trace_LR_ece']:8.3f} {r.get('best_baseline_ece_platt', float('nan')):9.3f} "
              f"{r.get('best_calibrated_baseline_platt',''):<22s} "
              f"{str(r.get('trace_LR_still_best_after_calibration','')):>9s}")
    n_yes = sum(1 for r in summary if r.get("trace_LR_still_best_after_calibration"))
    print(f"\ntrace_LR best ECE after Platt calibration: {n_yes}/{len(summary)} cells")
    print(f"Outputs -> {OUT.relative_to(L.PROJECT)}/")


def _json_safe(df: pd.DataFrame):
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v)):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


def _write_readme():
    (OUT / "README.md").write_text(_README, encoding="utf-8")


_README = """# Calibration Check (additive, non-destructive)

Answers: **does `trace_LR_combined` still have the best ECE once the baselines
are also calibrated with a fitted 1-D map?**

The original ECE comparison is fitted (`trace_LR`, a logistic regression) vs
raw (the baselines were never fit to the data). To make it fitted-vs-fitted,
each raw baseline is calibrated with a 1-D map (Platt scaling = primary;
isotonic = secondary), fit **inside the same 5-fold stratified CV** Stage 4
used (`StratifiedKFold(5, shuffle=True, random_state=42)`), train-folds only,
applied to the held-out fold. ECE is recomputed on the calibrated out-of-fold
predictions with the **same** ECE implementation and `n_bins=10`.

`trace_LR_combined` and `full_LR` are **not refit** — their existing ECE
(read from each cell's `stage4/ece.csv`) is carried forward unchanged.

## Files

- `<dataset>_calibrated_ece.{csv,json}` — one row per (model, method):
  - `ece_raw` — existing pre-calibration ECE (from Stage 4)
  - `ece_platt` — ECE after Platt scaling (PRIMARY)
  - `ece_isotonic` — ECE after isotonic regression (secondary)
  - `n` — rows used (NaN-in-method dropped, same as Stage 4)
  - `method_role` — `baseline_calibrated` or `fitted_reference`
- `summary_trace_lr_still_best.{csv,json}` — one row per (model, dataset):
  - `trace_LR_ece`, `best_calibrated_baseline_platt`, `best_baseline_ece_platt`
  - **`trace_LR_still_best_after_calibration`** — the headline boolean
    (trace_LR ECE < min Platt-calibrated baseline ECE)
  - `margin_platt` — best-baseline-ECE − trace_LR-ECE (positive = trace_LR wins)
  - isotonic equivalents alongside

## Notes
- Platt scaling is invariant to affine transforms of its input, so the
  `1 - H/log2(5)` confidence form used for `answer_semantic_entropy` gives the
  same Platt-calibrated ECE as calibrating raw H — the log-K choice is moot
  for the calibrated number.
- Lower ECE = better calibration. `trace_LR` only loses a cell if a calibrated
  baseline reaches a strictly lower ECE there.

## IMPORTANT — read `*_proper_scores.*` before trusting the ECE grid

ECE alone is **gameable**: a constant base-rate predictor scores ECE ~ 0 while
being useless (AUROC 0.5). Platt-scaling a weakly-discriminative baseline
collapses its predictions toward the base rate, which drives ECE toward zero
for the wrong reason. The `_proper_scores` files compute **Brier** and **NLL**
(proper scoring rules = calibration + sharpness, ungameable) plus a
`base_rate_constant` reference row that demonstrates the artifact directly
(e.g. p_true-after-Platt has Brier/NLL *identical* to the base-rate constant
while showing near-zero ECE). The proper-score comparison is the one to cite.
"""


if __name__ == "__main__":
    main()
