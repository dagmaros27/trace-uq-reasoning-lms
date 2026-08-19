"""
Aux table — trace_LR_greedy vs baselines on AUROC + AURC.

Pulls greedy numbers from `02c_greedy_vs_sampled/T2c.1.csv` and baseline
numbers from `05_vs_baselines/T5.1.csv`. No new modelling.

Saves to results_for_paper/02c_greedy_vs_sampled/trace_LR_greedy_vs_baselines.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

OUT_DIR = L.PROJECT / "results_for_paper" / "02c_greedy_vs_sampled"
T2C_PATH = OUT_DIR / "T2c.1.csv"
T51_PATH = L.PROJECT / "results_for_paper" / "05_vs_baselines" / "T5.1.csv"

DATASETS_ORDER = ["medqa", "mmlu_pro", "trivia_qa"]
REASONING_ORDER = ["qwen3-4b", "r1-distill-llama-8b", "qwq-32b"]
CONTROL_ORDER   = ["qwen3-4b-nothink", "llama-3.1-8b-instruct"]
MODELS_ORDER    = REASONING_ORDER + CONTROL_ORDER


def main():
    t2c = pd.read_csv(T2C_PATH)
    t51 = pd.read_csv(T51_PATH)

    rows = []
    for _, r in t2c.iterrows():
        m, d = r["model"], r["dataset"]
        # baseline metrics, on each baseline's own pool (T5.1 columns)
        def baseline_metric(method, col):
            sub = t51[(t51["model"] == m) & (t51["dataset"] == d) &
                      (t51["method"] == method)]
            return float(sub.iloc[0][col]) if not sub.empty else float("nan")

        row = {
            "dataset": d, "model": m,
            "n_greedy": int(r["n_greedy"]),
            # trace_LR_greedy
            "auroc_trace_LR_greedy": round(float(r["auroc_greedy"]), 4),
            "aurc_trace_LR_greedy":  round(float(r["aurc_greedy"]),  4),
            # baselines (raw-oriented, from T5.1 — each on its own pool)
            "auroc_semantic_entropy":      round(baseline_metric("semantic_entropy", "auroc"), 4),
            "aurc_semantic_entropy":       round(baseline_metric("semantic_entropy", "aurc"),  4),
            "auroc_p_true":                round(baseline_metric("p_true", "auroc"), 4),
            "aurc_p_true":                 round(baseline_metric("p_true", "aurc"),  4),
            "auroc_verbalized_confidence": round(baseline_metric("verbalized_confidence", "auroc"), 4),
            "aurc_verbalized_confidence":  round(baseline_metric("verbalized_confidence", "aurc"),  4),
        }
        # convenience deltas (greedy − each baseline, AUROC)
        row["delta_auroc_greedy_minus_SE"]   = round(
            row["auroc_trace_LR_greedy"] - row["auroc_semantic_entropy"], 4)
        row["delta_auroc_greedy_minus_p_true"] = round(
            row["auroc_trace_LR_greedy"] - row["auroc_p_true"], 4)
        row["delta_auroc_greedy_minus_verbal"] = round(
            row["auroc_trace_LR_greedy"] - row["auroc_verbalized_confidence"], 4)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort: dataset, then reasoning models before non-reasoning
    df["_d_order"] = df["dataset"].map({d: i for i, d in enumerate(DATASETS_ORDER)})
    df["_m_order"] = df["model"].map({m: i for i, m in enumerate(MODELS_ORDER)})
    df = df.sort_values(["_d_order", "_m_order"]).drop(columns=["_d_order", "_m_order"])
    df = df.reset_index(drop=True)

    csv_path = OUT_DIR / "trace_LR_greedy_vs_baselines.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path.relative_to(L.PROJECT)} ({len(df)} rows)")

    # Console pretty-print
    print()
    print(f"{'dataset':10s} {'model':22s} {'n':>5s}  "
          f"{'gAUROC':>7s} {'SEau':>7s} {'pT au':>7s} {'vCau':>7s}    "
          f"{'gAURC':>7s} {'SEar':>7s} {'pT ar':>7s} {'vCar':>7s}   "
          f"{'g-SE':>7s} {'g-pT':>7s} {'g-vC':>7s}")
    for _, r in df.iterrows():
        print(f"{r['dataset']:10s} {r['model']:22s} {int(r['n_greedy']):5d}  "
              f"{r['auroc_trace_LR_greedy']:7.4f} "
              f"{r['auroc_semantic_entropy']:7.4f} "
              f"{r['auroc_p_true']:7.4f} "
              f"{r['auroc_verbalized_confidence']:7.4f}    "
              f"{r['aurc_trace_LR_greedy']:7.4f} "
              f"{r['aurc_semantic_entropy']:7.4f} "
              f"{r['aurc_p_true']:7.4f} "
              f"{r['aurc_verbalized_confidence']:7.4f}   "
              f"{r['delta_auroc_greedy_minus_SE']:+7.4f} "
              f"{r['delta_auroc_greedy_minus_p_true']:+7.4f} "
              f"{r['delta_auroc_greedy_minus_verbal']:+7.4f}")


if __name__ == "__main__":
    main()
