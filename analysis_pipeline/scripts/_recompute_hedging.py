"""Recompute hedging + connector density columns from the JSONL records using the
current lexicon (v2.0). Preserves all other columns in the parquet — in particular
the `trace_divergence` BGE-M3 embeddings, which are expensive."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import _lib as L
from stage3_features import features_for_trace

HEDGING_COLS = ["hedging_formal", "hedging_reasoning", "hedging_combined",
                "connector_density"]


def recompute(model_short: str, dataset: str = "medqa") -> Path:
    records = L.load_records(model_short, dataset)
    parquet_path = L.FEATURES_DIR / model_short / f"{dataset}.parquet"
    df = pd.read_parquet(parquet_path)
    if len(df) != len(records):
        print(f"WARN: {model_short} parquet has {len(df)} rows but {len(records)} JSONL records")

    # Build map from question_id -> recomputed row
    new_vals = {f: [] for f in HEDGING_COLS}
    qids = []
    for rec in records:
        per_sample = [features_for_trace(s["reasoning_trace"], model_short) for s in rec["samples"]]
        def mean_of(key):
            vals = [s[key] for s in per_sample
                    if not np.isnan(s.get(key, float('nan')))]
            return float(np.mean(vals)) if vals else float('nan')
        new_vals["hedging_formal"].append(   mean_of("hedging_formal_density"))
        new_vals["hedging_reasoning"].append(mean_of("hedging_reasoning_density"))
        new_vals["hedging_combined"].append( mean_of("hedging_combined_density"))
        new_vals["connector_density"].append(mean_of("connectors_logical_density"))
        qids.append(rec["question_id"])

    update = pd.DataFrame({"question_id": qids, **new_vals})
    # Merge back: drop old columns then merge by question_id
    df = df.drop(columns=HEDGING_COLS).merge(update, on="question_id", how="left")
    # Reorder to keep schema consistent (optional)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(parquet_path.with_suffix(".csv"), index=False)
    return parquet_path


if __name__ == "__main__":
    import time
    for m in L.MODELS:
        t = time.time()
        p = recompute(m)
        print(f"  {m}: updated {p.name} in {time.time() - t:.1f}s")
    print("Done. Hedging features now reflect lexicon v2.0 (relational + reasoning, no propositional).")
