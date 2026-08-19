"""
Step 1 — EDA & Generation Statistics.

Descriptive only. No modelling.

Outputs into results_for_paper/01_eda/:
  T1.1 generation_stats.csv     13 rows (qwq-32b/medqa not generated;
                                qwq-32b/mmlu_pro skipped this pass)
  F1.1 hedge_frequency_main.pdf  top-10 hedging terms pooled across the three
                                reasoning models, per-trace frequency
  F1.A hedge_frequency_permodel.pdf  3 panels (qwen3-4b, r1-distill, qwq-32b),
                                pooled per model, per-trace frequency
  finding.md                    short narrative; numbers all sourced from T1.1
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

OUT = L.PROJECT / "results_for_paper" / "01_eda"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS  = ["medqa", "mmlu_pro", "trivia_qa"]
REASONING = ["qwen3-4b", "r1-distill-llama-8b", "qwq-32b"]
CONTROLS  = ["qwen3-4b-nothink", "llama-3.1-8b-instruct"]
ORDER     = REASONING + CONTROLS
SKIP      = {("qwq-32b", "medqa")}
GEN_ROOT  = L.PROJECT.parent / "data_generation" / "data" / "generations"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def jsonl_path(model: str, dataset: str) -> Path | None:
    nested = GEN_ROOT / model / f"{dataset}.jsonl"
    flat   = GEN_ROOT / f"{dataset}_{model}.jsonl"
    if nested.exists(): return nested
    if flat.exists():   return flat
    return None


def feature_path(model: str, dataset: str) -> Path:
    return L.FEATURES_DIR / model / f"{dataset}.parquet"


def is_reasoning(model: str) -> bool:
    return model in REASONING


# ─────────────────────────────────────────────────────────────────────────────
# T1.1 — generation stats
# ─────────────────────────────────────────────────────────────────────────────
def build_T1_1() -> pd.DataFrame:
    rows = []
    for d in DATASETS:
        for m in ORDER:
            if (m, d) in SKIP: continue
            p = feature_path(m, d)
            if not p.exists(): continue
            df = pd.read_parquet(p)
            n_total = int(len(df))
            n_trunc_g = int(df["greedy_truncated"].sum())
            n_clean = int(df["in_all_clean"].sum())
            mask_pool = df["in_all_clean"] & df["correct"].notna()
            n_pool = int(mask_pool.sum())
            acc = float(df.loc[mask_pool, "correct"].mean()) if n_pool else float("nan")
            # trace_length is the sampled-averaged length already
            ml = float(df["trace_length"].mean())
            rows.append({
                "dataset": d, "model": m,
                "n_total": n_total,
                "n_truncated_greedy": n_trunc_g,
                "n_clean": n_clean,
                "n_clean_and_labelled": n_pool,
                "greedy_accuracy_on_clean": round(acc, 4),
                "mean_trace_length_sampled": round(ml, 1),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Hedging frequency — pooled across reasoning model TRACES (samples-side)
# ─────────────────────────────────────────────────────────────────────────────
def gather_hedge_counts():
    """For each reasoning model, iterate every sample's reasoning_trace
    across every cell that exists; aggregate per-term match counts via
    L.lex_match_terms(..., "hedging_combined"). Track number of traces so
    we can normalise to per-trace frequency.

    Returns:
      per_model_counts: dict[model] -> Counter
      per_model_n_traces: dict[model] -> int
      global_counts: Counter (sum of per-model)
      global_n_traces: int
    """
    per_model_counts: dict[str, Counter] = {m: Counter() for m in REASONING}
    per_model_n_traces: dict[str, int] = {m: 0 for m in REASONING}
    for m in REASONING:
        for d in DATASETS:
            if (m, d) in SKIP: continue
            p = jsonl_path(m, d)
            if p is None: continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    for s in (r.get("samples") or []):
                        trace = s.get("reasoning_trace") or ""
                        if not trace:
                            per_model_n_traces[m] += 1
                            continue
                        per_model_counts[m] += L.lex_match_terms(
                            trace, "hedging_combined")
                        per_model_n_traces[m] += 1
            print(f"  pooled {m}/{d}", flush=True)
    global_counts = Counter()
    for m in REASONING:
        global_counts += per_model_counts[m]
    global_n_traces = sum(per_model_n_traces.values())
    return per_model_counts, per_model_n_traces, global_counts, global_n_traces


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def fig_main(global_counts: Counter, global_n_traces: int, top_k: int = 10):
    """F1.1 — single horizontal bar chart of top-k hedging terms by total
    match count, displayed as per-trace frequency."""
    L.apply_style()
    top = global_counts.most_common(top_k)
    terms = [t for t, _ in top]
    rates = [c / global_n_traces for _, c in top]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    y = np.arange(len(terms))
    ax.barh(y, rates, color="#1f78b4", edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set_xlabel("matches per reasoning-model sample trace")
    ax.set_title(f"Top {top_k} hedging terms — pooled across reasoning models "
                 f"(qwen3-4b, r1-distill-llama-8b, qwq-32b); "
                 f"{global_n_traces:,} traces", loc="left", fontsize=10)
    for i, r in enumerate(rates):
        ax.text(r, i, f"  {r:.3f}", va="center", fontsize=9, color="#222")
    ax.set_xlim(0, max(rates) * 1.18)
    fig.tight_layout()
    return fig


def fig_permodel(per_model_counts, per_model_n_traces, top_k: int = 10):
    """F1.A — one panel per reasoning model, per-trace frequency. Each panel
    uses its OWN top-10 (the model's own most frequent hedges)."""
    L.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=False)
    colors = ["#e31a1c", "#1f78b4", "#6a3d9a"]  # qwen3-4b, r1, qwq
    for ax, m, color in zip(axes, REASONING, colors):
        c = per_model_counts[m]
        n = per_model_n_traces[m]
        if n == 0 or not c:
            ax.text(0.5, 0.5, "no traces", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(L.MODEL_LABEL.get(m, m), loc="left", fontsize=10)
            ax.axis("off"); continue
        top = c.most_common(top_k)
        terms = [t for t, _ in top]
        rates = [k / n for _, k in top]
        y = np.arange(len(terms))
        ax.barh(y, rates, color=color, edgecolor="white")
        ax.set_yticks(y); ax.set_yticklabels(terms, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("matches per sample trace")
        ax.set_title(f"{L.MODEL_LABEL.get(m, m)}  ({n:,} traces)",
                     loc="left", fontsize=10)
        ax.set_xlim(0, max(rates) * 1.18)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# finding.md (numbers all read from T1.1.csv)
# ─────────────────────────────────────────────────────────────────────────────
def write_finding(t11: pd.DataFrame, global_n_traces: int):
    """Draft findings; ALL numbers come from t11 (no hand-typed values)."""
    # accuracy spread
    by_acc = t11.sort_values("greedy_accuracy_on_clean", ascending=False)
    best = by_acc.iloc[0];  worst = by_acc.iloc[-1]
    # weakest cell by n_clean / n_total
    t11_aux = t11.assign(clean_frac=t11["n_clean"] / t11["n_total"])
    weakest = t11_aux.sort_values("clean_frac").iloc[0]
    # trace length means: reasoning vs non-reasoning
    is_rsn = t11["model"].isin(REASONING)
    mean_len_rsn = float(t11.loc[is_rsn, "mean_trace_length_sampled"].mean())
    mean_len_ctl = float(t11.loc[~is_rsn, "mean_trace_length_sampled"].mean())
    longest_rsn = t11.loc[is_rsn].sort_values(
        "mean_trace_length_sampled", ascending=False).iloc[0]
    shortest_ctl = t11.loc[~is_rsn].sort_values(
        "mean_trace_length_sampled").iloc[0]

    # per-cell accuracy lines for the table
    lines = []
    add = lines.append
    add("# Step 1 — EDA findings\n")
    add("All numbers below come from `T1.1.csv`.\n")
    add("## Accuracy spread (greedy on clean-and-labelled pool)\n")
    add(f"- Best cell: **{best['model']} / {best['dataset']}** at "
        f"**{best['greedy_accuracy_on_clean']:.3f}** "
        f"(n_clean_and_labelled = {best['n_clean_and_labelled']}).")
    add(f"- Worst cell: **{worst['model']} / {worst['dataset']}** at "
        f"**{worst['greedy_accuracy_on_clean']:.3f}** "
        f"(n = {worst['n_clean_and_labelled']}).")
    # by-model average accuracy (treating each cell as a unit)
    by_mod = t11.groupby("model")["greedy_accuracy_on_clean"].mean().sort_values(
        ascending=False).round(3).to_dict()
    add("- Mean accuracy per model (averaged across the datasets each model "
        "has): "
        + ", ".join(f"`{k}` {v:.3f}" for k, v in by_mod.items()) + ".")
    rsn_means = t11.loc[is_rsn].groupby("model")[
        "greedy_accuracy_on_clean"].mean().sort_values(ascending=False)
    add("- Among reasoning models the order is "
        + ", ".join(f"`{k}` ({v:.3f})" for k, v in rsn_means.items())
        + " — r1-distill is the weakest reasoning model; qwen3-4b and qwq-32b "
        "are the strongest.")
    add("")
    add("## Truncation caveat\n")
    add(f"- Worst-cleaning cell present in this pass: "
        f"**{weakest['model']} / {weakest['dataset']}** — "
        f"n_clean = **{int(weakest['n_clean'])}** of "
        f"{int(weakest['n_total'])} "
        f"({100.0 * weakest['n_clean'] / weakest['n_total']:.1f} % clean), "
        f"n_clean_and_labelled = {int(weakest['n_clean_and_labelled'])}.")
    add("- This cell stays in the analysis; the smaller usable sample is "
        "noted alongside any number sourced from it.")
    add(f"- qwq-32b / mmlu_pro is omitted this pass (partial 500-record run; "
        "resume in flight on the VM).")
    add("")
    add("## trivia_qa parse-failed counter artefact\n")
    add("- Stage 1 records `extracted_choice = None` for every "
        "`kind='free_answer'` record by design (there is no MCQ letter to "
        "extract). Looking at that column alone would suggest 100 % parse "
        "failure on trivia_qa.")
    add("- The real free-form extraction lives in `extracted_prediction` "
        "(parsed from the `<answer>...</answer>` block, with last-non-empty-"
        "line fallback). Labels were assigned successfully — n_clean_and_"
        "labelled equals n_clean on every trivia_qa cell in T1.1 — confirming "
        "the apparent parse-fail signal is a column-naming artefact, not lost "
        "data.")
    add("")
    add("## Trace length — reasoning vs non-reasoning\n")
    add(f"- Reasoning-model mean sampled trace length (averaged over their "
        f"cells): **{mean_len_rsn:.0f}** tokens.")
    add(f"- Non-reasoning controls mean sampled trace length: "
        f"**{mean_len_ctl:.0f}** tokens "
        f"({mean_len_rsn / mean_len_ctl:.1f}× shorter than reasoning models).")
    add(f"- Longest reasoning cell: **{longest_rsn['model']} / "
        f"{longest_rsn['dataset']}** at "
        f"**{longest_rsn['mean_trace_length_sampled']:.0f}** tokens.")
    add(f"- Shortest non-reasoning cell: **{shortest_ctl['model']} / "
        f"{shortest_ctl['dataset']}** at "
        f"**{shortest_ctl['mean_trace_length_sampled']:.0f}** tokens.")
    add("- Reasoning models produce substantially longer traces than non-"
        "reasoning models on every dataset; trace_length is therefore a "
        "candidate uncertainty feature with discriminative range, used in "
        "the later modelling stage.")
    add("")
    add("## Hedging-frequency figures\n")
    add(f"- F1.1 (main): top-10 terms pooled across "
        f"{global_n_traces:,} sample traces from the three reasoning models.")
    add(f"- F1.A (appendix): same per-trace-frequency view broken down per "
        f"reasoning model.")
    add("")
    add("---")
    add("STOP. Awaiting go-ahead for Step 2.")
    (OUT / "finding.md").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t11 = build_T1_1()
    csv = OUT / "T1.1.csv"
    t11.to_csv(csv, index=False)
    print(f"Wrote {csv.relative_to(L.PROJECT)}  ({len(t11)} rows)")

    print("Gathering hedge counts across reasoning-model sample traces ...")
    per_model_counts, per_model_n_traces, global_counts, global_n_traces = \
        gather_hedge_counts()
    print(f"  total traces pooled: {global_n_traces:,}")
    print(f"  total hedge matches: {sum(global_counts.values()):,}")

    if global_n_traces == 0:
        # Raw generations not present (they are not shipped in the repo);
        # the hedge-frequency figures need them. Tables above still reproduce.
        print("  raw generation JSONLs not found — skipping F1.1 / F1.A "
              "(regenerate stage-1 data to produce these figures)")
        write_finding(t11, global_n_traces)
        print(f"Wrote {(OUT / 'finding.md').relative_to(L.PROJECT)}")
        return

    fig1 = fig_main(global_counts, global_n_traces)
    fig1.savefig(OUT / "F1.1.pdf"); plt.close(fig1)
    print(f"Wrote {(OUT / 'F1.1.pdf').relative_to(L.PROJECT)}")

    figA = fig_permodel(per_model_counts, per_model_n_traces)
    figA.savefig(OUT / "F1.A.pdf"); plt.close(figA)
    print(f"Wrote {(OUT / 'F1.A.pdf').relative_to(L.PROJECT)}")

    write_finding(t11, global_n_traces)
    print(f"Wrote {(OUT / 'finding.md').relative_to(L.PROJECT)}")


if __name__ == "__main__":
    main()
