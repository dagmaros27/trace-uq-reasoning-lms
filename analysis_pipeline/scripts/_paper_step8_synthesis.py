"""
Step 8 — Reasoning vs Non-Reasoning synthesis.

Aggregates the canonical outputs from Steps 5, 6, 7 (and Step 1's T1.1 for
context). Computes no new models. Tags every cell on two axes:
  model_type ∈ {reasoning, non_reasoning}
  task_type  ∈ {mcq, free_form}

Outputs in results_for_paper/08_reasoning_vs_nonreasoning/:
  T8.1.csv         14-row synthesis grid (every metric pulled from existing tables)
  T8.2.csv         4-group summary (model_type x task_type), characterisation blank
  F8.1.pdf         (trace_LR − SE) heatmap across all 14 cells, * marks CI > 0
  finding.md       per-group counts + medians, no over-conclusion
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

OUT = L.PROJECT / "results_for_paper" / "08_reasoning_vs_nonreasoning"
OUT.mkdir(parents=True, exist_ok=True)
T1_DIR = L.PROJECT / "results_for_paper" / "01_eda"
T5_DIR = L.PROJECT / "results_for_paper" / "05_vs_baselines"
T6_DIR = L.PROJECT / "results_for_paper" / "06_combined"
T7_DIR = L.PROJECT / "results_for_paper" / "07_calibration"

REASONING = {"qwen3-4b", "qwq-32b", "r1-distill-llama-8b"}
NON_REASONING = {"qwen3-4b-nothink", "llama-3.1-8b-instruct"}
MCQ_DATASETS  = {"medqa", "mmlu_pro"}
FREE_DATASETS = {"trivia_qa"}

DATASETS_ORDER  = ["medqa", "mmlu_pro", "trivia_qa"]
MODELS_ORDER    = ["qwen3-4b", "qwq-32b", "r1-distill-llama-8b",
                   "qwen3-4b-nothink", "llama-3.1-8b-instruct"]


def model_type(m: str) -> str:
    return "reasoning" if m in REASONING else "non_reasoning"


def task_type(d: str) -> str:
    return "mcq" if d in MCQ_DATASETS else "free_form"


# ─── Precondition: 14 cells in every canonical output ──────────────────────
def precondition_check():
    t31 = pd.read_csv(L.PROJECT / "results_for_paper" / "03_feature_set" / "T3.2.csv")
    t52 = pd.read_csv(T5_DIR / "T5.2.csv")
    t61 = pd.read_csv(T6_DIR / "T6.1.csv")
    t73 = pd.read_csv(T7_DIR / "T7.3.csv")
    counts = {"T3.2": len(t31), "T5.2": len(t52),
              "T6.1": len(t61), "T7.3": len(t73)}
    if any(v != 14 for v in counts.values()):
        print(f"PRECONDITION FAILED: cell counts = {counts}")
        sys.exit(2)
    print(f"  precondition OK: {counts}")


# ─── Build T8.1 — 14-row synthesis grid ────────────────────────────────────
def build_T81():
    t11 = pd.read_csv(T1_DIR / "T1.1.csv")
    t51 = pd.read_csv(T5_DIR / "T5.1.csv")
    t52 = pd.read_csv(T5_DIR / "T5.2.csv")
    t61 = pd.read_csv(T6_DIR / "T6.1.csv")
    t73 = pd.read_csv(T7_DIR / "T7.3.csv")

    rows = []
    # Iterate every (model, dataset) cell present in T5.2 (the paired bootstrap)
    for _, p in t52.iterrows():
        m, d = p["model"], p["dataset"]
        # n_clean from T1.1 (n_clean_and_labelled column there)
        t11_row = t11[(t11["model"] == m) & (t11["dataset"] == d)]
        if t11_row.empty:
            # qwq/mmlu_pro is not in the n=13 Step-1 T1.1; pull n from T5.2 instead
            n_clean = int(p["n"])
        else:
            n_clean = int(t11_row.iloc[0]["n_clean_and_labelled"])
        # trace_LR + SE AUROC / AURC / acc@80 from T5.1
        tr_auc = float(t51[(t51["model"] == m) & (t51["dataset"] == d)
                            & (t51["method"] == "trace_LR")].iloc[0]["auroc"])
        tr_aurc = float(t51[(t51["model"] == m) & (t51["dataset"] == d)
                            & (t51["method"] == "trace_LR")].iloc[0]["aurc"])
        tr_acc80 = float(t51[(t51["model"] == m) & (t51["dataset"] == d)
                            & (t51["method"] == "trace_LR")].iloc[0]["acc_at_80"])
        se_auc = float(t51[(t51["model"] == m) & (t51["dataset"] == d)
                            & (t51["method"] == "semantic_entropy")].iloc[0]["auroc"])
        se_aurc = float(t51[(t51["model"] == m) & (t51["dataset"] == d)
                            & (t51["method"] == "semantic_entropy")].iloc[0]["aurc"])
        se_acc80 = float(t51[(t51["model"] == m) & (t51["dataset"] == d)
                            & (t51["method"] == "semantic_entropy")].iloc[0]["acc_at_80"])
        # CI-based trace_beats_se from T5.2 (paired bootstrap CI low > 0)
        ci_low = float(p["ci_low"])
        median_delta = float(p["median_delta_auroc"])
        trace_beats_se = bool(ci_low > 0 and median_delta > 0)

        # Brier / NLL winners from T7.3 (delta_X_SE_minus_trace > 0 ⇒ trace better)
        t73_row = t73[(t73["model"] == m) & (t73["dataset"] == d)]
        if t73_row.empty:
            brier_winner = "n/a"; nll_winner = "n/a"
        else:
            t73_row = t73_row.iloc[0]
            d_b = float(t73_row["delta_brier_SE_minus_trace"])
            d_b_lo = float(t73_row["delta_brier_ci_low"])
            d_b_hi = float(t73_row["delta_brier_ci_high"])
            d_n = float(t73_row["delta_nll_SE_minus_trace"])
            d_n_lo = float(t73_row["delta_nll_ci_low"])
            d_n_hi = float(t73_row["delta_nll_ci_high"])
            def _winner(d, lo, hi):
                if lo > 0:  return "trace"
                if hi < 0:  return "SE"
                return "tie"
            brier_winner = _winner(d_b, d_b_lo, d_b_hi)
            nll_winner   = _winner(d_n, d_n_lo, d_n_hi)

        # combined verdict from T6.1 (rule reapplied; matches Step 6's finding)
        t61_row = t61[(t61["model"] == m) & (t61["dataset"] == d)].iloc[0]
        f_auc = float(t61_row["auroc_full_LR"])
        t_auc = float(t61_row["auroc_trace_LR"])
        s_auc = float(t61_row["auroc_semantic_entropy"])
        max_alt = max(t_auc, s_auc)
        d_ft_lo = float(t61_row["delta_auroc_full_vs_trace_ci_low"])
        d_fs_lo = float(t61_row["delta_auroc_full_vs_SE_ci_low"])
        comp = (f_auc >= max_alt + 0.005) and (d_ft_lo > 0) and (d_fs_lo > 0)
        red  = (abs(f_auc - max_alt) <= 0.01)
        combined_verdict = "complementary" if comp else (
            "redundant" if red else "mixed")

        rows.append({
            "dataset": d, "model": m,
            "model_type": model_type(m),
            "task_type":  task_type(d),
            "n_clean":    n_clean,
            "trace_LR_auroc":   round(tr_auc, 4),
            "se_auroc":         round(se_auc, 4),
            "trace_minus_se_auroc": round(median_delta, 4),
            "trace_minus_se_ci_low":  round(ci_low,                            4),
            "trace_minus_se_ci_high": round(float(p["ci_high"]),                4),
            "trace_beats_se":   trace_beats_se,
            "trace_LR_aurc":    round(tr_aurc, 4),
            "se_aurc":          round(se_aurc, 4),
            "trace_LR_acc80":   round(tr_acc80, 4),
            "se_acc80":         round(se_acc80, 4),
            "brier_winner":     brier_winner,
            "nll_winner":       nll_winner,
            "combined_verdict": combined_verdict,
        })

    # Order: reasoning first, mcq datasets first
    def sort_key(r):
        return (r["model_type"] != "reasoning",          # reasoning first
                MODELS_ORDER.index(r["model"])
                  if r["model"] in MODELS_ORDER else 99,
                r["task_type"] != "mcq",                 # mcq first
                DATASETS_ORDER.index(r["dataset"])
                  if r["dataset"] in DATASETS_ORDER else 99)
    rows = sorted(rows, key=sort_key)
    return pd.DataFrame(rows)


# ─── T8.2 — group summary ─────────────────────────────────────────────────
def build_T82(t81: pd.DataFrame):
    rows = []
    for mt in ["reasoning", "non_reasoning"]:
        for tt in ["mcq", "free_form"]:
            sub = t81[(t81["model_type"] == mt) & (t81["task_type"] == tt)]
            rows.append({
                "model_type": mt, "task_type": tt,
                "n_cells":    int(len(sub)),
                "n_cells_trace_beats_se": int(sub["trace_beats_se"].sum()),
                "median_trace_minus_se_auroc": (
                    round(float(sub["trace_minus_se_auroc"].median()), 4)
                    if len(sub) else float("nan")),
                "characterisation": "",   # left BLANK on purpose for the student
            })
    return pd.DataFrame(rows)


# ─── F8.1 heatmap ──────────────────────────────────────────────────────────
def fig_synthesis_heatmap(t81: pd.DataFrame):
    L.apply_style()
    # Row order: reasoning models grouped first
    row_models = [m for m in MODELS_ORDER if m in REASONING] + \
                 [m for m in MODELS_ORDER if m in NON_REASONING]
    col_datasets = ["medqa", "mmlu_pro", "trivia_qa"]
    nR, nC = len(row_models), len(col_datasets)
    A = np.full((nR, nC), np.nan)
    star = np.zeros((nR, nC), dtype=bool)
    for _, r in t81.iterrows():
        i = row_models.index(r["model"])
        j = col_datasets.index(r["dataset"])
        A[i, j] = float(r["trace_minus_se_auroc"])
        star[i, j] = bool(r["trace_beats_se"])

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    vmax = float(np.nanmax(np.abs(A)))
    im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(nC))
    ax.set_xticklabels(col_datasets, fontsize=9)
    ax.set_yticks(range(nR))
    ax.set_yticklabels(
        [L.MODEL_LABEL.get(m, m) + ("  (reasoning)" if m in REASONING
                                     else "  (non-reasoning)")
         for m in row_models], fontsize=9)
    # block separator
    boundary = sum(1 for m in row_models if m in REASONING) - 0.5
    ax.axhline(boundary, color="#222", lw=1.2)
    # task-type column separator (mcq | free_form)
    ax.axvline(1.5, color="#222", lw=0.9, linestyle=":")

    for i in range(nR):
        for j in range(nC):
            v = A[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="#999")
            else:
                txt = f"{v:+.3f}" + ("*" if star[i, j] else "")
                color = "white" if abs(v) > vmax * 0.55 else "#111"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("AUROC(trace_LR) − AUROC(semantic_entropy)")
    ax.set_title("Step 8 synthesis — trace_LR vs SE across the 14-cell grid\n"
                 "(* = paired-bootstrap 95 % CI entirely above 0)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    return fig


# ─── finding.md ────────────────────────────────────────────────────────────
def write_finding(t81: pd.DataFrame, t82: pd.DataFrame):
    lines = []; add = lines.append
    add("# Step 8 — Reasoning vs Non-Reasoning Synthesis\n")
    add("All numbers below come from `T8.1.csv` and `T8.2.csv`, which "
        "in turn pull from `T5.1`, `T5.2`, `T6.1`, `T7.3` and `T1.1`. "
        "**No new models are fit in this step.** 14 cells.\n")

    add("## Per-group pattern (T8.2)\n")
    add("| model_type | task_type | n_cells | n_cells_trace_beats_se | median Δ AUROC |")
    add("|---|---|---|---|---|")
    for _, r in t82.iterrows():
        add(f"| {r['model_type']} | {r['task_type']} | {int(r['n_cells'])} | "
            f"**{int(r['n_cells_trace_beats_se'])}** | "
            f"{r['median_trace_minus_se_auroc']:+.4f} |")
    add("\n*(`trace_beats_se` is CI-based — paired bootstrap 95 % CI strictly "
        "above 0.)*\n")

    # Reasoning-MCQ trio explicitly
    add("## The reasoning × MCQ trio (the central evidence)\n")
    trio_cells = [("qwen3-4b", "medqa"),
                  ("qwen3-4b", "mmlu_pro"),
                  ("qwq-32b",  "mmlu_pro")]
    add("| cell | n | trace_LR | SE | Δ trace − SE | 95 % CI | brier winner | nll winner |")
    add("|---|---|---|---|---|---|---|---|")
    for m, d in trio_cells:
        r = t81[(t81["model"] == m) & (t81["dataset"] == d)]
        if r.empty: continue
        r = r.iloc[0]
        add(f"| {m} / {d} | {int(r['n_clean'])} | "
            f"{r['trace_LR_auroc']:.4f} | {r['se_auroc']:.4f} | "
            f"**{r['trace_minus_se_auroc']:+.4f}** | "
            f"[{r['trace_minus_se_ci_low']:+.4f}, "
            f"{r['trace_minus_se_ci_high']:+.4f}] | "
            f"{r['brier_winner']} | {r['nll_winner']} |")
    add("")

    # r1-distill exception
    add("## r1-distill — the reasoning model that does NOT show the MCQ effect\n")
    r1 = t81[t81["model"] == "r1-distill-llama-8b"]
    add("| cell | n_clean | trace_LR | SE | Δ trace − SE | 95 % CI | trace_beats_se? |")
    add("|---|---|---|---|---|---|---|")
    for _, r in r1.iterrows():
        add(f"| {r['model']} / {r['dataset']} | {int(r['n_clean'])} | "
            f"{r['trace_LR_auroc']:.4f} | {r['se_auroc']:.4f} | "
            f"{r['trace_minus_se_auroc']:+.4f} | "
            f"[{r['trace_minus_se_ci_low']:+.4f}, "
            f"{r['trace_minus_se_ci_high']:+.4f}] | "
            f"{'YES' if r['trace_beats_se'] else 'no'} |")
    # Add context from T1.1
    t11 = pd.read_csv(T1_DIR / "T1.1.csv")
    r1_ctx = t11[t11["model"] == "r1-distill-llama-8b"]
    add("\n**Associated context (T1.1, descriptive only — not a causal claim):**\n")
    add("| dataset | accuracy on clean | n_clean | truncation (greedy) |")
    add("|---|---|---|---|")
    for _, r in r1_ctx.iterrows():
        add(f"| {r['dataset']} | {float(r['greedy_accuracy_on_clean']):.3f} | "
            f"{int(r['n_clean_and_labelled'])} | "
            f"{int(r['n_truncated_greedy'])} / {int(r['n_total'])} "
            f"({100.0 * r['n_truncated_greedy'] / r['n_total']:.1f} %) |")
    add("")

    add("## Free-form (trivia_qa) — SE wins for every model\n")
    free = t81[t81["task_type"] == "free_form"]
    add("| cell | trace_LR | SE | Δ trace − SE | 95 % CI |")
    add("|---|---|---|---|---|")
    for _, r in free.iterrows():
        add(f"| {r['model']} / {r['dataset']} | {r['trace_LR_auroc']:.4f} | "
            f"{r['se_auroc']:.4f} | {r['trace_minus_se_auroc']:+.4f} | "
            f"[{r['trace_minus_se_ci_low']:+.4f}, "
            f"{r['trace_minus_se_ci_high']:+.4f}] |")
    add("")

    add("## Non-reasoning models — SE wins throughout\n")
    nr = t81[t81["model_type"] == "non_reasoning"]
    add("| cell | trace_LR | SE | Δ trace − SE | 95 % CI |")
    add("|---|---|---|---|---|")
    for _, r in nr.iterrows():
        add(f"| {r['model']} / {r['dataset']} | {r['trace_LR_auroc']:.4f} | "
            f"{r['se_auroc']:.4f} | {r['trace_minus_se_auroc']:+.4f} | "
            f"[{r['trace_minus_se_ci_low']:+.4f}, "
            f"{r['trace_minus_se_ci_high']:+.4f}] |")
    add("")

    # Sanity spot check
    add("## Sanity — every number traces back; no recomputation\n")
    r = t81[(t81["model"] == "qwen3-4b") & (t81["dataset"] == "mmlu_pro")].iloc[0]
    add("Spot check (`qwen3-4b / mmlu_pro`):\n")
    add(f"- `trace_LR_auroc = {r['trace_LR_auroc']:.4f}` ← from T5.1 row "
        f"(method == 'trace_LR')")
    add(f"- `se_auroc = {r['se_auroc']:.4f}` ← from T5.1 row "
        f"(method == 'semantic_entropy')")
    add(f"- `trace_minus_se_auroc = {r['trace_minus_se_auroc']:+.4f}` and "
        f"`CI = [{r['trace_minus_se_ci_low']:+.4f}, "
        f"{r['trace_minus_se_ci_high']:+.4f}]` ← T5.2 paired bootstrap row")
    add(f"- `brier_winner = '{r['brier_winner']}'`, "
        f"`nll_winner = '{r['nll_winner']}'` ← derived from T7.3 deltas + CIs")
    add(f"- `combined_verdict = '{r['combined_verdict']}'` ← derived from T6.1 "
        "by reapplying the Step-6 rule\n")

    # Stop short of the thesis verdict
    add("---\n")
    add("**Important framing reminder.** This step reports the pattern in the "
        "numbers. The sentence that closes the thesis claim — *the trace-feature "
        "discrimination signal is a property of reasoning models on multiple-"
        "choice tasks* — is the student's to write, supported by this table and "
        "F8.1. The script does NOT write that conclusion. The reasoning × MCQ "
        f"group has **{int(t82.iloc[0]['n_cells_trace_beats_se'])} of "
        f"{int(t82.iloc[0]['n_cells'])} cells with CI-based trace wins**; "
        "every other group has 0.\n")
    add("**Pending**: r1-distill / mmlu_pro is included but is a known weakest "
        "cell (T1.1 accuracy ≈ 0.50, n_clean = 599 after heavy truncation). "
        "Its non-win is associated with these characteristics; the causal "
        "interpretation is reserved for the writeup.")
    add("\nSTOP. Awaiting joint review.")
    (OUT / "finding.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Step 8 — synthesis")
    precondition_check()
    t81 = build_T81()
    t82 = build_T82(t81)
    t81.to_csv(OUT / "T8.1.csv", index=False)
    t82.to_csv(OUT / "T8.2.csv", index=False)
    print(f"  wrote T8.1.csv ({len(t81)} rows)")
    print(f"  wrote T8.2.csv ({len(t82)} rows)")

    fig = fig_synthesis_heatmap(t81)
    fig.savefig(OUT / "F8.1.pdf"); plt.close(fig)
    print(f"  wrote F8.1.pdf")

    write_finding(t81, t82)
    print(f"  wrote finding.md")


if __name__ == "__main__":
    main()
