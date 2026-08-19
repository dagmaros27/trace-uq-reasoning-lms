"""
Step 2a — Correlation pruning (redundancy decision, NOT performance-based).

Outputs into results_for_paper/02_features/:
  corr_<model>.csv               per-model Pearson correlation matrix
                                  (5 files, one per model)
  F2.1.pdf                        main-text heatmap for qwen3-4b
  F2.1.A.pdf                      appendix: heatmap per model (5 panels)
  pruning_decision.md             pairs |r|>0.95, rep_n rule applied,
                                  hedging correlations, trace_length/rep_5
                                  note, PROPOSED survivors (not frozen)
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

OUT = L.PROJECT / "results_for_paper" / "02_features"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS  = ["medqa", "mmlu_pro", "trivia_qa"]
REASONING = ["qwen3-4b", "r1-distill-llama-8b", "qwq-32b"]
CONTROLS  = ["qwen3-4b-nothink", "llama-3.1-8b-instruct"]
MODELS    = REASONING + CONTROLS
SKIP      = {("qwq-32b", "medqa"), ("qwq-32b", "mmlu_pro")}

FEATURES  = ["trace_length", "rep_3", "rep_4", "rep_5",
             "hedging_formal", "hedging_reasoning", "hedging_combined",
             "connector_density", "trace_divergence"]

# Thresholds + pre-declared decisions
HIGH_R    = 0.95
EXPECTED_REPN_KEEP    = "rep_5"
EXPECTED_REPN_DROP    = ["rep_3", "rep_4"]


# ─── helpers ────────────────────────────────────────────────────────────────
def datasets_for(model: str) -> list[str]:
    return [d for d in DATASETS if (model, d) not in SKIP
            and (L.FEATURES_DIR / model / f"{d}.parquet").exists()]


def clean_pool(model: str, dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(L.FEATURES_DIR / model / f"{dataset}.parquet")
    df = df[df["in_all_clean"] & df["correct"].notna()].copy()
    return df[FEATURES + ["dataset", "model"]].dropna(subset=FEATURES)


def per_model_pool(model: str) -> pd.DataFrame:
    parts = [clean_pool(model, d) for d in datasets_for(model)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def per_dataset_corrs(model: str) -> dict[str, pd.DataFrame]:
    """For sign-flip check: corr matrix computed on each dataset of the model."""
    return {d: clean_pool(model, d)[FEATURES].corr(method="pearson")
            for d in datasets_for(model)}


# ─── per-model correlation matrices ─────────────────────────────────────────
def write_per_model_corrs() -> dict[str, pd.DataFrame]:
    pooled = {}
    for m in MODELS:
        pool = per_model_pool(m)
        if pool.empty:
            continue
        C = pool[FEATURES].corr(method="pearson").round(3)
        C.to_csv(OUT / f"corr_{m}.csv")
        pooled[m] = C
        print(f"  wrote corr_{m}.csv  (pooled n={len(pool)}, "
              f"datasets={datasets_for(m)})")
    return pooled


# ─── sign-flip detector ─────────────────────────────────────────────────────
def sign_flip_report(model: str) -> list[dict]:
    """Per-model: for each feature pair, list datasets where the sign of r
    differs from the pooled sign. Returns one record per flipping pair."""
    pds = per_dataset_corrs(model)
    if not pds:
        return []
    pooled = per_model_pool(model)[FEATURES].corr(method="pearson")
    flags = []
    for a, b in combinations(FEATURES, 2):
        r_pool = pooled.loc[a, b]
        if np.isnan(r_pool):
            continue
        signs = {d: np.sign(C.loc[a, b]) for d, C in pds.items()
                 if not np.isnan(C.loc[a, b])}
        if len({s for s in signs.values()}) > 1:
            flags.append({"model": model, "feat_a": a, "feat_b": b,
                          "r_pooled": round(r_pool, 3),
                          "per_dataset": {d: round(C.loc[a, b], 3)
                                           for d, C in pds.items()}})
    return flags


# ─── heatmap plotting ───────────────────────────────────────────────────────
def heatmap(ax, C: pd.DataFrame, title: str):
    A = C.values
    im = ax.imshow(A, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(FEATURES, fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            r = A[i, j]
            if np.isnan(r): continue
            color = "white" if abs(r) > 0.55 else "black"
            ax.text(j, i, f"{r:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    ax.set_title(title, loc="left", fontsize=10)
    return im


def fig_main(pooled_corrs: dict[str, pd.DataFrame]):
    L.apply_style()
    m = "qwen3-4b"
    C = pooled_corrs[m]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = heatmap(ax, C, f"{L.MODEL_LABEL[m]} — feature Pearson r "
                        f"(pooled across that model's datasets)")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    return fig


def fig_appendix(pooled_corrs: dict[str, pd.DataFrame]):
    L.apply_style()
    n = len(pooled_corrs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5.5 * rows))
    axes = np.array(axes).reshape(-1)
    last_im = None
    for ax, (m, C) in zip(axes, pooled_corrs.items()):
        last_im = heatmap(ax, C, L.MODEL_LABEL.get(m, m))
    for ax in axes[n:]:
        ax.axis("off")
    fig.colorbar(last_im, ax=axes[:n].tolist(),
                 fraction=0.04, pad=0.04, shrink=0.7)
    fig.suptitle("Feature Pearson r — pooled per model "
                 "(appendix)", fontsize=11, y=1.00)
    return fig


# ─── pruning_decision.md ────────────────────────────────────────────────────
def write_pruning_decision(pooled_corrs: dict[str, pd.DataFrame]):
    """Build the markdown report. Numbers all sourced from pooled_corrs and
    per-dataset corrs."""
    lines = []
    add = lines.append
    add("# Step 2a — Pruning Decision (PROPOSED, not frozen)\n")
    add("Redundancy view only — *no AUROC or performance signal used here*. "
        "Freezing happens in Step 3 after single-feature AUROC + LOFO.\n")
    add(f"Features in scope ({len(FEATURES)}): "
        + ", ".join(f"`{f}`" for f in FEATURES) + ".\n")

    # ── 1. All pairs |r|>HIGH_R, grouped by pair, across models
    add(f"## 1. Pairs with |r| > {HIGH_R} (across all model pools)\n")
    add("Each row = one feature pair; columns = the per-model pooled "
        "correlations. Bold rows hit the threshold in at least one model.\n")
    header_models = list(pooled_corrs.keys())
    add("| pair | " + " | ".join(header_models) + " | hits threshold? |")
    add("|" + "---|" * (len(header_models) + 2))
    pair_rows = []
    for a, b in combinations(FEATURES, 2):
        rs = [float(pooled_corrs[m].loc[a, b]) for m in header_models]
        hit = any(abs(r) > HIGH_R for r in rs if not np.isnan(r))
        pair_rows.append((a, b, rs, hit))
    # Sort by max-abs r descending so the strongest pairs surface first
    pair_rows.sort(key=lambda t: -max(abs(r) for r in t[2] if not np.isnan(r)))
    for a, b, rs, hit in pair_rows:
        cells = []
        for r in rs:
            if np.isnan(r): cells.append("—")
            elif abs(r) > HIGH_R: cells.append(f"**{r:+.3f}**")
            else: cells.append(f"{r:+.3f}")
        mark = "**YES**" if hit else "no"
        add(f"| `{a}` × `{b}` | " + " | ".join(cells) + f" | {mark} |")
    add("")

    # ── 2. rep_n pre-declared rule
    add("## 2. Pre-declared rule — repetition group (rep_3, rep_4, rep_5)\n")
    add("Rule: keep `rep_5` as the single repetition representative; "
        "drop `rep_3` and `rep_4`. Confirming the high pairwise r:\n")
    add("| model | r(rep_3, rep_5) | r(rep_4, rep_5) | r(rep_3, rep_4) |")
    add("|---|---|---|---|")
    for m in header_models:
        C = pooled_corrs[m]
        add(f"| {m} | {C.loc['rep_3','rep_5']:+.3f} | "
            f"{C.loc['rep_4','rep_5']:+.3f} | "
            f"{C.loc['rep_3','rep_4']:+.3f} |")
    add("\n**Decision (applied):** drop `rep_3`, `rep_4`; keep `rep_5`.")
    add("")

    # ── 3. Hedging correlations
    add("## 3. Hedging variants — correlations only (decision deferred)\n")
    add("`hedging_combined` is the formal+reasoning union; the formal/"
        "reasoning split is a robustness contrast. The numbers below decide "
        "whether the split adds anything independent.\n")
    add("| model | r(formal, combined) | r(reasoning, combined) | "
        "r(formal, reasoning) |")
    add("|---|---|---|---|")
    for m in header_models:
        C = pooled_corrs[m]
        add(f"| {m} | {C.loc['hedging_formal','hedging_combined']:+.3f} | "
            f"{C.loc['hedging_reasoning','hedging_combined']:+.3f} | "
            f"{C.loc['hedging_formal','hedging_reasoning']:+.3f} |")
    add("\n*Decision deferred to Step 3. Headline plan is `hedging_combined` "
        "as primary with the formal/reasoning split kept as a robustness "
        "variant; the formal–reasoning cross-correlation says whether the two "
        "actually carry independent signal.*")
    add("")

    # ── 4. trace_length vs rep_5 — expected ~0.8, NOT redundant
    add("## 4. `trace_length` × `rep_5` — correlated but not redundant\n")
    add("Pre-declared expectation: ~0.8 on qwen3-4b, below the 0.95 "
        "redundancy threshold. Keep BOTH. Actual per-model values:\n")
    add("| model | r(trace_length, rep_5) | exceeds 0.95? |")
    add("|---|---|---|")
    for m in header_models:
        r = float(pooled_corrs[m].loc["trace_length", "rep_5"])
        mark = "**yes — flag**" if abs(r) > HIGH_R else "no"
        add(f"| {m} | {r:+.3f} | {mark} |")
    add("\n**Decision (applied):** keep both `trace_length` and `rep_5`.")
    add("")

    # ── 5. Unanticipated pairs with |r|>0.95
    add("## 5. Unanticipated pairs above threshold (flagged, not auto-applied)\n")
    unanticipated = []
    rep_set = {"rep_3", "rep_4", "rep_5"}
    for a, b, rs, hit in pair_rows:
        if not hit:
            continue
        if {a, b}.issubset(rep_set):
            continue  # rep_n: handled in section 2
        unanticipated.append((a, b, rs))
    if not unanticipated:
        add("None. Every pair above |r|=0.95 is the pre-declared rep_n "
            "block. No further pruning candidates from this matrix.")
    else:
        add("These pairs exceed the threshold on at least one model and were "
            "NOT pre-declared. Reported here for joint review.\n")
        for a, b, rs in unanticipated:
            per_model = ", ".join(f"{m}: {r:+.3f}" for m, r in
                                  zip(header_models, rs) if not np.isnan(r))
            add(f"- `{a}` × `{b}`  →  {per_model}. "
                f"*Proposed representative — pending decision.*")
    add("")

    # ── 6. Sign-flip flags across datasets
    add("## 6. Sign-flip flags (pair sign differs across datasets within a model)\n")
    any_flags = False
    for m in header_models:
        flips = sign_flip_report(m)
        if not flips:
            continue
        any_flags = True
        add(f"\n### {m}")
        for f in flips:
            per = ", ".join(f"{k}: {v:+.3f}" for k, v in f["per_dataset"].items())
            add(f"- `{f['feat_a']}` × `{f['feat_b']}`  pooled r = "
                f"{f['r_pooled']:+.3f};  per-dataset → {per}")
    if not any_flags:
        add("No sign-flipping pairs detected.")
    add("")

    # ── 7. Proposed survivor set
    survivors = [f for f in FEATURES if f not in EXPECTED_REPN_DROP]
    add("## 7. PROPOSED survivor set (not frozen)\n")
    add("After applying the rep_n rule and keeping both `trace_length` and "
        "`rep_5`, the proposed survivors are:\n")
    add("```")
    for f in survivors:
        add(f"  {f}")
    add("```")
    add(f"\nDropped from this matrix: {', '.join('`'+f+'`' for f in EXPECTED_REPN_DROP)}. "
        "Hedging split kept pending Step 3.")
    add("")
    add("---")
    add("STOP. Awaiting joint review of pruning_decision.md before freezing "
        "the feature set in Step 3.")
    (OUT / "pruning_decision.md").write_text("\n".join(lines), encoding="utf-8")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("Step 2a — Correlation pruning")
    pooled = write_per_model_corrs()
    fig1 = fig_main(pooled)
    fig1.savefig(OUT / "F2.1.pdf"); plt.close(fig1)
    print(f"  wrote {(OUT / 'F2.1.pdf').relative_to(L.PROJECT)}")
    figA = fig_appendix(pooled)
    figA.savefig(OUT / "F2.1.A.pdf"); plt.close(figA)
    print(f"  wrote {(OUT / 'F2.1.A.pdf').relative_to(L.PROJECT)}")
    write_pruning_decision(pooled)
    print(f"  wrote {(OUT / 'pruning_decision.md').relative_to(L.PROJECT)}")


if __name__ == "__main__":
    main()
