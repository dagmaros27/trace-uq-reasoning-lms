"""
Stage 2 — Inspection. Eyeball-the-data stats, distributions, example dumps.

Outputs per (model, dataset) under results/{model_short}/{dataset}/inspection/:
  - stats.csv                 — counts, accuracies, rates
  - truncation_breakdown.csv  — greedy/sample/verb_conf/p_true
  - tag_parse_status.csv
  - choice_method.csv
  - examples.md               — 2 per category for manual review
  - fig_data_triage.pdf       — Sankey-flavoured triage flow (BOTH models)
  - fig_truncation_heatmap.pdf — questions × generation matrix
  - fig_score_distributions.pdf — confidence scores by correctness
  - fig_joint_baselines.pdf   — P(True) vs verb_conf, coloured by correctness
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import _lib as L


# ─── Stats ────────────────────────────────────────────────────────────────────
def compute_stats(records: list[dict]) -> dict:
    total = len(records)
    all_clean   = [r for r in records if L.is_all_clean(r)]
    greedy_clean = [r for r in records if not L.greedy_truncated(r)]
    parse_fail  = [r for r in records if L.parse_failed(r)]
    truncated   = [r for r in records if L.greedy_truncated(r) or L.n_samples_truncated(r) > 0]

    def acc(subset):
        labeled = [L.is_correct(r) for r in subset if L.is_correct(r) is not None]
        return float(np.mean(labeled)) if labeled else float("nan"), len(labeled)

    acc_overall,  n_lab_overall  = acc(records)
    acc_clean,    n_lab_clean    = acc(all_clean)
    acc_trunc,    n_lab_trunc    = acc(truncated)

    return {
        "n_total":              total,
        "n_all_clean":          len(all_clean),
        "pct_all_clean":        100 * len(all_clean) / total,
        "n_greedy_clean":       len(greedy_clean),
        "pct_greedy_clean":     100 * len(greedy_clean) / total,
        "n_parse_fail":         len(parse_fail),
        "pct_parse_fail":       100 * len(parse_fail) / total,
        "n_truncated_any":      len(truncated),
        "pct_truncated_any":    100 * len(truncated) / total,
        "acc_overall":          acc_overall,
        "acc_all_clean":        acc_clean,
        "acc_truncated":        acc_trunc,
        "n_labeled_overall":    n_lab_overall,
        "n_labeled_clean":      n_lab_clean,
        "n_labeled_truncated":  n_lab_trunc,
    }


def truncation_breakdown(records: list[dict]) -> pd.DataFrame:
    """One row per generation stage; counts of length-stops."""
    rows = []
    n_q = len(records)
    n_g = n_q                          # 1 greedy / question
    n_s = n_q * 10                     # 10 samples / question
    n_v = n_q                          # 1 verb_conf / question
    n_p = n_q                          # 1 p_true / question
    counts = {
        "greedy":    sum(1 for r in records if r["greedy"]["finish_reason"]              == "length"),
        "sample":    sum(1 for r in records for s in r["samples"] if s["finish_reason"]   == "length"),
        "verb_conf": sum(1 for r in records if r["verbalized_confidence"]["finish_reason"] == "length"),
        "p_true":    sum(1 for r in records if r["ptrue"]["finish_reason"]               == "length"),
    }
    denom = {"greedy": n_g, "sample": n_s, "verb_conf": n_v, "p_true": n_p}
    for stage, n in counts.items():
        rows.append({"stage": stage, "n_truncated": n, "n_total_gens": denom[stage],
                     "pct": 100 * n / denom[stage]})
    return pd.DataFrame(rows)


def distribution(records: list[dict], key_path: list[str]) -> pd.Series:
    """key_path like ['greedy', 'tag_parse_status']."""
    vals = []
    for r in records:
        cur = r
        for k in key_path:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        vals.append(cur)
    return pd.Series(vals).fillna("None").value_counts()


# ─── Example dumps ────────────────────────────────────────────────────────────
def pick_examples(records: list[dict], k: int = 2, seed: int = L.SEED) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    correct, incorrect, trunc_greedy, parse_fail = [], [], [], []
    for r in records:
        if L.parse_failed(r):
            parse_fail.append(r)
        elif L.greedy_truncated(r):
            trunc_greedy.append(r)
        elif L.is_correct(r) is True:
            correct.append(r)
        elif L.is_correct(r) is False:
            incorrect.append(r)
    return {
        "correct":         rng.sample(correct,       min(k, len(correct))),
        "incorrect":       rng.sample(incorrect,     min(k, len(incorrect))),
        "truncated_greedy": rng.sample(trunc_greedy,  min(k, len(trunc_greedy))),
        "parse_fail":      rng.sample(parse_fail,    min(k, len(parse_fail))),
    }


def render_examples_md(examples: dict[str, list[dict]], model_short: str) -> str:
    out = [f"# Example records — {model_short}\n",
           f"Seed = {L.SEED}; 2 examples per category. Reasoning traces shown as ",
           "first 1000 + last 600 chars when long.\n"]
    for cat, recs in examples.items():
        out.append(f"\n---\n## {cat.upper()}  (n shown = {len(recs)})\n")
        for r in recs:
            g = r["greedy"]
            trace = g["reasoning_trace"]
            if len(trace) > 1700:
                trace = trace[:1000] + "\n\n[ ... truncated for dump ... ]\n\n" + trace[-600:]
            out.append(f"\n### {r['question_id']}\n")
            out.append(f"**Question**: {r['question'][:400]}{'...' if len(r['question']) > 400 else ''}\n\n")
            out.append("**Options**: " + ", ".join(f"{k}. {v[:60]}" for k, v in r["options"].items()) + "\n\n")
            out.append(f"**Gold**: `{r['gold_answer']}` | **Greedy extracted**: `{g['extracted_choice']}` "
                       f"({g['choice_method']}) | **finish_reason**: `{g['finish_reason']}`\n\n")
            out.append(f"**Verbalized conf**: {r['verbalized_confidence']['parsed_confidence']}  |  "
                       f"**P(True) normalized**: {r['ptrue']['p_true_normalized']}\n\n")
            out.append("**Reasoning trace** (think region):\n\n")
            out.append("```\n" + trace + "\n```\n")
            out.append(f"\n**Final answer**:\n\n> {g['final_answer'][:500]}\n")
    return "".join(out)


# ─── Figures ──────────────────────────────────────────────────────────────────
def fig_data_triage(all_records: dict[str, list[dict]], out_dir: Path):
    """Two-column waterfall: each model, N -> all-clean+correct / all-clean+wrong /
       greedy-truncated / parse-fail. Stacked horizontal bars, value labels."""
    import matplotlib.pyplot as plt
    L.apply_style()

    fig, axes = plt.subplots(1, len(all_records), figsize=(11, 4.0),
                             sharex=False, gridspec_kw={"wspace": 0.35})
    if len(all_records) == 1:
        axes = [axes]

    for ax, (model, records) in zip(axes, all_records.items()):
        total = len(records)
        bins = {
            "clean correct":   sum(1 for r in records if L.is_all_clean(r) and L.is_correct(r) is True),
            "clean incorrect": sum(1 for r in records if L.is_all_clean(r) and L.is_correct(r) is False),
            "trunc (greedy)":  sum(1 for r in records if L.greedy_truncated(r) and not L.parse_failed(r)),
            "trunc (samples)": sum(1 for r in records if not L.greedy_truncated(r) and L.n_samples_truncated(r) > 0),
            "parse fail":      sum(1 for r in records if L.parse_failed(r)),
        }
        colors = [L.PALETTE["correct"], L.PALETTE["incorrect"], "#fdb863", "#e6e6e6", "#9e9e9e"]
        labels = list(bins.keys()); vals = list(bins.values())

        # Horizontal stacked single bar (each labelled inside)
        left = 0
        for v, c, lbl in zip(vals, colors, labels):
            ax.barh(0, v, left=left, color=c, edgecolor="white", linewidth=2, height=0.55)
            if v / total > 0.025:
                ax.text(left + v / 2, 0, f"{lbl}\n{v}", ha="center", va="center",
                        fontsize=10, color="white" if c in (L.PALETTE["correct"], L.PALETTE["incorrect"]) else "#222")
            left += v

        ax.set_xlim(0, total)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel("number of questions")
        ax.set_title(L.MODEL_LABEL[model], loc="left", pad=10)
        ax.set_axisbelow(True); ax.grid(axis="x", alpha=0.3)
        # Headline summary above the bar
        clean = bins["clean correct"] + bins["clean incorrect"]
        ax.text(0, 0.55, f"All-clean: {clean}/{total}  ({100*clean/total:.1f}%)",
                fontsize=10, fontweight="semibold", color="#222")

    fig.suptitle("Stage-2 data triage — where each question lands",
                 fontsize=13, fontweight="bold", y=1.04)
    return fig


def fig_truncation_heatmap(records: list[dict], model_short: str, out_dir: Path):
    """Questions × stages matrix: red cell = generation was truncated.
       Sort questions by total truncations descending so worst rises to the top."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    L.apply_style()
    n = len(records)
    stages = ["greedy"] + [f"s{i+1}" for i in range(10)] + ["verb_conf", "p_true"]
    mat = np.zeros((n, len(stages)), dtype=np.float32)
    for i, r in enumerate(records):
        mat[i, 0] = (r["greedy"]["finish_reason"] == "length")
        for j, s in enumerate(r["samples"]):
            mat[i, 1 + j] = (s["finish_reason"] == "length")
        mat[i, 11] = (r["verbalized_confidence"]["finish_reason"] == "length")
        mat[i, 12] = (r["ptrue"]["finish_reason"] == "length")

    n_trunc_per_q = mat.sum(axis=1)
    order = np.argsort(-n_trunc_per_q, kind="mergesort")
    mat = mat[order]
    n_trunc_per_q = n_trunc_per_q[order]
    n_affected = int((n_trunc_per_q > 0).sum())

    fig = plt.figure(figsize=(11, 4.8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 16], wspace=0.04)
    ax_side = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])

    # Side column: how many truncations per question (horizontal lollipop-ish bars)
    affected_idx = np.arange(n_affected)
    ax_side.barh(affected_idx, n_trunc_per_q[:n_affected], color="#444", height=0.9)
    ax_side.set_xlim(0, 13); ax_side.set_ylim(n, -0.5)
    ax_side.invert_xaxis()
    ax_side.set_xticks([0, 5, 10]); ax_side.tick_params(axis="x", labelsize=8)
    ax_side.set_yticks([]); ax_side.set_xlabel("# trunc.")
    ax_side.set_title("affected\nquestions", loc="left", fontsize=10)

    # Main heatmap: red where truncated, very light grey where clean.
    cmap = mcolors.ListedColormap(["#f5f5f5", "#d73027"])
    ax_main.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
    ax_main.set_xticks(range(len(stages)))
    ax_main.set_xticklabels(stages, rotation=0, fontsize=9)
    ax_main.axvline(0.5, color="#999", lw=0.6)
    ax_main.axvline(10.5, color="#999", lw=0.6)
    ax_main.axvline(11.5, color="#999", lw=0.6)
    ax_main.set_yticks([])
    ax_main.set_ylabel(f"{n} questions, sorted by # truncations")
    ax_main.set_title(f"{L.MODEL_LABEL[model_short]} — truncation map  "
                      f"({n_affected}/{n} questions affected; "
                      f"{int(mat.sum())} truncated generations total)",
                      loc="left")
    return fig


def fig_score_distributions(records: list[dict], model_short: str):
    """Raincloud (half-violin + jittered strip + box) for P(True) and verbal-conf,
       split correct vs incorrect, on the all-clean set."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    L.apply_style()
    rows = []
    for r in records:
        if not L.is_all_clean(r): continue
        c = L.is_correct(r)
        if c is None: continue
        rows.append({
            "correct": "correct" if c else "incorrect",
            "P(True)":           r["ptrue"]["p_true_normalized"],
            "verbalized conf":   None if r["verbalized_confidence"]["parsed_confidence"] is None
                                  else r["verbalized_confidence"]["parsed_confidence"] / 100.0,
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, col in zip(axes, ["P(True)", "verbalized conf"]):
        sub = df[["correct", col]].dropna()
        if sub.empty:
            ax.text(0.5, 0.5, f"no data for {col}", ha="center", va="center", transform=ax.transAxes)
            continue
        order = ["correct", "incorrect"]
        palette = {"correct": L.PALETTE["correct"], "incorrect": L.PALETTE["incorrect"]}

        # Half-violin
        for i, cat in enumerate(order):
            vals = sub[sub["correct"] == cat][col].dropna().values
            if len(vals) == 0: continue
            parts = ax.violinplot(vals, positions=[i], showextrema=False, widths=0.85)
            for body in parts["bodies"]:
                body.set_facecolor(palette[cat]); body.set_alpha(0.35); body.set_edgecolor("none")
                # half violin: cut the right side
                paths = body.get_paths()[0]
                verts = paths.vertices
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, i)

        # Jittered strip
        rng = np.random.default_rng(L.SEED)
        for i, cat in enumerate(order):
            vals = sub[sub["correct"] == cat][col].dropna().values
            jitter = rng.normal(i + 0.18, 0.05, len(vals))
            ax.scatter(jitter, vals, s=10, color=palette[cat], alpha=0.5, edgecolor="none")

        # Box with median + IQR
        for i, cat in enumerate(order):
            vals = sub[sub["correct"] == cat][col].dropna().values
            if len(vals) == 0: continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([i + 0.32, i + 0.32], [q1, q3], color="black", lw=2)
            ax.plot([i + 0.27, i + 0.37], [med, med], color="black", lw=2.5)

        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=10)
        ax.set_title(col, loc="left")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("score (all-clean set)")
    fig.suptitle(f"{L.MODEL_LABEL[model_short]} — baseline confidence by correctness",
                 fontsize=13, fontweight="bold", y=1.02)
    return fig


def fig_joint_baselines(records: list[dict], model_short: str):
    """P(True) vs verbalized conf, coloured by correctness, with marginal hists."""
    import matplotlib.pyplot as plt
    L.apply_style()

    rows = []
    for r in records:
        if not L.is_all_clean(r): continue
        c = L.is_correct(r)
        if c is None: continue
        vc = r["verbalized_confidence"]["parsed_confidence"]
        rows.append({
            "p_true": r["ptrue"]["p_true_normalized"],
            "verb":   None if vc is None else vc / 100.0,
            "correct": c,
        })
    df = pd.DataFrame(rows).dropna()
    if df.empty: return None

    fig = plt.figure(figsize=(7, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1.5], height_ratios=[1.5, 5],
                          hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    for c, col in [(True, L.PALETTE["correct"]), (False, L.PALETTE["incorrect"])]:
        sub = df[df["correct"] == c]
        ax.scatter(sub["p_true"], sub["verb"], s=30, color=col,
                   alpha=0.5, edgecolor="white", linewidth=0.4,
                   label="correct" if c else "incorrect")
        ax_top.hist(sub["p_true"], bins=20, color=col, alpha=0.5)
        ax_right.hist(sub["verb"], bins=20, color=col, alpha=0.5, orientation="horizontal")

    ax.set_xlabel("P(True) — greedy"); ax.set_ylabel("verbalized confidence — greedy")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax_top.set_axis_off(); ax_right.set_axis_off()
    fig.suptitle(f"{L.MODEL_LABEL[model_short]} — baseline agreement & correctness",
                 fontsize=13, fontweight="bold", y=0.98)
    return fig


# ─── Main per-model runner ────────────────────────────────────────────────────
def run_for_model(model_short: str, dataset: str = "medqa") -> dict:
    L.set_seeds()
    out_dir = L.RESULTS_DIR / model_short / dataset / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stage 2 — inspecting {model_short} / {dataset} ===")
    records = L.load_records(model_short, dataset)
    print(f"  loaded {len(records)} records")

    # 1. Stats
    stats = compute_stats(records)
    pd.DataFrame([stats]).to_csv(out_dir / "stats.csv", index=False)
    print("  wrote stats.csv")
    print(f"    all-clean: {stats['n_all_clean']}/{stats['n_total']} ({stats['pct_all_clean']:.1f}%)")
    print(f"    acc on all-clean: {stats['acc_all_clean']:.3f}  vs on truncated: {stats['acc_truncated']:.3f}")

    # 2. Truncation breakdown
    trunc_df = truncation_breakdown(records)
    trunc_df.to_csv(out_dir / "truncation_breakdown.csv", index=False)
    print("  wrote truncation_breakdown.csv:")
    for _, row in trunc_df.iterrows():
        print(f"    {row['stage']:<10s}  {row['n_truncated']:>5d} / {row['n_total_gens']:<6d}  ({row['pct']:.2f}%)")

    # 3. Tag-parse + choice-method distributions
    tag_dist    = distribution(records, ["greedy", "tag_parse_status"])
    choice_dist = distribution(records, ["greedy", "choice_method"])
    tag_dist.to_csv(out_dir    / "tag_parse_status_dist.csv")
    choice_dist.to_csv(out_dir / "choice_method_dist.csv")

    # 4. Examples
    examples = pick_examples(records)
    (out_dir / "examples.md").write_text(render_examples_md(examples, model_short), encoding="utf-8")
    print(f"  wrote examples.md  ({sum(len(v) for v in examples.values())} records)")

    # 5. Figures — functions return fig; we save PDF + close here.
    import matplotlib.pyplot as plt
    for name, fig in [
        (f"fig_truncation_heatmap_{model_short}",   fig_truncation_heatmap(records, model_short, out_dir)),
        (f"fig_score_distributions_{model_short}",  fig_score_distributions(records, model_short)),
        (f"fig_joint_baselines_{model_short}",      fig_joint_baselines(records, model_short)),
    ]:
        if fig is not None:
            L.save_fig(fig, name, subdir="stage2")
            plt.close(fig)
    print(f"  wrote 3 figures to {L.RESULTS_DIR / 'stage2'}")

    return {"model": model_short, "stats": stats,
            "out_dir": str(out_dir)}


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(L.MODELS.keys()))
    ap.add_argument("--dataset", default="medqa")
    args = ap.parse_args()

    L.apply_style()
    L.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = {}
    all_records = {}
    for m in args.models:
        summaries[m] = run_for_model(m, args.dataset)
        all_records[m] = L.load_records(m, args.dataset)

    # Cross-model triage figure (one PDF for both)
    import matplotlib.pyplot as plt
    fig_dt = fig_data_triage(all_records, L.RESULTS_DIR / "stage2")
    if fig_dt is not None:
        L.save_fig(fig_dt, "fig_data_triage", subdir="stage2")
        plt.close(fig_dt)
    print(f"\nWrote cross-model triage figure.")

    print("\n=== Stage 2 done. Per-model summaries: ===")
    for m, s in summaries.items():
        st = s["stats"]
        print(f"  {m}: clean {st['n_all_clean']}/{st['n_total']} "
              f"({st['pct_all_clean']:.1f}%), acc-clean {st['acc_all_clean']:.3f}")


if __name__ == "__main__":
    main()
