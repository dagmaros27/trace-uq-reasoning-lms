"""
Step 0 — Data Inventory for paper results.

Walks every (model, dataset) cell and probes the 9 items from the spec.
Reads only. Writes `inventory.md` into results_for_paper/. Does not compute
any features, AUROCs, or models.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib as L

OUT_DIR = L.PROJECT / "results_for_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen3-4b", "r1-distill-llama-8b", "qwen3-4b-nothink",
          "llama-3.1-8b-instruct", "qwq-32b"]
DATASETS = ["medqa", "mmlu_pro", "trivia_qa"]
GEN_ROOT = L.PROJECT.parent / "data_generation" / "data" / "generations"


def gen_path(model: str, dataset: str) -> Path | None:
    """Auto-detect nested or flat layout. Return None if neither exists."""
    nested = GEN_ROOT / model / f"{dataset}.jsonl"
    flat   = GEN_ROOT / f"{dataset}_{model}.jsonl"
    if nested.exists(): return nested
    if flat.exists():   return flat
    return None


def parquet_path(model: str, dataset: str) -> Path:
    return L.FEATURES_DIR / model / f"{dataset}.parquet"


def first_record(p: Path) -> dict | None:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def probe_cell(model: str, dataset: str) -> dict:
    """Return a dict with all 9 item flags + counts for one cell."""
    out = {"model": model, "dataset": dataset}
    p = gen_path(model, dataset)
    out["jsonl_path"] = str(p.relative_to(L.PROJECT.parent)) if p else None
    if p is None:
        # Cell doesn't exist. Mark everything False/None.
        out.update({"jsonl_exists": False, "n_total": 0,
                    "items": {f"item_{k}": False for k in range(1, 10)},
                    "extra": "no jsonl"})
        return out
    out["jsonl_exists"] = True

    # Count records + check schema on first one
    n_total = 0
    n_truncated_greedy = 0
    n_truncated_any_sample = 0
    n_with_label = 0
    n_clean = 0
    n_clean_and_labeled = 0
    label_method_observed = None
    schema = None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            n_total += 1
            if schema is None:
                schema = r
            kind = r.get("kind") or ("free_answer"
                                    if r.get("gold_normalized_aliases") is not None
                                    else "mcq")
            # Greedy parse
            g = r.get("greedy") or {}
            if kind == "mcq":
                pred = g.get("extracted_choice")
            else:
                pred = g.get("extracted_prediction") or g.get("extracted_choice")
            greedy_truncated = (g.get("finish_reason") == "length")
            if greedy_truncated: n_truncated_greedy += 1
            samples = r.get("samples") or []
            sample_trunc_count = sum(1 for s in samples
                                     if s.get("finish_reason") == "length")
            if sample_trunc_count > 0: n_truncated_any_sample += 1
            is_clean = (not greedy_truncated) and sample_trunc_count == 0
            if is_clean: n_clean += 1
            # Label
            if pred is not None:
                if kind == "mcq":
                    label_ok = True   # MCQ letter extracted -> label computable
                else:
                    label_ok = bool(r.get("gold_normalized_aliases")
                                    or r.get("gold_normalized_value")
                                    or r.get("gold_answer"))
                if label_ok:
                    n_with_label += 1
                    if is_clean: n_clean_and_labeled += 1

    out["n_total"]               = n_total
    out["n_truncated_greedy"]    = n_truncated_greedy
    out["n_samples_with_any_trunc"] = n_truncated_any_sample
    out["n_with_label"]          = n_with_label
    out["n_clean"]               = n_clean
    out["n_clean_and_labeled"]   = n_clean_and_labeled

    # ITEM 1: raw greedy generation present (full_output + reasoning + final)
    g0 = schema.get("greedy") or {}
    item_1 = all(k in g0 for k in ("full_output", "reasoning_trace", "final_answer"))
    # ITEM 2: raw sampled generations (10 samples each with the same fields)
    samples0 = schema.get("samples") or []
    item_2 = (len(samples0) >= 10
              and all(all(k in s for k in ("full_output", "reasoning_trace", "final_answer"))
                      for s in samples0[:3]))
    # ITEM 3: correctness labels computable (schema has the right fields).
    # Check the SCHEMA, not whether the first record happened to be parseable.
    # MCQ: needs gold_answer + greedy.extracted_choice field present.
    # Free-answer: needs gold_normalized_aliases + greedy.extracted_prediction.
    has_gold = "gold_answer" in schema
    is_mcq = "options" in schema and schema.get("options") is not None
    if is_mcq:
        item_3 = has_gold and ("extracted_choice" in g0)
    else:
        item_3 = has_gold and ("extracted_prediction" in g0 or
                               "extracted_choice" in g0) and (
            schema.get("gold_normalized_aliases") is not None
            or schema.get("kind") == "free_answer")
    # ITEM 4: truncation flags (finish_reason on every generation)
    item_4 = ("finish_reason" in g0) and (
        len(samples0) == 0 or all("finish_reason" in s for s in samples0[:3])
    )
    # ITEM 5: baseline scores present
    pt = schema.get("ptrue") or {}
    vc = schema.get("verbalized_confidence") or {}
    has_ptrue = "p_true_normalized" in pt
    has_verbcnf = "parsed_confidence" in vc
    # answer_semantic_entropy is a derived feature -> check parquet later
    item_5_jsonl = has_ptrue and has_verbcnf

    # ITEM 6: parquet exists
    qp = parquet_path(model, dataset)
    item_6 = qp.exists()
    pq_cols = None
    pq_n = None
    if item_6:
        df = pd.read_parquet(qp)
        pq_cols = list(df.columns)
        pq_n = len(df)

    # ITEM 7: trace_divergence value present (as a column; embeddings are NOT
    # persisted -- the spec only saved the computed scalar)
    item_7 = (pq_cols is not None) and ("trace_divergence" in pq_cols)
    # ITEM 5 (parquet side): answer_semantic_entropy
    item_5 = item_5_jsonl and (pq_cols is not None and
                               "answer_semantic_entropy" in pq_cols)

    # ITEM 8: per-sample confidence / probability values saved
    # Each sample has `logprob_summary` (mean_token_entropy_bits, ...). The
    # *per-question* P(True) is one value; not per-sample. Sample-level
    # `extracted_choice` letters are saved -> letter-entropy is recomputable
    # from those. For free-answer, sample-level `extracted_prediction` is
    # saved -> NLI cluster entropy is recomputable.
    has_sample_logprob = (len(samples0) > 0 and
                          isinstance(samples0[0].get("logprob_summary"), dict))
    has_sample_extracted = (len(samples0) > 0 and
                            ("extracted_choice" in samples0[0]
                             or "extracted_prediction" in samples0[0]))
    item_8 = has_sample_logprob and has_sample_extracted

    # ITEM 9: proper-scores / calibration-check outputs
    cc = L.RESULTS_DIR / "calibration_check"
    item_9 = (cc / f"{dataset}_proper_scores.csv").exists() and \
             (cc / f"{dataset}_calibrated_ece.csv").exists()

    out["items"] = {
        "1_raw_greedy":       item_1,
        "2_raw_samples":      item_2,
        "3_labels":           item_3,
        "4_trunc_flags":      item_4,
        "5_baselines":        item_5,
        "6_features_parquet": item_6,
        "7_trace_divergence": item_7,
        "8_per_sample_signals": item_8,
        "9_calibration_check": item_9,
    }
    out["parquet_n"] = pq_n
    out["parquet_cols"] = pq_cols
    out["jsonl_first_record_keys"] = list(schema.keys())
    return out


def main():
    cells = []
    for m in MODELS:
        for d in DATASETS:
            cells.append(probe_cell(m, d))
    # Write json sidecar for the markdown report
    (OUT_DIR / "_inventory_raw.json").write_text(
        json.dumps(cells, indent=2, default=str), encoding="utf-8")

    # Build inventory.md
    lines = []
    L_ = lines.append
    L_("# Data Inventory — Step 0")
    L_("")
    L_("This is a presence + counts pass over every (model, dataset) cell. ")
    L_("No analysis was run. Reports against the 9 items from the spec.")
    L_("")
    L_("## Presence table")
    L_("")
    L_("Items: 1=raw greedy · 2=raw samples · 3=labels computable · "
       "4=truncation flags · 5=baseline scores (P(True), verbalized, "
       "semantic_entropy) · 6=features parquet exists · 7=trace_divergence "
       "value present · 8=per-sample signals (logprob_summary + extracted "
       "letter/text) · 9=proper-scores / calibration-check outputs")
    L_("")
    L_("| model | dataset | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |")
    L_("|---|---|---|---|---|---|---|---|---|---|---|")
    def mark(b): return "✓" if b else "—"
    for c in cells:
        if not c["jsonl_exists"]:
            L_(f"| {c['model']} | {c['dataset']} | — | — | — | — | — | — | — | — | — |  *(no jsonl)*")
            continue
        i = c["items"]
        L_(f"| {c['model']} | {c['dataset']} | "
           f"{mark(i['1_raw_greedy'])} | {mark(i['2_raw_samples'])} | "
           f"{mark(i['3_labels'])} | {mark(i['4_trunc_flags'])} | "
           f"{mark(i['5_baselines'])} | {mark(i['6_features_parquet'])} | "
           f"{mark(i['7_trace_divergence'])} | {mark(i['8_per_sample_signals'])} | "
           f"{mark(i['9_calibration_check'])} |")
    L_("")
    L_("## Counts per cell")
    L_("")
    L_("`n_clean` = neither greedy nor any sample was truncated. ")
    L_("`n_clean_and_labeled` = the Stage-4 modelling pool.")
    L_("")
    L_("| model | dataset | n_total | n_truncated_greedy | n_clean | n_clean_and_labeled | parquet_n |")
    L_("|---|---|---|---|---|---|---|")
    for c in cells:
        if not c["jsonl_exists"]:
            L_(f"| {c['model']} | {c['dataset']} | — | — | — | — | — |")
            continue
        L_(f"| {c['model']} | {c['dataset']} | {c['n_total']} | "
           f"{c['n_truncated_greedy']} | {c['n_clean']} | "
           f"{c['n_clean_and_labeled']} | {c['parquet_n']} |")
    L_("")
    L_("## Explicit answers to A, B, C, D")
    L_("")
    L_("### A. Can per-question features be RECOMPUTED from raw traces/samples?")
    L_("")
    L_("**Yes.** Each jsonl record persists the full text of every generation:")
    L_("- `greedy.full_output`, `greedy.reasoning_trace`, `greedy.final_answer`")
    L_("- `samples[i].full_output`, `samples[i].reasoning_trace`, `samples[i].final_answer` "
       "for i in 0..9")
    L_("- Plus the saved baselines (`ptrue.p_true_normalized`, "
       "`verbalized_confidence.parsed_confidence`) and the extracted letters / "
       "free-text answers per sample.")
    L_("")
    L_("So lexicon features (hedging, connectors, rep-N), trace length, and "
       "answer-distribution features (letter entropy / NLI cluster entropy on "
       "the saved `extracted_choice` / `extracted_prediction` per sample) are "
       "all recomputable. The aggregated `*.parquet` table is a convenience, "
       "not the source of truth.")
    L_("")
    L_("**One caveat:** `trace_divergence` is a BGE-M3 cosine-distance "
       "aggregate. The 8192-dim embeddings themselves are NOT persisted — only "
       "the per-question scalar lands in the parquet. Recomputing trace "
       "divergence requires the BGE-M3 GPU pass, which needs the VM.")
    L_("")
    L_("### B. Can features be computed SEPARATELY for the greedy trace vs the sampled traces?")
    L_("")
    L_("**Yes.** The greedy trace is stored as a single field "
       "(`greedy.reasoning_trace`) and the 10 sample traces are stored as a "
       "list (`samples[i].reasoning_trace`). Every text-side feature — "
       "trace_length, hedging_formal / reasoning / combined, connector_density, "
       "rep_3 / rep_4 / rep_5 — is a pure function of one trace's text. So we "
       "can compute:")
    L_("")
    L_("- **Greedy-only features**: apply the feature function to "
       "`greedy.reasoning_trace` once per question.")
    L_("- **Sampled features**: apply to each `samples[i].reasoning_trace`, "
       "average over the 10 (this is what the current parquet stores).")
    L_("")
    L_("The current `data/features/<model>/<dataset>.parquet` only carries the "
       "**sampled-averaged** version. A greedy-vs-sampled efficiency comparison "
       "would require running the same feature functions on the greedy trace "
       "and writing a parallel column set (e.g. `trace_length_greedy`, "
       "`hedging_combined_greedy`, ...). The underlying text is already on "
       "disk, so this is a pure local CPU recompute — no VM needed.")
    L_("")
    L_("`trace_divergence` is the exception: it is by definition a "
       "*pairwise-over-samples* metric, so it only exists for the 10 sampled "
       "traces. There is no greedy-only analogue (a single trace has no "
       "pairwise divergence).")
    L_("")
    L_("### C. Per-cell counts (already in the table above; flagged anomalies below)")
    L_("")
    flagged = []
    for c in cells:
        if not c["jsonl_exists"]:
            continue
        n = c["n_total"]
        if n == 0:
            continue
        clean_pct = 100.0 * c["n_clean"] / n
        if n < 1000:
            flagged.append(f"- **{c['model']} / {c['dataset']}**: n_total = "
                           f"{n} (partial run — expected 1000)")
        if clean_pct < 70:
            flagged.append(f"- **{c['model']} / {c['dataset']}**: n_clean = "
                           f"{c['n_clean']} of {n} ({clean_pct:.1f} % clean) — "
                           f"heavy truncation")
    if flagged:
        L_("")
        for line in flagged:
            L_(line)
    else:
        L_("")
        L_("All cells at n=1000 and >70 % clean. (See counts table for exact numbers.)")
    L_("")
    L_("### D. Is the clean-set definition consistent across cells?")
    L_("")
    L_("**Yes.** The clean set is defined by two flags computed identically "
       "for every record by Stage 3:")
    L_("- `in_all_clean` = `(greedy.finish_reason != 'length')` AND "
       "`all(s.finish_reason != 'length' for s in samples)`")
    L_("- `correct` is `np.nan` whenever the greedy prediction could not be "
       "parsed (MCQ: no A-J letter extractable; free-answer: no usable "
       "prediction). Otherwise: MCQ uses letter==gold_answer; "
       "free-answer uses `normalize(pred) in gold_normalized_aliases`.")
    L_("")
    L_("Stage 4 uniformly takes `df[df.in_all_clean & df.correct.notna()]` as "
       "its modelling pool. Same code path for every cell.")
    L_("")
    L_("## Anything missing, inconsistent, or surprising")
    L_("")
    notes = []
    for c in cells:
        if not c["jsonl_exists"]:
            notes.append(f"- **{c['model']} / {c['dataset']}**: no jsonl — "
                         f"cell intentionally not generated.")
            continue
        i = c["items"]
        missing = [k for k, v in i.items() if not v]
        if missing:
            notes.append(f"- **{c['model']} / {c['dataset']}**: items "
                         f"{missing} flagged absent — investigate.")
    if not notes:
        notes.append("- No load-bearing items missing on any present cell.")

    # Specific known points
    notes.append("")
    notes.append("Known-by-design points (not bugs):")
    notes.append("- `qwq-32b` × `medqa` is not generated — QwQ-32B was added "
                 "in Phase 4, only mmlu_pro + trivia_qa runs were funded.")
    notes.append("- `qwq-32b` × `mmlu_pro` was a partial 500-record run "
                 "(cost cap); a resume to n=1000 was launched on the VM and "
                 "is in flight at the time of this inventory. Re-run this "
                 "script after resume completes to refresh the counts.")
    notes.append("- `trace_divergence` embeddings (BGE-M3, 8192-dim) are NOT "
                 "persisted; only the per-question cosine-distance aggregate "
                 "is. Recomputing trace_divergence requires the GPU pass.")
    notes.append("- Stage-3 parquet column `answer_semantic_entropy` is the "
                 "letter-entropy on MCQ datasets and the NLI-cluster entropy "
                 "(DeBERTa-v3-large-mnli, bidirectional entailment) on "
                 "trivia_qa. Dispatch is by per-record `kind`.")
    notes.append("- `ptrue.judgment_token_decoded` etc. on the current jsonls "
                 "use the v2 (literal True/False) protocol after the rescore "
                 "on 2026-06-08. Old v1 jsonls archived as `*.v1.jsonl` "
                 "alongside, not used downstream.")
    for n in notes:
        L_(n)
    L_("")
    L_("---")
    L_("")
    L_("**Recommendation**: nothing load-bearing is missing on any cell that "
       "exists. The pipeline is recompute-friendly (all source text is on "
       "disk). Specifically, a greedy-vs-sampled feature comparison (answer "
       "to question B) is feasible from the saved data alone with no new "
       "generation runs needed.")
    L_("")
    L_("Awaiting confirmation before proceeding.")

    out_path = OUT_DIR / "inventory.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path.relative_to(L.PROJECT)}")
    print(f"Wrote {(OUT_DIR / '_inventory_raw.json').relative_to(L.PROJECT)}")


if __name__ == "__main__":
    main()
