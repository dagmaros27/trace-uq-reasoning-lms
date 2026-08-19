"""
Stage 3 — Feature generation.

One row per question, saved as:
  data/features/{model_short}/{dataset}.parquet  (+ .csv mirror)

Features (per spec; all per-question):
  trace_length              — model tokenizer, think region, mean over 10 samples
  hedging_formal            — lexicon matches per token, mean over samples
  hedging_reasoning         — same, reasoning extension lexicon
  hedging_combined          — formal ∪ reasoning, per token, mean
  connector_density         — connectors per token, mean (neutral framing)
  rep_3, rep_4, rep_5       — repetition score per trace, mean over samples
  trace_divergence          — mean pairwise cosine distance among BGE-M3 embeddings
  answer_semantic_entropy   — for MCQ: discrete letter entropy across 10 samples.
                              For free-answer (TriviaQA): NLI-cluster entropy
                              (Kuhn et al. 2023) using bidirectional entailment
                              from a DeBERTa-v3-large-mnli model. Sample answers
                              are clustered by entailment connected-components
                              and Shannon entropy is computed over cluster
                              sizes.
  p_true                    — greedy p_true_normalized
  verbalized_confidence     — greedy parsed/100

Metadata: question_id, model, dataset, kind, correct, in_all_clean,
          greedy_truncated, n_samples_clean, gold_answer, greedy_prediction,
          n_samples_with_prediction.

NaN-explicit: nulls left as NaN; never silently imputed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import _lib as L


# ─── Per-trace, per-feature ───────────────────────────────────────────────────
def features_for_trace(trace_text: str, model_short: str) -> dict:
    """Returns dict with counts + per-token densities for one reasoning trace."""
    n_tok = L.count_tokens(trace_text, model_short)
    counts = {
        "hedging_formal":     L.lex_match_count(trace_text, "hedging_formal"),
        "hedging_reasoning":  L.lex_match_count(trace_text, "hedging_reasoning"),
        "hedging_combined":   L.lex_match_count(trace_text, "hedging_combined"),
        "connectors_logical": L.lex_match_count(trace_text, "connectors_logical"),
    }
    # Per-token density (NaN if trace is empty/zero tokens)
    densities = {f"{k}_density": (v / n_tok if n_tok > 0 else float("nan"))
                 for k, v in counts.items()}
    return {
        "n_tokens": n_tok,
        **counts,
        **densities,
        "rep_3": L.rep_n_score(trace_text, 3),
        "rep_4": L.rep_n_score(trace_text, 4),
        "rep_5": L.rep_n_score(trace_text, 5),
    }


# ─── Embedding pass for trace_divergence ──────────────────────────────────────
EMBEDDER_MODEL = "BAAI/bge-m3"   # 8192-token context, CPU-friendly, no flash-attn dep
                                  # (swapped from jinaai/jina-embeddings-v3 — see README impl notes)


def load_embedder():
    """Load BGE-M3 embedder (CPU is OK; ~2.3 GB)."""
    print(f"Loading {EMBEDDER_MODEL} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER_MODEL)
    return model


def trace_divergence_for_question(traces: list[str], embedder, batch_size: int = 8) -> float:
    """Embed each of the (up to 10) traces, return mean pairwise cosine distance."""
    traces = [t if t else "" for t in traces]
    if len(traces) < 2:
        return float("nan")
    embs = embedder.encode(
        traces, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    # cosine distance = 1 - cosine similarity (unit vectors -> dot product)
    sim = embs @ embs.T
    n = sim.shape[0]
    # upper triangle (excluding diagonal)
    iu = np.triu_indices(n, k=1)
    dists = 1.0 - sim[iu]
    return float(np.mean(dists))


# ─── NLI-based semantic entropy for free-answer datasets ──────────────────────
# Kuhn et al. (Semantic Uncertainty, 2023): cluster samples by bidirectional
# entailment, then compute Shannon entropy over the cluster size distribution.
# We use MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli — a popular
# DeBERTa-v3-large MNLI model, ~440M params (~1.5 GB), runs comfortably on
# A100 alongside BGE-M3.
NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


class _NLI:
    """Lazy-loaded NLI scorer. label_id->name mapping read from the model
    config; we treat "entailment" specifically."""
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        print(f"Loading {NLI_MODEL_ID} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        # Map "entailment" name to its label id (DeBERTa-v3-mnli uses
        # {entailment: 0, neutral: 1, contradiction: 2} but read from config
        # to be safe).
        id2label = self.model.config.id2label  # {0: 'entailment', ...}
        self.entail_id = next(i for i, n in id2label.items()
                              if str(n).lower().startswith("entail"))

    def entail_pairs(self, premises: list[str], hypotheses: list[str],
                     batch_size: int = 32) -> list[bool]:
        """Returns True iff the predicted class is 'entailment' for each (p, h)."""
        import torch
        assert len(premises) == len(hypotheses)
        out = []
        with torch.inference_mode():
            for i in range(0, len(premises), batch_size):
                bp = premises[i:i + batch_size]
                bh = hypotheses[i:i + batch_size]
                enc = self.tokenizer(bp, bh, return_tensors="pt",
                                     truncation=True, padding=True,
                                     max_length=256).to(self.device)
                logits = self.model(**enc).logits
                preds  = logits.argmax(-1).cpu().tolist()
                out.extend([p == self.entail_id for p in preds])
        return out


_nli_singleton: Optional[_NLI] = None  # type: ignore[name-defined]


def get_nli() -> "_NLI":
    global _nli_singleton
    if _nli_singleton is None:
        _nli_singleton = _NLI()
    return _nli_singleton


def _connected_components(n: int, edges) -> list[int]:
    """Standard union-find. edges is iterable of (i, j) i<j."""
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return [find(i) for i in range(n)]


def semantic_entropy_free_answer(
    question: str,
    predictions: list[Optional[str]],
    nli: "_NLI",
) -> tuple[float, int]:
    """Kuhn-style semantic-entropy proxy:
      - drop None predictions
      - for every unordered pair (i, j) with i != j, classify pair (cleaned
        premise = "<question> <answer_i>", hypothesis = "<question> <answer_j>")
        in BOTH directions; an edge i-j exists iff BOTH directions are
        entailment.
      - connected components of the entailment graph = semantic clusters.
      - entropy = -sum (n_k / N) log2(n_k / N).

    Returns (entropy_bits, n_predictions_used). NaN if <2 valid predictions.
    """
    preds = [p for p in predictions if p is not None and str(p).strip()]
    n = len(preds)
    if n < 2:
        return (float("nan") if n == 0 else 0.0), n

    # Build candidate pairs (i < j).
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return 0.0, n

    # Build NLI batches. We frame both sides as a short statement glued to
    # the question to give the NLI model the same context for both legs.
    def frame(a: str) -> str:
        return f"In response to: {question}\nAnswer: {a}"

    p1 = [frame(preds[i]) for (i, _) in pairs]
    h1 = [frame(preds[j]) for (_, j) in pairs]
    p2 = h1[:]
    h2 = p1[:]

    e_fwd = nli.entail_pairs(p1, h1)
    e_bwd = nli.entail_pairs(p2, h2)

    edges = [(pairs[k][0], pairs[k][1])
             for k in range(len(pairs)) if e_fwd[k] and e_bwd[k]]
    comp_ids = _connected_components(n, edges)

    # Cluster sizes
    from collections import Counter
    sizes = list(Counter(comp_ids).values())
    p = np.array(sizes, dtype=np.float64) / float(n)
    h_bits = float(-(p * np.log2(p + 1e-12)).sum())
    return h_bits, n


# ─── Main per-question feature aggregation ────────────────────────────────────
def per_question_row(rec: dict, model_short: str, embedder, nli=None) -> dict:
    """Kind-aware. For MCQ, answer_semantic_entropy is the discrete letter
    entropy across samples. For free_answer, it is the NLI-cluster entropy
    described in the module docstring (requires `nli` to be a loaded _NLI
    instance)."""
    kind = L.record_kind(rec)

    # Trace-side features: aggregated over the samples' reasoning_trace.
    sample_feats = [features_for_trace(s["reasoning_trace"], model_short)
                    for s in rec["samples"]]

    def mean_of(key, default=float("nan")):
        vals = [f[key] for f in sample_feats
                if not np.isnan(f.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else default

    trace_length    = mean_of("n_tokens")
    hed_formal      = mean_of("hedging_formal_density")
    hed_reasoning   = mean_of("hedging_reasoning_density")
    hed_combined    = mean_of("hedging_combined_density")
    conn_density    = mean_of("connectors_logical_density")
    rep_3           = mean_of("rep_3")
    rep_4           = mean_of("rep_4")
    rep_5           = mean_of("rep_5")

    divergence = trace_divergence_for_question(
        [s["reasoning_trace"] for s in rec["samples"]], embedder)

    # Kind-dispatched semantic entropy
    if kind == "mcq":
        letters = [s.get("extracted_choice") for s in rec["samples"]]
        h_bits, n_with_pred = L.letter_entropy(letters)
        greedy_pred = rec["greedy"].get("extracted_choice")
    else:
        # free_answer -- requires nli to be loaded
        preds = [s.get("extracted_prediction") for s in rec["samples"]]
        if nli is None:
            h_bits, n_with_pred = float("nan"), sum(1 for p in preds if p)
        else:
            h_bits, n_with_pred = semantic_entropy_free_answer(
                rec["question"], preds, nli)
        greedy_pred = (rec["greedy"].get("extracted_prediction")
                       or rec["greedy"].get("extracted_choice"))

    # Baselines from greedy
    p_true = rec["ptrue"].get("p_true_normalized")
    vc = rec["verbalized_confidence"].get("parsed_confidence")
    verb_conf = (vc / 100.0) if vc is not None else float("nan")

    correct_or_none = L.is_correct(rec)
    return {
        "question_id":              rec["question_id"],
        "model":                    model_short,
        "dataset":                  rec["dataset"],
        "kind":                     kind,
        # metadata
        "gold_answer":              rec["gold_answer"],
        "greedy_prediction":        greedy_pred,
        # legacy column name kept for back-compat with existing parquets:
        "greedy_choice":            (greedy_pred if kind == "mcq" else None),
        "correct":                  np.nan if correct_or_none is None else bool(correct_or_none),
        "in_all_clean":             L.is_all_clean(rec),
        "greedy_truncated":         L.greedy_truncated(rec),
        "n_samples_clean":          L.n_samples_clean(rec),
        "n_samples_with_prediction": n_with_pred,
        # back-compat alias:
        "n_samples_with_letter":    n_with_pred,
        # trace features
        "trace_length":             trace_length,
        "hedging_formal":           hed_formal,
        "hedging_reasoning":        hed_reasoning,
        "hedging_combined":         hed_combined,
        "connector_density":        conn_density,
        "rep_3":                    rep_3,
        "rep_4":                    rep_4,
        "rep_5":                    rep_5,
        "trace_divergence":         divergence,
        # baselines
        "answer_semantic_entropy":  h_bits if not (isinstance(h_bits, float) and np.isnan(h_bits)) else float("nan"),
        "p_true":                   p_true if p_true is not None else float("nan"),
        "verbalized_confidence":    verb_conf,
    }


def run_for_model(model_short: str, dataset: str, embedder,
                  limit: int = 0, nli=None) -> pd.DataFrame:
    print(f"\n=== Stage 3 — features for {model_short} / {dataset} ===")
    records = L.load_records(model_short, dataset)
    if limit and limit > 0:
        records = records[:limit]

    # If any record is free_answer, we need an NLI model for semantic entropy.
    needs_nli = any(L.record_kind(r) == "free_answer" for r in records)
    if needs_nli and nli is None:
        nli = get_nli()

    print(f"  processing {len(records)} records  "
          f"(kinds present: "
          f"{sorted({L.record_kind(r) for r in records})}) ...")
    t0 = time.time()
    rows = []
    for i, rec in enumerate(records):
        rows.append(per_question_row(rec, model_short, embedder, nli))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(records)}  ({(time.time() - t0)/60:.1f} min elapsed)")
    df = pd.DataFrame(rows)
    print(f"  done in {(time.time() - t0)/60:.1f} min")
    return df


def save(df: pd.DataFrame, model_short: str, dataset: str) -> Path:
    out_dir = L.FEATURES_DIR / model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / f"{dataset}.parquet"
    csv     = out_dir / f"{dataset}.csv"
    df.to_parquet(parquet, index=False)
    df.to_csv(csv, index=False)
    return parquet


def write_manifest(model_short_list: list[str], dataset: str):
    """Stage 3 manifest: lexicon hash + embedder + counts. Per spec."""
    import importlib.metadata as md
    pkgs = ["transformers", "torch", "sentence-transformers",
            "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "pyarrow"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = md.version(p)
        except Exception:
            versions[p] = "n/a"

    manifest = {
        "stage": "stage3_features",
        "dataset": dataset,
        "models": model_short_list,
        "embedder": EMBEDDER_MODEL,
        "nli_model": NLI_MODEL_ID,
        "nli_used_when": "kind == 'free_answer' for semantic-entropy clustering",
        "lexicon_file": str(L.LEXICONS_PATH.relative_to(L.ROOT)),
        "lexicon_sha256": L.lexicons_hash(),
        "lexicon_terms_count": {k: len(v) for k, v in L.all_lexicon_terms().items()},
        "seed": L.SEED,
        "library_versions": versions,
    }
    out = L.PROJECT / "results" / "stage3_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest: {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(L.MODELS.keys()))
    ap.add_argument("--dataset", default="medqa")
    ap.add_argument("--limit", type=int, default=0, help="dev cap (0 = all)")
    args = ap.parse_args()

    L.set_seeds()
    L.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    embedder = load_embedder()

    written = []
    for m in args.models:
        df = run_for_model(m, args.dataset, embedder, args.limit)
        p = save(df, m, args.dataset)
        written.append(str(p.relative_to(L.ROOT)))
        print(f"  wrote {p.name}  ({len(df)} rows, {df.shape[1]} cols)")

    write_manifest(args.models, args.dataset)
    print("\n=== Stage 3 done. ===")
    for p in written: print(f"  {p}")


if __name__ == "__main__":
    main()
