"""
Shared utilities for the analysis pipeline (stages 2–5).

Every script imports from here so the loading conventions, clean-set
definition, lexicon hashing, plot styling, and bootstrap routines are
defined once.

Reused/adapted from methodology_poc:
  - hedging lexicon match idea + word-boundary regex pattern (extended here)
  - matplotlib style nudges
Everything else is new for this stricter analysis spec.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# ─── Paths and conventions ───────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]      # D:\new_final_project (or ~/datagen on VM)
PROJECT    = ROOT / "analysis_pipeline"

def _find_gen_root() -> Path:
    """Look for the generations dir. Supports:
       - laptop layout:  <ROOT>/data_generation/data/generations
       - VM/flat layout: <ROOT>/data/generations
       - override:       env var ANALYSIS_GEN_ROOT
    """
    env = os.environ.get("ANALYSIS_GEN_ROOT")
    if env and Path(env).exists():
        return Path(env)
    for c in (ROOT / "data_generation" / "data" / "generations",
              ROOT / "data" / "generations"):
        if c.exists():
            return c
    return ROOT / "data_generation" / "data" / "generations"

GEN_ROOT      = _find_gen_root()
LEXICONS_PATH = PROJECT / "lexicons.json"
FEATURES_DIR  = PROJECT / "data" / "features"
RESULTS_DIR   = PROJECT / "results"

MODELS = {
    # short_name -> HF id (used for tokenizer loading)
    "r1-distill-llama-8b":     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "qwen3-4b":                "Qwen/Qwen3-4B",
    # Non-reasoning controls (Stage 1 generated separately, analysis is segregated)
    "qwen3-4b-nothink":        "Qwen/Qwen3-4B",
    "llama-3.1-8b-instruct":   "meta-llama/Llama-3.1-8B-Instruct",
    # Phase 4: bigger reasoning model. mmlu_pro is n=500 (partial run, cost cap).
    "qwq-32b":                 "Qwen/QwQ-32B",
}
CONTROL_MODELS = ["qwen3-4b-nothink", "llama-3.1-8b-instruct"]
DATASETS = ["medqa", "mmlu_pro", "trivia_qa"]

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ─── Input loading ───────────────────────────────────────────────────────────
def jsonl_path(model_short: str, dataset: str) -> Path:
    """Auto-detect nested or flat input layout."""
    nested = GEN_ROOT / model_short / f"{dataset}.jsonl"
    flat   = GEN_ROOT / f"{dataset}_{model_short}.jsonl"
    if nested.exists():  return nested
    if flat.exists():    return flat
    raise FileNotFoundError(f"No JSONL for {model_short}/{dataset} at {nested} or {flat}")


def manifest_path(model_short: str, dataset: str) -> Path:
    nested = GEN_ROOT / model_short / f"{dataset}_manifest.json"
    flat   = GEN_ROOT / f"{dataset}_{model_short}_manifest.json"
    if nested.exists():  return nested
    if flat.exists():    return flat
    raise FileNotFoundError(f"No manifest for {model_short}/{dataset}")


def load_records(model_short: str, dataset: str = "medqa") -> list[dict]:
    p = jsonl_path(model_short, dataset)
    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_manifest(model_short: str, dataset: str = "medqa") -> dict:
    return json.loads(manifest_path(model_short, dataset).read_text())


# ─── Clean-set / truncation / parse-fail ─────────────────────────────────────
def greedy_truncated(rec: dict) -> bool:
    return rec["greedy"]["finish_reason"] == "length"


def record_kind(rec: dict) -> str:
    """`"mcq"` or `"free_answer"`. Older MedQA records that predate the
    multi-dataset refactor don't carry an explicit `kind` -- those are MCQ."""
    return rec.get("kind") or ("free_answer"
                                if rec.get("gold_normalized_aliases") is not None
                                else "mcq")


def _greedy_prediction(rec: dict):
    """Returns the model's greedy prediction in the right shape per kind.
    For MCQ that's the extracted_choice letter; for free that's the
    extracted_prediction free-text. Either may be None for parse failure."""
    g = rec["greedy"]
    if record_kind(rec) == "mcq":
        return g.get("extracted_choice")
    return g.get("extracted_prediction") or g.get("extracted_choice")


def parse_failed(rec: dict) -> bool:
    return _greedy_prediction(rec) is None


def n_samples_truncated(rec: dict) -> int:
    return sum(1 for s in rec["samples"] if s["finish_reason"] == "length")


def n_samples_clean(rec: dict) -> int:
    return sum(1 for s in rec["samples"] if s["finish_reason"] != "length")


def is_all_clean(rec: dict) -> bool:
    """Per spec: NO generation (greedy + all 10 samples) was truncated."""
    if greedy_truncated(rec):
        return False
    return n_samples_truncated(rec) == 0


# SQuAD/TriviaQA-style answer normalization for free-answer correctness.
import re as _re
_ARTICLES_RE_LIB = _re.compile(r"\b(a|an|the)\b", _re.IGNORECASE)
_PUNCT_RE_LIB    = _re.compile(r"[^\w\s]")


def normalize_free_answer(text):
    if not text:
        return ""
    s = str(text).strip().lower()
    s = _ARTICLES_RE_LIB.sub(" ", s)
    s = _PUNCT_RE_LIB.sub(" ", s)
    return " ".join(s.split())


def is_correct(rec: dict) -> Optional[bool]:
    """None if parse failed (cannot label).

    MCQ:  predicted letter == gold letter.
    Free: normalised(pred) in gold_normalized_aliases (TriviaQA convention)."""
    if parse_failed(rec):
        return None
    kind = record_kind(rec)
    if kind == "mcq":
        return rec["greedy"]["extracted_choice"] == rec["gold_answer"]
    # free_answer
    pred = _greedy_prediction(rec)
    aliases = rec.get("gold_normalized_aliases") or []
    # Some legacy records may only carry gold_answer (canonical). Fall back to
    # normalising both sides if the alias list is missing.
    if not aliases:
        gold = rec.get("gold_normalized_value") or rec.get("gold_answer")
        return normalize_free_answer(pred) == normalize_free_answer(gold)
    return normalize_free_answer(pred) in set(aliases)


# ─── Lexicons ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_lexicons() -> dict:
    return json.loads(LEXICONS_PATH.read_text(encoding="utf-8"))


def lexicons_hash() -> str:
    """SHA-256 of the canonical lexicon file. For the run manifest."""
    return hashlib.sha256(LEXICONS_PATH.read_bytes()).hexdigest()


def all_lexicon_terms() -> dict[str, list[str]]:
    """Return the flat term lists per feature: hedging_formal, hedging_reasoning,
       hedging_combined, connectors_logical.
    v2.0: hedging_formal = relational ONLY (propositional dropped).
    Backward-compatible with v1.0 files that still have 'propositional'."""
    lex = load_lexicons()
    formal_block = lex["hedging_formal"]
    formal_terms = formal_block.get("relational", [])
    # v1.0 fallback: if the file still carries propositional, merge it. v2.0 files don't.
    if "propositional" in formal_block:
        formal_terms = formal_terms + formal_block["propositional"]
    reasoning_terms = lex["hedging_reasoning"]["terms"]
    connectors      = lex["connectors_logical"]["terms"]
    return {
        "hedging_formal":     formal_terms,
        "hedging_reasoning":  reasoning_terms,
        "hedging_combined":   sorted(set(formal_terms + reasoning_terms), key=len, reverse=True),
        "connectors_logical": connectors,
    }


def _build_lex_pattern(terms: Iterable[str]) -> re.Pattern:
    """Case-insensitive, word-boundary-aware, multi-word phrase-aware."""
    sorted_terms = sorted(set(terms), key=len, reverse=True)
    parts = []
    for t in sorted_terms:
        esc = re.escape(t.lower())
        if " " in t:
            parts.append(esc)
        else:
            parts.append(rf"\b{esc}\b")
    return re.compile("|".join(parts), re.IGNORECASE)


@lru_cache(maxsize=8)
def _pattern_for(feature_name: str) -> re.Pattern:
    return _build_lex_pattern(all_lexicon_terms()[feature_name])


def lex_match_count(text: str, feature_name: str) -> int:
    if not text:
        return 0
    return len(_pattern_for(feature_name).findall(text.lower()))


def lex_match_terms(text: str, feature_name: str) -> Counter:
    """For inspection: which exact terms matched, how many times each."""
    if not text:
        return Counter()
    return Counter(m.group(0).lower() for m in _pattern_for(feature_name).finditer(text.lower()))


# ─── Per-trace token utilities ───────────────────────────────────────────────
@lru_cache(maxsize=4)
def get_tokenizer(model_short: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODELS[model_short], trust_remote_code=True)


def count_tokens(text: str, model_short: str) -> int:
    if not text:
        return 0
    return len(get_tokenizer(model_short).encode(text, add_special_tokens=False))


# ─── Repetition (rep-N) ──────────────────────────────────────────────────────
_WS_RE = re.compile(r"\s+")


def _normalize_for_ngrams(text: str) -> list[str]:
    """Lowercase, collapse whitespace, whitespace-tokenize. Per spec: 'lowercase
       + collapse consecutive whitespace to a single space, then tokenize for
       n-grams'. Standard whitespace split matches Welleck et al. 2020."""
    if not text:
        return []
    return _WS_RE.sub(" ", text.lower()).strip().split(" ")


def rep_n_score(text: str, n: int) -> float:
    """Repetition score: 1 − distinct n-grams / total n-grams. 0 = fully distinct.
       Trace with < n tokens returns 0 per spec."""
    toks = _normalize_for_ngrams(text)
    if len(toks) < n:
        return 0.0
    ngrams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    total = len(ngrams)
    if total == 0:
        return 0.0
    return 1.0 - (len(set(ngrams)) / total)


# ─── Answer-letter entropy ───────────────────────────────────────────────────
def letter_entropy(letters: Sequence[Optional[str]]) -> tuple[float, int]:
    """Shannon entropy (bits) over the discrete choice distribution across samples.
       Excludes None per our locked policy. Returns (entropy_bits, n_with_letter)."""
    valid = [l for l in letters if l is not None]
    n = len(valid)
    if n == 0:
        return float("nan"), 0
    counts = Counter(valid)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h, n


# ─── Bootstrap utilities ─────────────────────────────────────────────────────
def bootstrap_auroc_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int = 1000,
                       seed: int = SEED, ci: float = 0.95) -> dict:
    """Return point AUROC + bootstrap CI. Resamples paired (score, label) with replacement."""
    from sklearn.metrics import roc_auc_score
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    mask = ~np.isnan(scores)
    scores, labels = scores[mask], labels[mask]
    if len(set(labels)) < 2 or len(scores) == 0:
        return {"auroc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n": len(scores), "boots": np.array([])}
    point = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    n = len(scores)
    boots = np.empty(n_boot)
    valid = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        s, l = scores[idx], labels[idx]
        if len(set(l)) < 2:
            boots[i] = np.nan; continue
        boots[i] = roc_auc_score(l, s)
        valid += 1
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [(1 - ci) * 50, 100 - (1 - ci) * 50])
    return {"auroc": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n": n, "boots": boots}


def bootstrap_auroc_diff(s_a: np.ndarray, s_b: np.ndarray, labels: np.ndarray,
                         n_boot: int = 1000, seed: int = SEED) -> dict:
    """Paired bootstrap of AUROC(A) - AUROC(B). Returns dict with median,
       95% CI of the difference, and win-fraction P(A > B)."""
    from sklearn.metrics import roc_auc_score
    s_a = np.asarray(s_a, dtype=float)
    s_b = np.asarray(s_b, dtype=float)
    labels = np.asarray(labels)
    mask = ~(np.isnan(s_a) | np.isnan(s_b))
    s_a, s_b, labels = s_a[mask], s_b[mask], labels[mask]
    if len(set(labels)) < 2 or len(s_a) == 0:
        return {"diff_median": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "win_fraction": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    n = len(labels)
    valid = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        l = labels[idx]
        if len(set(l)) < 2:
            diffs[i] = np.nan; continue
        diffs[i] = roc_auc_score(l, s_a[idx]) - roc_auc_score(l, s_b[idx])
        valid += 1
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    win = float((diffs > 0).mean())
    return {"diff_median": float(np.median(diffs)), "ci_lo": float(lo),
            "ci_hi": float(hi), "win_fraction": win, "n": n}


# ─── ECE ─────────────────────────────────────────────────────────────────────
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> dict:
    """Equal-width bin ECE. Returns ECE and per-bin info for reliability diagrams."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = ~np.isnan(probs)
    probs, labels = probs[mask], labels[mask]
    if len(probs) == 0:
        return {"ece": float("nan"), "n_bins": n_bins,
                "bin_centers": np.array([]), "bin_acc": np.array([]),
                "bin_conf": np.array([]), "bin_count": np.array([])}
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    ece = 0.0
    total = len(probs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        n = int(in_bin.sum())
        centers.append((lo + hi) / 2)
        counts.append(n)
        if n == 0:
            accs.append(float("nan")); confs.append(float("nan")); continue
        acc = float(labels[in_bin].mean()); conf = float(probs[in_bin].mean())
        accs.append(acc); confs.append(conf)
        ece += (n / total) * abs(acc - conf)
    return {"ece": ece, "n_bins": n_bins, "bin_centers": np.array(centers),
            "bin_acc": np.array(accs), "bin_conf": np.array(confs),
            "bin_count": np.array(counts)}


# ─── Risk-coverage curve ────────────────────────────────────────────────────
def risk_coverage_curve(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Sort by confidence descending; sweep coverage from 1/N -> 1 and compute
       running error rate. Returns AURC and accuracy at 80% coverage."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = ~np.isnan(scores)
    scores, labels = scores[mask], labels[mask]
    n = len(scores)
    if n == 0:
        return {"coverage": np.array([]), "risk": np.array([]),
                "aurc": float("nan"), "acc_at_80": float("nan")}
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    cum_correct = np.cumsum(labels_sorted)
    cov = np.arange(1, n + 1) / n
    risk = 1.0 - cum_correct / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, cov))   # NumPy 2.x rename of np.trapz
    idx80 = int(np.argmin(np.abs(cov - 0.8)))
    acc_at_80 = float(1.0 - risk[idx80])
    return {"coverage": cov, "risk": risk, "aurc": aurc, "acc_at_80": acc_at_80}


# ─── Plot style ──────────────────────────────────────────────────────────────
PALETTE = {
    "correct":         "#1a9850",
    "incorrect":       "#d73027",
    "neutral":         "#525252",
    "trace":           "#2c7fb8",
    "baseline":        "#fdae6b",
    "highlight":       "#762a83",
    "muted":           "#bdbdbd",
}
MODEL_COLOR = {
    "r1-distill-llama-8b":     "#1f78b4",
    "qwen3-4b":                "#e31a1c",
    "qwen3-4b-nothink":        "#fb9a99",
    "llama-3.1-8b-instruct":   "#33a02c",
    "qwq-32b":                 "#6a3d9a",
}
MODEL_LABEL = {
    "r1-distill-llama-8b":     "DeepSeek-R1-Distill-Llama-8B",
    "qwen3-4b":                "Qwen3-4B",
    "qwen3-4b-nothink":        "Qwen3-4B (thinking off, CoT)",
    "llama-3.1-8b-instruct":   "Llama-3.1-8B-Instruct (CoT)",
    "qwq-32b":                 "QwQ-32B",
}

# Paper-friendly display labels. Hedging variants kept as "Hedging (X)" per user
# request (no fancy "epistemic"/"backtracking" rebranding). All other internal
# names are mapped to descriptive labels for figures and tables.
DISPLAY_LABELS = {
    # features
    "trace_length":             "Trace Length (tokens)",
    "hedging_formal":           "Hedging (formal)",
    "hedging_reasoning":        "Hedging (reasoning)",
    "hedging_combined":         "Hedging (combined)",
    "connector_density":        "Logical Connectors",
    "rep_3":                    "3-gram Repetition",
    "rep_4":                    "4-gram Repetition",
    "rep_5":                    "5-gram Repetition",
    "trace_divergence":         "Trace Divergence",
    "answer_semantic_entropy":  "Semantic Entropy",
    "p_true":                   "P(True)",
    "verbalized_confidence":    "Verbalized Confidence",
    # methods / LR models
    "trace_LR":                 "Trace-Feature Model",
    "trace_LR_combined":        "Trace-Feature Model",        # legacy reasoning name
    "trace_LR_split":           "Trace-Feature Model (hedges-split variant)",
    "full_LR":                  "Trace + baselines (combined)",
    # metadata column for correlation matrix
    "correct":                  "correct (label)",
}

def label_for(name: str) -> str:
    """Return paper-friendly label for an internal feature/method name."""
    return DISPLAY_LABELS.get(name, name)


def apply_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi":          120,
        "savefig.dpi":         200,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.15,
        "font.family":         "DejaVu Sans",
        "font.size":           10.5,
        "axes.titlesize":      12,
        "axes.labelsize":      11,
        "axes.titleweight":    "semibold",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.0,
        "axes.edgecolor":      "#222222",
        "axes.titlepad":       12,
        "axes.labelpad":       6,
        "axes.grid":           True,
        "grid.color":          "#cccccc",
        "grid.linestyle":      "--",
        "grid.linewidth":      0.7,
        "grid.alpha":          0.5,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      False,
        "legend.fontsize":     9.5,
        "lines.linewidth":     1.8,
    })


def save_fig(fig, name: str, subdir: Optional[str] = None):
    """Save a figure as **PDF only** (vector, per spec). Does NOT close the
    figure — caller is responsible. Returns the PDF path."""
    out_dir = RESULTS_DIR if subdir is None else RESULTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf)
    return pdf
