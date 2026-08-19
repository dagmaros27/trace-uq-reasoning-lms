"""
Stage 1 — Generation pipeline for trace-based UQ.

Implements README_generation_pipeline.md exactly:
  per question →  1 greedy (T=0) + 10 samples (T=0.7, top_p=0.95)
                + 1 verbalized-confidence call (0-100, Tian-style)
                + 1 reason-then-judge P(True) call

Captures top-k logprobs everywhere they may be needed downstream.

Run, e.g.:
    python stage1_generate.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --split test --n-questions 5 --output-dir data/generations

    python stage1_generate.py \
        --model Qwen/Qwen3-4B \
        --split test --n-questions 200 --output-dir data/generations
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# vLLM is imported lazily so --help works on a machine without it
def _lazy_vllm():
    from vllm import LLM, SamplingParams  # noqa: F401
    return LLM, SamplingParams


# ────────────────────────────────────────────────────────────────────────────
# Per-model adapter
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelAdapter:
    """Knobs that vary across reasoning and non-reasoning control models."""
    hf_id: str
    short_name: str
    force_think_prefix: bool          # R1-Distill: yes; Qwen3: no (handled by template)
    enable_thinking_kw: bool          # Qwen3: pass enable_thinking=True; R1: ignore
    chat_template_kwargs: dict = field(default_factory=dict)
    split_strategy: str = "think_tags"  # think_tags | inline_cot
    mcq_instruction: str = ""


MODEL_REGISTRY = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ModelAdapter(
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        short_name="r1-distill-llama-8b",
        force_think_prefix=True,
        enable_thinking_kw=False,
    ),
    "Qwen/Qwen3-4B": ModelAdapter(
        hf_id="Qwen/Qwen3-4B",
        short_name="qwen3-4b",
        force_think_prefix=False,
        enable_thinking_kw=True,
        chat_template_kwargs={"enable_thinking": True},
    ),
    "Qwen/Qwen3-4B:no-think": ModelAdapter(
        hf_id="Qwen/Qwen3-4B",
        short_name="qwen3-4b-nothink",
        force_think_prefix=False,
        enable_thinking_kw=True,
        chat_template_kwargs={"enable_thinking": False},
        split_strategy="inline_cot",
    ),
    "meta-llama/Llama-3.1-8B-Instruct": ModelAdapter(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        short_name="llama-3.1-8b-instruct",
        force_think_prefix=False,
        enable_thinking_kw=False,
        split_strategy="inline_cot",
    ),
    # ─── Phase 4 (2026-06-12): bigger reasoning models. Need TP=2 on 2x A100-40GB. ───
    "Qwen/QwQ-32B": ModelAdapter(
        # Qwen reasoning model, 32B params (~64 GB BF16). Reasons in plain text
        # ending with "Answer: X" -- NO <think>...</think> tags, despite being a
        # reasoning model (verified via 5-Q smoke on 2026-06-12). Uses standard
        # Qwen chat template, no force_think_prefix, no enable_thinking kwarg.
        hf_id="Qwen/QwQ-32B",
        short_name="qwq-32b",
        force_think_prefix=False,
        enable_thinking_kw=False,
        chat_template_kwargs={},
        split_strategy="inline_cot",
    ),
    "openai/gpt-oss-20b": ModelAdapter(
        # OpenAI open-weight reasoning model, 20B (~40 GB BF16). Uses "harmony"
        # channel format (analysis / final). Smoke-test required to decide
        # between think_tags / inline_cot / a new harmony-aware split.
        hf_id="openai/gpt-oss-20b",
        short_name="gpt-oss-20b",
        force_think_prefix=False,
        enable_thinking_kw=False,
        chat_template_kwargs={},
        # Provisional: inline_cot, then promote to a `harmony` strategy once
        # we've seen the actual output text. The split_generation dispatcher
        # falls through to inline_cot's cue-based regex, which is the safest
        # default for unknown formats.
        split_strategy="inline_cot",
    ),
}


# ────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ────────────────────────────────────────────────────────────────────────────
def load_medqa(split: str, data_dir: Path) -> list[dict]:
    """Load MedQA from a local jsonl. Returns list of normalized MCQ records."""
    path = data_dir / "medqa" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"MedQA split not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append({
                "question_id":  f"medqa_{split}_{i:05d}",
                "dataset":      "medqa",
                "kind":         "mcq",
                "answerable":   True,
                "question":     r["question"],
                "options":      r["options"],          # dict, may be A..E
                "gold_answer":  r["answer_idx"],       # letter
                "_raw_index":   i,
            })
    return records


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_mmlu_pro(split: str, data_dir: Path) -> list[dict]:
    """Load MMLU-Pro via HF datasets. 10-option MCQ (some records have 6-9).
    Letters A..J are derived from option list order. cot_content is ignored."""
    from datasets import load_dataset
    # MMLU-Pro provides "test" and "validation"; map our "test"/"dev" cleanly.
    hf_split = {"test": "test", "dev": "validation",
                "validation": "validation"}.get(split, split)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=hf_split)

    records = []
    for i, ex in enumerate(ds):
        opt_list = list(ex["options"])
        if not opt_list:
            continue
        options = {_LETTERS[k]: opt_list[k] for k in range(len(opt_list))}
        records.append({
            "question_id":  f"mmlu_pro_{hf_split}_{i:05d}",
            "dataset":      "mmlu_pro",
            "kind":         "mcq",
            "answerable":   True,
            "question":     ex["question"],
            "options":      options,
            "gold_answer":  ex["answer"],            # already a letter
            "category":     ex.get("category"),
            "src":          ex.get("src"),
            "_raw_index":   i,
        })
    return records


def load_trivia_qa(split: str, data_dir: Path) -> list[dict]:
    """Load TriviaQA (rc.nocontext config) for closed-book free-answer QA.
    Stores `gold_answer` (canonical) + `gold_aliases` (raw aliases) +
    `gold_normalized_aliases` (lowercased, article-stripped) for the
    SQuAD/TriviaQA-style scoring at label-assignment time."""
    from datasets import load_dataset
    hf_split = {"test": "validation",        # TriviaQA test labels not public
                "dev":  "validation",
                "validation": "validation",
                "train": "train"}.get(split, split)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=hf_split)

    records = []
    for i, ex in enumerate(ds):
        ans = ex["answer"]
        records.append({
            "question_id":            f"trivia_qa_{hf_split}_{i:05d}",
            "dataset":                "trivia_qa",
            "kind":                   "free_answer",
            "answerable":             True,
            "question":               ex["question"],
            "options":                None,
            "gold_answer":            ans["value"],
            "gold_normalized_value":  ans["normalized_value"],
            "gold_aliases":           list(ans["aliases"]),
            "gold_normalized_aliases": list(ans["normalized_aliases"]),
            "answer_type":            ans.get("type"),
            "trivia_qa_qid":          ex["question_id"],
            "_raw_index":             i,
        })
    return records


# Dataset registry: short_name -> dict of properties + load fn.
# `kind` is the prompting/extraction style for everything downstream of load.
DATASET_REGISTRY: dict[str, dict] = {
    "medqa": {
        "kind":      "mcq",
        "default_split": "test",
        "load_fn":   load_medqa,
        "needs_local_data": True,
    },
    "mmlu_pro": {
        "kind":      "mcq",
        "default_split": "test",
        "load_fn":   load_mmlu_pro,
        "needs_local_data": False,
    },
    "trivia_qa": {
        "kind":      "free_answer",
        "default_split": "validation",
        "load_fn":   load_trivia_qa,
        "needs_local_data": False,
    },
}


def subsample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Stable seeded subsample, nested: subsample(pool, k, s) is a prefix of
    subsample(pool, K, s) for k <= K. This makes a 5-Q smoke a strict subset
    of a 200-Q run, so smoke records survive resume into the full run."""
    if n <= 0 or n >= len(records):
        return list(records)
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[:n]


# ────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ────────────────────────────────────────────────────────────────────────────
MCQ_INSTRUCTION = (
    "You are answering a multiple-choice question. Reason carefully, then "
    "give your final answer as a single letter (one of the options below) "
    "on the last line in the form 'Answer: X'."
)

CONTROL_COT_MCQ_INSTRUCTION = (
    "Answer the following multiple-choice question. Let's think step by step, "
    "then give your final answer as a single letter on the last line in the "
    "form 'Answer: X'."
)


def render_mcq_user_message(rec: dict, adapter: Optional[ModelAdapter] = None) -> str:
    opt_lines = [f"{k}. {v}" for k, v in rec["options"].items()]
    instruction = (
        adapter.mcq_instruction
        if adapter and adapter.mcq_instruction
        else CONTROL_COT_MCQ_INSTRUCTION
        if adapter and adapter.split_strategy == "inline_cot"
        else MCQ_INSTRUCTION
    )
    return (
        f"{instruction}\n\n"
        f"Question: {rec['question']}\n\n"
        + "\n".join(opt_lines)
    )


def render_verb_conf_user_message(rec: dict, proposed_answer_letter: str) -> str:
    opt_lines = [f"{k}. {v}" for k, v in rec["options"].items()]
    return (
        f"Question: {rec['question']}\n\n"
        + "\n".join(opt_lines)
        + f"\n\nYour proposed answer: {proposed_answer_letter}\n\n"
        "How confident are you, on a scale of 0 to 100, that your proposed "
        "answer is correct? Reply with only the integer score on the final line "
        "in the form 'Confidence: NN'."
    )


def render_ptrue_user_message(rec: dict, proposed_answer_letter: str) -> str:
    opt_lines = [f"{k}. {v}" for k, v in rec["options"].items()]
    proposed_text = rec["options"].get(proposed_answer_letter, "[no choice]")
    return (
        f"Question: {rec['question']}\n\n"
        + "\n".join(opt_lines)
        + f"\n\nProposed answer: {proposed_answer_letter}. {proposed_text}\n\n"
        "Is the proposed answer correct? Reason through it, then on the final "
        "line answer with a single word, exactly 'True' or 'False' (no letter "
        "prefix, no punctuation)."
    )


# ────────────────────────────────────────────────────────────────────────────
# Free-answer (TriviaQA-style) prompts and extraction
# ────────────────────────────────────────────────────────────────────────────
# Standard SQuAD/TriviaQA answer-normalisation: lowercase, strip articles,
# strip punctuation, collapse whitespace. Both predictions and gold aliases
# are passed through this before comparison.
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_RE    = re.compile(r"[^\w\s]")


def normalize_free_answer(text: Optional[str]) -> str:
    if not text:
        return ""
    s = text.strip().lower()
    s = _ARTICLES_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def trivia_match(predicted: Optional[str], normalized_aliases: list[str]) -> bool:
    """Standard TriviaQA correctness: any alias (already normalised) matches
    the normalised prediction."""
    norm = normalize_free_answer(predicted)
    if not norm:
        return False
    # gold list is already normalised by the dataset itself.
    return norm in set(normalized_aliases)


FREE_ANSWER_INSTRUCTION = (
    "Answer the following question. Reason freely first if you need to. "
    "At the very end of your response, output your final short answer "
    "inside <answer>...</answer> tags. Put nothing else inside those tags."
)


def render_free_user_message(rec: dict, adapter: Optional[ModelAdapter] = None) -> str:
    return (
        f"{FREE_ANSWER_INSTRUCTION}\n\n"
        f"Question: {rec['question']}"
    )


def render_free_verb_conf_user_message(rec: dict, proposed_answer: str) -> str:
    return (
        f"Question: {rec['question']}\n\n"
        f"Your proposed answer: {proposed_answer}\n\n"
        "How confident are you, on a scale of 0 to 100, that your proposed "
        "answer is correct? Reply with only the integer score on the final "
        "line in the form 'Confidence: NN'."
    )


def render_free_ptrue_user_message(rec: dict, proposed_answer: str) -> str:
    return (
        f"Question: {rec['question']}\n\n"
        f"Proposed answer: {proposed_answer}\n\n"
        "Is the proposed answer correct? Reason through it, then on the final "
        "line answer with a single word, exactly 'True' or 'False' (no letter "
        "prefix, no punctuation)."
    )


_ANSWER_TAG_RE = re.compile(
    r"<answer\s*>(.*?)</answer\s*>",
    re.IGNORECASE | re.DOTALL,
)


def extract_free_answer(full_or_final: str) -> tuple[Optional[str], str]:
    """
    Pull the last <answer>...</answer> block from the model's output.
    Returns (text, status) where status ∈ {"tag", "fallback_last_line",
    "empty"}.

    If the tag is missing we fall back to "the last non-empty line of the
    output" (lossy, but better than nothing). The tag rate should be very
    high once we condition on the structural instruction in the prompt.
    """
    if not full_or_final:
        return None, "empty"
    matches = _ANSWER_TAG_RE.findall(full_or_final)
    if matches:
        ans = matches[-1].strip()
        return (ans if ans else None), "tag"
    # fallback
    for line in reversed(full_or_final.splitlines()):
        if line.strip():
            return line.strip(), "fallback_last_line"
    return None, "empty"


def build_prompt(tokenizer, user_message: str, adapter: ModelAdapter) -> str:
    """Apply chat template, optionally enable_thinking, optionally force <think>\\n prefix."""
    messages = [{"role": "user", "content": user_message}]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if adapter.enable_thinking_kw:
        template_kwargs.update(adapter.chat_template_kwargs)
    rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
    if adapter.force_think_prefix:
        # Per the R1-Distill card: prefilling "<think>\n" makes it actually reason.
        # We add this to the prompt; the model's first generated token will follow.
        if not rendered.rstrip().endswith("<think>"):
            rendered = rendered + "<think>\n"
    return rendered


# ────────────────────────────────────────────────────────────────────────────
# Byte-level BPE marker cleanup
# ────────────────────────────────────────────────────────────────────────────
# vLLM's incremental decode leaves raw GPT-2-style byte markers in `output.text`
# for some tokenizers (notably Llama-3 / DeepSeek-R1-Distill). We strip them.
_BPE_MARKERS = {"Ġ": " ", "Ċ": "\n", "ĉ": "\t"}


def clean_vllm_text(text: str) -> str:
    if not text:
        return text
    for k, v in _BPE_MARKERS.items():
        if k in text:
            text = text.replace(k, v)
    return text


# ────────────────────────────────────────────────────────────────────────────
# Parsers (think tag, MCQ letter, confidence integer)
# ────────────────────────────────────────────────────────────────────────────
THINK_OPEN_RE  = re.compile(r"<think>",  re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
INLINE_FINAL_RE = re.compile(
    r"(?:^|\n|(?<=[.!?])\s+)(?:"
    r"(?:final\s+)?answer|"
    r"the\s+(?:final\s+)?answer\s+is|"
    r"therefore,\s*the\s+(?:final\s+)?answer\s+is|"
    r"so,\s*the\s+(?:final\s+)?answer\s+is"
    r")\s*[:\-]?\s*(?:\(?\*?\*?\s*[A-J]\b|\\boxed\{\s*[A-J]\s*\})[^\n]*",
    re.IGNORECASE,
)


def split_think_tags(full_output: str, force_think_prefix: bool) -> tuple[str, str, str]:
    """
    Return (reasoning_trace, final_answer, parse_status).
    parse_status ∈ {"strict", "no_open_tag", "no_close_tag", "no_tags"}.

    If force_think_prefix was used, the prefilled "<think>\n" is NOT part of
    full_output (it was in the prompt). So a well-formed generation will
    contain only </think> + the final answer, and the reasoning is everything
    before </think>. We handle both layouts.
    """
    text = full_output
    open_m = THINK_OPEN_RE.search(text)
    close_m = THINK_CLOSE_RE.search(text)

    if force_think_prefix and not open_m:
        # Reasoning starts at position 0 (prefilled <think>\n is upstream)
        if close_m:
            return text[:close_m.start()].strip(), text[close_m.end():].strip(), "strict"
        # No close tag — fall back to "everything is reasoning except the last 1–2 lines"
        return _fallback_split(text, "no_close_tag")

    if open_m and close_m and close_m.start() > open_m.end():
        return (
            text[open_m.end():close_m.start()].strip(),
            text[close_m.end():].strip(),
            "strict",
        )
    if open_m and not close_m:
        return _fallback_split(text[open_m.end():], "no_close_tag")
    if close_m and not open_m:
        return text[:close_m.start()].strip(), text[close_m.end():].strip(), "no_open_tag"
    return _fallback_split(text, "no_tags")


def split_inline_cot(full_output: str) -> tuple[str, str, str]:
    """
    Split zero-shot CoT output into reasoning text and final answer text.

    Non-reasoning controls have no <think> tags. We treat the text before the
    last clear concluding answer statement as the reasoning trace. If the model
    only emits a bare answer, the reasoning trace is empty and the parse status
    records that honestly.
    """
    text = full_output.strip()
    if not text:
        return "", "", "empty"

    matches = list(INLINE_FINAL_RE.finditer(text))
    if matches:
        m = matches[-1]
        reasoning = text[:m.start()].strip()
        final = m.group(0).strip()
        status = "inline_answer_line" if reasoning else "no_reasoning"
        return reasoning, final, status

    # If the final non-empty line contains a parsable choice, use that as the
    # answer statement and keep the preceding body as the trace.
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) >= 2:
        final = lines[-1].strip()
        if any(p.search(final) for p in CHOICE_PATTERNS):
            return "\n".join(lines[:-1]).strip(), final, "inline_last_line"

    # Bare one-line answer. Keep the raw text in final_answer so choice parsing
    # can still work; log as no_reasoning instead of inventing a trace.
    if len(lines) == 1 and any(p.search(lines[0]) for p in CHOICE_PATTERNS):
        return "", lines[0].strip(), "no_reasoning"

    return _fallback_split(text, "inline_fallback")


def split_generation(full_output: str, adapter: ModelAdapter) -> tuple[str, str, str]:
    if adapter.split_strategy == "inline_cot":
        return split_inline_cot(full_output)
    return split_think_tags(full_output, adapter.force_think_prefix)


def _fallback_split(text: str, status: str) -> tuple[str, str, str]:
    """Heuristic: pull the last line as the answer; everything before = reasoning."""
    last_answer_match = re.search(
        r"(?:^|\n)\s*(?:Answer|Final[\s\-]*Answer)\s*[:\-]\s*([^\n]*)",
        text, re.IGNORECASE,
    )
    if last_answer_match:
        ans = last_answer_match.group(0).strip()
        return text[:last_answer_match.start()].strip(), ans, status
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) >= 2:
        return "\n".join(lines[:-1]).strip(), lines[-1].strip(), status
    return text.strip(), "", status


CHOICE_PATTERNS = [
    re.compile(r"\banswer\s*[:=]\s*\(?\*?\*?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"\banswer\s+is\s*\(?\*?\*?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"\\boxed\{\s*([A-J])\s*\}"),
    re.compile(r"\bfinal\s+answer\s*[:=]?\s*\(?\*?\*?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"\boption\s*\(?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"\(\s*([A-J])\s*\)"),
    re.compile(r"\b([A-J])\s*[.\)\]]\s*$", re.MULTILINE),
    re.compile(r"\*\*\s*([A-J])\s*\*\*"),
]


def extract_choice(final_answer: str, valid_letters: set[str]) -> tuple[Optional[str], str]:
    """Return (letter or None, method_used). The pattern bank caps at A-J;
    every individual match is then filtered through `valid_letters` so a
    5-option question (MedQA) won't accidentally pick up an F-J that appears
    incidentally in the text."""
    if not final_answer:
        return None, "empty"
    for pat in CHOICE_PATTERNS:
        for m in pat.finditer(final_answer):
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter, pat.pattern[:40]
    # Last-ditch: any standalone letter token, take the last one
    for m in reversed(list(re.finditer(r"\b([A-J])\b", final_answer))):
        letter = m.group(1).upper()
        if letter in valid_letters:
            return letter, "standalone"
    return None, "unparsed"


CONFIDENCE_RE = re.compile(r"confidence\s*[:=]?\s*(\d{1,3})", re.IGNORECASE)


def parse_confidence(text: str) -> Optional[int]:
    if not text:
        return None
    m = CONFIDENCE_RE.search(text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v
    # Fallback — any standalone integer in [0, 100]
    for m in re.finditer(r"\b(\d{1,3})\b", text):
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v
    return None


# ────────────────────────────────────────────────────────────────────────────
# Tokenizer probes (True/False tokens for P(True))
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class JudgmentTokens:
    true_tokens:  dict[int, str]   # token_id -> readable
    false_tokens: dict[int, str]


def probe_judgment_tokens(tokenizer) -> JudgmentTokens:
    """
    Find token ids that map to literal 'True' (positive verdict) and 'False'
    (negative verdict) in this tokenizer. We collect several casing/leading-
    space spellings because models vary in what they emit first.

    NOTE: we deliberately do NOT include 'A'/'B' / '(A)' / '(B)' here. Earlier
    versions of this script paired the prompt "answer (A) True or (B) False"
    with A/B as judgment tokens; this collided with the MCQ option letters
    A..E and made the position-finder grab MCQ letters from the reasoning
    body instead of the actual verdict on inline-CoT outputs. Standard
    Kadavath setup: literal True / False only.
    """
    true_candidates  = ["True", " True", "true", " true"]
    false_candidates = ["False", " False", "false", " false"]

    def first_token_id(s: str) -> Optional[int]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None

    true_map, false_map = {}, {}
    for s in true_candidates:
        tid = first_token_id(s)
        if tid is not None:
            true_map[tid] = s
    for s in false_candidates:
        tid = first_token_id(s)
        if tid is not None and tid not in true_map:
            false_map[tid] = s

    print("Probed True-encoding tokens:")
    for tid, s in true_map.items():
        print(f"  id={tid:6d}  {s!r}")
    print("Probed False-encoding tokens:")
    for tid, s in false_map.items():
        print(f"  id={tid:6d}  {s!r}")
    return JudgmentTokens(true_tokens=true_map, false_tokens=false_map)


def compute_ptrue(vllm_output, judgment_tokens: JudgmentTokens) -> dict:
    """
    Reason-then-judge: scan token-by-token in the generated output for the
    verdict position, then aggregate the True/False top-k mass there.

    For think-tag models, scanning starts after </think>. For inline-CoT
    controls, scanning starts at the final answer/verdict cue when present, so
    an early "A" or "B" in the reasoning body is not mistaken for the judgment.
    """
    token_ids     = list(vllm_output.token_ids or [])
    logprobs_list = list(vllm_output.logprobs or [])
    if not token_ids or not logprobs_list:
        return {
            "judgment_token_index":   None,
            "judgment_token_id":      None,
            "judgment_token_decoded": None,
            "p_true_token_prob":      0.0,
            "p_false_token_prob":     0.0,
            "p_true_normalized":      None,
        }

    # 1) Decode token pieces and build char offsets for cue detection.
    pieces = []
    cum_lengths = []
    cum = ""
    for i, tid in enumerate(token_ids):
        piece = logprobs_list[i].get(tid) if logprobs_list[i] else None
        decoded = clean_vllm_text(piece.decoded_token if piece else "")
        pieces.append(decoded)
        cum += decoded
        cum_lengths.append(len(cum))

    full_text_lower = cum.lower()
    scan_start_idx = 0

    think_pos = full_text_lower.find("</think>")
    if think_pos >= 0:
        target_char = think_pos + len("</think>")
        for i, end_char in enumerate(cum_lengths):
            if end_char >= target_char:
                scan_start_idx = i + 1
                break
    else:
        verdict_cues = list(re.finditer(
            r"(?:^|\n)\s*(?:answer|final\s+answer|verdict)\s*[:\-]?\s*(?:true|false)",
            cum,
            re.IGNORECASE,
        ))
        if verdict_cues:
            target_char = verdict_cues[-1].start()
            for i, end_char in enumerate(cum_lengths):
                if end_char >= target_char:
                    scan_start_idx = i
                    break

    # 2) Find the first sampled True/False candidate at or after the cue.
    all_judgment_ids = (set(judgment_tokens.true_tokens) |
                       set(judgment_tokens.false_tokens))
    judgment_token_index = None
    for j in range(scan_start_idx, len(token_ids)):
        if token_ids[j] in all_judgment_ids:
            judgment_token_index = j
            break

    # 3) Aggregate top-k true/false probabilities AT the judgment position.
    p_true_total  = 0.0
    p_false_total = 0.0
    sampled_tok_id      = None
    sampled_tok_decoded = None
    if judgment_token_index is not None:
        sampled_tok_id = token_ids[judgment_token_index]
        dist = logprobs_list[judgment_token_index] or {}
        sampled = dist.get(sampled_tok_id)
        sampled_tok_decoded = sampled.decoded_token if sampled else None
        for tid, lp_obj in dist.items():
            p = float(np.exp(lp_obj.logprob))
            if tid in judgment_tokens.true_tokens:
                p_true_total  += p
            elif tid in judgment_tokens.false_tokens:
                p_false_total += p

    denom = p_true_total + p_false_total
    p_true_norm = (p_true_total / denom) if denom > 0 else None

    return {
        "judgment_token_index":   judgment_token_index,
        "judgment_token_id":      sampled_tok_id,
        "judgment_token_decoded": sampled_tok_decoded,
        "p_true_token_prob":      p_true_total,
        "p_false_token_prob":     p_false_total,
        "p_true_normalized":      p_true_norm,
    }


# ────────────────────────────────────────────────────────────────────────────
# Logprob serialization: full per-token (heavy) vs summary stats (light)
# ────────────────────────────────────────────────────────────────────────────
def serialize_logprobs_full(vllm_output):
    """Heavy: convert every position's top-k logprob dict to a JSON-able dict."""
    out = []
    for pos_dict in (vllm_output.logprobs or []):
        if pos_dict is None:
            out.append(None); continue
        out.append({
            str(tid): {
                "logprob":        float(lp.logprob),
                "rank":           getattr(lp, "rank", None),
                "decoded_token":  lp.decoded_token,
            }
            for tid, lp in pos_dict.items()
        })
    return out


def summarize_logprobs(vllm_output) -> dict:
    """
    Lightweight per-generation summary stats. ~4 floats vs ~5 MB.

    - mean_token_entropy_bits: average Shannon entropy of top-k distribution
      across all positions (normalized over the top-k, so this is a LOWER
      BOUND on the true entropy — but consistent across runs).
    - mean_neg_logprob: average -log p(sampled token), aka NLL/token.
      exp() of this is per-token perplexity.
    - max_token_entropy_bits: highest-entropy position (most uncertain step).
    - min_token_logprob: lowest single-token logprob (most surprised step).
    """
    logprobs_list = vllm_output.logprobs or []
    token_ids     = vllm_output.token_ids or []
    if not logprobs_list:
        return {
            "mean_token_entropy_bits": None,
            "max_token_entropy_bits":  None,
            "mean_neg_logprob":        None,
            "min_token_logprob":       None,
            "n_positions":             0,
        }
    entropies, neglogs, logps = [], [], []
    for pos_dict, tid in zip(logprobs_list, token_ids):
        if not pos_dict:
            continue
        lps = np.fromiter((lp.logprob for lp in pos_dict.values()), dtype=np.float64)
        if lps.size == 0:
            continue
        # Renormalize the top-k to a proper distribution
        mx = lps.max()
        ps = np.exp(lps - mx)
        ps = ps / ps.sum()
        # Shannon entropy in bits (avoid log2 of 0)
        h = -np.sum(ps * np.log2(ps + 1e-12))
        entropies.append(float(h))
        if tid in pos_dict:
            lp_sampled = float(pos_dict[tid].logprob)
            logps.append(lp_sampled)
            neglogs.append(-lp_sampled)
    return {
        "mean_token_entropy_bits": float(np.mean(entropies)) if entropies else None,
        "max_token_entropy_bits":  float(np.max(entropies))  if entropies else None,
        "mean_neg_logprob":        float(np.mean(neglogs))   if neglogs   else None,
        "min_token_logprob":       float(np.min(logps))      if logps     else None,
        "n_positions":             len(entropies),
    }


# Set by main() from the CLI flag
LOGPROB_MODE = "summary"   # one of: summary, full, none


def logprob_payload(vllm_output):
    if LOGPROB_MODE == "full":
        return {"logprobs": serialize_logprobs_full(vllm_output)}
    if LOGPROB_MODE == "none":
        return {}
    return {"logprob_summary": summarize_logprobs(vllm_output)}


def output_to_dict(vllm_output) -> dict:
    return {
        "text":          clean_vllm_text(vllm_output.text),
        "finish_reason": vllm_output.finish_reason,
        "token_count":   len(vllm_output.token_ids) if vllm_output.token_ids else 0,
        **logprob_payload(vllm_output),
    }


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF model id; must be in MODEL_REGISTRY")
    ap.add_argument("--dataset", default="medqa",
                    choices=list(DATASET_REGISTRY.keys()))
    ap.add_argument("--split", default=None,
                    help="dataset split to read; defaults to the registry's "
                         "default_split (test for medqa/mmlu_pro, validation "
                         "for trivia_qa).")
    ap.add_argument("--n-questions", type=int, default=1500,
                    help="<=0 means use all in the split")
    ap.add_argument("--seed", type=int, default=42,
                    help="Used for question subsample (fixed across models)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="data/generations")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--sample-temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--logprobs-k", type=int, default=5,
                    help="top-k logprobs to capture per token")
    ap.add_argument("--ptrue-logprobs-k", type=int, default=20,
                    help="broader top-k for the P(True) call so True/False land in top-k")
    ap.add_argument("--logprob-mode", default="summary",
                    choices=["summary", "full", "none"],
                    help="summary: 4 floats/generation. full: per-token top-k (README spec; ~50x heavier). "
                         "none: drop entirely. P(True) always keeps the judgment-position dist.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="Split the model across this many GPUs via vLLM "
                         "tensor parallelism. Use >1 only when the model "
                         "doesn't fit on a single GPU (e.g. 32B on 2x A100-40GB).")
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="Process questions in chunks; partial JSONL appended after each")
    ap.add_argument("--resume", action="store_true",
                    help="Skip question_ids already in the output file")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + first prompt; do not load vLLM")
    ap.add_argument("--ptrue-only", action="store_true",
                    help="Re-score only P(True) on an existing medqa.jsonl. "
                         "Skips greedy/samples/verbalized; reuses the saved "
                         "greedy.extracted_choice per record. Writes a sibling "
                         "<dataset>_<model>.ptrue_v2.jsonl with the same row "
                         "shape but a fresh `ptrue` field.")
    ap.add_argument("--input-jsonl", type=str, default=None,
                    help="Path to the existing per-question jsonl to re-score. "
                         "Required with --ptrue-only.")
    args = ap.parse_args()

    global LOGPROB_MODE
    LOGPROB_MODE = args.logprob_mode

    if args.model not in MODEL_REGISTRY:
        print(f"ERROR: model {args.model!r} not registered. "
              f"Add it to MODEL_REGISTRY first.", file=sys.stderr)
        sys.exit(2)
    adapter = MODEL_REGISTRY[args.model]

    # ── Dataset ───────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    ds_meta = DATASET_REGISTRY[args.dataset]
    ds_kind = ds_meta["kind"]
    split   = args.split or ds_meta["default_split"]
    records = ds_meta["load_fn"](split, data_dir)
    pool = subsample(records, args.n_questions, args.seed)
    print(f"Loaded {len(records)} from {args.dataset}/{split} "
          f"(kind={ds_kind}); using {len(pool)} (seed={args.seed})")

    # ── Output path & resume ──────────────────────────────────────────────
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{adapter.short_name}.jsonl"
    done_ids = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["question_id"])
        print(f"Resume: {len(done_ids)} already in {out_path.name}")
        pool = [r for r in pool if r["question_id"] not in done_ids]

    print(f"Output: {out_path}")
    print(f"Remaining to process: {len(pool)}")

    if args.dry_run:
        # Show what the first prompt would look like and exit.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter.hf_id, trust_remote_code=True)
        rec = pool[0]
        if ds_kind == "mcq":
            user_msg = render_mcq_user_message(rec, adapter)
        else:  # free_answer
            user_msg = render_free_user_message(rec, adapter)
        prompt = build_prompt(tokenizer, user_msg, adapter)
        print("\n" + "=" * 70)
        print(f"DRY RUN — first prompt for {rec['question_id']} "
              f"(kind={ds_kind}):")
        print("=" * 70)
        print(prompt)
        print("=" * 70)
        if ds_kind == "mcq":
            print(f"\nValid letters for this question: "
                  f"{sorted(rec['options'].keys())}")
        else:
            print(f"\nGold answer: {rec['gold_answer']!r}")
            print(f"Gold normalized aliases: "
                  f"{rec.get('gold_normalized_aliases')!r}")
        probe_judgment_tokens(tokenizer)
        return

    # ── vLLM load ─────────────────────────────────────────────────────────
    LLM, SamplingParams = _lazy_vllm()
    print(f"Loading vLLM with {adapter.hf_id} ...")
    llm = LLM(
        model=adapter.hf_id,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    judgment_tokens = probe_judgment_tokens(tokenizer)

    greedy_params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, logprobs=args.logprobs_k,
        n=1,
    )
    sample_params = SamplingParams(
        temperature=args.sample_temp, top_p=args.top_p, max_tokens=args.max_tokens,
        logprobs=args.logprobs_k, n=args.n_samples,
    )
    verb_conf_params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, logprobs=args.logprobs_k, n=1,
    )
    ptrue_params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, logprobs=args.ptrue_logprobs_k, n=1,
    )

    # ── --ptrue-only short-circuit ───────────────────────────────────────
    if args.ptrue_only:
        if not args.input_jsonl:
            print("ERROR: --ptrue-only requires --input-jsonl", file=sys.stderr)
            sys.exit(2)
        in_path = Path(args.input_jsonl)
        if not in_path.exists():
            print(f"ERROR: input jsonl not found: {in_path}", file=sys.stderr)
            sys.exit(2)
        v2_out = in_path.with_suffix(".ptrue_v2.jsonl")
        v2_manifest = in_path.with_suffix(".ptrue_v2_manifest.json")

        # Load every saved record into memory (~1k rows, fine).
        saved = []
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    saved.append(json.loads(line))
        print(f"[ptrue-only] loaded {len(saved)} records from {in_path}")

        # Optional resume: skip qids already in v2_out.
        already = set()
        if args.resume and v2_out.exists():
            with v2_out.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        already.add(json.loads(line)["question_id"])
            print(f"[ptrue-only] resume: {len(already)} already in {v2_out.name}")
        todo = [r for r in saved if r["question_id"] not in already]
        print(f"[ptrue-only] remaining: {len(todo)}")

        v2_manifest.write_text(json.dumps({
            "started_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model":           adapter.hf_id,
            "model_short":     adapter.short_name,
            "source_jsonl":    str(in_path),
            "n_records":       len(saved),
            "ptrue_logprobs_k": args.ptrue_logprobs_k,
            "max_tokens":      args.max_tokens,
            "true_token_ids":  list(judgment_tokens.true_tokens.keys()),
            "false_token_ids": list(judgment_tokens.false_tokens.keys()),
            "prompt_change":   "literal 'True'/'False' (no '(A)/(B)' letter prefix)",
        }, indent=2))

        t_pt_start = time.time()
        for chunk_start in range(0, len(todo), args.chunk_size):
            chunk = todo[chunk_start: chunk_start + args.chunk_size]
            print(f"\n[ptrue-only] chunk {chunk_start // args.chunk_size + 1} "
                  f"({len(chunk)} questions)")

            prompts_pt = []
            for r in chunk:
                # Reuse the saved greedy letter; fall back to '?' if missing.
                letter = (r.get("greedy") or {}).get("extracted_choice") or "?"
                # Rebuild the minimum record shape needed by render_ptrue_user_message.
                stub = {
                    "question": r["question"],
                    "options":  r["options"],
                }
                user_msg = render_ptrue_user_message(stub, letter)
                prompts_pt.append(build_prompt(tokenizer, user_msg, adapter))

            t0 = time.time()
            pt_results = llm.generate(prompts_pt, ptrue_params)
            print(f"  [p_true] done in {time.time() - t0:.1f}s")

            with v2_out.open("a", encoding="utf-8") as f:
                for r, prompt_text, res in zip(chunk, prompts_pt, pt_results):
                    out0 = res.outputs[0]
                    pt = compute_ptrue(out0, judgment_tokens)
                    pt["prompt_used"]   = prompt_text
                    pt["raw_response"]  = clean_vllm_text(out0.text)
                    pt["finish_reason"] = out0.finish_reason
                    new_rec = dict(r)
                    new_rec["ptrue"] = pt
                    new_rec.setdefault("generation_config", {})[
                        "ptrue_v2_rescored_at"
                    ] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
        print(f"\n[ptrue-only] total {time.time() - t_pt_start:.1f}s")
        print(f"[ptrue-only] wrote {v2_out}")
        print(f"[ptrue-only] manifest {v2_manifest}")
        return

    # ── Manifest ─────────────────────────────────────────────────────────
    manifest = {
        "started_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model":           adapter.hf_id,
        "model_short":     adapter.short_name,
        "dataset":         args.dataset,
        "dataset_kind":    ds_kind,
        "split":           split,
        "n_questions":     len(pool),
        "seed":            args.seed,
        "sample_temp":     args.sample_temp,
        "top_p":           args.top_p,
        "n_samples":       args.n_samples,
        "max_tokens":      args.max_tokens,
        "logprobs_k":      args.logprobs_k,
        "ptrue_logprobs_k": args.ptrue_logprobs_k,
        "logprob_mode":    args.logprob_mode,
        "force_think_prefix": adapter.force_think_prefix,
        "enable_thinking_kw": adapter.enable_thinking_kw,
        "chat_template_kwargs": adapter.chat_template_kwargs,
        "split_strategy":  adapter.split_strategy,
        "true_token_ids":   list(judgment_tokens.true_tokens.keys()),
        "false_token_ids":  list(judgment_tokens.false_tokens.keys()),
    }
    manifest_path = out_dir / f"{args.dataset}_{adapter.short_name}_manifest.json"

    # ── Per-chunk processing ──────────────────────────────────────────────
    t_start = time.time()
    truncation_count = 0
    parse_fail_count = 0

    def run_batch(prompts: list[str], params, label: str):
        print(f"  [{label}] {len(prompts)} prompt(s) -> vLLM ...", flush=True)
        t0 = time.time()
        out = llm.generate(prompts, params)
        print(f"  [{label}] done in {time.time() - t0:.1f}s")
        return out

    # ── Per-kind dispatch helpers ─────────────────────────────────────────
    def render_q_user_message(rec):
        if ds_kind == "mcq":
            return render_mcq_user_message(rec, adapter)
        return render_free_user_message(rec, adapter)

    def parse_generation(full: str, rec: dict):
        """Returns (reasoning, final, parse_status, predicted, predicted_method).
        For MCQ: predicted is the letter. For free: predicted is the free-text
        answer extracted from <answer>...</answer> (or last-line fallback)."""
        reasoning, final, tag_status = split_generation(full, adapter)
        if ds_kind == "mcq":
            valid_letters = set(rec["options"].keys())
            pred, method = extract_choice(final, valid_letters)
        else:
            # The <answer>...</answer> tag is supposed to be at the very END
            # of the response (after </think> for reasoning models). We scan
            # the FULL output (`full`), not `final`, because the tag itself
            # may sit just after </think> -- both layouts work either way.
            pred, method = extract_free_answer(full)
        return reasoning, final, tag_status, pred, method

    for chunk_start in range(0, len(pool), args.chunk_size):
        chunk = pool[chunk_start: chunk_start + args.chunk_size]
        print(f"\n=== Chunk {chunk_start // args.chunk_size + 1} "
              f"({len(chunk)} questions, {chunk_start + 1}..{chunk_start + len(chunk)}) ===")

        # 1) Greedy
        prompts_greedy = [build_prompt(tokenizer, render_q_user_message(r), adapter) for r in chunk]
        greedy_results = run_batch(prompts_greedy, greedy_params, "greedy")

        greedy_records = []
        for rec, res in zip(chunk, greedy_results):
            out0 = res.outputs[0]
            full = clean_vllm_text(out0.text)
            reasoning, final, parse_status, pred, pred_method = parse_generation(full, rec)
            if pred is None:
                parse_fail_count += 1
            if out0.finish_reason == "length":
                truncation_count += 1
            greedy_records.append({
                "full_output":         full,
                "reasoning_trace":     reasoning,
                "final_answer":        final,
                # MCQ legacy field kept as alias for backward compat; for
                # free-answer it holds the free-text prediction.
                "extracted_choice":    pred if ds_kind == "mcq" else None,
                "extracted_prediction": pred,
                "choice_method":       pred_method,
                "tag_parse_status":    parse_status,
                "finish_reason":       out0.finish_reason,
                "token_count":         len(out0.token_ids) if out0.token_ids else 0,
                **logprob_payload(out0),
            })

        # 2) Samples (n=N each, single batched call)
        prompts_samples = prompts_greedy
        sample_results = run_batch(prompts_samples, sample_params, "samples")

        all_samples = []
        for rec, res in zip(chunk, sample_results):
            sample_list = []
            for s_out in res.outputs:
                full = clean_vllm_text(s_out.text)
                reasoning, final, parse_status, pred, pred_method = parse_generation(full, rec)
                if pred is None:
                    parse_fail_count += 1
                if s_out.finish_reason == "length":
                    truncation_count += 1
                sample_list.append({
                    "full_output":         full,
                    "reasoning_trace":     reasoning,
                    "final_answer":        final,
                    "extracted_choice":    pred if ds_kind == "mcq" else None,
                    "extracted_prediction": pred,
                    "choice_method":       pred_method,
                    "tag_parse_status":    parse_status,
                    "finish_reason":       s_out.finish_reason,
                    "token_count":         len(s_out.token_ids) if s_out.token_ids else 0,
                    **logprob_payload(s_out),
                })
            all_samples.append(sample_list)

        # 3) Verbalized confidence (uses greedy prediction, or "?" if null)
        prompts_vc = []
        for rec, gr in zip(chunk, greedy_records):
            pred = gr["extracted_prediction"] or "?"
            if ds_kind == "mcq":
                user_msg = render_verb_conf_user_message(rec, pred)
            else:
                user_msg = render_free_verb_conf_user_message(rec, pred)
            prompts_vc.append(build_prompt(tokenizer, user_msg, adapter))
        vc_results = run_batch(prompts_vc, verb_conf_params, "verb_conf")

        verb_conf_records = []
        for prompt_text, res in zip(prompts_vc, vc_results):
            out0 = res.outputs[0]
            full = clean_vllm_text(out0.text)
            _, final, _ = split_generation(full, adapter)
            parsed = parse_confidence(final or full)
            verb_conf_records.append({
                "prompt_used":         prompt_text,
                "raw_response":        full,
                "post_think_text":     final,
                "parsed_confidence":   parsed,
                "finish_reason":       out0.finish_reason,
            })

        # 4) P(True) — reason-then-judge
        prompts_pt = []
        for rec, gr in zip(chunk, greedy_records):
            pred = gr["extracted_prediction"] or "?"
            if ds_kind == "mcq":
                user_msg = render_ptrue_user_message(rec, pred)
            else:
                user_msg = render_free_ptrue_user_message(rec, pred)
            prompts_pt.append(build_prompt(tokenizer, user_msg, adapter))
        pt_results = run_batch(prompts_pt, ptrue_params, "p_true")

        ptrue_records = []
        for prompt_text, res in zip(prompts_pt, pt_results):
            out0 = res.outputs[0]
            pt = compute_ptrue(out0, judgment_tokens)
            pt["prompt_used"]   = prompt_text
            pt["raw_response"]  = clean_vllm_text(out0.text)
            pt["finish_reason"] = out0.finish_reason
            ptrue_records.append(pt)

        # ── Assemble + append ──
        with out_path.open("a", encoding="utf-8") as f:
            for rec, greedy, samples, vc, pt in zip(
                chunk, greedy_records, all_samples, verb_conf_records, ptrue_records
            ):
                record = {
                    "question_id":          rec["question_id"],
                    "dataset":              rec["dataset"],
                    "kind":                 ds_kind,
                    "answerable":           rec["answerable"],
                    "question":             rec["question"],
                    "options":              rec.get("options"),
                    "gold_answer":          rec["gold_answer"],
                    "greedy":               greedy,
                    "samples":              samples,
                    "verbalized_confidence": vc,
                    "ptrue":                pt,
                    "generation_config": {
                        "model":         adapter.hf_id,
                        "greedy_temp":   0.0,
                        "sample_temp":   args.sample_temp,
                        "top_p":         args.top_p,
                        "n_samples":     args.n_samples,
                        "max_tokens":    args.max_tokens,
                        "seed_or_run_id": args.seed,
                        "ptrue_version": "v2",
                    },
                }
                # Carry free-answer gold metadata so Stage 3 can do alias
                # matching without re-loading the HF dataset.
                if ds_kind == "free_answer":
                    record["gold_normalized_value"]   = rec.get("gold_normalized_value")
                    record["gold_aliases"]            = rec.get("gold_aliases")
                    record["gold_normalized_aliases"] = rec.get("gold_normalized_aliases")
                    record["answer_type"]             = rec.get("answer_type")
                # MCQ extras
                if ds_kind == "mcq":
                    if rec.get("category") is not None:
                        record["category"] = rec["category"]
                    if rec.get("src") is not None:
                        record["src"] = rec["src"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        elapsed = time.time() - t_start
        print(f"  Saved chunk. cumulative elapsed: {elapsed/60:.1f} min")

    # ── Manifest ─────────────────────────────────────────────────────────
    manifest["finished_at"]      = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["elapsed_seconds"]  = time.time() - t_start
    manifest["truncation_count"] = truncation_count
    manifest["parse_fail_count"] = parse_fail_count
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {manifest_path}")
    print(f"Truncations : {truncation_count}")
    print(f"Parse fails : {parse_fail_count}")
    print(f"Wall clock  : {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
