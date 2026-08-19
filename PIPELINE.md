# The Pipeline — What We Did and Why

A narrative walkthrough of the study, from research question to headline
table. Read this before diving into the scripts.

---

## 1. The research question

Reasoning language models (OpenAI o1, DeepSeek-R1, QwQ-32B, Qwen3-think)
generate a long "thinking" trace before committing to an answer. Recent work
shows these models are *worse* at knowing when they're wrong: reasoning
fine-tuning degrades abstention (Kirichenko et al. 2025) and more reasoning
increases overconfidence on wrong answers (Mei et al. 2025). Yet both papers
observe that the *trace itself* shows the doubt — hedging, backtracking,
"wait, actually…" — even when the final answer is stated confidently.

Existing uncertainty-quantification (UQ) methods look only at the final
answer. **Question: can simple, interpretable features of the reasoning trace
predict answer correctness better than the standard final-answer methods?**

## 2. Study design

**Grid.** 5 models × 3 datasets = 15 cells, minus (QwQ-32B, MedQA) which was
not generated for cost reasons → **14 cells**. Every analysis is per-cell;
nothing is pooled, because the hypothesis predicts the effect only in a
specific regime.

| group | models |
|---|---|
| RL-tuned reasoning | Qwen3-4B (thinking on), QwQ-32B |
| distilled reasoning | DeepSeek-R1-Distill-Llama-8B |
| controls (no reasoning fine-tuning) | Qwen3-4B (thinking **off** — same weights!), Llama-3.1-8B-Instruct |

| dataset | format | domain |
|---|---|---|
| MedQA | multiple choice (5) | medical exams |
| MMLU-Pro | multiple choice (~10) | multi-domain, hard |
| TriviaQA | free-form | open-domain trivia |

The Qwen3-4B thinking-on/off pair is the key control: same weights, same
questions, only the reasoning behaviour differs. Any effect that appears with
thinking on but not off is attributable to the reasoning mode, not to the
questions or the base model.

**Per question (stage 1, vLLM on A100s):**
- 1 greedy generation (T=0)
- 10 sampled generations (T=0.7, top_p=0.95)
- 1 verbalised-confidence probe (two-turn, Tian et al. 2023 style: fix the
  answer first, then ask "how confident, 0–100?")
- 1 P(True) probe (reason-then-judge, Kadavath et al. 2022 style)

**Clean set.** A question enters analysis only if no generation was truncated
and the answer was parseable. This costs the most on hard MCQ reasoning cells
(truncation 15–25% on R1-Distill and QwQ at MMLU-Pro); n_clean per cell is
600–1000.

## 3. The features (frozen before evaluation)

Five features, declared from the hypothesis and literature *before* any
performance was measured, then never trimmed:

| feature | what it measures | intuition |
|---|---|---|
| `trace_length` | tokens in the trace | uncertain models reason longer |
| `rep_5` | 1 − unique 5-grams / total 5-grams | looping/backtracking = no confident path |
| `hedging_formal` | hedge-lexicon hits per token | "maybe", "not sure", "wait" |
| `connector_density` | discourse connectors per token | structural churn |
| `trace_divergence` | mean pairwise cosine distance among the 10 samples' BGE-M3 embeddings | unstable reasoning across samples |

Per-sample features are averaged over the 10 samples. The hedge lexicon is
versioned in `lexicons.json`; it includes "wait" because Kirichenko et al.
2025 and Muennighoff et al. 2025 identify it as a reasoning-specific marker —
literature-led, not data-led. A redundancy pass (|r| > 0.95) removed rep_3/
rep_4 (duplicates of rep_5) and hedging variants before freezing; nothing was
removed for performance reasons.

**trace_LR** = logistic regression on the 5 features, fit per cell with
StratifiedKFold(5, shuffle, seed=42), standardised inside training folds
only, producing out-of-fold (OOF) predictions for every question.

## 4. Baselines

| method | source | cost | access |
|---|---|---|---|
| semantic entropy | Kuhn et al. 2023 | 10 generations | black-box |
| P(True) | Kadavath et al. 2022 | 1 extra call | black-box |
| verbalised confidence | Tian et al. 2023 | 1 extra call | black-box |

Semantic entropy is the strong baseline: for MCQ, discrete entropy over the
10 samples' answer letters; for TriviaQA, NLI-cluster entropy (DeBERTa-v3
bidirectional entailment, Kuhn's original recipe). All baselines are
black-box because that is the deployment regime the method targets.

## 5. Evaluation protocol

- **Discrimination:** AUROC (primary), with paired non-parametric bootstrap
  (1000 resamples, same indices applied to both methods, percentile 95% CIs,
  win fractions).
- **Selective prediction:** AURC (area under the risk–coverage curve) and
  accuracy at 80% coverage.
- **Calibration:** ECE (equal-width, 10 bins) — but reported with a
  demonstration that ECE alone is gameable: a constant base-rate predictor
  scores ECE = 0 with AUROC = 0.5. Method ranking therefore uses proper
  scoring rules (Brier, NLL) after Platt-scaling **every** method inside CV
  folds (1-D LR on train folds only — no leakage, no favouritism).

## 6. The analysis chain (scripts ↔ paper sections)

| script | produces | paper |
|---|---|---|
| `_paper_step1_eda.py` | dataset stats, accuracy, trace lengths, truncation (T1.x) | §4.1 |
| `_paper_step2a_corr.py` | feature correlation / redundancy check | §3 |
| `_paper_step2b_perfeat.py` | single-feature AUROC + Cohen's d (T2b.x) | §4.2 |
| `_paper_step2d_lofo.py` | leave-one-feature-out marginal contributions | §4.3 |
| `_paper_step3_freeze.py` | **canonical frozen trace_LR fit → OOF predictions (T3.1)** | §4.3 |
| `_paper_step5_baselines.py` | trace_LR vs 3 baselines, paired bootstrap (T5.x) | §4.4 |
| `_paper_step6_combined.py` | full_LR (trace + SE) upper bound | §4.6 |
| `_paper_step7_calibration.py` | ECE / Brier / NLL, Platt-scaled | §4.7 |
| `_paper_step8_synthesis.py` | the headline synthesis: wins by regime (T8.1) | §4.4, §4.8 |
| `_paper_step2c_greedy.py` | single-greedy-trace variant vs 10-sample variant | §4.5 |
| `_paper_aux_feature_importance.py` | LOFO ranking on the frozen 5-feature model | §4.3 |
| `_paper_aux_greedy_vs_baselines.py` | greedy trace_LR vs baselines table | §4.5 |

Step 3 is the hub: every later step consumes its OOF predictions, so all
methods are compared on identical questions under an identical protocol.

## 7. What we found

**Headline.** trace_LR beats semantic entropy on **3 of the 5 reasoning ×
MCQ cells** — Qwen3-4B on MedQA (+0.094) and MMLU-Pro (+0.097), and QwQ-32B
on MMLU-Pro (+0.147, the largest gap in the study) — with paired-bootstrap
CIs strictly above zero and 100% win fractions. It wins on **0 of the 9
cells outside that regime**. The effect is regime-specific, and the regime
was predicted in advance.

**Controls.** Same-weights Qwen3-4B with thinking *off* shows no effect →
the signal comes from reasoning fine-tuning behaviour, not from question
difficulty or the base model. Non-reasoning Llama-3.1 shows no effect on any
dataset.

**The principled exception.** R1-Distill (reasoning, MCQ) only ties with
semantic entropy. It is the lowest-accuracy model in the study and the only
*distilled* (non-RL-trained) reasoner — consistent with the view that the
reasoning training recipe, not the trace format, drives the signal.

**Free-form boundary.** On TriviaQA semantic entropy wins for every model,
including reasoners: clustering distinct answer strings is a strong signal
when the answer space is open, and short trivia traces carry less structure.

**Feature attribution (LOFO on the frozen model).** rep_5 (repetition) is
the primary signal, hedging_formal second and independent; trace_length is
strong alone but redundant with rep_5; trace_divergence matters little on
MCQ but is the top feature on free-form; connector_density is marginal.

**Efficiency.** A single greedy trace (1 generation instead of 10) loses
only 0.011 AUROC on Qwen3-4B/MMLU-Pro (CI crosses zero — statistically a
tie) and still beats P(True) and verbalised confidence, at one-tenth the
cost of semantic entropy.

**Calibration.** After Platt scaling, trace_LR has the best Brier/NLL on
the trio cells by direction, though those CIs cross zero; the ranking claim
rests on discrimination (AUROC/AURC), and ECE is explicitly retired as a
ranking metric via the base-rate-constant demonstration.

## 8. Honest limitations

- The headline rests on 3 cells; the regime (RL-tuned reasoning × MCQ) is
  small. More RL-trained reasoners are the obvious next test.
- (QwQ-32B, MedQA) was never generated; the reasoning × MCQ group has 5
  cells, not 6.
- The clean-set filter drops truncated (hardest) questions — likely biasing
  *against* trace_LR, but a selection effect nonetheless.
- Uncertainty is treated operationally ("probability of being wrong"); no
  aleatoric/epistemic decomposition.
- Closed-weight reasoning APIs mostly hide the raw trace, which limits
  direct deployment of this method to open-weight models today.
