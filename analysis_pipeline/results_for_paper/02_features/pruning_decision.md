# Step 2a — Pruning Decision (PROPOSED, not frozen)

Redundancy view only — *no AUROC or performance signal used here*. Freezing happens in Step 3 after single-feature AUROC + LOFO.

Features in scope (9): `trace_length`, `rep_3`, `rep_4`, `rep_5`, `hedging_formal`, `hedging_reasoning`, `hedging_combined`, `connector_density`, `trace_divergence`.

## 1. Pairs with |r| > 0.95 (across all model pools)

Each row = one feature pair; columns = the per-model pooled correlations. Bold rows hit the threshold in at least one model.

| pair | qwen3-4b | r1-distill-llama-8b | qwq-32b | qwen3-4b-nothink | llama-3.1-8b-instruct | hits threshold? |
|---|---|---|---|---|---|---|
| `hedging_formal` × `hedging_combined` | **+0.951** | **+0.987** | +0.914 | **+0.993** | **+0.995** | **YES** |
| `rep_4` × `rep_5` | **+0.993** | **+0.992** | **+0.984** | **+0.986** | **+0.989** | **YES** |
| `rep_3` × `rep_4` | **+0.991** | **+0.989** | **+0.982** | **+0.978** | **+0.985** | **YES** |
| `rep_3` × `rep_5` | **+0.969** | **+0.964** | +0.940 | +0.937 | **+0.954** | **YES** |
| `trace_length` × `rep_3` | +0.750 | +0.735 | +0.848 | +0.628 | +0.387 | no |
| `trace_length` × `rep_4` | +0.707 | +0.707 | +0.818 | +0.565 | +0.379 | no |
| `hedging_reasoning` × `hedging_combined` | +0.816 | +0.594 | +0.781 | +0.074 | +0.091 | no |
| `trace_length` × `rep_5` | +0.661 | +0.677 | +0.773 | +0.521 | +0.381 | no |
| `trace_length` × `hedging_reasoning` | -0.082 | +0.058 | +0.727 | +0.014 | -0.067 | no |
| `rep_3` × `hedging_reasoning` | +0.333 | +0.348 | +0.680 | +0.077 | +0.079 | no |
| `rep_4` × `hedging_reasoning` | +0.357 | +0.328 | +0.632 | +0.088 | +0.087 | no |
| `hedging_reasoning` × `trace_divergence` | +0.593 | +0.394 | +0.605 | +0.148 | +0.062 | no |
| `hedging_formal` × `hedging_reasoning` | +0.604 | +0.462 | +0.463 | -0.047 | -0.013 | no |
| `rep_5` × `hedging_reasoning` | +0.376 | +0.308 | +0.576 | +0.092 | +0.087 | no |
| `trace_length` × `hedging_formal` | -0.340 | -0.508 | +0.167 | +0.131 | -0.060 | no |
| `hedging_combined` × `trace_divergence` | +0.470 | +0.440 | +0.495 | +0.021 | +0.155 | no |
| `connector_density` × `trace_divergence` | -0.476 | -0.265 | -0.402 | -0.107 | -0.170 | no |
| `trace_length` × `hedging_combined` | -0.286 | -0.455 | +0.452 | +0.133 | -0.066 | no |
| `rep_3` × `connector_density` | +0.019 | +0.364 | -0.050 | +0.279 | +0.434 | no |
| `hedging_formal` × `trace_divergence` | +0.344 | +0.414 | +0.317 | +0.003 | +0.150 | no |
| `rep_4` × `connector_density` | -0.003 | +0.365 | -0.068 | +0.240 | +0.405 | no |
| `trace_length` × `trace_divergence` | -0.081 | -0.193 | +0.398 | +0.073 | +0.035 | no |
| `hedging_combined` × `connector_density` | -0.394 | -0.023 | -0.372 | -0.008 | -0.175 | no |
| `hedging_reasoning` × `connector_density` | -0.379 | +0.188 | -0.317 | +0.049 | +0.060 | no |
| `rep_5` × `connector_density` | -0.024 | +0.362 | -0.080 | +0.212 | +0.375 | no |
| `rep_3` × `hedging_combined` | +0.097 | -0.146 | +0.371 | +0.092 | -0.175 | no |
| `hedging_formal` × `connector_density` | -0.366 | -0.082 | -0.334 | -0.014 | -0.182 | no |
| `rep_4` × `hedging_combined` | +0.137 | -0.138 | +0.344 | +0.077 | -0.161 | no |
| `rep_3` × `trace_divergence` | +0.177 | -0.065 | +0.341 | -0.069 | -0.217 | no |
| `rep_4` × `trace_divergence` | +0.205 | -0.060 | +0.340 | -0.054 | -0.184 | no |
| `rep_5` × `trace_divergence` | +0.231 | -0.056 | +0.327 | -0.043 | -0.153 | no |
| `rep_5` × `hedging_combined` | +0.168 | -0.130 | +0.312 | +0.056 | -0.146 | no |
| `rep_3` × `hedging_formal` | -0.031 | -0.232 | +0.080 | +0.083 | -0.184 | no |
| `rep_4` × `hedging_formal` | +0.010 | -0.221 | +0.072 | +0.067 | -0.171 | no |
| `rep_5` × `hedging_formal` | +0.042 | -0.208 | +0.062 | +0.045 | -0.156 | no |
| `trace_length` × `connector_density` | +0.152 | +0.134 | -0.172 | +0.118 | -0.040 | no |

## 2. Pre-declared rule — repetition group (rep_3, rep_4, rep_5)

Rule: keep `rep_5` as the single repetition representative; drop `rep_3` and `rep_4`. Confirming the high pairwise r:

| model | r(rep_3, rep_5) | r(rep_4, rep_5) | r(rep_3, rep_4) |
|---|---|---|---|
| qwen3-4b | +0.969 | +0.993 | +0.991 |
| r1-distill-llama-8b | +0.964 | +0.992 | +0.989 |
| qwq-32b | +0.940 | +0.984 | +0.982 |
| qwen3-4b-nothink | +0.937 | +0.986 | +0.978 |
| llama-3.1-8b-instruct | +0.954 | +0.989 | +0.985 |

**Decision (applied):** drop `rep_3`, `rep_4`; keep `rep_5`.

## 3. Hedging variants — correlations only (decision deferred)

`hedging_combined` is the formal+reasoning union; the formal/reasoning split is a robustness contrast. The numbers below decide whether the split adds anything independent.

| model | r(formal, combined) | r(reasoning, combined) | r(formal, reasoning) |
|---|---|---|---|
| qwen3-4b | +0.951 | +0.816 | +0.604 |
| r1-distill-llama-8b | +0.987 | +0.594 | +0.462 |
| qwq-32b | +0.914 | +0.781 | +0.463 |
| qwen3-4b-nothink | +0.993 | +0.074 | -0.047 |
| llama-3.1-8b-instruct | +0.995 | +0.091 | -0.013 |

*Decision deferred to Step 3. Headline plan is `hedging_combined` as primary with the formal/reasoning split kept as a robustness variant; the formal–reasoning cross-correlation says whether the two actually carry independent signal.*

## 4. `trace_length` × `rep_5` — correlated but not redundant

Pre-declared expectation: ~0.8 on qwen3-4b, below the 0.95 redundancy threshold. Keep BOTH. Actual per-model values:

| model | r(trace_length, rep_5) | exceeds 0.95? |
|---|---|---|
| qwen3-4b | +0.661 | no |
| r1-distill-llama-8b | +0.677 | no |
| qwq-32b | +0.773 | no |
| qwen3-4b-nothink | +0.521 | no |
| llama-3.1-8b-instruct | +0.381 | no |

**Decision (applied):** keep both `trace_length` and `rep_5`.

## 5. Unanticipated pairs above threshold (flagged, not auto-applied)

These pairs exceed the threshold on at least one model and were NOT pre-declared. Reported here for joint review.

- `hedging_formal` × `hedging_combined`  →  qwen3-4b: +0.951, r1-distill-llama-8b: +0.987, qwq-32b: +0.914, qwen3-4b-nothink: +0.993, llama-3.1-8b-instruct: +0.995. *Proposed representative — pending decision.*

## 6. Sign-flip flags (pair sign differs across datasets within a model)


### qwen3-4b
- `trace_length` × `hedging_formal`  pooled r = -0.340;  per-dataset → medqa: +0.408, mmlu_pro: -0.243, trivia_qa: +0.041
- `trace_length` × `hedging_combined`  pooled r = -0.286;  per-dataset → medqa: +0.557, mmlu_pro: -0.160, trivia_qa: +0.292
- `trace_length` × `connector_density`  pooled r = +0.152;  per-dataset → medqa: -0.109, mmlu_pro: +0.102, trivia_qa: -0.452
- `rep_3` × `hedging_formal`  pooled r = -0.031;  per-dataset → medqa: +0.326, mmlu_pro: -0.058, trivia_qa: +0.107
- `rep_3` × `connector_density`  pooled r = +0.019;  per-dataset → medqa: +0.079, mmlu_pro: +0.154, trivia_qa: -0.487
- `rep_3` × `trace_divergence`  pooled r = +0.177;  per-dataset → medqa: -0.002, mmlu_pro: +0.157, trivia_qa: +0.464
- `rep_4` × `hedging_formal`  pooled r = +0.010;  per-dataset → medqa: +0.310, mmlu_pro: -0.040, trivia_qa: +0.117
- `rep_4` × `connector_density`  pooled r = -0.003;  per-dataset → medqa: +0.074, mmlu_pro: +0.146, trivia_qa: -0.483
- `rep_4` × `trace_divergence`  pooled r = +0.205;  per-dataset → medqa: -0.009, mmlu_pro: +0.150, trivia_qa: +0.465
- `rep_5` × `hedging_formal`  pooled r = +0.042;  per-dataset → medqa: +0.294, mmlu_pro: -0.034, trivia_qa: +0.124
- `rep_5` × `connector_density`  pooled r = -0.024;  per-dataset → medqa: +0.064, mmlu_pro: +0.139, trivia_qa: -0.472
- `rep_5` × `trace_divergence`  pooled r = +0.231;  per-dataset → medqa: -0.012, mmlu_pro: +0.144, trivia_qa: +0.460
- `hedging_formal` × `trace_divergence`  pooled r = +0.344;  per-dataset → medqa: +0.072, mmlu_pro: -0.004, trivia_qa: +0.293
- `hedging_reasoning` × `connector_density`  pooled r = -0.379;  per-dataset → medqa: +0.020, mmlu_pro: -0.112, trivia_qa: -0.613

### r1-distill-llama-8b
- `trace_length` × `hedging_formal`  pooled r = -0.508;  per-dataset → medqa: +0.355, mmlu_pro: -0.118, trivia_qa: +0.028
- `trace_length` × `hedging_combined`  pooled r = -0.455;  per-dataset → medqa: +0.506, mmlu_pro: -0.008, trivia_qa: +0.150
- `trace_length` × `connector_density`  pooled r = +0.134;  per-dataset → medqa: +0.044, mmlu_pro: +0.372, trivia_qa: -0.325
- `rep_3` × `hedging_formal`  pooled r = -0.232;  per-dataset → medqa: +0.287, mmlu_pro: -0.211, trivia_qa: -0.091
- `rep_3` × `hedging_combined`  pooled r = -0.146;  per-dataset → medqa: +0.507, mmlu_pro: -0.094, trivia_qa: +0.055
- `rep_3` × `connector_density`  pooled r = +0.364;  per-dataset → medqa: +0.368, mmlu_pro: +0.411, trivia_qa: -0.140
- `rep_4` × `hedging_formal`  pooled r = -0.221;  per-dataset → medqa: +0.257, mmlu_pro: -0.217, trivia_qa: -0.106
- `rep_4` × `hedging_combined`  pooled r = -0.138;  per-dataset → medqa: +0.474, mmlu_pro: -0.107, trivia_qa: +0.037
- `rep_4` × `connector_density`  pooled r = +0.365;  per-dataset → medqa: +0.385, mmlu_pro: +0.386, trivia_qa: -0.133
- `rep_5` × `hedging_formal`  pooled r = -0.208;  per-dataset → medqa: +0.231, mmlu_pro: -0.221, trivia_qa: -0.105
- `rep_5` × `hedging_combined`  pooled r = -0.130;  per-dataset → medqa: +0.443, mmlu_pro: -0.116, trivia_qa: +0.031
- `rep_5` × `connector_density`  pooled r = +0.362;  per-dataset → medqa: +0.394, mmlu_pro: +0.363, trivia_qa: -0.123
- `hedging_formal` × `connector_density`  pooled r = -0.082;  per-dataset → medqa: +0.064, mmlu_pro: -0.255, trivia_qa: -0.343
- `hedging_formal` × `trace_divergence`  pooled r = +0.414;  per-dataset → medqa: +0.104, mmlu_pro: -0.186, trivia_qa: +0.348
- `hedging_reasoning` × `connector_density`  pooled r = +0.188;  per-dataset → medqa: +0.335, mmlu_pro: +0.399, trivia_qa: -0.190
- `hedging_combined` × `connector_density`  pooled r = -0.023;  per-dataset → medqa: +0.209, mmlu_pro: -0.124, trivia_qa: -0.335
- `hedging_combined` × `trace_divergence`  pooled r = +0.440;  per-dataset → medqa: +0.184, mmlu_pro: -0.147, trivia_qa: +0.403

### qwen3-4b-nothink
- `trace_length` × `hedging_formal`  pooled r = +0.131;  per-dataset → medqa: +0.080, mmlu_pro: -0.224, trivia_qa: +0.166
- `trace_length` × `hedging_combined`  pooled r = +0.133;  per-dataset → medqa: +0.086, mmlu_pro: -0.205, trivia_qa: +0.185
- `trace_length` × `connector_density`  pooled r = +0.118;  per-dataset → medqa: -0.071, mmlu_pro: +0.041, trivia_qa: +0.123
- `rep_3` × `hedging_formal`  pooled r = +0.083;  per-dataset → medqa: -0.017, mmlu_pro: -0.114, trivia_qa: +0.108
- `rep_3` × `hedging_combined`  pooled r = +0.092;  per-dataset → medqa: -0.003, mmlu_pro: -0.101, trivia_qa: +0.134
- `rep_3` × `connector_density`  pooled r = +0.279;  per-dataset → medqa: +0.294, mmlu_pro: -0.007, trivia_qa: +0.181
- `rep_3` × `trace_divergence`  pooled r = -0.069;  per-dataset → medqa: -0.052, mmlu_pro: +0.076, trivia_qa: +0.180
- `rep_4` × `hedging_formal`  pooled r = +0.067;  per-dataset → medqa: -0.025, mmlu_pro: -0.071, trivia_qa: +0.102
- `rep_4` × `hedging_combined`  pooled r = +0.077;  per-dataset → medqa: -0.011, mmlu_pro: -0.058, trivia_qa: +0.130
- `rep_4` × `connector_density`  pooled r = +0.240;  per-dataset → medqa: +0.265, mmlu_pro: -0.014, trivia_qa: +0.120
- `rep_4` × `trace_divergence`  pooled r = -0.054;  per-dataset → medqa: -0.037, mmlu_pro: +0.079, trivia_qa: +0.193
- `rep_5` × `hedging_formal`  pooled r = +0.045;  per-dataset → medqa: -0.033, mmlu_pro: -0.069, trivia_qa: +0.085
- `rep_5` × `hedging_combined`  pooled r = +0.056;  per-dataset → medqa: -0.020, mmlu_pro: -0.056, trivia_qa: +0.112
- `rep_5` × `connector_density`  pooled r = +0.212;  per-dataset → medqa: +0.243, mmlu_pro: -0.009, trivia_qa: +0.074
- `rep_5` × `trace_divergence`  pooled r = -0.043;  per-dataset → medqa: -0.028, mmlu_pro: +0.093, trivia_qa: +0.187
- `hedging_formal` × `connector_density`  pooled r = -0.014;  per-dataset → medqa: -0.026, mmlu_pro: -0.033, trivia_qa: +0.004
- `hedging_formal` × `trace_divergence`  pooled r = +0.003;  per-dataset → medqa: +0.007, mmlu_pro: -0.014, trivia_qa: -0.025
- `hedging_combined` × `connector_density`  pooled r = -0.008;  per-dataset → medqa: -0.012, mmlu_pro: -0.024, trivia_qa: +0.006
- `hedging_combined` × `trace_divergence`  pooled r = +0.021;  per-dataset → medqa: +0.010, mmlu_pro: -0.004, trivia_qa: +0.003
- `connector_density` × `trace_divergence`  pooled r = -0.107;  per-dataset → medqa: -0.128, mmlu_pro: +0.086, trivia_qa: -0.050

### llama-3.1-8b-instruct
- `trace_length` × `rep_3`  pooled r = +0.387;  per-dataset → medqa: -0.083, mmlu_pro: +0.495, trivia_qa: +0.491
- `trace_length` × `rep_4`  pooled r = +0.379;  per-dataset → medqa: -0.096, mmlu_pro: +0.497, trivia_qa: +0.505
- `trace_length` × `rep_5`  pooled r = +0.381;  per-dataset → medqa: -0.090, mmlu_pro: +0.496, trivia_qa: +0.529
- `trace_length` × `hedging_formal`  pooled r = -0.060;  per-dataset → medqa: +0.071, mmlu_pro: -0.145, trivia_qa: -0.056
- `trace_length` × `hedging_reasoning`  pooled r = -0.067;  per-dataset → medqa: -0.123, mmlu_pro: +0.020, trivia_qa: +0.091
- `trace_length` × `hedging_combined`  pooled r = -0.066;  per-dataset → medqa: +0.062, mmlu_pro: -0.144, trivia_qa: -0.043
- `rep_3` × `hedging_formal`  pooled r = -0.184;  per-dataset → medqa: -0.308, mmlu_pro: -0.147, trivia_qa: +0.044
- `rep_3` × `hedging_combined`  pooled r = -0.175;  per-dataset → medqa: -0.294, mmlu_pro: -0.145, trivia_qa: +0.065
- `rep_3` × `trace_divergence`  pooled r = -0.217;  per-dataset → medqa: -0.154, mmlu_pro: -0.135, trivia_qa: +0.009
- `rep_4` × `hedging_formal`  pooled r = -0.171;  per-dataset → medqa: -0.308, mmlu_pro: -0.127, trivia_qa: +0.070
- `rep_4` × `hedging_combined`  pooled r = -0.161;  per-dataset → medqa: -0.293, mmlu_pro: -0.125, trivia_qa: +0.095
- `rep_4` × `trace_divergence`  pooled r = -0.184;  per-dataset → medqa: -0.143, mmlu_pro: -0.096, trivia_qa: +0.086
- `rep_5` × `hedging_formal`  pooled r = -0.156;  per-dataset → medqa: -0.298, mmlu_pro: -0.109, trivia_qa: +0.089
- `rep_5` × `hedging_combined`  pooled r = -0.146;  per-dataset → medqa: -0.283, mmlu_pro: -0.107, trivia_qa: +0.117
- `rep_5` × `trace_divergence`  pooled r = -0.153;  per-dataset → medqa: -0.137, mmlu_pro: -0.059, trivia_qa: +0.143
- `hedging_formal` × `hedging_reasoning`  pooled r = -0.013;  per-dataset → medqa: -0.133, mmlu_pro: -0.058, trivia_qa: +0.081
- `hedging_formal` × `trace_divergence`  pooled r = +0.150;  per-dataset → medqa: +0.122, mmlu_pro: -0.004, trivia_qa: +0.221
- `hedging_reasoning` × `hedging_combined`  pooled r = +0.091;  per-dataset → medqa: -0.051, mmlu_pro: +0.028, trivia_qa: +0.210
- `hedging_reasoning` × `connector_density`  pooled r = +0.060;  per-dataset → medqa: +0.301, mmlu_pro: +0.013, trivia_qa: -0.026
- `hedging_reasoning` × `trace_divergence`  pooled r = +0.062;  per-dataset → medqa: -0.033, mmlu_pro: -0.151, trivia_qa: +0.230
- `hedging_combined` × `trace_divergence`  pooled r = +0.155;  per-dataset → medqa: +0.120, mmlu_pro: -0.017, trivia_qa: +0.246
- `connector_density` × `trace_divergence`  pooled r = -0.170;  per-dataset → medqa: -0.037, mmlu_pro: +0.034, trivia_qa: -0.192

## 7. PROPOSED survivor set (not frozen)

After applying the rep_n rule and keeping both `trace_length` and `rep_5`, the proposed survivors are:

```
  trace_length
  rep_5
  hedging_formal
  hedging_reasoning
  hedging_combined
  connector_density
  trace_divergence
```

Dropped from this matrix: `rep_3`, `rep_4`. Hedging split kept pending Step 3.

---
STOP. Awaiting joint review of pruning_decision.md before freezing the feature set in Step 3.