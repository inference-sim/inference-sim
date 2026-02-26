# Step-Time Prediction Research Design

**Status:** Draft
**Date:** 2026-02-26
**Type:** Research methodology design
**Target venue:** Top-tier AI/systems conference (MLSys, OSDI, EuroSys, NSDI)

## Problem Statement

BLIS has two step-time estimation modes: blackbox (3-coefficient linear regression) and roofline (analytical FLOPs/bandwidth with MFU lookup). Both have accuracy limitations — the blackbox model uses only 3 features and ignores batch composition details, while the roofline model requires hand-tuned correction factors and kernel-level benchmark data that may not generalize across hardware.

We have ~165K step-level ground-truth data points from instrumented vLLM (v0.15.1) with journey/step tracing and KV events, covering 4 model families (dense + MoE), 4 workload types, and multiple TP configurations on H100 GPUs.

**Algorithm scope:** Not restricted to ML. Research ideas may propose statistical, analytical, physics-informed, machine learning, or hybrid approaches. Every idea must cite relevant prior work and justify its algorithmic choice with first-principles reasoning. Each approach must be novel or distinct from prior art in the systems/ML literature.

**Goal:** Design prediction algorithms — statistical, analytical, ML-based, or hybrid — that achieve <10% MAPE (excellent rating) while generalizing across models, workloads, hardware, and vLLM configurations. Each approach must cite relevant literature and meet the rigor standards of a top-tier AI/systems venue (MLSys, OSDI, EuroSys, NSDI).

## Ground-Truth Data

### Source

`tektonc-data-collection/results/` — 20 experiments (16 main + 4 sweep) produced by a Tekton pipeline running the instrumented vLLM fork (`github.com/inference-sim/vllm`).

### Step-Level Schema (from `traces.json` step.BATCH_SUMMARY events)

| Feature | Type | Description |
|---------|------|-------------|
| `step.duration_us` | int | **Ground truth target** — step execution time in microseconds |
| `batch.prefill_tokens` | int | Prefill tokens in this step |
| `batch.decode_tokens` | int | Decode tokens in this step |
| `batch.num_prefill_reqs` | int | Prefill request count |
| `batch.num_decode_reqs` | int | Decode request count |
| `batch.scheduled_tokens` | int | Total tokens scheduled |
| `queue.running_depth` | int | Active batch size (requests) |
| `queue.waiting_depth` | int | Queue depth |
| `kv.usage_gpu_ratio` | float | GPU KV cache utilization |
| `kv.blocks_free_gpu` | int | Free GPU KV blocks |
| `kv.blocks_total_gpu` | int | Total GPU KV blocks |
| `batch.num_finished` | int | Requests finishing this step |
| `batch.num_preempted` | int | Preemptions this step |
| `step.ts_start_ns` | int | Step start timestamp (nanoseconds) |
| `step.ts_end_ns` | int | Step end timestamp (nanoseconds) |
| `step.id` | int | Sequential step identifier |

**Sampling note:** 10% of steps are traced (`step_tracing_sample_rate=0.1`). The ~165K sampled steps represent ~1.65M total steps.

### Model/GPU/TP Coverage

| Model | Architecture | TP | Workloads |
|-------|-------------|-----|-----------|
| Llama-2-7B | Dense | 1 | general, codegen, roleplay, reasoning |
| Llama-2-70B | Dense | 4 | general, codegen, roleplay, reasoning |
| Mixtral-8x7B-v0.1 | MoE (8 experts, top-2) | 2 | general, codegen, roleplay, reasoning |
| CodeLlama-34B | Dense | 2 | general, codegen, roleplay, reasoning |

All on H100 80GB, vLLM v0.15.1, `max_model_len=4096`, `max_num_batched_tokens=2048`, `max_num_seqs=128`, chunked prefill enabled.

### Additional Data Sources

- **KV events** (`kv_events.jsonl`): BlockStored, CacheStoreCommitted, TransferInitiated/Completed — useful for KV cache offloading research.
- **Per-request lifecycle** (`per_request_lifecycle_metrics.json`): Per-token timestamps, input/output token counts — useful for request-level validation.
- **MFU benchmarks** (`InferSim/bench_data/`): Kernel-level GEMM and attention MFU data by GPU/attention-config — useful for physics-informed features.

## Pipeline Architecture

Four phases, each using a specific skill:

### Phase 1: Ideation (`/research-ideas`)

Generate research ideas for step-time prediction algorithms (statistical, analytical, ML, or hybrid), iteratively reviewed by multiple LLM judges. Each idea must cite relevant prior work and meet top-tier conference rigor.

**problem.md scope:**
- Dense + MoE model step-time prediction
- KV cache dynamics as features (offloading, utilization)
- Single-instance step-time error reduction
- Literature-grounded algorithmic design with citations
- Must address all 10 evaluation dimensions: accuracy, generalization (workloads, vLLM config, LLMs, hardware), ease of use, retraining story, vLLM version robustness, overhead, reproducibility

**Context sources:**
- This repo (BLIS simulator, existing latency models, calibration infrastructure)
- `tektonc-data-collection/results/` (ground-truth data schema)
- `InferSim/bench_data/` (kernel benchmark data)
- Existing roofline hypotheses (`hypotheses/h-roofline/`)
- Targeted web search for academic literature (10 specific queries)

**Output:** `research.md` with 3+ ranked ideas, each with literature citations + LLM reviews.

### Phase 2: Hypothesis Selection

Extract top ideas from `research.md` and map each to a hypothesis family with 2-3 sub-hypotheses. Each idea's HYPOTHESIS.md must include the literature citations and algorithmic justification from `research.md`.

### Phase 3: Experimentation (`/hypothesis-test` adapted)

Per idea, scaffold and run:
1. **H-stepml-N.1: Feature Engineering** — which features matter?
2. **H-stepml-N.2: Model Training** — does the model achieve <10% MAPE?
3. **H-stepml-N.3: Generalization** — does it hold across configs?

### Phase 4: Comparison & Selection

Cross-idea leaderboard on held-out test set.

## Hypothesis Structure

### Directory Layout

```
hypotheses/h-stepml/
├── README.md                        # Index + leaderboard
├── shared/                          # Shared Python infrastructure
│   ├── data_loader.py               # Parse traces.json → DataFrame
│   ├── split.py                     # Stratified train/valid/test (60/20/20)
│   ├── evaluation.py                # MAPE, Pearson r, per-model breakdown
│   ├── baseline.py                  # Current blackbox + roofline baselines
│   └── requirements.txt            # Python deps (pandas, scikit-learn, xgboost, torch, etc.)
│
├── idea-1-<name>/
│   ├── h1-features/
│   │   ├── HYPOTHESIS.md            # "Feature set X reduces MAPE below raw features"
│   │   ├── run.sh                   # Feature extraction + ablation
│   │   ├── analyze.py               # Feature importance + MAPE by feature set
│   │   ├── features.py              # Idea-specific feature engineering
│   │   └── FINDINGS.md
│   ├── h2-model/
│   │   ├── HYPOTHESIS.md            # "Model Y achieves <10% MAPE on validation"
│   │   ├── run.sh                   # Training + hyperparameter search
│   │   ├── analyze.py               # MAPE, Pearson r, learning curves
│   │   ├── train.py                 # Training script
│   │   ├── model/                   # Saved artifacts
│   │   └── FINDINGS.md
│   └── h3-generalization/
│       ├── HYPOTHESIS.md            # "Model generalizes across model/TP/workload"
│       ├── run.sh                   # Leave-one-model-out + cross-workload eval
│       ├── analyze.py               # Per-config MAPE breakdown
│       └── FINDINGS.md
│
├── idea-2-<name>/
│   └── (same structure)
└── ...
```

### Dependency Chain (per idea)

```
h1-features ──→ h2-model ──→ h3-generalization
```

Short-circuit rule: if h1 shows features yield MAPE > 30% even with the best model, skip h2 and h3 for that idea.

## Evaluation Framework

### Primary Metrics

| Metric | Target | Source |
|--------|--------|--------|
| MAPE (step_duration_us) | < 10% | `shared/evaluation.py` |
| Pearson r | > 0.95 | `shared/evaluation.py` |

### Diagnostic Breakdowns

| Dimension | Splits |
|-----------|--------|
| Per-model | Llama-7B, Llama-70B, Mixtral-8x7B, CodeLlama-34B |
| Per-architecture | Dense vs MoE |
| Per-workload | general, codegen, roleplay, reasoning |
| Per-phase | Prefill-heavy steps vs decode-heavy steps |
| Per-load | Low QPS vs high QPS |

### Evaluation Dimensions (beyond accuracy)

| Dimension | What We Measure |
|-----------|----------------|
| **Accuracy** | MAPE < 10%, Pearson r > 0.95 on test set |
| **Generalization: workloads** | MAPE variance across codegen, reasoning, roleplay, general |
| **Generalization: vLLM config** | Sensitivity to max_num_seqs, max_num_batched_tokens changes |
| **Generalization: LLMs** | Leave-one-model-out MAPE (dense + MoE) |
| **Generalization: hardware** | Transferability across GPU generations (H100 → A100 etc.) |
| **Ease of use** | Number of inputs required, configuration complexity |
| **Retraining story** | When retraining is needed, data requirements, training time |
| **vLLM version robustness** | Which vLLM scheduler changes would invalidate the model |
| **Overheads** | Data collection time, training time, inference latency of the model |
| **Reproducibility** | Fixed seeds, deterministic training, documented dependencies |

### Baselines

Every `analyze.py` must compare against:
1. **Blackbox (current):** `beta0 + beta1*prefill_tokens + beta2*decode_tokens`
2. **Roofline (current):** Analytical FLOPs/bandwidth model (when model config available)
3. **Naive mean:** Always predict mean `step_duration_us`

### Leaderboard Format (README.md)

| Idea | Algorithm | Key Citations | Val MAPE | Test MAPE | Pearson r | Dense MAPE | MoE MAPE | Generalization | Novelty |
|------|-----------|--------------|----------|-----------|-----------|------------|----------|----------------|---------|

## Data Split Strategy

**One-time split, shared across all ideas:**

```python
# Stratified by (model, workload_type) to ensure balanced representation
train (60%) / valid (20%) / test (20%)
```

- Stratification ensures each split contains data from all 16 model×workload combinations
- Test set is NEVER seen during any idea's training or hyperparameter tuning
- The split is saved as indices (not data) so it's reproducible: `shared/split_indices.json`
- For generalization testing (h3), additional leave-one-model-out splits are created

## Execution Plan

### Wave-Based Parallel Execution

```
Step 1: Build shared infrastructure
  └── data_loader.py, split.py, evaluation.py, baseline.py
  └── Parse all 20 experiments into unified dataset
  └── Compute baseline MAPE for reference

Step 2: Run /research-ideas (sequential)
  └── problem.md with full context
  └── 3 iterations, multiple LLM judges
  └── Output: research.md with ranked ideas

Step 3: Scaffold hypotheses (parallel across ideas)
  └── Extract top 3-5 ideas
  └── Create directory structure
  └── Write HYPOTHESIS.md for each sub-hypothesis

Step 4 - Wave 1: Feature engineering (parallel across ideas)
  └── All h1-features run simultaneously
  └── Each idea extracts its unique features
  └── Short-circuit check: drop ideas with MAPE > 30%

Step 5 - Wave 2: Model training (parallel across surviving ideas)
  └── All h2-model run simultaneously
  └── Each uses its h1's best feature set
  └── Output: trained models + validation MAPE

Step 6 - Wave 3: Generalization (parallel)
  └── All h3-generalization run simultaneously
  └── Leave-one-model-out evaluation
  └── Cross-workload MAPE breakdown

Step 7: Leaderboard + selection (sequential)
  └── Compare all ideas on test set
  └── Evaluate on all dimensions (accuracy, ease of use, etc.)
  └── Select best approach, document findings
```

### Parallelism Model

- **Across ideas:** Fully parallel (independent data, independent models)
- **Within an idea:** Sequential dependency chain (h1 → h2 → h3)
- **Within a hypothesis:** Parallel configs/ablations within `run.sh`
- **Short-circuiting:** After Wave 1, drop ideas with feature MAPE > 30%

## Skill Invocation Specifications

Each skill invocation below specifies the exact inputs for every decision point (screen/question). No ad-hoc prompting — every answer is derived from the requirements in this document.

---

### S1. `/research-ideas` — Phase 1: Ideation

**Purpose:** Generate 3+ ML research ideas for step-time prediction, iteratively reviewed by external LLM judges.

**Pre-requisite:** Write `problem.md` (content specified below) before invoking the skill.

#### Screen 1: Problem

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Where is your research problem defined?" | **"Use problem.md"** (option 1) | We write problem.md first with the exact content below |

#### problem.md Content (written before skill invocation)

```markdown
# Step-Time Prediction Algorithms for LLM Inference Serving Simulation

## Problem

Discrete-event simulators for LLM inference serving (e.g., BLIS, Vidur, SimLLM,
LLMServingSim) need accurate step-time estimation to predict end-to-end latency
metrics (TTFT, ITL, E2E) without requiring real GPU execution. Step time — the wall-
clock duration of a single vLLM scheduler iteration processing a heterogeneous batch
of prefill and decode requests — is the fundamental unit of simulation fidelity.

Current approaches fall into two categories, both with known limitations:

1. **Parametric regression** (e.g., BLIS blackbox mode): 3-coefficient linear model
   `beta0 + beta1*prefill_tokens + beta2*decode_tokens`. Ignores batch composition,
   KV cache state, and architecture-specific compute patterns. MAPE often >20%.

2. **Analytical roofline** (e.g., BLIS roofline mode, Vidur): FLOPs/bandwidth analysis
   with empirical MFU correction factors from kernel microbenchmarks. Requires per-GPU
   per-attention-config benchmark sweeps. Struggles with MoE architectures, mixed-phase
   batches, and scheduler overhead. Sensitive to MFU interpolation errors.

Neither approach is grounded in real scheduler-step observations from production
serving systems.

## Available Data

We have ~165K step-level ground-truth observations from instrumented vLLM (v0.15.1,
custom fork: github.com/inference-sim/vllm) with journey/step tracing and KV cache
event capture.

**Target variable:** `step.duration_us` — wall-clock microseconds per scheduler step.

**Observable features per step (16 attributes):**
- Batch composition: `prefill_tokens`, `decode_tokens`, `num_prefill_reqs`,
  `num_decode_reqs`, `scheduled_tokens`
- Queue state: `running_depth` (active batch size in requests), `waiting_depth`
- KV cache state: `kv_usage_gpu_ratio`, `kv_blocks_free_gpu`, `kv_blocks_total_gpu`
- Step lifecycle: `num_finished`, `num_preempted`, `step_id`, `ts_start_ns`, `ts_end_ns`
- Experiment metadata (per-experiment, not per-step): model architecture (dense vs MoE),
  tensor parallelism degree, workload type, vLLM config

**Data coverage (4 models × 4 workloads = 16 main experiments + 4 sweep):**

| Model | Arch | Params | TP | Experts |
|-------|------|--------|-----|---------|
| Llama-2-7B | Dense | 7B | 1 | — |
| Llama-2-70B | Dense | 70B | 4 | — |
| Mixtral-8x7B-v0.1 | MoE | 46.7B (12.9B active) | 2 | 8 (top-2) |
| CodeLlama-34B | Dense | 34B | 2 | — |

Hardware: NVIDIA H100 80GB HBM3 (all experiments).
vLLM config: max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
chunked prefill + prefix caching enabled.
Workloads: general, codegen, roleplay, reasoning — varying input/output length
distributions and multi-turn conversation patterns.
Sampling: 10% of steps traced (step_tracing_sample_rate=0.1).

**Supplementary data sources:**
- **KV cache events** (kv_events.jsonl, ~42K-137K events/experiment): Block-level
  GPU/CPU storage, cache store commits, transfer initiation/completion — captures
  offload/reload dynamics
- **Per-request lifecycle** (per_request_lifecycle_metrics.json, ~5K-17K requests/
  experiment): Per-output-token arrival timestamps — enables validation against
  observed TTFT/ITL/E2E
- **Kernel microbenchmarks** (InferSim/bench_data/): GEMM MFU by (M,K,N), attention
  MFU by (seq_len) for prefill and (batch_size, kv_len, tp) for decode, across
  H100 and H20 GPUs

## Research Goal

Design prediction algorithms for per-batch step time that achieve <10% MAPE on a
held-out test set. Algorithms are NOT restricted to ML — they may be:

- **Statistical**: regression with domain-informed basis functions, quantile regression,
  Bayesian hierarchical models, Gaussian processes, non-parametric methods
- **Analytical/physics-informed**: roofline-based with learned corrections, piecewise
  linear models exploiting compute/memory boundedness transitions, operator-level
  cost models with empirical calibration
- **Machine learning**: gradient-boosted trees, neural networks, attention-based
  sequence models, mixture-of-experts prediction models
- **Hybrid**: analytical backbone with learned residual corrections, feature
  engineering combining physical insight with data-driven terms

Each approach MUST:
1. Cite relevant prior work from the systems/ML literature (MLSys, OSDI, SOSP,
   EuroSys, NSDI, ISCA, MICRO, ATC, SoCC, NeurIPS systems, SIGCOMM)
2. Justify the algorithmic choice with first-principles reasoning about *why*
   this approach should work for this problem domain
3. Identify what makes this approach novel or distinct from prior art

## Evaluation Dimensions

Reviewers from a top-tier AI/systems conference will judge each approach on:

1. **Accuracy**: MAPE < 10%, Pearson r > 0.95 on held-out test set
2. **Generalization across workloads**: consistent MAPE across codegen, reasoning,
   roleplay, general — not overfit to one workload distribution
3. **Generalization across vLLM configs**: robust to max_num_seqs,
   max_num_batched_tokens, chunked prefill on/off
4. **Generalization across LLM architectures**: dense (Llama, CodeLlama) AND
   MoE (Mixtral) — fundamentally different compute patterns
5. **Generalization across hardware**: transferable across GPU generations
   (H100 → A100 → future accelerators). What needs to be re-measured vs.
   what transfers?
6. **Ease of use**: minimal required inputs, simple calibration procedure
7. **Retraining/recalibration story**: under what circumstances? How much data?
   How long does it take? Can it be done with a small probe workload?
8. **vLLM version robustness**: which vLLM scheduler/batch-formation changes
   would invalidate the model? How to detect model staleness?
9. **Overheads**: data collection burden, training/fitting time, per-prediction
   inference latency (must be <1ms for simulator integration)
10. **Reproducibility**: fixed seeds, deterministic fitting, documented dependencies

## Constraints

- Prediction uses only features observable at step time (no future lookahead)
- The algorithm replaces `StepTime(batch []*Request) int64` in a Go discrete-event
  simulator — inference must be fast (<1ms per prediction) and side-effect-free
- Training data is 10%-sampled — account for potential sampling bias
- Dense and MoE have fundamentally different compute patterns (MoE: top-2 of 8
  experts → sparse activation, different GEMM shapes, different memory access)
- The same algorithm should handle pure-prefill, pure-decode, and mixed batches
  (chunked prefill produces mixed batches every step)

## Baselines to Beat

1. **Parametric regression (BLIS blackbox)**: 3 coefficients, trained offline
2. **Analytical roofline (BLIS roofline)**: per-GEMM + per-attention FLOPs/bandwidth
   with MFU lookup from kernel benchmarks
3. **Naive mean**: predict the unconditional mean step_duration_us

## What Constitutes a Strong Idea

Each research idea should propose:
1. **Algorithmic design**: the specific prediction method with theoretical motivation
2. **Feature representation**: what features to extract/engineer and why (physically
   motivated, not just "try all combinations")
3. **Fitting/training methodology**: how to estimate parameters, validate, select
   hyperparameters
4. **Generalization mechanism**: how the algorithm handles unseen models, hardware,
   or configurations — what is architecture-agnostic vs. what needs recalibration
5. **Related work positioning**: how this approach relates to prior work on latency
   prediction in serving systems (Vidur, SimLLM, splitwise, DistServe, etc.)

Ideas must address ALL 10 evaluation dimensions, not just accuracy. Conference
reviewers will scrutinize generalization claims and novelty.
```

#### Conditional Screen: Existing Research

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Found existing research.md..." | **"Start fresh"** (option 1) | Clean start for this research campaign |

#### Screen 2: Background

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Which sources for background context?" | **Multi-select: options 1, 2, 4, 6** | See below |

Selected sources and their follow-up inputs:

| Source | Follow-up Input | What It Provides |
|--------|----------------|------------------|
| **Current repository** (option 1) | (none needed) | BLIS architecture, existing latency models (`sim/latency/`), MFU database, calibration infrastructure, roofline hypotheses |
| **Other local repositories** (option 2) | Paths: `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/tektonc-data-collection`, `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/InferSim` | Ground-truth data schema, kernel benchmark data, Tekton pipeline structure |
| **Local documents** (option 4) | Path: `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/docs/design/roofline.md` | Existing roofline model documentation |
| **Web search** (option 6) | **"I'll provide specific queries"** (option 2) — see query list below | Targeted academic literature search for conference-quality prior work |

**Web search queries (provided manually for precision):**

```
1. "LLM inference serving latency prediction model" site:arxiv.org
2. "Vidur simulator step time estimation" MLSys
3. "roofline model GPU kernel performance prediction transformer"
4. "vLLM scheduler batch iteration time modeling"
5. "discrete event simulation inference serving latency"
6. "GPU kernel performance prediction machine learning"
7. "mixture of experts inference latency cost model"
8. "splitwise DistServe Sarathi latency model"
9. "SimLLM LLMServingSim performance model"
10. "Bayesian performance modeling GPU workloads"
```

**Rationale for manual queries:** Auto-generated queries would be too broad. These 10 queries target: (a) the exact problem domain (LLM serving latency), (b) known competing simulators (Vidur, SimLLM, LLMServingSim), (c) specific algorithmic families (roofline, Bayesian, kernel prediction), and (d) MoE-specific challenges. This ensures the `/research-ideas` background section contains citable prior work that conference reviewers would expect to see.

**Not selected and why:**
- GitHub repositories (option 3): The instrumented vLLM fork is too large to scan usefully via API; the local repo clone is more efficient
- Remote papers/URLs (option 5): We use web search with specific queries instead — more likely to find relevant papers than guessing URLs
- Skip background (option 7): Background with literature is essential for conference-quality ideas

#### Screen 3: Judges

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Which AI models should review?" | **Multi-select: all 3 available models** | Maximum review diversity: Claude for reasoning depth, GPT-4o for breadth, Gemini Flash for speed. 3 judges is the recommended minimum. |

Selected judges: Claude Opus, GPT-4o, Gemini 2.5 Flash.

#### Screen 4: Iterations

| Question | Answer | Rationale |
|----------|--------|-----------|
| "How many idea iterations?" | **"3 iterations"** (option 1, recommended) | 3 iterations produces 3 ideas, each building on feedback from the prior. More iterations have diminishing returns for this problem scope. |

#### Expected Output

`research.md` containing:
- Problem statement + background context
- 3 research ideas, each with:
  - Feature engineering approach
  - Model architecture
  - Training methodology
  - Generalization strategy
  - Reviews from 3 LLM judges
- Executive summary ranking the ideas

---

### S2. `/hypothesis-test` — Phase 3: Experimentation

**Purpose:** Scaffold and run experiments for each research idea. Invoked once per idea (not once for all ideas).

**Pre-requisite:** Shared infrastructure (`hypotheses/h-stepml/shared/`) must be built first. Research ideas must be extracted from `research.md` and mapped to hypothesis directories.

**Adaptation:** The skill generates hypotheses for the whole project by default. We override this by providing pre-written `HYPOTHESIS.md` files and using the "pending" detection mechanism — the skill detects existing pending hypotheses and offers to test them.

#### Pre-invocation Setup (per idea)

Before invoking `/hypothesis-test` for Idea N, manually create:

```
hypotheses/h-stepml/idea-N-<name>/h1-features/HYPOTHESIS.md   # Status: Pending
hypotheses/h-stepml/idea-N-<name>/h2-model/HYPOTHESIS.md      # Status: Pending
hypotheses/h-stepml/idea-N-<name>/h3-generalization/HYPOTHESIS.md  # Status: Pending
```

Each `HYPOTHESIS.md` follows the project template with content derived from the research idea. It MUST include a Related Work section with citations from `research.md`. Example for h1-features:

```markdown
# Hypothesis: [Idea N] Feature Engineering

**Status:** Pending
**Family:** Performance-regime
**Type:** Type 2 (Statistical — Dominance)

## Claim

[Extracted from research.md idea N]: Feature set X (e.g., "physics-informed features
including compute intensity ratio, memory boundedness index, and prefill/decode token
interaction terms") achieves lower validation MAPE than raw batch features alone when
used with a baseline Ridge regression model.

## Refuted If

The engineered feature set achieves validation MAPE within 2 percentage points of
raw features (no statistically significant improvement at p < 0.05).

## Related Work

[Extracted from research.md idea N — required citations]:
- [Author et al., Venue Year]: How this work informs our feature design
- [Author et al., Venue Year]: What features they used and why ours differ
- ...

## Algorithmic Justification

[First-principles reasoning for why this feature set should work for step-time
prediction — e.g., "Compute intensity ratio captures the transition between
compute-bound and memory-bound regimes that determines which hardware resource
is the bottleneck for a given batch composition" (cf. Williams et al., roofline
model, 2009)]
```

#### Screen 1: Project + Focus + Count

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Which project?" | **"Current directory"** (option 1) | Project root is the BLIS repo containing `hypotheses/` |
| "What area?" | **"Specific component"** → type: `hypotheses/h-stepml/idea-N-<name>/` | Scope generation to this idea's directory. Prevents generating unrelated hypotheses about the Go simulator. |
| "How many hypotheses?" | **"3"** (option 1) | We want exactly 3 per idea (h1-features, h2-model, h3-generalization). The skill will detect our 3 pending HYPOTHESIS.md files. |

**Critical note:** If the skill generates NEW hypotheses instead of detecting our pending ones, we select only the pending ones in Screen 3 (see below).

#### Screen 2: Dashboard (autonomous)

No input needed. The skill either generates new hypotheses (which we'll ignore) or detects our pending ones.

#### Screen 3: Select + Execution

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Which hypotheses to test?" | **Select only the 3 pending hypotheses** (h1-features, h2-model, h3-generalization) | Ignore any newly generated hypotheses. We want exactly our pre-written ones. |
| "Parallel or sequential?" | **"Sequential"** (option 2) | Within an idea, h2 depends on h1's output. Sequential ensures dependency chain. Cross-idea parallelism is handled at the wave level by `dispatching-parallel-agents`. |

#### Screen 4: Approve

The skill scaffolds `run.sh`, `analyze.py`, `FINDINGS.md` for each hypothesis.

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Which experiments to run?" | **Review each design, then "All of them"** (option 1) | But first verify: (1) `run.sh` uses Python + shared infrastructure, (2) `analyze.py` computes MAPE via `shared/evaluation.py`, (3) experiment configs use our stratified split. |

**Verification checklist before approving:**
- [ ] `run.sh` sources shared data loader, not custom parsing
- [ ] `run.sh` uses the shared train/valid/test split (not its own)
- [ ] `analyze.py` imports `shared/evaluation.py` for MAPE computation
- [ ] `analyze.py` computes baseline comparisons (blackbox, roofline, naive mean)
- [ ] `FINDINGS.md` template includes all evaluation dimensions
- [ ] No hardcoded paths; uses relative paths from hypothesis directory

#### Screen 5: Dashboard (autonomous)

No input needed. The skill executes experiments sequentially (h1 → h2 → h3).

#### Screen 6: Commit

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Commit results?" | **"Commit all"** (option 1) | Commit all findings, including failed experiments (they're diagnostic information). |

---

### S3. `convergence-review` — Review Gates

**Purpose:** Quality gates at three points per idea. Each gate ensures experiment rigor before proceeding.

#### Gate 1: Hypothesis Design Review (after writing HYPOTHESIS.md, before scaffolding)

```
/convergence-review h-design --model sonnet
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gate type | `h-design` | Hypothesis design review (5 perspectives) |
| Artifact path | (none — uses conversation context) | h-design reads from current conversation |
| Model | `sonnet` | Good balance of rigor and cost for design review. 5 perspectives × sonnet is reasonable. |

**Convergence criteria:** Zero CRITICAL, zero IMPORTANT findings across 5 perspectives.

**Perspectives evaluated:**
1. Hypothesis quality (testable, falsifiable, specific)
2. ED-1 through ED-6 rigor (controlled comparison, rate awareness, etc.)
3. Parameter calibration (realistic values from ground-truth data)
4. Control completeness (all confounds addressed)
5. DES fit (applicable to discrete-event simulation context)

#### Gate 2: Experiment Code Review (after scaffolding, before running)

```
/convergence-review h-code hypotheses/h-stepml/idea-N-<name>/h1-features --model sonnet
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gate type | `h-code` | Hypothesis code review (5 perspectives) |
| Artifact path | Path to the hypothesis directory being reviewed | Points to `run.sh` + `analyze.py` |
| Model | `sonnet` | Catches implementation bugs without opus cost |

**Convergence criteria:** Zero CRITICAL, zero IMPORTANT findings across 5 perspectives.

**Perspectives evaluated:**
1. Parser-output agreement (analyze.py correctly parses run.sh output)
2. CLI flags and commands (correct Python invocations, paths)
3. YAML/config fields (correct data loading, feature names)
4. Config diff (matches experiment design)
5. Seed/determinism (reproducible results)

#### Gate 3: Findings Review (after running, before leaderboard inclusion)

```
/convergence-review h-findings hypotheses/h-stepml/idea-N-<name>/h2-model/FINDINGS.md --model opus
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gate type | `h-findings` | Hypothesis FINDINGS review (10 perspectives) |
| Artifact path | Path to the specific FINDINGS.md | The most important review — validates conclusions |
| Model | `opus` | 10 perspectives on findings quality warrants opus-level rigor. This is the gate that determines if an idea's results are trustworthy. |

**Convergence criteria:** Zero CRITICAL, zero IMPORTANT findings across 10 perspectives.

**Perspectives evaluated:**
1. Code verifier (results match what the code actually produces)
2. Experiment designer (design is sound)
3. Statistical rigor (MAPE computation, significance tests)
4. Control auditor (baselines compared correctly)
5. Standards compliance (BLIS experiment standards)
6. Substance/logic (conclusions follow from evidence)
7. DES mechanism (step-time modeling is physically meaningful)
8. Reproducibility (can be re-run independently)
9. Cross-experiment consistency (findings consistent with other ideas)
10. User guidance (actionable recommendations)

---

### S4. `dispatching-parallel-agents` — Wave Execution

**Purpose:** Run multiple ideas in parallel during each wave.

**Note:** This skill is a methodology guide, not an interactive workflow. It provides no screens or questions. The principled input is the task decomposition.

#### Wave 1 Dispatch: Feature Engineering

**Independence check:** All h1-features experiments are independent because:
- Each reads the same shared dataset (read-only)
- Each writes to its own `idea-N-<name>/h1-features/` directory (no shared write state)
- No idea's feature engineering depends on another idea's features

**Task definitions for parallel agents:**

| Agent | Directory | Task |
|-------|-----------|------|
| Agent 1 | `hypotheses/h-stepml/idea-1-<name>/h1-features/` | Run `./run.sh`, then `python3 analyze.py`. Report MAPE results. |
| Agent 2 | `hypotheses/h-stepml/idea-2-<name>/h1-features/` | Run `./run.sh`, then `python3 analyze.py`. Report MAPE results. |
| Agent 3 | `hypotheses/h-stepml/idea-3-<name>/h1-features/` | Run `./run.sh`, then `python3 analyze.py`. Report MAPE results. |

**Post-wave integration:** After all agents complete, the orchestrator:
1. Collects each agent's MAPE results
2. Applies short-circuit rule: drop ideas with MAPE > 30%
3. Updates leaderboard README.md
4. Proceeds to Wave 2 with surviving ideas

#### Wave 2 Dispatch: Model Training

Same pattern as Wave 1, but for `h2-model/` directories. Only surviving ideas participate.

#### Wave 3 Dispatch: Generalization

Same pattern, for `h3-generalization/` directories. Only ideas that passed Wave 2 (<10% MAPE on validation) participate.

---

### S5. `writing-plans` — Implementation Planning

**Purpose:** Create the detailed implementation plan for the shared infrastructure (Phase 0) before any skill is invoked.

**Invocation:** After this design doc is approved, invoke `/writing-plans` with this design doc as context.

**The skill asks one question after plan creation:**

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Subagent-Driven or Parallel Session?" | **"Subagent-Driven (this session)"** | Shared infrastructure is a sequential dependency chain (data_loader → split → evaluation → baseline). Subagent-driven allows review between tasks and fast iteration in the current session. |

**Plan scope (what writing-plans should plan):**

The plan covers only the shared infrastructure (Step 1 of the execution plan). Subsequent steps (research-ideas, hypothesis-test, etc.) are orchestrated by this design doc's skill specifications, not by the implementation plan.

Tasks for the plan:
1. Parse all 20 experiments' `traces.json` → unified Parquet dataset
2. Implement stratified train/valid/test split (60/20/20 by model×workload)
3. Implement MAPE + Pearson r evaluation harness
4. Implement blackbox + roofline + naive mean baselines
5. Verify baselines produce expected MAPE ranges on the dataset

---

### Skill Invocation Summary

| Phase | Skill | Invocations | Total Agent Cost |
|-------|-------|-------------|-----------------|
| 0. Infrastructure | `/writing-plans` | 1× | ~5 tasks × 1 agent |
| 1. Ideation | `/research-ideas` | 1× | 3 iterations × 3 judges = 9 review agents + 3 background agents |
| 2. Selection | Manual | 1× | 0 agents |
| 3a. Design review | `/convergence-review h-design` | 1× per idea (3-5×) | 5 perspectives × 3-5 ideas = 15-25 agents (sonnet) |
| 3b. Scaffolding | `/hypothesis-test` | 1× per idea (3-5×) | 3 hypotheses × 3-5 ideas = 9-15 scaffold agents |
| 3c. Code review | `/convergence-review h-code` | 3× per idea (9-15×) | 5 perspectives × 9-15 = 45-75 agents (sonnet) |
| 3d. Execution | `dispatching-parallel-agents` | 3 waves | 3-5 agents per wave × 3 waves = 9-15 agents |
| 3e. Findings review | `/convergence-review h-findings` | Up to 3× per idea | 10 perspectives × up to 15 = up to 150 agents (opus) |
| 4. Leaderboard | Manual | 1× | 0 agents |

## Success Criteria

1. At least one research idea achieves **<10% MAPE** on the held-out test set
2. The winning model generalizes across **dense and MoE** architectures
3. The retraining story is **documented and practical** (data requirements, training time)
4. Results are **reproducible** (fixed seeds, saved models, documented dependencies)
5. The winning model **outperforms both baselines** (blackbox + roofline) on test MAPE
