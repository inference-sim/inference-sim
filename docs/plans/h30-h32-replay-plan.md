# Micro Plan: H30/H31/H32 — BLIS Replay vs Real vLLM

**Species:** Hypothesis experiment batch (3 hypotheses)
**Branch:** `h30-h32-replay`
**Date:** 2026-03-03
**Status:** Findings complete, PR pending

## Executive Summary

Three hypothesis experiments comparing BLIS's crossmodel latency backend against 16 real vLLM experiments (4 models × 4 profiles) from the training data pipeline. Uses a three-way comparison (crossmodel vs per-model blackbox vs real vLLM) to isolate error sources between coefficient accuracy and BLIS scheduler fidelity.

**Key finding:** BLIS throughput prediction is excellent (±2.5%) across all models and load levels. TTFT is systematically underpredicted (-17% to -56%) by BOTH backends equally, revealing that the error is in BLIS's scheduling model (processes requests faster than real vLLM), not in the crossmodel coefficients. At saturation (ρ > 1.0), a 22% throughput overestimate causes 2000× TTFT error through queueing dynamics collapse.

## What This PR Contains

### Hypothesis directories (new)
- `hypotheses/h30-replay-fidelity/` — HYPOTHESIS.md, FINDINGS.md, run_all.sh, analyze.py
- `hypotheses/h31-replay-generalization/` — HYPOTHESIS.md, FINDINGS.md, run.sh, analyze.py
- `hypotheses/h32-replay-capacity/` — HYPOTHESIS.md, FINDINGS.md, run.sh, analyze.py

### Training pipeline script (new)
- `training/generate_replay_specs.py` — Generates per-experiment inference-perf workload specs and ground truth from training data. Depends on `training/split.py` (existing on training branch).

### Files NOT included
- `training/cmd/replay/main.go` — Go replay binary (written but superseded by CLI approach)
- `training/extract_replay.py` — superseded by `generate_replay_specs.py`
- `training/replay_data/` — generated output (can be regenerated)
- Raw training data (`training/default_args/`) — large data files on training branch

## Behavioral Contracts

**BC-1:** FINDINGS.md documents report the same metrics shown in the experiment runs (no post-hoc number changes).

**BC-2:** The three-way comparison (crossmodel vs blackbox vs real) is included in H30 FINDINGS to isolate error sources.

**BC-3:** Each FINDINGS.md includes a Verdict section with clear Confirmed/Refuted/Partially Confirmed status and evidence.

**BC-4:** Reproducing instructions are included with exact CLI commands.

**BC-5:** The `enable_multi_turn_chat` semantic mismatch is documented as a discovered bug.

## Tasks

### Task 1: Clean up experiment files
- Remove unused Go replay binary (`training/cmd/replay/`)
- Remove superseded `training/extract_replay.py`
- Ensure run scripts reference correct paths
- Verify all FINDINGS.md are internally consistent

### Task 2: Update hypotheses/README.md
- Add H30, H31, H32 entries to the catalog

### Task 3: Commit and create PR
- Stage hypothesis directories + training script
- Create PR targeting main branch

## Deviation Log

- **Multi-turn disabled**: Original specs used `enable_multi_turn_chat: true` from profiles. Disabled after discovering semantic mismatch (BLIS accumulates context; real inference-perf uses it for chat template). Real data confirms constant input tokens.
- **Go replay binary abandoned**: Originally planned for per-request trace replay. Switched to CLI-based same-specification comparison after user feedback that the existing `blis convert inference-perf` + `blis run` path is the right abstraction.
- **Three-way comparison added**: Not in original hypothesis design. Added to isolate whether TTFT error is from crossmodel coefficients or BLIS scheduler. Run via `./blis run --beta-coeffs <per-model>` for blackbox. This turned out to be the most important finding.
- **Shared harness bypassed**: These experiments use `blis convert inference-perf` + `blis run` pipeline rather than the standard `hypotheses/lib/harness.sh` `blis_run()` wrapper. The conversion step doesn't fit the harness's single-command model. The standard harness is designed for direct `blis run` invocations with flag-based workloads.
- **Throughput claim scope**: At sub-saturation, throughput ≈ arrival rate (all requests complete). The ±2.5% measures Poisson noise, not step-time accuracy. Meaningful throughput divergence appears only at saturation (reasoning experiments).
- **TTFT attribution**: The systematic TTFT underprediction is attributed to BLIS's zero inter-step overhead model (`SchedulingProcessingTime()` returns 0), not to coefficient accuracy. Conclusive attribution would require a "perfect-beta" control (injecting real step durations into BLIS), which was not performed.
