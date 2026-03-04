# Idea 1: Trace-Driven Simulation with Lifecycle Replay

## Overview

Bypass BLIS's workload-spec generation by replaying ground-truth request traces directly, eliminating the arrival-pattern mismatch that causes 31,906% mean TTFT error. This idea attacks BC-1 (the primary blocker) at its source by testing whether the error is in the workload specification or in BLIS's scheduling/batch-formation logic.

**LatencyModel methods covered:** All 5 (inherited from Round 2's best model). This idea's contribution is isolating simulation-level error from step-time error.

**Go integration path:** No new Go code needed — uses existing TraceV2 replay infrastructure (`sim/workload/replay.go`, `sim/workload/tracev2.go`).

## Prior Round Context

- **Round 2 (Idea 2, H2):** BLIS E2E validation revealed 427.8% mean E2E error, dominated by 31,906% mean TTFT error. ITL was 33.6% (5/10 < 10%). The TTFT mismatch is identified as the PRIMARY binding constraint (BC-1).
- **Round 2 infrastructure:** `convert_lifecycle_to_traces.py` already exists in shared/ for converting lifecycle data to BLIS trace CSV format. `validate_blis.py` supports both workload-spec and trace-based BLIS runs.
- **Literature:** Revati [arXiv 2601.00397] showed reimplemented schedulers diverge from real vLLM. DistServe [arXiv 2401.09670] showed TTFT diverges exponentially near capacity via M/D/1 queuing: `TTFT = D + RD²/(2(1-RD))`.

## Training Strategy

No model training required — this idea uses the existing Round 2 regime ensemble coefficients (the best available StepML artifact per model). The focus is on controlling the *input* to BLIS (trace replay vs workload spec) and measuring the resulting E2E error difference.

**Data split:** Not applicable (no training). All experiments used for validation.

---

## Sub-Hypothesis H1: Trace Replay Reduces TTFT Error

### Claim

Replaying ground-truth request arrival times and token lengths via BLIS's TraceV2 format reduces mean TTFT error from 31,906% to < 100%, confirming that the workload-spec generator is the dominant source of TTFT mismatch.

### Rationale

The 31,906% TTFT error in Round 2 could stem from two sources: (a) BLIS's workload-spec generator producing different arrival patterns/request sizes than reality, or (b) BLIS's internal scheduling/batch-formation diverging from real vLLM. If (a) dominates, trace replay — which feeds BLIS the exact ground-truth arrivals — should dramatically reduce TTFT error. The DistServe M/D/1 model predicts that even a small arrival-rate mismatch near capacity causes exponential TTFT divergence, so fixing arrivals should have outsized impact.

### Method

1. Convert each experiment's per-request lifecycle data to BLIS TraceV2 format using `convert_lifecycle_to_traces.py`
2. Run BLIS with trace replay + Round 2's best StepML artifact per model (regime ensemble + overhead floor)
3. Use same vLLM serving parameters (max_model_len, max_num_seqs, max_num_batched_tokens, KV blocks) as in Round 2 validation
4. Measure E2E/TTFT/ITL errors per experiment
5. Compare against Round 2 H2 results (workload-spec mode) side by side

### Refutation Criteria

- **Supported:** Mean TTFT error < 100% with trace replay (vs 31,906% with workload spec). At least 5/10 experiments show TTFT < 50%.
- **Refuted:** Mean TTFT error > 500% with trace replay — the workload spec is NOT the dominant error source; scheduling/batch-formation divergence dominates.

### Diagnostics

- Per-experiment TTFT/ITL/E2E error table (trace replay vs workload spec)
- Arrival rate comparison: BLIS workload-spec rate vs ground-truth rate per experiment
- Request count comparison: BLIS-generated requests vs ground-truth requests per experiment
- Utilization comparison: BLIS utilization (requests in system / capacity) vs ground-truth

---

## Sub-Hypothesis H2: Error Attribution with Trace Replay

### Claim

With trace replay eliminating workload-spec error, the remaining E2E error is < 25% mean, attributable primarily to step-time prediction inaccuracy rather than simulation-level mismatch.

### Rationale

If H1 is supported (trace replay fixes TTFT), the remaining E2E error comes from: (i) step-time prediction errors accumulating over a request's lifetime, (ii) BLIS batch-formation differences from real vLLM, and (iii) KV allocation/preemption timing differences. By measuring E2E error with trace replay, we establish the *best achievable* E2E accuracy with the current step-time model — the "step-time error floor." This quantifies how much further step-time improvement (Idea 2) is needed.

### Method

1. Use trace replay results from H1
2. Decompose E2E error into TTFT component and ITL component per experiment
3. For experiments where E2E > 15%: analyze whether the error is TTFT-dominated (suggesting remaining scheduling mismatch) or ITL-dominated (suggesting step-time model improvement needed)
4. Compute the "step-time improvement headroom" = E2E_error_trace_replay - ITL_error_trace_replay. If headroom is small, step-time is not the bottleneck even with trace replay.

### Refutation Criteria

- **Supported:** Mean E2E error < 25% with trace replay. At least 3/10 experiments < 10%.
- **Refuted:** Mean E2E error > 50% even with trace replay — BLIS scheduling divergence is a co-dominant error source alongside workload-spec mismatch.

### Diagnostics

- Per-experiment E2E/TTFT/ITL error breakdown (trace replay mode)
- Step-time improvement headroom per experiment
- Batch composition comparison: BLIS batch sizes per step vs ground-truth batch sizes (if extractable from traces)

---

## Sub-Hypothesis H3: Workload-Spec Parameter Diagnosis

### Claim

The workload-spec generator's dominant error source can be identified and is one of: (a) arrival rate mismatch, (b) token length distribution mismatch, or (c) horizon/duration mismatch.

### Rationale

If H1 confirms the workload spec is the error source, we need to diagnose *which* parameter is wrong. This enables targeted fixes to the workload-spec pipeline rather than requiring trace replay for all future experiments.

### Method

1. Extract ground-truth statistics: mean arrival rate, mean/std input tokens, mean/std output tokens, total duration, total requests
2. Extract BLIS workload-spec statistics from the generated profile.yaml files used in Round 2 validation
3. Compare pairwise for each experiment. Compute relative error on each parameter.
4. Identify the parameter(s) with largest relative error — these are the calibration targets
5. If arrival rate is the dominant mismatch: test whether using the ground-truth arrival rate in the workload spec (keeping distributions from profile.yaml) closes most of the TTFT gap

### Refutation Criteria

- **Supported:** One or two parameters explain > 80% of the TTFT error variance across experiments. The workload-spec error has a clear fix.
- **Refuted:** Multiple parameters contribute roughly equally, and no simple fix to the workload spec pipeline is viable — trace replay remains necessary.

### Diagnostics

- Parameter comparison table: GT vs workload-spec for each experiment (arrival rate, input tokens, output tokens, duration, request count)
- Correlation analysis: which parameter mismatch correlates most with TTFT error
- If applicable: BLIS run with corrected arrival rate to validate the diagnosis
