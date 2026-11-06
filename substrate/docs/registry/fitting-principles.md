# Fitting BLIS to the real world: principles and a repeatable procedure

This document states how a coefficient, or a pricing mechanism, earns its place in BLIS. It
complements [methodology.md](methodology.md), which defines what a coefficient is and how each
family was fitted, and [reproducing-coefficients.md](reproducing-coefficients.md), which says how
to regenerate any value. Where this document and methodology.md §3 differ on the role of
end-to-end data, this document records the newer rule and §3.1 below says what changed.

## 1. The principle

**A fit that does well across latency, KV, queueing and throughput together is the evidence
that BLIS's mechanisms mimic the real ones. A fit that does well on latency alone is not.**

A simulator can match a median latency for the wrong reason: an over-large KV pool hiding an
under-priced step, or an optimistic queue hiding a pessimistic kernel. Two errors of opposite
sign cancel in one metric and show up in every other. So a mechanism is accepted only when
the quantities it governs agree with the measurement simultaneously, at the levels where the
mechanism acts:

| level | what to compare | what a disagreement means |
|---|---|---|
| GPU step | step time against batch composition; operator timings | a kernel or composition term is wrong |
| memory | KV pool size, KV usage over time, host-tier usage | a memory term, the cache layout, or admission is wrong |
| scheduling | running and waiting requests, prefix-hit rate, prompt tokens by source | admission, eviction, chunking or routing is wrong |
| queueing | per-request queue wait, its distribution | arrival handling, client pacing, or a capacity term is wrong |
| end to end | TTFT, TPOT/ITL, E2E, output throughput, interactivity | the composition of all of the above |

A KV deviation is often the clearest diagnosis: a simulated KV usage of 10% against a measured
90% names a wrong pool, eviction or admission long before the latency error says why.

## 2. Compare apples to apples, or not at all

Every comparison is between the same quantity on both sides. Before any number is trusted:

1. **Same definition.** Map each measured field to the simulator's field by its source code,
   not its name. Examples found the hard way:
   - InferenceX's ITL and TPOT are the same per-request quantity; interactivity is
     1 / ITL percentile.
   - vLLM's logged "GPU KV cache size" for DeepSeek-V4 is not pool tokens. It is
     `num_blocks / blocks_reserved_per_request x max_model_len`, summed over every KV group
     (`vllm/v1/core/kv_cache_utils.py`). Compare blocks with blocks.
   - A log whose stated KV size contradicts its own "maximum concurrency" line is excluded,
     not compared.
2. **Same window and request set.** Profile phase on both sides; warmup excluded on both;
   cancelled requests handled identically. The code asserts it, and a mismatch is recorded by
   name instead of compared.
3. **Same offered load.** The replay must reproduce the client's pacing. The anchor runs used
   aiperf's `--trace-idle-gap-cap-seconds 300`; a replay without the cap carried 2.6-4x fewer
   requests in flight at the same nominal concurrency, and its latency error was a load
   artefact, not a model error.
4. **Same engine behaviour.** Resolve every version-dependent behaviour from the run's own
   engine build, never the newest default: SWA bounded replay, model runner, async scheduling,
   frontend, capture sizes. A behaviour the schema cannot state becomes a named mismatch on
   that point.
5. **Same model and hardware.** Use the checkpoint the run actually served (from its recipe or
   log), and the device memory the driver reports, not the marketing figure.
6. **Check parity with KV first.** A gross KV deviation (the pool, or the usage distribution)
   is checked first as a parity defect: in the setup (memory budget, max_model_len, block size,
   KV dtype, layout, prefix caching, the checkpoint served, KV groups), the workload (ISL/OSL,
   prefix sharing, offered load, window) or the definitions. Only after those are ruled out is
   it treated as a model error. Points with an unresolved parity defect are excluded from fits
   with a named reason. Because KV capacity drives admission, queueing and preemption, closing
   KV parity usually improves the other metrics too, where capacity binds.

## 3. Data roles

| source class | role |
|---|---|
| operator and microbenchmark data (AISimulate sweeps, InferenceX OperatorX / CollectiveX / ubenchX, vendor kernel READMEs verified against their repos) | fit the primitive it measures |
| forward-pass data (FPM) | fit composed terms; validate composition |
| end-to-end runs, FIT share | fit terms no operator data can supply: host and scheduler cost, admission, queueing, collectives at scale, KV transfer, offload |
| end-to-end runs, HELD share | evaluation only |
| named anchor sets | evaluation only, reported separately |

### 3.1 The end-to-end split

End-to-end runs are partitioned once, before any scoring, by a seeded, documented,
diversity-stratified random draw: about 40% FIT and 60% HELD. Every level of every factor
(model, GPU, engine build, parallel layout, P/D mode, speculation, offload tier, frontend,
scenario, concurrency band) appears in both sets wherever its stratum has at least two
points. A point keeps its request pairs and its seed repeats in one set. Named anchors are
HELD in full. The partition is hashed, and new runs are added by the same seeded rule without
re-drawing existing assignments.

This replaces methodology.md §3's "evaluation only" rule for end-to-end rows. The protection
that rule gave is kept by the HELD share and the anchors: no HELD result is read by a fit
until that fit is frozen.

### 3.2 Reputation of a source

Authoritative sources are vendors and benchmark organisations publishing their own
measurements (NVIDIA, the engine projects, InferenceX, MLCommons, model labs), verified as the
real organisation. Research data with a published method is usable with stated caveats and
never as a coefficient's only source. Anonymous, re-uploaded, synthetic or unlicensed data is
not used. A figure forwarded from a secondary summary is verified against its primary file at
a pinned commit before use.

## 4. The loss: distributions, on informative metrics

1. **Fit distributions, not points.** Per run and metric, use the median-normalised
   Wasserstein-1 distance between the measured and simulated empirical distributions, plus a
   tail term (quantile errors at p50/p90/p99 or KS). Where server time series exist, include
   the distributions of running requests, waiting requests and KV usage, so a fit must
   reproduce queueing as well as latency. A one-to-one request join (same trace turn, input and
   output length) gives per-request comparisons where the data allows.
2. **Choose the metrics that carry the term's signal.** For each term, a metric enters its loss
   only if it passes all of:
   - signal: variance across configurations well above variance across seed repeats;
   - sensitivity: it moves when the parameter moves (check by perturbation on FIT runs);
   - independence: it is not a restatement of a metric already in the loss;
   - reliability: no definitional mismatch, window effect or aggregation quirk on either side;
   - level: it observes the level where the term's mechanism acts.
   Weight the remaining metrics by inverse noise variance, and record the choice before fitting.
3. **Few, bounded, physical parameters.** A distribution loss can be gamed by a flexible term,
   so no free-form corrections. Each parameter has a physical meaning and bounds.

## 4a. Why distributions, not aggregates

An aggregate (a mean, a median, one percentile) answers "is the typical value right". A
distribution answers "is the process that produced the values right", and that is the question
a simulator has to answer.

1. **Aggregates hide compensating errors; distributions expose them.** A median latency can
   match while the simulated distribution is too narrow (no queueing tail) and the measured one
   is bimodal (some requests stalled). The medians agree for the wrong reason. Example from the
   DeepSeek-V4.1 H100 anchor at its saturation point: the measured decode spent most of its time
   in chunk gaps over 250 ms, a second mode the simulator never produced. No single percentile
   names that; the distribution shape does.
2. **The shape carries the mechanism.**
   - A long right tail in TTFT is queueing; its weight says how often the system is saturated.
   - A second mode in ITL is stalls: preemption, a blocking transfer, a sync point.
   - A distribution of KV usage pinned near 100% is a capacity regime; one spread across the
     range is a lightly loaded system.
   - A width that grows with load is contention; a width that does not is a fixed overhead.
   Matching the shape constrains the mechanism. Matching the median constrains only one point
   of it.
3. **Tails are what deployments are judged on.** Interactivity is quoted at P90 and SLOs at
   P90/P99. A model fitted to medians is unconstrained exactly where it is used.
4. **More information per run.** A run with thousands of requests is one point to an aggregate
   loss and thousands of observations to a distribution loss. That identifies more parameters
   per run without needing more runs. Uncertainty is still clustered by run, since requests in
   one run are not independent.
5. **Robustness.** A median-normalised Wasserstein-1 distance is in units of the metric,
   comparable across models and loads, and less sensitive to a few outliers than a mean error.

**What distributions cost, and the guard rails.**
- They need the same population on both sides. A distribution comparison over different request
  sets (warmup on one side, a different window, different cancellation handling) is
  meaningless, so the request set is asserted (§2.2).
- They are noisier per run than a median, so a distance enters a loss only after its
  seed-to-seed spread has been measured (§4.2, signal), and is weighted by inverse noise
  variance.
- A flexible term can fit any shape, so distributional losses make the "few, bounded, physical
  parameters" rule (§4.3) stricter, not looser.

**How both are used.**

| question | measure |
|---|---|
| is a deployment's typical cost right (what a user plans capacity on) | aggregate error: MAPE per metric, signed median |
| are the mechanisms right (what makes a prediction trustworthy off the training set) | distribution distance: W1, KS, p10/p50/p90/p99 quantile errors; server-state distributions |
| is something grossly broken | gross-deviation flags (e.g. KV usage off by more than 30 points or 2x, hit rate off by more than 0.2) |

Fits minimise distribution distance on the informative metrics. Both aggregate error and
distribution distance are reported for every held-out set, and the landing gate (§7) requires
both to improve. An improvement in one with a loss in the other means a mechanism is still
wrong, and the change does not land.

## 5. Where direct data is missing: model, don't assume

When no measurement covers an operator at the needed hardware, dtype or shape, build a
mechanistic model from the kernel's implementation: tiling, FLOPs and bytes per tile, wave
quantisation over SMs, launch and sync overhead, the roofline. Fit its few physical parameters
on all indirect data (the same kernel on other hardware through a declared transfer, sibling
kernels, forward-pass data, end-to-end FIT decomposition). Report identifiability, and run a
parameter-recovery check. A parameter the data cannot identify is fixed at a stated physical
value with a range; it is never called fitted. Such coefficients carry the evidence class
`modelled`, distinct from `measured`, `fitted` and `assumed`.

## 6. Version discipline

A coefficient fitted on data from engine version V may be used for version W only if every
function on the term's code path is unchanged between V and W, shown by a per-function diff
that ignores comments, typing and logging. A branch that changed counts as unchanged for a
particular point only when that point's recorded configuration provably disables it. Terms
that do not pass through engine code (collectives, NVMe, host link, GPU kernel floors) may use
any version's data, with the reason recorded.

## 7. The landing gate

Every change is pre-registered (data, split, metric choice, pass condition) before any held-out
number is computed. It lands only if:

1. held-out error improves, per factor where a stratum has at least 10 points;
2. held-out distribution distance improves;
3. nothing else gets worse: the other metrics, the named anchors, the transfer check, and
   seed-averaged comparisons within the pooled standard error;
4. the code change carries tests that fail when the fix is reverted.

A change that fails is parked with its evidence, or lands as a recorded deviation only when the
failure is shown to be measurement noise or a definitional artefact, and the record says so.

## 8. The repeatable procedure

1. Enumerate every source; classify it (§3); check its reputation and whether it has updates.
2. Build or extend the run inventory; record each run's blocker and what would unlock it.
3. Draw or extend the FIT/HELD partition (§3.1). Hash it.
4. Resolve every run's engine behaviour from its own build (§2.4).
5. Replay FIT points apples to apples (§2); assert definitions, windows and load.
6. Triage gross KV deviations for parity (§2.6) before anything else; exclude points with an
   unresolved parity defect, with a named reason.
7. Keep a per-coefficient data ledger: every source, used or not, and why. An unused relevant
   source is a defect to close.
8. Diagnose on FIT points across every level of §1 before fitting anything. A term fitted to
   absorb a different mechanism's error is wrong.
9. Choose the metrics per term (§4.2); pre-register the gate (§7).
10. Fit; check identifiability; freeze.
11. Evaluate on HELD; report per factor, with stratified sampling fractions and run-clustered
    bootstrap intervals; report the count of single-run strata, which make intervals optimistic.
12. Land or park (§7). Record every landing and deviation where the next fitter will find it.
