# H16: Gamma vs Poisson Tail Latency

**Status:** Confirmed with nuance
**Resolution:** Partial confirmation with surprise
**Family:** Workload/arrival
**VV&UQ:** Validation
**Tier:** 2
**Type:** Statistical (Dominance)
**Date:** 2026-02-22
**Rounds:** 2

## Hypothesis

> "Bursty (Gamma) arrivals should produce worse tail latency than Poisson at the same average rate."

## Experiment Design

**Classification:** Statistical/Dominance (Gamma TTFT p99 > Poisson TTFT p99 across all seeds)

**Configurations compared:**
- A (Poisson): `--model meta-llama/llama-3.1-8b-instruct --num-instances 4 --seed $SEED --workload-spec poisson_wl.yaml --routing-policy least-loaded --scheduler fcfs --priority-policy constant --admission-policy always-admit --log error --summarize-trace --trace-level decisions`
  - Workload YAML: `aggregate_rate: $RATE`, `num_requests: $NUM_REQS`, `arrival: { process: poisson }`, `input_distribution: { type: gaussian, params: { mean: 256, std_dev: 50, min: 64, max: 512 } }`, `output_distribution: { type: gaussian, params: { mean: 128, std_dev: 30, min: 32, max: 256 } }`
- B (Gamma): Identical CLI flags and workload YAML except `arrival: { process: gamma, cv: 3.5 }`

**Experiments:**
| Experiment | Rate | Requests | rho (approx) | Purpose |
|------------|------|----------|--------------|---------|
| Exp 1 (Round 1) | 1000 | 500 | 2.95x | Core hypothesis test |
| Exp B2 (Round 2) | 200 | 500 | 0.59x | Sub-saturation control (ED-2) |
| Exp C1 (Round 2) | 1000 | 2000 | 2.95x | Larger sample for robust p99 |

**Controlled variables:** Instances (4), routing (least-loaded), scheduler (fcfs), priority (constant), admission (always-admit), input/output distributions (gaussian), KV blocks (default, abundant)

**Varied variable:** Arrival process (poisson vs gamma with CV=3.5)

**Seeds:** 42, 123, 456

**Preconditions verified:**
- Gamma CV=3.5 produces shape=0.082, above the 0.01 fallback threshold (`sim/workload/arrival.go:128`), so the Gamma sampler is actually used
- Rate 1000 with 4 instances at step_time ~11.8ms creates ~2.95x overload; Rate 200 creates ~0.59x utilization
- No KV pressure (default blocks), no preemptions observed across all 18 runs, no cache hit confound
- Both configurations use identical workload YAML seeds for the RNG

**Statistical note:** scipy is not available in this environment. Significance is assessed by consistent directionality across seeds and effect size magnitude, not formal hypothesis tests.

## Results

### Exp 1: Core (rate=1000, 500 requests, 3x overload)

| Seed | Arrival | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | E2E p99 (ms) |
|------|---------|----------------|----------------|----------------|---------------|
| 42   | Poisson | 122.69         | 193.34         | 1543.65        | 2197.26       |
|      | Gamma   | 130.35         | 244.46         | 1551.59        | 2222.67       |
|      | Ratio   | 1.06x          | **1.26x**      | 1.01x          | 1.01x         |
| 123  | Poisson | 138.59         | 230.76         | 1565.39        | 2264.53       |
|      | Gamma   | 188.56         | 325.11         | 1634.27        | 2231.27       |
|      | Ratio   | 1.36x          | **1.41x**      | 1.04x          | 0.99x         |
| 456  | Poisson | 130.87         | 257.35         | 1546.88        | 2210.78       |
|      | Gamma   | 150.87         | 280.84         | 1566.30        | 2243.56       |
|      | Ratio   | 1.15x          | **1.09x**      | 1.01x          | 1.01x         |

Cross-seed: Gamma TTFT p99 worse in **3/3** seeds, avg ratio **1.25x** (range: 1.09x-1.41x).
Note: Seed 456 at 1.09x is below the 10% threshold. Seed 123 E2E p99 ratio of 0.99x is within noise.

### Exp B2: Sub-saturation control (rate=200, 500 requests, 0.59x utilization)

| Seed | Arrival | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | E2E p99 (ms) |
|------|---------|----------------|----------------|----------------|---------------|
| 42   | Poisson | 21.24          | 33.10          | 1333.98        | 2028.94       |
|      | Gamma   | 29.47          | 51.69          | 1366.38        | 2150.92       |
|      | Ratio   | 1.39x          | **1.56x**      | 1.02x          | 1.06x         |
| 123  | Poisson | 21.10          | 32.28          | 1366.22        | 2167.40       |
|      | Gamma   | 31.20          | 51.91          | 1437.26        | 2146.31       |
|      | Ratio   | 1.48x          | **1.61x**      | 1.05x          | 0.99x         |
| 456  | Poisson | 21.87          | 35.77          | 1363.33        | 2082.05       |
|      | Gamma   | 32.24          | 64.78          | 1361.63        | 2122.80       |
|      | Ratio   | 1.47x          | **1.81x**      | 1.00x          | 1.02x         |

**SURPRISE:** Gamma effect is *stronger* at sub-saturation (avg 1.66x vs 1.25x at overload). Effect does NOT vanish as predicted.

Cross-seed: Gamma TTFT p99 worse in **3/3** seeds, avg ratio **1.66x** (range: 1.56x-1.81x).

### Exp C1: Larger sample (rate=1000, 2000 requests, 3x overload)

| Seed | Arrival | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | E2E p99 (ms) |
|------|---------|----------------|----------------|----------------|---------------|
| 42   | Poisson | 708.92         | 1648.08        | 2746.86        | 3750.66       |
|      | Gamma   | 823.09         | 1903.72        | 2876.48        | 3881.20       |
|      | Ratio   | 1.16x          | **1.16x**      | 1.05x          | 1.03x         |
| 123  | Poisson | 750.17         | 1754.95        | 2793.82        | 3829.43       |
|      | Gamma   | 654.67         | 1460.23        | 2710.98        | 3819.33       |
|      | Ratio   | 0.87x          | **0.83x**      | 0.97x          | 1.00x         |
| 456  | Poisson | 774.71         | 1737.32        | 2816.13        | 3852.56       |
|      | Gamma   | 819.23         | 1838.63        | 2871.17        | 3887.44       |
|      | Ratio   | 1.06x          | **1.06x**      | 1.02x          | 1.01x         |

**SURPRISE:** Effect is greatly attenuated with larger sample. Seed 123 shows Gamma *better* (0.83x). TTFT means jumped from ~130ms (500 req) to ~750ms (2000 req) because queue grows linearly under sustained overload, drowning out the burst signal.

Cross-seed: Gamma TTFT p99 worse in **2/3** seeds, avg ratio **1.02x** (range: 0.83x-1.16x).

### Cross-Experiment Summary

| Experiment | Rate | Reqs | Gamma wins | Avg TTFT p99 ratio | Range |
|------------|------|------|-----------|-------------------|-------|
| Exp 1      | 1000 | 500  | 3/3       | 1.25x             | 1.09x-1.41x |
| Exp B2     | 200  | 500  | 3/3       | 1.66x             | 1.56x-1.81x |
| Exp C1     | 1000 | 2000 | 2/3       | 1.02x             | 0.83x-1.16x |

**Conservation (INV-1)**: PASS for all 18 runs (6 per experiment).
**Preemptions**: 0 across all 18 runs. **Cache hits**: 0. No confounds.

## Root Cause Analysis

### Mechanism: Gamma burstiness creates transient queue depth spikes (RCV-1, RCV-3)

The causal chain from arrival process to TTFT tail latency traces through five code paths:

**Step 1 — Gamma inter-arrival time generation** (`sim/workload/arrival.go:116-132`):
`NewArrivalSampler` computes shape = 1/CV^2 = 1/12.25 = 0.0816 and scale = mean_IAT * CV^2 (`arrival.go:125-127`). Since shape < 1, `gammaRand` at `arrival.go:52-55` uses the Ahrens-Dieter transformation: `Gamma(0.082) = Gamma(1.082) * U^(1/0.082)`. The exponent 1/0.082 = 12.2 means `U^12.2` is heavily concentrated near zero for most draws (e.g., U=0.5 produces 0.5^12.2 = 0.00021). This produces many near-zero inter-arrival times (burst clusters) interspersed with occasional large gaps.

**Step 2 — Request generation with IAT accumulation** (`sim/workload/generator.go:134-140`):
The generator accumulates `currentTime += iat` in a loop (`generator.go:139-140`). Near-zero IATs from the Gamma sampler produce clusters of requests with nearly identical `ArrivalTime` values. Poisson's exponential IATs produce more uniform spacing.

**Step 3 — Cluster routing distributes burst-arriving requests** (`sim/cluster/cluster_event.go:85-93`, `148-189`):
Each request triggers `ClusterArrivalEvent.Execute` at `cluster_event.go:85` which chains through `AdmissionDecisionEvent` (`cluster_event.go:109`) and `RoutingDecisionEvent` (`cluster_event.go:148`). `LeastLoaded.Route` at `sim/routing.go:107-124` selects the instance with minimum `EffectiveLoad()` (`routing.go:23-25`: QueueDepth + BatchSize + PendingRequests). The `pendingRequests` counter incremented at `cluster_event.go:185` prevents pile-on for same-timestamp requests. However, burst-arriving requests still create N/4 requests per instance (N = burst size), all entering the queue faster than the step loop can drain.

**Step 4 — Queue accumulation and wait time** (`sim/event.go:52-63`, `sim/simulator.go:304-305`):
`QueuedEvent.Execute` at `event.go:52` calls `sim.EnqueueRequest` (`simulator.go:304-305`) which appends to the FIFO WaitQ. If a step is already in progress, these requests wait until the current step completes and the next batch is formed. During a burst, the WaitQ grows by the burst size (minus the current batch capacity), and requests at the back of the queue wait multiple step durations.

**Step 5 — TTFT computation** (`sim/simulator.go:432-436`):
When a request's prefill completes, `FirstTokenTime = now + currStepAdvance + OutputTokenProcessingTime() - req.ArrivalTime` (`simulator.go:434`). Requests that queued during a burst have `now` many milliseconds after `ArrivalTime`, producing large TTFT values. The p99 captures the worst burst events — the last requests in the largest clusters.

### Why the effect is STRONGER at sub-saturation (RCV-2 — surprise)

First-principles calculation: At rate=200 (0.59x utilization), the baseline TTFT is ~21ms for Poisson (single step + alpha overhead, minimal queuing). A Gamma burst depositing 5-6 requests on one instance in < 1ms creates a transient queue of 4-5. At ~11.8ms per step, this queue takes 4 * 11.8 = 47ms to drain. The burst tail sees TTFT of ~50-65ms, which is 1.5-1.9x the Poisson baseline of ~33ms. This matches the observed 1.66x.

At overload (rate=1000, 2.95x), the Poisson baseline TTFT is already elevated (~130-230ms) because a persistent queue exists even without bursts. The Gamma bursts add the same absolute ~30-50ms increment, but on a higher baseline, yielding a smaller *relative* ratio (1.25x). The absolute TTFT difference is comparable (~50-90ms in Exp 1 vs ~20-30ms in Exp B2), but the relative ratio flips because the baseline differs by 7x.

### Why the effect VANISHES at 2000 requests (RCV-2 — surprise)

At rate=1000 with 2000 requests, the system is overloaded for 4x as long. The queue grows linearly throughout the simulation: with arrival rate exceeding capacity by ~660 req/s, the queue depth at request N is approximately `N * (1 - capacity/rate)`. By request 2000, the queue has accumulated ~1300 pending requests. TTFT for late-arriving requests is dominated by this linear queue backlog (~750ms mean), not by transient burst spikes. The Gamma burst signal (50-100ms absolute) becomes noise against the 750ms baseline, producing ratios near 1.0x. Seed 123 showing 0.83x (Gamma better) is plausible because a Gamma gap (long inter-arrival pause) allows the queue to partially drain, producing a lower p99 than Poisson's relentless steady arrivals.

### Control experiment (RCV-4)

The Poisson configuration is the control — it disables ONLY the burstiness mechanism (CV=1 vs CV=3.5). Exp B2 (sub-saturation) was intended as a second control to test load-dependence, but instead revealed the mechanism operates through relative (not absolute) queue depth spikes.

## Devil's Advocate (RCV-5)

**If this is "Confirmed with nuance," argue why it might be Refuted:**

The 2000-request experiment (C1) shows the effect is not robust to longer time horizons. Seed 123 at 0.83x directly contradicts the hypothesis. The core experiment (Exp 1) has one seed (456) below the 10% significance threshold. The sub-saturation result contradicts the original mechanism explanation (queue buildup under overload), suggesting the mechanism is incompletely understood. The 500-request p99 is computed from ~5 data points, making the 1.25x average unreliable. A formal statistical test might find no significant difference.

**If this were "Refuted," argue why it might be Confirmed:**

The directional consistency (3/3 seeds in Exp 1 and B2, 5/6 across both) with a mean 1.25x-1.66x effect is unlikely by chance alone. The sub-saturation result actually *strengthens* the finding — bursts affect relative TTFT more when the baseline is low. The C1 anomaly is explained by a well-understood mechanism (linear queue growth drowning out burst signal), not by a failure of the burst mechanism itself. The mechanism is traced through 5 concrete code paths.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Gamma CV=3.5 produces 9%-41% worse TTFT p99 than Poisson at 500 req (Exp 1) | Confirmation | Documented here |
| Effect is stronger (56%-81%) at sub-saturation, not weaker (Exp B2) | Surprise | Documented here |
| Effect vanishes under sustained overload at 2000 req (Exp C1) | Surprise | Documented here |
| E2E p99 is insensitive to arrival burstiness across all experiments | Confirmation | Documented here |
| Seed 123 E2E p99 ratio 0.99x (Exp 1) is within noise | Confirmation | Documented here |
| Gamma shape=0.082 engages Ahrens-Dieter transformation correctly | Confirmation | Documented here |
| The burst mechanism is relative-queue-spike, not absolute-queue-buildup | Surprise | Documented here |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? INV-1 (conservation) confirmed across all 18 runs

## Scope and Limitations (RCV-6)

- **Operating point tested:** rate=200 (0.59x), rate=1000 (2.95x), 4 instances, gaussian input (mean=256), gaussian output (mean=128), least-loaded routing, always-admit, default KV blocks, 500 and 2000 requests
- **Parameters findings depend on:** Effect magnitude depends on request count (stronger at 500, vanishes at 2000 under overload) and relative (not absolute) load level
- **What was NOT tested:**
  - Intermediate request counts (750, 1000) to find the crossover point
  - Different CV values (1.5, 2.0, 5.0) for monotonicity characterization
  - Different routing policies (round-robin would not balance bursts)
  - Weibull arrivals (alternative bursty distribution)
  - Higher seed count for formal statistical power
- **Generalizability:** The directional finding (Gamma = worse TTFT tails) holds at 500 requests across both load levels. It does NOT hold at 2000 requests under sustained overload. This limits practical applicability to short bursts / moderate workloads.
- **Uncertainty quantification:** scipy not available (legacy threshold exemption). 3 seeds, 500-2000 requests each. Exp 1: 3/3 seeds directionally consistent but 1 below 10% threshold. Exp B2: 3/3 seeds, all above 50% threshold. Exp C1: 2/3 seeds, including one reversal. Medium confidence in the directional finding for short workloads; low confidence for sustained overload.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT p99 ratio, Exp 1 (500 req, 3x overload) | 1.25x avg (1.09x-1.41x) | Medium -- consistent direction, one seed below threshold |
| TTFT p99 ratio, Exp B2 (500 req, sub-sat) | 1.66x avg (1.56x-1.81x) | High -- consistent direction, large effect |
| TTFT p99 ratio, Exp C1 (2000 req, 3x overload) | 1.02x avg (0.83x-1.16x) | Low -- mixed direction, effect vanished |
| Sample size | 3 seeds x (500 + 500 + 2000) req x 2 configs = 18 runs | Medium |
| Mechanism | Gamma burst clustering -> relative queue spikes -> TTFT p99 | High -- 5 code paths traced, first-principles matches data |

## Implications for Users

- **Workload modeling matters for short workloads**: Poisson arrivals underestimate tail latency for bursty real-world traffic. At 500 requests, Gamma (CV=3.5) produces 9%-41% worse TTFT p99 at overload and 56%-81% worse at sub-saturation.
- **Effect is load-duration dependent**: Under sustained overload (2000 requests), linear queue growth dominates and arrival burstiness becomes irrelevant. For capacity planning of long-running workloads, Poisson may be a sufficient approximation.
- **Sub-saturation surprise**: Burstiness affects TTFT *more* at low utilization (1.66x) than at high utilization (1.25x). This is because the relative magnitude of burst-induced queue spikes is larger when the baseline queue is shallow. Users running below saturation should still model bursty arrivals if TTFT p99 matters.
- **TTFT is the sensitive metric**: E2E latency is dominated by decode time and is insensitive to arrival patterns across all experiments (E2E p99 ratio ~1.0x everywhere).

## Reproducing

```
cd hypotheses/h16-gamma-vs-poisson
./run.sh
```
