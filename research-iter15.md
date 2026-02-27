# Iteration 15: Epoch-Based Online Weight Adaptation

## Problem
Static routing weights (even Bayesian-optimized pa:4,qd:3) are optimal for ONE load level. In production, load varies continuously (diurnal patterns, traffic spikes, model switching). No single weight configuration is optimal across all regimes. Prior iterations proved this: pa:3,qd:2 wins at low load, pa:4,qd:3 wins at high load, and pa:4,qd:2 is catastrophic at any load.

## Prior Art
- **Iter 13**: Bayesian optimization found pa:4,qd:3 as globally optimal at rate=2000 (131.8ms, 69.3% goodput)
- **Iter 14**: PA:QD sweep confirmed the safety rule: ratio ≤ 1.33. pa:4,qd:2 causes 3570ms cascade
- **Iter 8**: KV-utilization scorer is counterproductive — drop it
- **Iter 11**: SLO-gated admission is the breakthrough 3rd lever at high load
- **Discussion #451**: Priority policy is the primary SLO differentiator with orthogonal workloads

## The Core Idea: Rejection Rate as a Free Online Learning Signal

The SLO-gated admission controller already computes a rejection rate (requests rejected / total arriving). This rate is:
- **0%** when the system has headroom → safe to increase PA weight (exploit cache)
- **>10%** when the system is overloaded → should increase QD weight (spread load)
- **>30%** when severely overloaded → should maximize QD weight

This signal is:
1. **Free** — already computed by SLO-gated admission
2. **Real-time** — updates every request
3. **Holistic** — captures all system effects (load, KV pressure, cache state, workload mix)
4. **Self-calibrating** — the rejection rate naturally reflects the current operating point

---

# Idea 1: EpochAdaptiveScoring

## Algorithm

```go
type EpochAdaptiveScoring struct {
    currentPA     float64  // current prefix-affinity weight
    currentQD     float64  // current queue-depth weight
    scorer        *WeightedScoring  // current scorer pipeline

    // Epoch tracking
    epochCounter  int      // requests in current epoch
    epochRejects  int      // rejections in current epoch

    // Shared state
    prefixIdx     *PrefixCacheIndex
    observers     []observerFunc

    config        EpochAdaptiveConfig
}

type EpochAdaptiveConfig struct {
    // Epoch size: adapt every N requests
    EpochSize     int     // default: 100, range: [20, 500]

    // Rejection rate thresholds for adaptation
    HighThreshold float64 // default: 0.10 (10%), range: [0.05, 0.30]
    LowThreshold  float64 // default: 0.02 (2%), range: [0.00, 0.10]

    // Weight adjustment per epoch
    StepSize      float64 // default: 0.5, range: [0.1, 2.0]

    // Weight bounds (enforce PA:QD ≤ 1.33 safety rule)
    PAMin float64  // default: 1.0
    PAMax float64  // default: 5.0
    QDMin float64  // default: 2.0
    QDMax float64  // default: 5.0

    // Initial weights (starting point, will be adapted)
    InitialPA float64  // default: 3.0
    InitialQD float64  // default: 2.0
}
```

### Route Method

```go
func (e *EpochAdaptiveScoring) Route(req *Request, state *RouterState) RoutingDecision {
    // Track epoch
    e.epochCounter++

    // Check if epoch is complete
    if e.epochCounter >= e.config.EpochSize {
        rejectionRate := float64(e.epochRejects) / float64(e.epochCounter)

        if rejectionRate > e.config.HighThreshold {
            // Overloaded: decrease PA, increase QD (more load balance)
            e.currentPA = max(e.config.PAMin, e.currentPA - e.config.StepSize)
            e.currentQD = min(e.config.QDMax, e.currentQD + e.config.StepSize)
        } else if rejectionRate < e.config.LowThreshold {
            // Headroom: increase PA, decrease QD (more cache exploitation)
            e.currentPA = min(e.config.PAMax, e.currentPA + e.config.StepSize)
            e.currentQD = max(e.config.QDMin, e.currentQD - e.config.StepSize)
        }
        // else: in the sweet spot, don't change

        // Enforce safety rule: PA:QD ≤ 1.33
        if e.currentPA / e.currentQD > 1.33 {
            e.currentQD = e.currentPA / 1.33
        }

        // Rebuild scorer with new weights
        e.rebuildScorer()

        // Reset epoch
        e.epochCounter = 0
        e.epochRejects = 0
    }

    // Delegate to current scorer
    return e.scorer.Route(req, state)
}

// RecordRejection is called by the admission controller when a request is rejected.
// This is the learning signal.
func (e *EpochAdaptiveScoring) RecordRejection() {
    e.epochRejects++
}
```

### The Admission-Routing Feedback Loop

```
                    ┌─────────────┐
     requests ──→  │  Admission   │──reject──→ rejected count
                    │  (SLO-gated) │             │
                    └──────┬──────┘             │
                           │ admit              │
                           ↓                    ↓
                    ┌──────────────┐    ┌──────────────┐
                    │   Routing    │←── │  Epoch       │
                    │ (PA:QD      │    │  Adapter     │
                    │  adaptive)   │    │ (adjusts     │
                    └──────┬──────┘    │  weights)    │
                           │            └──────────────┘
                           ↓
                    ┌──────────────┐
                    │  Scheduling  │
                    │ (SLO-class   │
                    │  priority)   │
                    └──────────────┘
```

The epoch adapter sits between admission and routing, using the rejection count from admission to adjust routing weights. This closes the loop:
- High rejection → weights shift to QD-heavy → less cache concentration → less queueing → lower rejection
- Low rejection → weights shift to PA-heavy → more cache exploitation → better TTFT

### Convergence Analysis

The system has a natural equilibrium:
- If PA is too high → cache concentration → deep queues → high rejection → PA decreases (corrective)
- If QD is too high → uniform distribution → no cache hits → longer prefill → high queue times → eventually high rejection too

The equilibrium point is where rejection rate = (LowThreshold + HighThreshold) / 2 ≈ 6%. At this point, the system admits ~94% of requests while maintaining cache-aware routing.

The step size controls convergence speed:
- Large step (1.0): fast adaptation, oscillation risk
- Small step (0.1): slow adaptation, may not respond to rapid load changes
- Default (0.5): moderate — converges in ~5-10 epochs (~500-1000 requests)

## Hypotheses

### H15-1: Epoch adaptation converges to pa≈4,qd≈3 at steady rate=2000
**Mechanism**: At rate=2000, the Bayesian-optimal is pa:4,qd:3. The epoch adapter should discover this via the rejection rate signal.
**Prediction**: After 5-10 epochs (500-1000 requests), weights stabilize near pa:4±0.5, qd:3±0.5.
**Metric**: GOODPUT = completed_within_SLO / total_arriving. Target: ≥ 65% (vs 69.3% for Bayesian-static).
**Control**: Static pa:4,qd:3 + SLO-gated at same workload.

### H15-2: Epoch adaptation matches Bayesian-optimal across MULTIPLE rates
**Mechanism**: At rate=500 (low load), PA should drift high (cache-heavy). At rate=2000 (high load), QD should increase (load-heavy). The adapter automatically finds the right point.
**Prediction**: Within 5% of Bayesian-optimal goodput at EACH rate in {500, 1000, 1500, 2000}.
**Metric**: Goodput across rate sweep. A UNIVERSAL strategy that's near-optimal everywhere.
**Control**: Static pa:4,qd:3 (optimized for rate=2000 only).

### H15-3: Under load ramp (rate 500→2000 over 1000 requests), epoch adaptation outperforms static
**Mechanism**: Static pa:4,qd:3 is over-conservative at rate=500 (wastes cache) and near-optimal at rate=2000. Epoch adaptation uses PA-heavy at start (cache-exploiting) then shifts to QD-heavy as load increases.
**Prediction**: 10-20% better goodput than static pa:4,qd:3 across the ramp.
**Control**: Static pa:4,qd:3 at the same ramp workload.

### H15-4: Safety rule (PA:QD ≤ 1.33) prevents the pa:4,qd:2 catastrophe
**Mechanism**: Even if the adapter tries to increase PA past the safety boundary, the enforced cap prevents cascade failure.
**Prediction**: No configuration explored by the adapter produces TTFT > 500ms.
**Control**: Same adapter WITHOUT the safety rule.

## Component Isolation Matrix (6 configs)

| # | Config | Routing | Adaptation | Admission | Scheduler | Priority |
|---|--------|---------|------------|-----------|-----------|----------|
| 1 | RR | round-robin | none | always-admit | fcfs | constant |
| 2 | Static optimal | pa:4,qd:3 | none | slo-gated | priority-fcfs | slo-class |
| 3 | **Epoch adaptive** | epoch-adaptive | rejection-rate | slo-gated | priority-fcfs | slo-class |
| 4 | Static conservative | pa:2,qd:3 | none | slo-gated | priority-fcfs | slo-class |
| 5 | Static aggressive | pa:5,qd:3 | none | slo-gated | priority-fcfs | slo-class |
| 6 | Epoch NO safety | epoch-adaptive (no cap) | rejection-rate | slo-gated | priority-fcfs | slo-class |

Test at rates: 500, 1000, 1500, 2000. Metric: GOODPUT. 3 seeds each.

## Bayesian Optimization Space (7 parameters)

1. `EpochSize` (20-500): adaptation frequency
2. `HighThreshold` (0.05-0.30): rejection rate to trigger QD increase
3. `LowThreshold` (0.00-0.10): rejection rate to trigger PA increase
4. `StepSize` (0.1-2.0): weight adjustment magnitude
5. `PAMin` (1-3): lower bound on PA weight
6. `QDMin` (2-4): lower bound on QD weight
7. `InitialPA` / `InitialQD` (starting weights — Bayesian can find cold-start optimum)

---

## Reviews for Idea 1

### GPT-4o Review (Iter 15)

**1. Multi-armed bandit framing: bang-bang vs. smoother update rules.**

The bang-bang step rule (fixed +/-0.5 per epoch) is not optimal. It is a discretized threshold controller, not a bandit algorithm. The fundamental issue: it treats the problem as having three zones (too high, too low, ok) rather than as a continuous optimization surface. Oscillation at the zone boundaries is inevitable -- when rejection rate hovers near 10%, the controller alternates between "increase QD" and "do nothing" on successive epochs, producing a limit cycle rather than convergence.

Better alternatives: (a) **Multiplicative weights** -- scale PA by `(1 - alpha * rejection_rate)` and QD by `(1 + alpha * rejection_rate)`, giving proportional response. (b) **EXP3-style** -- maintain a probability distribution over a discrete weight grid (e.g., PA in {1,2,3,4,5}), update weights exponentially based on observed reward. This handles the exploration/exploitation tradeoff the bang-bang controller ignores entirely. (c) **Gradient-based** -- use EWMA of rejection rate as a continuous signal and apply proportional-integral control. The dead zone between LowThreshold and HighThreshold wastes samples that carry information.

The convergence analysis claims equilibrium at `(LowThreshold + HighThreshold)/2 = 6%`. This is wrong for bang-bang control -- the system oscillates between the two thresholds, it does not settle at their midpoint. The average rejection rate across oscillation cycles will approximate 6%, but the instantaneous weights will not stabilize. With multiplicative weights, the equilibrium analysis becomes valid.

**2. Load ramp feasibility (H15-3).**

BLIS workload-spec YAML does **not** support time-varying rates. `aggregate_rate` is a single scalar. There is no ramp, schedule, or phase-change specification. To implement a 500-to-2000 ramp over 1000 requests, you would need a custom experiment harness that generates requests with non-stationary inter-arrival times. Concretely: pre-compute arrival times where the instantaneous rate increases linearly, write them as a trace v2 CSV (tracev2.go supports replay), and feed that via `--workload-spec`. This is doable but requires non-trivial harness code -- not a simple YAML configuration.

**3. The 5% goodput gap (H15-2).**

Unachievable. The Bayesian optimizer ran hundreds of evaluations per rate point with full hindsight. The epoch adapter has ~10 epochs of 100 requests to converge, starting from a potentially suboptimal initial point. Furthermore, at rate=500 (low load), rejection rate will be near-zero for ALL weight configurations, so the learning signal vanishes -- the adapter cannot distinguish pa:2,qd:3 from pa:5,qd:3 because neither triggers rejections. Expect 10-20% gap at low rates (where the signal is degenerate) and 5-10% gap at high rates (where the signal is informative). Revise the prediction.

**4. GOODPUT measurement.**

The cluster-level `RawMetrics` includes `PerSLOClass` distributions (`ComputePerSLODistributions` in `sim/cluster/metrics.go`), so per-SLO TTFT/E2E P99 is already available post-simulation. However, this is an **aggregate** measure -- you cannot compute goodput per-epoch during the simulation without adding instrumentation. For H15-1's convergence tracking, you need per-request E2E vs SLO target checked online, which `--results-path` per-request JSON provides (E2E + SLO class per request). Post-hoc binning by epoch is straightforward in `analyze.py`. You do **not** need new BLIS code for this -- parse the per-request output and bucket by request index.

The `RecordRejection()` callback is architecturally problematic: it requires the admission event (`cluster_event.go:124-125`) to call a method on the routing policy, violating the current separation where admission and routing are independent modules receiving `*RouterState`. Consider passing the rejection count via `RouterState` instead -- add an `EpochRejections int` field to `RouterState`, which is already the shared-state bridge between admission and routing. This keeps both modules stateless with respect to each other.
