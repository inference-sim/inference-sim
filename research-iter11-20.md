# Research: 3-Layer Optimization with Bayesian Parameter Search (Iterations 11-13)

## Protocol
Each iteration: Strategy (parameterized template) → Hypotheses → Implementation → Experiment → Bayesian Optimization → Review → Verdict

## Baseline
- Routing: `pa:3,qd:2,kv:2` (weighted) | Scheduler: FCFS | Priority: constant | Admission: always-admit
- Workload: Orthogonal mixed-SLO (critical/standard/sheddable with IDENTICAL token distributions)
- 8 instances, rate=high enough to create queueing

## Prior Art
- **Iters 1-10**: Static `pa:3,qd:2,kv:2` is optimal for routing. Scheduling adds nothing when routing separates traffic. KV-util scorer hurts under KV pressure. Per-SLO routing fragments cache affinity.
- **Discussion #451**: SLOTieredPriority achieves -50.8% critical TTFT P99 with orthogonal workloads. Priority policy is the primary lever. Compute floor at ~132ms.
- **Key gap**: Admission control is untested. SLO-differentiated admission operates orthogonally to routing (doesn't fragment cache).

---

# Idea 1 (Iter 11): SLO-Gated Admission with Priority Cascade

## Strategy Template

A 3-layer compound with 7 tunable parameters:

### Layer 1: SLO-Gated Admission (`SLOGatedAdmission`)
```
Admit(req, state):
  // Critical: always admit
  if req.SLOClass == "critical": return true

  // Compute cluster load signal
  totalQueue = sum(snap.EffectiveLoad() for snap in state.Snapshots)
  avgQueue = totalQueue / len(state.Snapshots)

  // Standard: admit if below standard threshold
  if req.SLOClass == "standard":
    return avgQueue < PARAM_standard_queue_threshold

  // Sheddable: admit if below sheddable threshold (tighter)
  return avgQueue < PARAM_sheddable_queue_threshold
```

**Parameters**:
- `PARAM_standard_queue_threshold` (range: 2-20, default: 10)
- `PARAM_sheddable_queue_threshold` (range: 1-10, default: 5)

### Layer 2: Routing (`pa:3,qd:2` — the iter 8 optimal)
Static weights, no KV-utilization scorer. Preserves cache affinity without KV penalty.

**Parameters**: None (fixed from iter 8 finding)

### Layer 3: Scheduling (`SLOClassPriority` + `priority-fcfs`)
```
priority = PARAM_base[slo_class] + PARAM_age_weight × age
```

**Parameters**:
- `PARAM_base_critical` (range: 5-20, default: 10)
- `PARAM_base_standard` (range: 2-10, default: 5)
- `PARAM_base_sheddable` (range: 0-5, default: 1)
- `PARAM_age_weight` (range: 1e-6 to 1e-4, default: 1e-5)
- `PARAM_starvation_cap` (range: 500ms-5000ms — max age before sheddable overrides critical)

**Total: 7 parameters** (within #451's recommended 5-7 sweet spot)

## Hypotheses

**H11-1: Critical TTFT P99 drops 40-50% vs baseline**
- *Mechanism*: Admission gate rejects sheddable when queues form → fewer requests competing for batch slots → critical gets scheduled faster. Priority cascade ensures critical goes first even when co-located.
- *Quantitative prediction*: If baseline is 269ms (from #451), expect 135-160ms for critical.
- *Why significant*: This approaches the compute floor (~132ms for 768-token prefill). The gap between 269ms and 132ms is pure queueing delay, which admission shedding + priority scheduling removes.

**H11-2: Throughput is preserved (≥95% of baseline)**
- *Mechanism*: Shedding only activates under overload (avgQueue > threshold). Under normal load, all requests are admitted. The rejected sheddable requests represent excess load that would have caused queueing anyway.
- *Quantitative prediction*: ≥16,338 tps (95% of 17,198).

**H11-3: Sheddable TTFT P99 degrades but within SLO budget**
- *Mechanism*: Sheddable requests that ARE admitted face longer queues (critical jumps ahead). Expected degradation: 50-100% worse than baseline. But with a 500ms SLO budget, even 2x baseline (540ms) is within budget.
- *Quantitative prediction*: 400-550ms (vs baseline 269ms).

**H11-4: The compound (admission + scheduling) is super-additive**
- *Mechanism*: Admission reduces total load → shorter queues for ALL classes. Priority reorders remaining queue → critical goes first. Neither alone achieves the full effect:
  - Admission alone: reduces load but critical still waits behind standard
  - Priority alone: reorders but doesn't reduce total load (all admitted)
  - Together: reduced load + optimal ordering = near-compute-floor for critical
- *Control*: Test admission-only (FCFS), priority-only (always-admit), compound
- *Prediction*: Compound > sum of individual improvements by 10-20%

## Experiment Design

### Orthogonal Workload (from #451)
All 3 SLO classes use IDENTICAL token distributions:
```yaml
version: "2"
aggregate_rate: RATE
clients:
  - id: critical-users
    slo_class: critical
    rate_fraction: 0.33
    prefix_group: shared-prompt
    prefix_length: 512
    arrival: {process: gamma, params: {shape: 0.25}}  # CV=2.0
    input_distribution: {type: gaussian, params: {mean: 256, std_dev: 100, min: 64, max: 512}}
    output_distribution: {type: exponential, params: {mean: 128, min: 16, max: 1024}}
  - id: standard-users
    slo_class: standard
    rate_fraction: 0.34
    prefix_group: shared-prompt
    prefix_length: 512
    arrival: {process: gamma, params: {shape: 0.25}}
    input_distribution: {type: gaussian, params: {mean: 256, std_dev: 100, min: 64, max: 512}}
    output_distribution: {type: exponential, params: {mean: 128, min: 16, max: 1024}}
  - id: sheddable-users
    slo_class: sheddable
    rate_fraction: 0.33
    prefix_group: shared-prompt
    prefix_length: 512
    arrival: {process: gamma, params: {shape: 0.25}}
    input_distribution: {type: gaussian, params: {mean: 256, std_dev: 100, min: 64, max: 512}}
    output_distribution: {type: exponential, params: {mean: 128, min: 16, max: 1024}}
```

### Component Isolation Matrix (6 configs)

| # | Config | Admission | Routing | Scheduler | Priority |
|---|--------|-----------|---------|-----------|----------|
| 1 | Baseline | always-admit | pa:3,qd:2,kv:2 | fcfs | constant |
| 2 | Admission-only | slo-gated | pa:3,qd:2,kv:2 | fcfs | constant |
| 3 | Priority-only | always-admit | pa:3,qd:2,kv:2 | priority-fcfs | slo-class |
| 4 | Routing-only (iter8) | always-admit | pa:3,qd:2 | fcfs | constant |
| 5 | Admission + Priority | slo-gated | pa:3,qd:2,kv:2 | priority-fcfs | slo-class |
| 6 | **Full compound** | slo-gated | pa:3,qd:2 | priority-fcfs | slo-class |

### Bayesian Optimization
After verifying default parameters, run scikit-optimize (30 calls × 3 seeds):
- **Objective**: Minimize critical TTFT P99
- **Constraints**: Throughput ≥ 95% of baseline, Sheddable rejection rate < 30%
- **Search space**: 7 parameters as defined above

---

## Reviews for Idea 1

### GPT-4o Review (Iter 11)

**1. EffectiveLoad vs QueueDepth for admission gating.**
EffectiveLoad = QueueDepth + BatchSize + PendingRequests. PendingRequests counts requests that have been routed but not yet absorbed into the WaitQ (the window between `RoutingDecisionEvent.Execute` incrementing the counter and the `QueuedEvent` decrementing it). At high arrival rates with gamma CV=2.0 bursts, multiple requests can arrive at the same timestamp. Since cluster events process Arrival(0) then Admission(1) then Routing(2) in priority order, a burst of N co-temporal arrivals produces N pending-but-not-yet-queued requests that inflate EffectiveLoad transiently. Using EffectiveLoad for admission means the gate reacts to routing-pipeline backpressure, not just actual queue depth -- this is arguably *more* conservative (admits fewer requests during bursts), which benefits critical TTFT. However, it also means the thresholds (2-20 for standard, 1-10 for sheddable) are measured against a *different* quantity than what users would intuit as "queue length." **Recommendation**: Use QueueDepth-only as the primary signal and test EffectiveLoad as a variant. At minimum, document which signal is used and why the threshold ranges were chosen against it.

**2. Starvation cap and AgeWeight crossover.**
The math checks out and the crossover is intentional. With AgeWeight=1e-5 and default bases (critical=10, sheddable=3), a sheddable request overtakes critical after (10-3)/(1e-5) = 700,000 ticks = 0.7 seconds. At starvation_cap=2000ms, a sheddable request reaches priority = 3 + 1e-5 * 2e6 = 23, well above critical's base of 10. But the cap is on the *Bayesian search range*, not a hard clamp in the priority formula. If the intent is that sheddable should *eventually* overtake critical (anti-starvation), this is correct by design. If the intent is that critical should *always* win, a hard ceiling is needed. **Recommendation**: State the design intent explicitly. If anti-starvation crossover is desired, the 0.7s crossover with default params seems too fast -- a fresh critical request would lose to a 0.7s-old sheddable request, which undermines the admission gate that just fought to protect critical. Consider either (a) widening the base gap (critical=20, sheddable=1) or (b) reducing AgeWeight to 1e-6 (crossover at 9s, matching `SLOBasedPriority`'s original calibration in `priority.go:102`).

**3. Missing rate sweep.**
The 6-config matrix tests mechanism isolation but at a single load point. The admission gate is a step function: below threshold it is invisible, above threshold it sheds. A single rate cannot characterize the crossover. **Recommendation**: Add a 5-point rate sweep (e.g., 0.5x, 0.8x, 1.0x, 1.5x, 2.0x of saturation capacity) for at least configs 1, 5, and 6. This adds 15 runs (trivial cost) and reveals the admission activation curve plus the throughput-vs-shedding tradeoff.

**4. Bayesian optimization feasibility.**
30 calls x 3 seeds = 90 runs. At 1000 requests, 8 instances, llama-8b/H100/TP=2: step time ~11.8ms (256/128 workload), capacity ~85 req/s per instance, so 1000 requests at high rate completes in ~2-3s of sim time. Wall-clock per run is dominated by Go startup + event processing, typically 1-5s. Total: ~2-8 minutes, not 45 minutes. The 45-minute estimate appears to assume 30s/run which would apply to 10,000+ request runs. **Recommendation**: The budget is generous. Consider increasing to 50 calls for better convergence in 7D space, or increasing to 2000 requests per run for more stable P99 estimates (P99 of 1000 requests = the 10th-worst value, noisy).

### Gemini Review (Iter 11)

**1. vLLM/llm-d state of the art on SLO-aware admission.**
Neither vLLM nor llm-d has SLO-aware admission control today. vLLM returns HTTP 503 when its internal queue is full (a hard capacity wall, not a differentiated policy). The Gateway API Inference Extension (GAIE), which llm-d builds on, defines exactly two criticality levels -- `Critical` and `Sheddable` -- in the `InferenceModel` CRD. The Endpoint Selection Extension (ESE) implements differentiated *filtering*, not admission gating: Critical requests pass if any endpoint has queue < 50; Sheddable requests require queue <= 5 AND KV cache <= 80%, and are dropped entirely if no endpoint qualifies. This happens at the endpoint picker (after the gateway, before the pod), not at an admission gate. BLIS's `SLOGatedAdmission` is therefore ahead of production -- it adds a cluster-wide admission gate that neither vLLM nor the GAIE currently implements. This is a strength for research but means there is no production system to validate against. **Recommendation**: Frame this as "what GAIE *should* add" rather than "what GAIE does." The GAIE filter chain is per-endpoint; BLIS's avgQueue is cluster-wide. Consider also testing a per-endpoint variant (reject sheddable if the *routed* instance's queue > threshold) to match how the ESE actually operates.

**2. Rejection semantics: BLIS vs Kubernetes reality.**
In BLIS, rejection is permanent and silent -- `rejectedRequests++` and the request vanishes (line 125 of `cluster_event.go`). In production Kubernetes, a rejected request triggers an HTTP 429 or 503, and clients retry with exponential backoff. This means BLIS's rejection rate maps to a *retry storm* in production, not a clean load shed. A 30% rejection rate constraint (in the Bayesian search) would produce 30% retry traffic in production, potentially *increasing* total load and creating a feedback loop the simulator does not model. **Recommendation**: Either (a) acknowledge that rejection rate X in BLIS maps to effective load increase of ~X/(1-X) in production (30% rejection = 43% retry amplification), or (b) model retries explicitly (re-inject rejected requests after a backoff delay). Option (b) would make the admission threshold calibration directly transferable to production. Without retries, the optimal thresholds found by Bayesian search will be too aggressive for deployment.

**3. Gateway-level vs per-instance admission.**
llm-d engineers would strongly prefer gateway-level admission (before routing) because: (a) it avoids wasting routing computation on doomed requests, (b) it provides a single rejection point for observability (one metric, one log), and (c) it matches the GAIE architecture where the ESE runs as a sidecar to the gateway, not per-pod. BLIS already places admission before routing (`AdmissionDecisionEvent` priority=1, `RoutingDecisionEvent` priority=2), which is architecturally correct. However, the admission signal (`avgQueue`) is derived from per-instance snapshots, which means it depends on snapshot freshness (INV-7). At `--snapshot-refresh-interval` > 5ms, stale KV/queue data could make the admission gate oscillate (admitting bursts during stale windows, then over-rejecting when the refresh catches up). **Recommendation**: Test with both 1ms and 50ms snapshot refresh intervals. If the admission gate is sensitive to staleness, that is a production-relevant finding -- real ESEs poll pod metrics at 1-5s intervals.

**4. The 7-parameter search is operationally unreasonable.**
Kubernetes operators tune 2-3 parameters per component, not 7 simultaneously. The GAIE exposes exactly 2 knobs for criticality-aware filtering: queue depth threshold and KV cache threshold (per criticality level). In practice, operators set these once based on SLO contracts and rarely re-tune. The 5 scheduling parameters (3 bases + age weight + starvation cap) are invisible to K8s operators entirely -- scheduling is a vLLM-internal concern. **Recommendation**: Split the search into two phases: (1) Fix scheduling params at the #451 `SLOTieredPriority` defaults (already validated) and search only the 2 admission thresholds. This gives the production-actionable result. (2) Then run the full 7-param search as a research exercise to quantify how much additional headroom scheduling tuning provides. The 2-param result is what goes in the llm-d demo; the 7-param result goes in the paper.

**5. Most compelling demo result.**
The single most compelling number for an llm-d demo is the *critical TTFT P99 vs rejection rate Pareto curve*. Not a single point, but the tradeoff: "At 5% sheddable rejection, critical TTFT P99 drops 35%. At 15% rejection, it drops 48%. At 0% rejection (baseline), no improvement." This directly answers the operator question: "How much do I sacrifice to protect my premium tier?" The rate sweep recommended in the GPT-4o review (point 3) generates this curve naturally. Plot it as a two-axis chart: x = sheddable rejection rate, y = critical TTFT P99, with each rate as a different point. This is immediately actionable for capacity planning.
