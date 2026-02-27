# Iterations 11-20: Principled Strategy Evolution with Bayesian Optimization

## Protocol (adopted from discussion #451)

Each iteration produces:
1. **Strategy**: Algorithm as a parameterized template (mechanism + tunable params)
2. **Hypotheses**: Clear predictions explaining WHY the strategy beats baseline SIGNIFICANTLY, with quantitative predictions
3. **Implementation**: Go code following extension recipes
4. **Experiment**: Run with 3+ seeds, compare against baseline
5. **Bayesian optimization**: scikit-optimize sweeps parameter space (30 calls × 3 seeds)
6. **Review**: 3 external LLM judges review each strategy
7. **Verdict**: Confirmed/Refuted with evidence

## Baseline (from discussion #451)

Workload: **Orthogonal multi-turn prefix-heavy** with identical token distributions across SLO tiers:
- 3 SLO classes (critical/standard/sheddable) × SAME token pattern
- prefix=512, input~Gaussian(256,100), output~Exp(128), gamma CV=2.0
- 8 instances, tiered KV (132K GPU + 44K CPU blocks)
- Rate: high enough to create meaningful queueing

Baseline routing: `pa:3,qd:2,kv:2` + FCFS + constant priority + always-admit

## Key Findings to Build On

### From our iterations 1-10:
- PA scorer is self-correcting (returns 0 on miss)
- Orthogonal PA+QD > pre-combined cost-benefit
- KV-utilization scorer is counterproductive under KV pressure
- Scheduling has zero effect when routing perfectly separates traffic
- SLO differentiation in routing fragments cache affinity

### From discussion #451:
- Priority policy is the primary SLO differentiator with orthogonal workloads
- Compute floor at ~132ms for 768-token prefill — scheduling can't beat physics
- SLO-priority routing scorer has marginal value (optimizer chose weight=1)
- Bayesian optimization converges quickly (call 2/30)
- 5-7 parameters is the sweet spot for optimization

### NEW: Admission control as third lever
- `AdmissionPolicy.Admit(req, *RouterState)` sees SLO class + all instance state
- Can selectively shed low-priority requests to protect critical TTFT
- TokenBucket already exists; SLO-aware variants needed

## Iteration Plan

### Iter 11: Adopt #451 protocol — orthogonal workload + baseline replication
- Replicate #451's baseline with orthogonal workloads
- Verify: all SLO classes get same TTFT (no differentiation)
- This establishes the starting point for principled improvement

### Iter 12: SLO-Tiered Priority Cascade (port from #451)
- **Strategy**: SLOTieredPriority with piecewise-linear urgency + grace periods
- **Hypothesis**: Critical TTFT P99 drops 40-50% vs baseline while throughput is preserved
- **Parameters**: base_critical, base_sheddable, age_weight, threshold_sheddable (4 params)
- **Bayesian**: Optimize for critical TTFT P99 subject to throughput ≥ 95% of baseline

### Iter 13: SLO-Aware Admission Control
- **Strategy**: SLOGatedAdmission — reject sheddable when cluster queue depth exceeds threshold
- **Hypothesis**: Shedding 10-20% of sheddable requests reduces critical TTFT P99 by 20-30%
- **Parameters**: queue_threshold, slo_class_to_shed, shed_probability (3 params)
- **Mechanism**: When avg(QueueDepth) > threshold, reject sheddable with probability p. Critical always admitted.
- **Why this wins**: Reduces queueing delay for critical by removing competing requests

### Iter 14: KV-Pressure Admission Gate
- **Strategy**: KVPressureGate — reject requests when FreeKVBlocks < threshold per-instance
- **Hypothesis**: Prevents cascading preemptions (our iter 6 finding: KV=1500 caused 22.7s TTFT)
- **Parameters**: free_blocks_threshold, per_instance_vs_cluster (2 params)
- **Mechanism**: Check if target instance (post-routing) has enough free blocks for this request

### Iter 15: Joint Routing + Scheduling + Admission (3-layer compound)
- **Strategy**: Combine #451's priority cascade + our KV-adaptive routing + SLO admission gate
- **Hypothesis**: The compound achieves critical TTFT P99 < 50% of baseline AND protects against KV pressure
- **Parameters**: All params from iters 12-14 (9-10 total)
- **Bayesian**: Multi-objective optimization (critical TTFT, throughput, sheddable SLO miss rate)

### Iter 16: Precise KV Routing (llm-d blog hypothesis)
- **Strategy**: Model precise KV routing by eliminating PrefixCacheIndex divergence
- **Hypothesis**: Precise routing achieves 5-10x TTFT improvement over approximate under KV pressure
- **Method**: Compare --snapshot-refresh-interval=0 vs stale snapshots under KV pressure

### Iter 17: Multi-Turn Session Affinity with Admission Backpressure
- **Strategy**: Hash SessionID for routing + admission backpressure when session instance is overloaded
- **Hypothesis**: Session affinity preserves cache across rounds (63.8% hit rate from H-Reasoning-KV)

### Iter 18: Bursty Arrivals + Adaptive Admission
- **Strategy**: Gamma CV=2-3.5 arrivals with admission control that throttles during bursts
- **Hypothesis**: Admission-controlled bursts reduce TTFT P99 spikes by 30-50%

### Iter 19: Full Parameter Sweep (Bayesian over ALL layers)
- **Strategy**: All 3 layers parameterized: routing (PA/QD/KV weights), scheduling (base scores, age weights), admission (thresholds, shed rates)
- **Bayesian**: 15-20 parameters, 50 calls × 3 seeds
- **Goal**: Find the globally optimal parameter configuration

### Iter 20: Final Synthesis + PR Update
- **Strategy**: The definitive compound strategy with Bayesian-optimized parameters
- **Comprehensive comparison** against all baselines across all workload types
- **PR update** with STRATEGY_LEDGER, code, experiments, and findings

## Success Criteria
- Critical TTFT P99 < 50% of baseline (matching #451's finding)
- Throughput preserved (≥ 95% of baseline)
- KV-pressure-resilient (no degradation at reduced blocks)
- Principled: every improvement explained by a verified hypothesis
