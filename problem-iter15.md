# Iteration 15: Epoch-Based Online Weight Adaptation + Multi-Turn Under Load

## The Genius Insight: Use Admission Rejection Rate as an Online Learning Signal

All prior iterations used STATIC routing weights (even the "adaptive" ones switch between two fixed profiles). The optimal PA:QD ratio depends on load level (iter 13: pa:4,qd:3 wins at rate=2000 but pa:3,qd:2 wins at rate=200). In production, load varies continuously. A truly adaptive strategy needs to LEARN the right weights from observed system behavior.

**The insight**: The SLO-gated admission controller's rejection rate is a FREE, REAL-TIME signal of system overload. When rejection rate is high, the system is overloaded → increase QD weight (spread load). When rejection rate is zero, the system has headroom → increase PA weight (exploit cache). This creates a simple feedback loop with a natural equilibrium.

## Strategy Template: EpochAdaptiveScoring

```
EpochAdaptiveScoring:
  Every PARAM_epoch_requests requests:
    1. Measure admission rejection rate over the epoch
    2. If rejection_rate > PARAM_high_threshold:
       PA_weight -= PARAM_step_size (min: PARAM_pa_min)
       QD_weight += PARAM_step_size (max: PARAM_qd_max)
    3. If rejection_rate < PARAM_low_threshold:
       PA_weight += PARAM_step_size (max: PARAM_pa_max)
       QD_weight -= PARAM_step_size (min: PARAM_qd_min)
    4. Rebuild WeightedScoring with new weights
```

This is:
- **Online**: adapts every epoch, not at startup
- **Physics-free**: no beta coefficients needed, no TTFT estimation
- **Self-calibrating**: the rejection rate naturally captures ALL system effects (load, KV pressure, cache hit rate)
- **Monotonic convergence**: at equilibrium, rejection rate stabilizes between thresholds → weights stabilize

## Parameters (7, for Bayesian optimization)
1. `epoch_requests` (50-500): adaptation frequency
2. `high_threshold` (0.1-0.5): rejection rate that triggers QD increase
3. `low_threshold` (0.0-0.1): rejection rate that triggers PA increase
4. `step_size` (0.1-1.0): weight adjustment per epoch
5. `pa_min`/`pa_max` (1-5): PA weight bounds
6. `qd_min`/`qd_max` (1-5): QD weight bounds

## Test Scenarios
1. **Steady high load** (rate=2000): does epoch adaptation converge to pa:4,qd:3?
2. **Load ramp** (rate=500→2000): does it adapt from cache-heavy to load-heavy?
3. **Multi-turn** (4 rounds, context accumulation): does it handle growing KV pressure?
4. **Bursty** (Gamma CV=2.0): does it respond to transient overload?

## Hypotheses

H15-1: Epoch adaptation CONVERGES to pa:4,qd:3 at steady rate=2000 (matching Bayesian optimal)
H15-2: Under load ramp, epoch adaptation achieves BETTER goodput than any static config because it matches weights to current load
H15-3: On multi-turn workloads, epoch adaptation responds to growing KV pressure by increasing QD weight (preventing the iter-6 degradation)
H15-4: Goodput of epoch-adaptive ≥ 95% of Bayesian-optimal at ANY load level (universal strategy)
