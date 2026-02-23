# Hypothesis Experiments

This directory contains validated hypothesis experiments for BLIS. Each hypothesis follows the methodology described in `docs/process/hypothesis.md`, with hypotheses drawn from the catalog in `docs/plans/research.md`:

1. **Identify hypothesis family** — which domain is being tested?
2. **Pose an intuitive, behavioral hypothesis** — conceptual claim, not code-grounded
3. **Design a controlled experiment** — configurations differing in exactly one dimension
4. **Code review experiment code** — verify parsers against simulator output format
5. **Run across multiple seeds** (42, 123, 456) for statistical rigor
6. **Analyze with three parallel external reviews** — iterate until convergence
7. **Document findings** — the experiment becomes a reproducible artifact with honest resolution

## Validated Hypotheses

| ID | Family | Hypothesis | Status | Key Finding |
|----|--------|-----------|--------|-------------|
| Prefix-Affinity | Cross-policy | Prefix-aware routing outperforms load-only for prefix-heavy workloads | **Confirmed** | 2.45x better TTFT; queue-depth destroys cache locality |
| H3 | Structural model | queue-depth distributes more evenly than kv-utilization at high rates | **Confirmed** | 200x better uniformity; DES event ordering causes kv-util staleness |
| H9 | Structural model | TTFT decreases monotonically as prefix_length increases | **Confirmed** | 95.8% TTFT reduction; cache hit rate linear with prefix fraction |
| H12 | Scheduler invariant | Request conservation holds across all policy configurations | **Confirmed** (with bug) | INV-1 holds (67 checks); preemption path panics on empty RunningBatch |
| H14 | Robustness | Pathological templates produce worse behavior; anomaly detectors fire | **Partially confirmed** | 4.5x worse TTFT; 3 detector bugs found |
| H8 | Performance-regime | Reducing KV blocks increases preemption frequency and worsens tail latency | **Confirmed** | Sharp cliff at ~2200 blocks; cascade amplifies preemptions |
| H5 | Robustness | Token-bucket admission smooths bursts under Gamma CV=3.5 | **Confirmed with nuance** | 56-69x TTFT improvement holds, but via 96% load shedding, not burst smoothing. Calibrated bucket (cap=100K) shows <5% — wrong mechanism, not wrong direction. |
| H10 | Structural model | Tiered KV cache reduces preemptions vs single-tier | **Confirmed** | Preemptions halved (17.5%→8.5%); `maybeOffload` preserves prefix hashes; 4 rounds to resolve |
| H13 | Scheduler invariant | Same seed produces byte-identical output | **Confirmed** | INV-6 holds for 5 policy configurations |
| H-Phase-Structure | Structural model | TTFT linear in input tokens, decode time linear in output tokens | **Confirmed** | R² = 1.000000 (adjusted); slopes match α/β predictions within <0.01% |
| H-MMK | Structural model | DES matches M/M/k analytical model under matching assumptions | **Partially confirmed** | Within 3.3% at ρ ≤ 0.3; diverges 28-71% at ρ ≥ 0.5 (discrete step processing) |
| H-Arrival | Workload/arrival | Poisson/Gamma/Weibull samplers match theoretical CDF | **Confirmed with limitation** | Poisson/low-CV pass KS; high-CV fail due to int64 μs clamping (42% for Gamma CV=3.5) |
| H-Liveness | Scheduler invariant | All schedulers satisfy liveness under admissible load | **Confirmed** | 45/45 pass (ρ=0.3-0.85); batching masks scheduler at default batch=256; SJF 31% faster under constrained batch |
| H-Overload | Robustness | 10x overload: no panics, conservation holds | **Confirmed** | 84/84 INV-1 checks pass; token-bucket rejects 70% at 10x; off-by-1 root-caused as alpha-model horizon edge |
| H1-SJF | Cross-policy | SJF reduces TTFT for short requests in bimodal workloads | **Confirmed** | 94% TTFT reduction for 32-token requests vs 1024-token (bimodal 32:1024); scheduling delay drops 98% |
| H2 | Cross-policy | Priority-FCFS with SLO-based priority should reduce realtime TTFT | **Refuted** | SLO-based priority is age-based, not class-aware — equivalent to FCFS (#347) |
| H-Overload-KV | Robustness | Conservation holds under combined overload + KV pressure | **Confirmed with nuance** | Cliff behavior between pressure regimes |
| H-Step-Quantum | Structural model | Step-time quantum causes DES-to-M/M/1 divergence | **Refuted** | Alpha/beta service time split is root cause, not discrete step quantization |
| H16 | Workload/arrival | Gamma arrivals produce worse tail latency than Poisson | **Confirmed with nuance** | Effect is load-duration dependent: 1.25x at 500 req, vanishes at 2000 req |
| H19 | Structural model | Roofline and blackbox produce same relative policy rankings | **Partially confirmed** | Mean rankings preserved; P99 diverges from alpha overhead |
| H21 | Robustness | Extreme scorer weight (100:1) behaves like single-scorer | **Refuted** | Cold-start cascade — tiebreaker is binary (present/absent), not proportional |
| H22 | Robustness | Zero KV blocks should panic at CLI, not inside simulation | **Confirmed** | Defense-in-depth validated |
| H24 | Robustness | Combined always-busiest + inverted-slo produces maximum anomalies | **Confirmed** | 4.9x TTFT degradation; super-additive priority inversions |
| H25 | Scheduler invariant | Full policy stack maintains conservation under stress | **Confirmed** | INV-1 holds with all features active simultaneously |
| H26 | Structural model | Admission latency delays E2E by exactly that amount at low load | **Confirmed** | Event pipeline causal ordering verified |
| H-Reasoning-KV | Performance-regime | Reasoning context accumulation shifts preemption cliff | **Refuted** | Mean demand drives cliff, not per-request peak; 63.8% prefix cache hits from context reuse |
| H7 | Performance-regime | Increasing instances from 4→8 should halve TTFT p99 at saturation | **Confirmed with nuance** | 0.297x ratio (better than 0.5x); queue growth rate (λ/k-μ) explains super-linear scaling |
| H20 | Workload/arrival | ParetoLogNormal should produce more preemptions than Gaussian | **Refuted** | Distribution median (not mean/tail) drives KV pressure; ParetoLN mixture creates breathing room |
| #377 | Cross-policy | Within-workload Pareto frontier emerges at high utilization | **Refuted** | Cache-heavy dominates all metrics even at 3x overload; session stickiness is inherently load-balanced |

## Running Experiments

Each hypothesis directory contains a `run.sh` script:

```bash
cd hypotheses/h3-signal-freshness
./run.sh
```

Scripts are self-contained — they build the binary, run all experiment variants, and print analysis to stdout. Requires Go 1.24+ and Python 3 (standard library only — no pip packages needed).

**To contribute a new experiment:** See `docs/process/hypothesis.md` for the full process, `docs/templates/hypothesis.md` for the FINDINGS.md template, and `docs/standards/experiments.md` for rigor requirements. To propose without implementing, file a [Hypothesis Proposal issue](https://github.com/inference-sim/inference-sim/issues/new?template=hypothesis.md).

## Coverage by Family

| Family | Done | Pending | Gaps |
|--------|------|---------|------|
| **Scheduler invariants** | H12 ✓, H13 ✓, H-Liveness ✓, H25 ✓ | — | **Complete** |
| **Structural model** | H3 ✓, H9 ✓, H10 ✓, H-Phase ✓, H-MMK ✓, H26 ✓, H-Step-Quantum ✓, H19 ✓ | — | **Complete** |
| **Robustness/failure-mode** | H14 ✓, H5 ✓, H-Overload ✓, H-Overload-KV ✓, H21 ✓, H22 ✓, H24 ✓ | — | **Complete** |
| **Cross-policy comparative** | Prefix-Affinity ✓, H1-SJF ✓, H2 ✓, H17 ✓, #377 ✓ | H4, H6, H15, H18, H23 | Fitness, fairness, baselines |
| **Performance-regime** | H7 ✓, H8 ✓, H11 ✓, H-Reasoning-KV ✓ | — | **Complete** |
| **Workload/arrival** | H-Arrival ✓, H16 ✓, H20 ✓ | — | **Complete** |

## Hypothesis Tiers (priority from research.md)

- **Tier 1**: Correctness baselines (H12 ✓, H13 ✓, H-Phase ✓, H-MMK ✓)
- **Tier 2**: High diagnostic value (H3 ✓, H9 ✓, H14 ✓, Prefix-Affinity ✓)
- **Tier 3**: System understanding (H1 ✓, H5 ✓, H10 ✓, H11 ✓)
- **Tier 4**: Research questions (H15, H17 ✓, H19 ✓, #377 ✓)
- **Tier 5**: Workload diversity (H2 ✓, H16 ✓, H18, H20 ✓)
- **Tier A**: Performance-regime (H7 ✓, H8 ✓, H-Reasoning-KV ✓)
- **Tier B**: Edge cases (H21 ✓, H22 ✓, H24 ✓, H25 ✓, H26 ✓)
