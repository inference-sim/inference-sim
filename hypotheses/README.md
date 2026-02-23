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
| H1-SJF | Cross-policy | SJF reduces TTFT for short requests in bimodal workloads | **Confirmed** | 94% TTFT reduction for 32-token requests vs 1024-token (bimodal 32:1024); scheduling delay drops 98% |
| H2 | Cross-policy | Priority-FCFS with SLO-based priority should reduce realtime TTFT at the cost of batch TTFT | **Refuted** | SLO-based priority is purely age-based (monotonic in arrival time), making it mathematically equivalent to FCFS |
| H3 | Structural model | queue-depth distributes more evenly than kv-utilization at high rates | **Confirmed** | 200x better uniformity; DES event ordering causes kv-util staleness |
| H5 | Robustness | Token-bucket admission smooths bursts under Gamma CV=3.5 | **Confirmed with nuance** | 56-69x TTFT improvement holds, but via 96% load shedding, not burst smoothing |
| H8 | Performance-regime | Reducing KV blocks increases preemption frequency and worsens tail latency | **Confirmed** | Sharp cliff at ~2200 blocks; cascade amplifies preemptions |
| H9 | Structural model | TTFT decreases monotonically as prefix_length increases | **Confirmed** | 95.8% TTFT reduction; cache hit rate linear with prefix fraction |
| H10 | Structural model | Tiered KV cache reduces preemptions vs single-tier | **Confirmed** | Preemptions halved (17.5%→8.5%); `maybeOffload` preserves prefix hashes |
| H11 | Performance-regime | Larger token budgets improve throughput but worsen ITL | **Confirmed with nuance** | Throughput +27%, ITL p99 worsens 5.8x, but ITL mean slightly decreases due to step composition shift |
| H12 | Scheduler invariant | Request conservation holds across all policy configurations | **Confirmed** (with bug) | INV-1 holds (67 checks); preemption path panics on empty RunningBatch |
| H13 | Scheduler invariant | Same seed produces byte-identical output | **Confirmed** | INV-6 holds for 5 policy configurations |
| H14 | Robustness | Pathological templates produce worse behavior; anomaly detectors fire | **Partially confirmed** | 4.5x worse TTFT; 3 detector bugs found |
| H16 | Workload/arrival | Bursty (Gamma) arrivals produce worse tail latency than Poisson at the same rate | **Confirmed with nuance** | Gamma CV=3.5 produces 1.25x worse TTFT p99 at overload, 1.66x at sub-saturation; vanishes under sustained load |
| H17 | Cross-policy | Multi-scorer weights produce a Pareto frontier | **Reclassified** | No within-workload Pareto frontier; cache-heavy routing dominates all metrics with any prefix overlap |
| H19 | Structural model | Roofline mode produces same relative policy rankings as blackbox mode | **Partially confirmed** | Mean rankings preserved (6/6); P99 rankings diverge (1/6) due to alpha overhead |
| H21 | Robustness | Extreme scorer weight (100:1) behaves identically to single-scorer routing | **Refuted** | Even 1/101 queue-depth weight prevents cold-start concentration cascade; single-scorer degenerates to all-to-one |
| H22 | Robustness | Zero KV blocks produces a clean CLI error, not a panic | **Confirmed** | All zero/negative KV configs caught at CLI boundary with clean logrus.Fatalf messages |
| H24 | Robustness | Combined always-busiest + inverted-slo produces maximum measurable anomalies | **Confirmed** | 4.9x TTFT degradation with super-additive priority inversions (9,963 combined vs 5,017 additive sum) |
| H25 | Scheduler invariant | Full policy stack maintains conservation invariants under combined load | **Confirmed** | INV-1, INV-6, INV-5 all hold under full policy stack |
| H26 | Structural model | Adding admission latency delays E2E by exactly that amount under low load | **Confirmed** | Exact additive offset to TTFT/E2E/scheduling delay; validates cluster event pipeline causal ordering |
| H-Arrival | Workload/arrival | Poisson/Gamma/Weibull samplers match theoretical CDF | **Confirmed with limitation** | Poisson/low-CV pass KS; high-CV fail due to int64 μs clamping (42% for Gamma CV=3.5) |
| H-Liveness | Scheduler invariant | All schedulers satisfy liveness under admissible load | **Confirmed** | 45/45 pass (ρ=0.3-0.85); batching masks scheduler at default batch=256; SJF 31% faster under constrained batch |
| H-MMK | Structural model | DES matches M/M/k analytical model under matching assumptions | **Partially confirmed** | Within 3.3% at ρ ≤ 0.3; diverges 28-71% at ρ ≥ 0.5 (discrete step processing) |
| H-Overload | Robustness | 10x overload: no panics, conservation holds | **Confirmed** | 84/84 INV-1 checks pass; token-bucket rejects 70% at 10x |
| H-Overload-KV | Robustness | Extreme overload combined with KV pressure maintains conservation | **Confirmed with nuance** | Conservation holds, but sharp cliff between 500-2000 KV blocks causes cascading preemption timeouts |
| H-Phase-Structure | Structural model | TTFT linear in input tokens, decode time linear in output tokens | **Confirmed** | R² = 1.000000 (adjusted); slopes match α/β predictions within <0.01% |
| H-Reasoning-KV | Performance-regime | Multi-turn reasoning triggers preemption cliff proportional to peak demand | **Refuted** | Mean demand drives cliff (1.09x shift), not peak; surprise 63.8% prefix cache hit rate from context accumulation |
| H-Step-Quantum | Structural model | Reducing step-time quantum proportionally reduces DES-to-M/M/1 divergence | **Refuted** | Divergence caused by alpha/beta split, not step quantization; reducing beta worsens divergence 47%→99% |

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
| **Scheduler invariants** | H12, H13, H-Liveness, H25 | — | Family complete |
| **Structural model** | H3, H9, H10, H-Phase, H-MMK, H26, H-Step-Quantum, H19 | — | Family complete |
| **Robustness/failure-mode** | H5, H14, H-Overload, H-Overload-KV, H21, H22, H24 | — | Family complete |
| **Performance-regime** | H8, H11, H-Reasoning-KV | H7 | Horizontal scaling |
| **Workload/arrival** | H-Arrival, H16 | H20 | Heavy-tailed distributions |
| **Cross-policy comparative** | Prefix-Affinity, H1-SJF, H2, H17 | H4, H6, H15, H18, H23 | 5 remaining |

## Hypothesis Tiers (priority from research.md)

- **Tier 1**: Correctness baselines (H12 ✓, H13 ✓, H-Phase ✓, H-MMK ✓)
- **Tier 2**: High diagnostic value (H3 ✓, H9 ✓, H14 ✓, Prefix-Affinity ✓)
- **Tier 3**: System understanding (H1 ✓, H5 ✓, H10 ✓, H11 ✓)
- **Tier 4**: Research questions (H15, H17 ✓, H19 ✓)
- **Tier 5**: Workload diversity (H2 ✓, H16 ✓, H18, H20)
