# Hypothesis Experiments

> **Canonical source:** [`docs/contributing/hypothesis.md`](../docs/contributing/hypothesis.md). If this step list diverges, hypothesis.md is authoritative.

This directory contains validated hypothesis experiments for BLIS. Each hypothesis follows the methodology described in `docs/contributing/hypothesis.md`, with hypotheses drawn from the catalog in `docs/plans/research.md`:

0. **Create isolated worktree** — `.worktrees/h-<name>` using worktree skill
1. **Select and classify hypothesis** — family, VV&UQ category, type
2. **Design experiment** — ED-1 through ED-6, then 5-perspective Design Review → convergence
3. **Human approval** — present design for approval before implementation
4. **Implement** — create `run.sh` and `analyze.py` using shared harness (`hypotheses/lib/`)
5. **Code Review** — 5-perspective review of experiment code → convergence
6. **Run** — execute across required seeds; verify reproducibility (ED-5)
7. **Analyze and document** — produce comparison tables, write FINDINGS.md
8. **FINDINGS Review** — 10-perspective review → convergence (iterate until clean)
9. **Self-audit** — 6 dimensions: logic, determinism, consistency, contradictions, guidance, issues
10. **Commit and PR** — verification gate (if code fixes), then `commit-push-pr` skill

## Note on File Path References

FINDINGS.md files in individual hypothesis directories may reference old file paths. These files were moved during package extraction and the FINDINGS.md paths reflect the codebase state at the time each experiment was conducted:

- `sim/kvcache.go` → `sim/kv/cache.go` (PKG-1, #421)
- `sim/kvcache_tiered.go` → `sim/kv/tiered.go` (PKG-1, #421)
- `sim/latency_model.go` → `sim/latency/latency.go` (PKG-2, #424)
- `sim/roofline_step.go` → `sim/latency/roofline.go` (PKG-2, #424)

## Staleness Caveat

Experiment findings reflect simulator behavior at the time of execution. Subsequent bug fixes or feature changes may shift quantitative results (e.g., #502 shifted KV preemption counts, #574 required an H29 erratum). Check the PR that introduced the experiment (linked in each FINDINGS.md) and compare against recent changes to the relevant code paths before citing specific numbers.

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
| H-Cross-Model | Structural model | All confirmed behavioral findings hold for Qwen/Qwen2.5-7B-Instruct | **Partially confirmed** | 12/15 findings generalize; invariants and policy ordering are model-agnostic; cache-related findings are parameter-dependent |
| H4 | Cross-policy | Round-robin matches least-loaded for uniform workloads at low rates | **Confirmed with nuance** | Mean metrics equivalent (<3.2% diff); TTFT p99 12-21% worse for LL due to tie-breaking bias; byte-identical results at high rate with constant tokens |
| H6 | Cross-policy | Counterfactual regret higher for RR than weighted under variable load | **Confirmed with wrong mechanism** | RR mean regret=3.2, Weighted=0.0 (structurally zero by design, not empirical); higher regret does NOT imply worse TTFT — RR 5-15% lower TTFT due to perfect distribution uniformity |
| H7 | Performance-regime | Increasing instances from 4 to 8 should roughly halve TTFT p99 for saturated workloads | **Confirmed** | 7.4x TTFT p99 improvement (4→8 instances) — super-linear due to non-linear queue growth rate reduction; E2E insensitive (1.06x); vanishes at sub-saturation |
| H15 | Cross-policy | Fitness evaluation ranks prefix-affinity higher than load-only for prefix-heavy workloads | **Confirmed with nuance** | Fitness correctly ranks prefix-aware higher (+4.4% TTFT-heavy, +0.7% throughput-heavy); control (no prefix) produces byte-identical results; 1/(1+x) normalization compresses raw 14-38% TTFT p99 improvement |
| H20 | Workload/arrival | Heavy-tailed (ParetoLogNormal) input distributions produce more preemptions and HOL blocking than Gaussian | **Refuted** | ParetoLogNormal produces FEWER preemptions (0.71x avg); distribution MEDIAN drives KV pressure, not mean or tail |
| H23 | Cross-policy | All routing policies produce equivalent TTFT at near-zero load | **Confirmed with nuance** | All 4 policies within 4.40% at rate=1; surprise: high-load control also equivalent with uniform workloads — differentiation requires workload heterogeneity, not just high load |
| H27 | Performance-regime | Chunked prefill (threshold=256) reduces short-request TTFT p99 by >=30% in bimodal workloads | **Confirmed** | 46-58% TTFT p99 reduction (avg 51.9%); chunking splits 2048-token prefills into 8 steps of ~11ms vs one ~43ms step; tradeoff: 60-69% TTFT increase for long requests |
| H28 | Performance-regime | Chunked prefill (threshold=512) improves mean ITL by >15% for concurrent decode requests | **Refuted** | ITL improvement is effectively zero (-0.5%); decode-dominated step count (~255 steps) drowns out the rare prefill-co-batched steps; unexpected -13.3% short-request TTFT improvement |
| H29 | Structural model | Stale routing snapshots (100ms vs 1ms) degrade TTFT p99 by >=20% for kv-utilization scorer | **Confirmed** | +242% to +548% TTFT p99 degradation; queue-depth negative control shows 0.0% change (Immediate mode); composite scorer mitigates ~99% of effect; dose-response monotonic with safe zone <5ms |
| H30 | Structural model | BLIS crossmodel backend matches real vLLM aggregate latency within 25% for training-set experiments | **Partially confirmed** | Passes 25% RE gate at moderate load; ~5% input token undercount from missing chat template/BOS/EOS overhead; multi-turn semantic mismatch (BLIS accumulates context; real inference-perf does not) |
| H31 | Structural model | BLIS completes ≥90% of requests with TTFT MAPE <30% for near-saturation reasoning workloads | **Refuted** | BLIS TTFT 45ms vs real 120,171ms; root cause: µ underestimate (3.93 vs 3.22 rps) shifts ρ from 124% to 102%, placing BLIS in barely-saturated regime while real server is heavily overloaded |
| H32 | Structural model | BLIS crossmodel meets validation gates on codellama but fails on mixtral reasoning | **Partially confirmed** | Codellama passes all gates (TTFT RE <25%, E2E RE <20%, throughput RE <10%); generalizes to unseen workload profiles (codegen, roleplay); failure mode prediction confirmed for near-saturation reasoning |
| H1-Burstiness | Workload/arrival | Bursty (Gamma CV=3) arrivals produce significantly higher TTFT and E2E than Poisson at equal throughput | **Confirmed** | 4.8–5.3x TTFT p99 ratios at all utilization ≥ 22%; mean ratio monotonically increases 3.20x→4.03x as ρ goes 0.22→0.93; falsification criterion not met at any utilization level |
| H-Perf-Wallclock | Performance-regime | Combining O(1) LRU, hash dedup, and SHA256 reuse reduces wall-clock time by >50% without changing simulation output | **Confirmed** | All three optimizations together exceed 50% reduction target; INV-6 determinism preserved across all 9 samples; negative control (load-balance only) confirms bottleneck is prefix-affinity-specific |
| H-SLO-Admission | Cross-policy | SLO-gated admission reduces critical TTFT P99 by >20% at 120% capacity | **Partially confirmed** | Primary refuted: +5.2% worse at 120% (multi-turn context accumulation dominates); non-zero-sum mechanism confirmed — sheddable rejection reduces queue depth for all classes; modest ~15% improvement at 80% load |
| H-Priority-Preemption | Cross-policy | Priority-based preemption reduces critical TTFT P99 by >50% over baseline at 120% capacity | **Partially confirmed** | 20.9% improvement at 120% load (exceeds 20% but not 50% target); circuit breaker (max 3 preemptions/step) limits leverage; inconclusive at 80% load due to seed variance |
| H-Compound-Strategy | Cross-policy | Full compound strategy (StaticClassWeight + SLOGatedAdmission + PriorityPreemption) reduces critical TTFT P99 by >25% at 120% capacity | **Confirmed with nuance** | 35.5% critical TTFT P99 improvement (exceeds 25% threshold); preemption is stronger individual lever (29.3%) than admission (10.9%); super-additivity seed-dependent; cluster-wide P99 favors admission-only |
| H-Deadline-Urgency | Cross-policy | Dynamic deadline-urgency weights outperform static class weights for SLO differentiation | **Refuted** | Deadline urgency adds no value over static class weights; class-awareness mechanism confirmed but deadline scaling is redundant given the existing priority ordering |
| H-Heterogeneous-Pools | Cross-policy | Physical fast-lane isolation achieves critical TTFT P99 < 100ms even under overload | **Confirmed with nuance** | 20.6x relative improvement over shared pool; isolation dominates queue-management (compound strategy achieves only 1.6x on same metric); absolute P99 target not met at extreme overload (2.8x system capacity) |
| H-Joint-KV-Scheduling | Cross-policy | SLO-aware KV eviction creates multiplicative (super-additive) interaction with elastic priority batching under KV pressure | **Confirmed with nuance** | Super-additive interaction confirmed at moderate KV pressure (1200–1500 blocks); mechanisms work independently at heavy pressure; no effect at abundant KV; elastic batching is the primary lever |
| H-Elastic-Batching | Strategy Evolution | Large batches (maxRunningReqs=64) with aggressive priority preemption can simultaneously achieve high SLO attainment and high GPU utilization | **Confirmed** | 4.7x better critical TTFT P99 vs large-batch (no preemption) at equivalent batch occupancy; resolves the SLO-throughput conflict from prior iterations with small batches |
| H-Elastic-Generalization | Strategy Evolution | Elastic priority batching dual-objective breakthrough generalizes across all major workload dimensions | **Confirmed** | Universal benefit across all 12 workload variants spanning load level (80%–150%), arrival process (Poisson/Gamma), session structure (stateless/multi-turn), and SLO mix |
| H-Elastic-Stress | Strategy Evolution | Elastic batching generalizes across cluster scale (2–16 instances), KV cache pressure, and asymmetric request sizes | **Confirmed with boundary** | Strong benefit in 6/8 variants; 2/8 show no effect (minimal KV pressure — test design artifact, not mechanism failure); benefit holds across all cluster scales tested |

## Running Experiments

Each hypothesis directory contains a `run.sh` script:

```bash
cd hypotheses/h3-signal-freshness
./run.sh
```

Scripts are self-contained — they build the binary, run all experiment variants, and print analysis to stdout. Requires Go 1.24+ and Python 3 (standard library only — no pip packages needed).

**To contribute a new experiment:** See `docs/contributing/hypothesis.md` for the full process, `docs/contributing/templates/hypothesis.md` for the FINDINGS.md template, and `docs/contributing/standards/experiments.md` for rigor requirements. To propose without implementing, file a [Hypothesis Proposal issue](https://github.com/inference-sim/inference-sim/issues/new?template=hypothesis.md).

## Coverage by Family

| Family | Done | Pending | Gaps |
|--------|------|---------|------|
| **Scheduler invariants** | H12, H13, H-Liveness, H25 | — | Family complete |
| **Structural model** | H3, H9, H10, H-Phase, H-MMK, H26, H-Step-Quantum, H19, H-Cross-Model, H29, H30, H31, H32 | — | Family complete |
| **Robustness/failure-mode** | H5, H14, H-Overload, H-Overload-KV, H21, H22, H24 | — | Family complete |
| **Performance-regime** | H7, H8, H11, H-Reasoning-KV, H27, H28, H-Perf-Wallclock | — | Family complete |
| **Workload/arrival** | H-Arrival, H1-Burstiness, H16, H20 | — | Family complete |
| **Cross-policy comparative** | Prefix-Affinity, H1-SJF, H2, H4, H6, H15, H17, H23, H-SLO-Admission, H-Priority-Preemption, H-Compound-Strategy, H-Deadline-Urgency, H-Heterogeneous-Pools, H-Joint-KV-Scheduling | H18 | 1 remaining |
| **Strategy Evolution** | H-Elastic-Batching, H-Elastic-Generalization, H-Elastic-Stress | — | Family complete |

## Hypothesis Tiers (priority from research.md)

- **Tier 1**: Correctness baselines (H12 ✓, H13 ✓, H-Phase ✓, H-MMK ✓)
- **Tier 2**: High diagnostic value (H3 ✓, H9 ✓, H14 ✓, Prefix-Affinity ✓)
- **Tier 3**: System understanding (H1 ✓, H5 ✓, H10 ✓, H11 ✓)
- **Tier 4**: Research questions (H15 ✓, H17 ✓, H19 ✓)
- **Tier 5**: Workload diversity (H2 ✓, H16 ✓, H18, H20 ✓)
