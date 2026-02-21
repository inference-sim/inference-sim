# Hypothesis Experiments

This directory contains validated hypothesis experiments for BLIS. Each hypothesis follows the methodology described in `docs/plans/research.md`:

1. **Pose an intuitive hypothesis** about system behavior
2. **Design a controlled experiment** — two configurations differing in exactly one dimension
3. **Run across multiple seeds** (42, 123, 456) for statistical rigor
4. **Analyze results** — confirm, refute, or surface bugs/design limitations
5. **Document findings** — the experiment becomes a reproducible artifact

## Validated Hypotheses

| ID | Hypothesis | Status | Key Finding |
|----|-----------|--------|-------------|
| Prefix-Affinity | Prefix-aware routing outperforms load-only for prefix-heavy workloads | **Confirmed** | 2.45x better TTFT for multi-turn; queue-depth actively destroys cache locality (23% vs 56% hit rate) |
| H3 | queue-depth distributes more evenly than kv-utilization at high rates | **Confirmed** | 200x better distribution uniformity; inherent DES event ordering causes kv-util staleness |
| H9 | TTFT decreases monotonically as prefix_length increases | **Confirmed** | 95.8% TTFT reduction at max prefix; cache hit rate precisely linear with prefix fraction; cache capacity has zero observable effect (model limitation) |
| H12 | Request conservation holds across all policy configurations | **Confirmed** (with bug) | INV-1 holds for 10 policy combinations (67 checks); preemption path panics on empty RunningBatch (simulator.go:383) |
| H14 | Pathological templates produce worse behavior; anomaly detectors fire | **Partially confirmed** | 4.5x worse TTFT confirmed; 3 bugs found (HOL blind spot, priority inversion false positives, double inversion cancellation) |
| H8 | Reducing KV blocks increases preemption frequency and worsens tail latency | **Confirmed** | Sharp cliff at ~2200 blocks: 4.7x TTFT p99 with 4.5% fewer blocks; cascade effect amplifies preemptions; conservation (INV-1) holds under pressure |
| H5 | Token-bucket admission reduces tail latency under Gamma burst | **Confirmed** | 56-69x TTFT p99 improvement across all seeds; 96% rejection at cap=500/refill=400; effect amplifies with rate (9x→69x from 200→2000 req/s) |
| H10 | Tiered KV cache reduces preemptions vs single-tier | **Partially confirmed** (surprise) | Zero preemptions in both tiers; tiered improves TTFT by 18-28% via capacity threshold effect; 100 CPU blocks = full benefit (threshold, not gradual) |
| H13 | Same seed produces byte-identical output | **Confirmed** | INV-6 holds for 5 policy configurations including stateful prefix-affinity scorer with LRU eviction |

## Running Experiments

Each hypothesis directory contains a `run.sh` script:

```bash
cd hypotheses/h3-signal-freshness
./run.sh
```

Scripts are self-contained — they build the binary, run all experiment variants, and print analysis to stdout. Requires Go 1.24+ and Python 3.

## Hypothesis Tiers (from research.md)

- **Tier 0**: Measurement audit (verify metrics exist in output)
- **Tier 1**: Correctness baselines (H12 conservation ✓, H13 determinism ✓)
- **Tier 2**: High diagnostic value (H3 signal freshness ✓, H9 prefix caching ✓, H14 pathological ✓)
- **Tier 3**: System understanding (H1, H5 ✓, H10 ✓, H11)
- **Tier 4**: Research questions (H15, H17, H19)
- **Tier 5**: Workload diversity (H2, H16, H18, H20)
