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

## Running Experiments

Each hypothesis directory contains a `run.sh` script:

```bash
cd hypotheses/h3-signal-freshness
./run.sh
```

Scripts are self-contained — they build the binary, run all experiment variants, and print analysis to stdout. Requires Go 1.24+ and Python 3.

## Hypothesis Tiers (from research.md)

- **Tier 0**: Measurement audit (verify metrics exist in output)
- **Tier 1**: Correctness baselines (H12 conservation, H13 determinism)
- **Tier 2**: High diagnostic value (H3 signal freshness, H9 prefix caching, H14 pathological)
- **Tier 3**: System understanding (H1, H5, H10, H11)
- **Tier 4**: Research questions (H15, H17, H19)
- **Tier 5**: Workload diversity (H2, H16, H18, H20)
