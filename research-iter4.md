# Research: Load-Adaptive Cache Scoring (Iteration 4)

## Problem
Static `pa:3,qd:2,kv:2` is the champion (74.15ms combined, 40.5% better than RR). But the PA scorer is a LINEAR function of cache match ratio — it doesn't account for queue depth. At high utilization, it may create persistent load imbalance because it pushes toward cached instances regardless of their queue state.

The PR #447 cost-benefit scorer solves this: `score = cacheSaving / (cacheSaving + queueDelay)`. This naturally adapts to load. But it's trapped in the `adaptive-weighted` policy factory.

## Idea 1: Composable Cost-Benefit Scorer

### Implementation
Wire `cost-benefit` into `newScorerWithObserver()` in `routing_scorers.go` so it can be used via `--routing-scorers cost-benefit:3,queue-depth:2`:

```go
case "cost-benefit":
    prefixIdx := NewPrefixCacheIndex(blockSize, defaultLRUCapacity)
    betaCoeffs := []float64{6910.42, 17.67, 2.84}
    scorer := newCostBenefitScorer(prefixIdx, betaCoeffs)
    obs := func(req *Request, targetInstance string) {
        if req != nil && len(req.InputTokens) > 0 {
            hashes := prefixIdx.ComputeBlockHashes(req.InputTokens)
            prefixIdx.RecordBlocks(hashes, targetInstance)
        }
    }
    return scorer, obs
```

### Hypothesis
**H-CB-1**: `cost-benefit:3,queue-depth:2` matches `pa:3,qd:2,kv:2` at moderate load (rate=250) and outperforms at high load (rate=400+).

**Mechanism**: At moderate load, cost-benefit ≈ PA (queue delay is small relative to cache saving). At high load, cost-benefit → 0 on loaded instances (queue delay dominates), naturally shifting traffic to less-loaded instances.

### Rate Sweep Design
Test at rates: 100, 200, 300, 400, 500 req/s with 8 instances on RAG workload (4096-token prefix).
- ρ estimates: 100→0.20, 200→0.41, 300→0.61, 400→0.82, 500→1.02
- Compare: cost-benefit:3,qd:2 vs pa:3,qd:2,kv:2 vs RR

### Additional Configurations to Test
- `cost-benefit:5,queue-depth:1` — heavy cache bias with cost-benefit adaptation
- `cost-benefit:1,queue-depth:3` — heavy load bias with cost-benefit adaptation
- `pa:5,qd:1,kv:1` — static heavy cache (known to be catastrophic from iter 1: 578ms)
