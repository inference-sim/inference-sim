# H3: Queue-Depth Distributes Requests More Evenly Than KV-Utilization at High Rates

**Status**: Confirmed
**Date**: 2026-02-20

## Hypothesis

> At high request rates, the queue-depth scorer should distribute requests more evenly than the kv-utilization scorer, because queue-depth updates synchronously (PendingRequests increments at routing time) while KV utilization only changes when batch formation allocates blocks (a lagging indicator). This staleness should cause kv-utilization to pile requests onto already-loaded instances.

**Refuted if:** KV-utilization TTFT mean is within 10% of queue-depth TTFT mean at rate=5000 with 4 instances, across all 3 seeds.
