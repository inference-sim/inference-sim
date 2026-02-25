# H29: Stale Routing Snapshots Degrade Tail Latency Under High Request Rates

**Status**: Confirmed
**Date**: 2026-02-25

## Hypothesis

> Increasing the snapshot refresh interval from 1ms to 100ms degrades TTFT p99 by at least 20% for weighted routing (kv-utilization scorer) at high request rates (>80% saturation, 4 instances), because stale load signals cause the router to repeatedly select already-loaded instances, creating transient load imbalance.

(Originally hypothesized for queue-depth; corrected to kv-utilization after discovering that --snapshot-refresh-interval only affects KVUtilization, not QueueDepth. See FINDINGS.md Critical Design Note.)

**Refuted if:** TTFT p99 difference between 1ms and 100ms snapshot refresh intervals is less than 10% across all 3 seeds at >80% saturation.
