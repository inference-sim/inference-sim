# H1-SJF: SJF Scheduling Reduces TTFT for Short Requests in Bimodal Workloads

**Status**: Confirmed
**Date**: 2026-02-21

## Hypothesis

> SJF scheduling should reduce TTFT for mixed-length workloads. If short requests get stuck behind long ones (head-of-line blocking), scheduling short jobs first should reduce average wait time -- the classic SJF result from operating systems. In a bimodal workload (50% short at 32 tokens, 50% long at 1024 tokens) at rate=3000 with 4 instances, SJF should dramatically reduce TTFT for short requests compared to FCFS.

**Refuted if:** SJF does not reduce interactive (short-request) TTFT mean by at least 20% compared to FCFS, across all 3 seeds.
