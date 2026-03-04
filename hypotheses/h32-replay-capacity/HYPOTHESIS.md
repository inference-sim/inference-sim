# H32: CrossModel Aggregate Capacity Planning Accuracy

**Status**: Partially Confirmed
**Date**: 2026-03-03

## Hypothesis

> BLIS crossmodel aggregate metrics will meet validation targets (TTFT p50 RE < 10%, TTFT p99 RE < 25%, throughput RE < 10%) on the codellama validation experiments (codegen and roleplay) where load is moderate and failure rate is zero, but will fail the TTFT p99 RE < 25% target on the mixtral reasoning validation experiment because the 69% failure rate creates a biased survivor completion set whose queueing dynamics diverge between BLIS and real vLLM.

**Refuted if:** The mixtral reasoning validation experiment achieves TTFT p99 RE < 25% despite 69% failure rate (overperformance), OR any codellama validation experiment fails any of the three aggregate targets.

## Context

This tests the practical user question: "At what arrival rate does TTFT p99 exceed X ms for model Y on H100?" Validation experiments: codellama-34b codegen (9K req, 5/10 RPS, 0% fail), codellama-34b roleplay (7.2K req, 6 RPS, 0% fail), mixtral-8x7b reasoning (4.8K req, 4 RPS, 69% fail). The codellama experiments test cross-profile generalization (architecture seen in training via general profile). The mixtral reasoning tests cross-regime generalization (overload). Real vLLM summary_lifecycle_metrics.json provides ground-truth p50/p99 and throughput for direct comparison.
