# Ground-Truth Trace Data Analysis for Reasoning Workloads

## Summary

Examined the trace data for Llama-2-7B reasoning experiment to understand what timing breakdowns are available.

**Bottom line**: The traces have **coarse-grained end-to-end timing** but **NO fine-grained internal breakdowns** for:
- Prefill execution time
- Queue/scheduler delay
- KV cache allocation time
- Attention kernel time
- Memory allocation time

## What's Available in Traces

### 1. Per-Request Lifecycle Metrics (`per_request_lifecycle_metrics.json`)
**Structure**: Array of 4800 requests with:
- `start_time`: Request arrival timestamp
- `end_time`: Request completion timestamp
- `request`: Input prompt
- `response`: Generated output
- `info`: Contains only:
  - `input_tokens`: Number of input tokens (confirms ~1082 tokens)
  - `output_tokens`: Number of output tokens
  - `output_token_times`: Array of timestamps for each output token (for ITL calculation)
- `error`: Error message if failed

**What's missing**: No internal timing breakdown between arrival and first token.

### 2. Summary Lifecycle Metrics (`summary_lifecycle_metrics.json`)
**Contains**:
- TTFT statistics: mean=106.6ms, p10=0.13ms, p90=215.4ms (huge variance!)
- Prompt length: mean=1082 tokens (confirms short context)
- Schedule delay: mean=0.476ms (very small, not the bottleneck)
- ITL statistics
- E2E latency statistics

**What's missing**: No breakdown of where the 106.6ms TTFT is spent.

### 3. OpenTelemetry Traces (`traces.json`)
**Contains**:
- `vllm.api` scope: `llm_request` spans with only ARRIVED/DEPARTED events
- `vllm.scheduler` scope: `llm_core` spans (minimal data)
- `vllm.scheduler.step` scope: Scheduler step spans (batch formation events)

**What's available in llm_request span**:
```json
{
  "name": "llm_request",
  "startTimeUnixNano": "1771377149658534007",
  "endTimeUnixNano": "1771377173925646417",
  "attributes": [
    {"key": "gen_ai.usage.prompt_tokens", "value": {"intValue": "1083"}},
    {"key": "gen_ai.request.max_tokens", "value": {"intValue": "1448"}}
  ],
  "events": [
    {"timeUnixNano": "1771377149658635892", "name": "api.ARRIVED"},
    {"timeUnixNano": "1771377173925633901", "name": "api.DEPARTED"}
  ]
}
```

**What's missing**: No events for:
- Queue entry/exit
- Prefill start/end
- KV allocation start/end
- First token generation
- Individual decode steps

### 4. KV Events (`kv_events.jsonl`)
**Contains**: Just timestamps, no detailed event types or request associations.

## Key Findings

### ✅ What We Can Measure
1. **E2E latency**: End-to-end request time (start to completion)
2. **TTFT**: Time to first token (from output_token_times[0] - start_time)
3. **ITL**: Inter-token latency (delta between consecutive output_token_times)
4. **TTFT variance**: p10=0.13ms, p90=215.4ms (1650× variance!)
5. **Prompt length**: Confirms ~1082 tokens (NOT 8K as hypothesized in iter3/4/5)

### ❌ What We CANNOT Measure
1. **Queuing delay**: Time request waits before prefill starts
2. **Prefill execution**: Actual GPU compute time for prefill
3. **KV allocation time**: Block allocation overhead
4. **Attention kernel time**: FlashAttention-2 execution time
5. **Memory allocation**: Activation buffer allocation
6. **Batch formation delay**: Time waiting for batch to fill
7. **Prefix cache hit/miss**: Whether shared system prompt was cached

## Implications for Iter7

### Why We Can't Decompose the 100-200ms TTFT

The missing 78.5-178.5ms in reasoning (β₆=21.5ms captured, actual=100-200ms) **cannot be identified from traces alone** because:

1. **No internal timing events**: Traces only have ARRIVED/DEPARTED, no intermediate events
2. **No scheduler instrumentation**: Can't measure batch formation delay
3. **No kernel-level timing**: Can't measure prefill compute vs queuing
4. **No memory profiling**: Can't measure allocation overhead

### What the Variance Tells Us

**TTFT variance** (p10=0.13ms, p90=215.4ms = 1650× range) strongly suggests:
- **Batching delay dominates**: Some requests processed immediately (p10=0.13ms), others wait for batch (p90=215ms)
- **Not compute-bound**: If prefill compute dominated, variance would be small (<2×)
- **Likely queue position dependent**: Early in batch → fast, late in batch → slow

### Three Approaches for Iter7

Since we **can't decompose from traces**, we have three options:

#### Option 1: Model Variance Explicitly (Workload-Agnostic)
Model batching delay as function of **concurrent requests** or **arrival rate**:
```go
// Higher concurrency → longer batching delay
queuing_delay_us = Beta[6] * (1.0 + concurrent_requests / 10.0) * 1000.0
// reasoning (multi-turn chat): concurrent_requests = 20-50 → 3-6× multiplier
// codegen (lower concurrency): concurrent_requests = 5-10 → 1.5-2× multiplier
```

**Pro**: Workload-agnostic (uses observable system state)
**Con**: Requires simulator to track concurrent requests

#### Option 2: Model p90 vs Mean Explicitly
Add variance term to capture batching delay distribution:
```go
// Use percentile instead of mean
queuing_delay_us = Beta[6] * percentile_factor * 1000.0
// percentile_factor: 1.0 for p50, 2.0 for p90, etc.
// Fit β₆ to p90 (215ms) instead of mean (21.5ms)
```

**Pro**: Captures full distribution, not just mean
**Con**: Requires modeling request position in batch

#### Option 3: Profile vLLM with nsys (Instrumentation Required)
Run reasoning experiment with NVIDIA Nsight Systems profiler:
```bash
nsys profile -o reasoning_profile python -m vllm.entrypoints.api_server ...
```

**Pro**: Would reveal exact breakdown (kernel launch, memory alloc, etc.)
**Con**: Requires new instrumentation, not available in existing traces

## Recommendation

**Use Option 1 (model variance via concurrent requests)** for iter7:
- Workload-agnostic (doesn't violate design constraint)
- Uses observable system state (concurrent requests is trackable)
- Explains why reasoning (high concurrency) differs from codegen (low concurrency)
- Can be fitted from existing traces by analyzing request arrival patterns

**If Option 1 fails**, fall back to Option 3 (nsys profiling) to get ground truth.

## Code Citation

Trace data location: `training/trainval_data/20260217-170634-llama-2-7b-tp1-reasoning/`
- Lifecycle metrics: `results/per_request_lifecycle_metrics.json`
- Summary: `results/summary_lifecycle_metrics.json`
- OpenTelemetry traces: `traces.json`
- KV events: `kv_events.jsonl`

vLLM instrumentation: Traces generated by vLLM OpenTelemetry integration, but lacks fine-grained scheduler/kernel events needed for bottleneck analysis.
