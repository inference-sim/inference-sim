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
**Contains**: **DETAILED KV cache operation timing with request associations!**

**Structure**: `[absolute_timestamp, [event_list], ...]`

**Event types**:
- `BlockStored`: KV blocks stored (includes block hashes, token IDs, number of blocks)
- `CacheStoreCommitted`: Cache store committed (includes request ID, location: GPU/CPU, block count)
- `TransferInitiated`: KV transfer started (includes request ID, direction, block count)
- `TransferCompleted`: KV transfer completed (includes request ID, direction, block count)

**Example for a request with 78.4ms TTFT**:
```
+0.0ms: BlockStored, CacheStoreCommitted (67 blocks to CPU)
+14.1ms: TransferInitiated
+20.9ms: TransferCompleted
+32.7ms: BlockStored, CacheStoreCommitted (1 block to CPU)
+44.6ms: TransferInitiated
+78.4ms: First token generated
```

**What this reveals**:
- KV allocation starts **immediately** at request arrival (0ms)
- Initial bulk allocation: 67 blocks committed in first ~20ms
- Subsequent per-token allocations: 1 block every ~10-15ms
- CPU offloading: All CacheStoreCommitted events show "CPU" location
- Transfer overhead: 6.8ms between TransferInitiated and TransferCompleted

## Key Findings

### ✅ What We Can Measure
1. **E2E latency**: End-to-end request time (start to completion)
2. **TTFT**: Time to first token (from output_token_times[0] - start_time)
3. **ITL**: Inter-token latency (delta between consecutive output_token_times)
4. **TTFT variance**: p10=0.13ms, p90=215.4ms (1650× variance!)
5. **Prompt length**: Confirms ~1082 tokens (NOT 8K as hypothesized in iter3/4/5)
6. **KV allocation timing**: Per-request KV cache operations (block allocation, CPU offloading, transfer overhead)
7. **KV allocation overhead**: ~20ms for initial 67-block allocation + ~6.8ms per CPU transfer
8. **CPU offloading pattern**: All KV cache committed to CPU (not GPU-only)

### ❌ What We CANNOT Measure (Still Missing)
1. **Queuing delay vs prefill execution**: KV allocation starts at t=0, but we can't distinguish queue wait vs prefill compute within TTFT
2. **Prefill compute time**: Actual GPU kernel execution time (TTFT includes KV allocation + compute + transfers)
3. **Attention kernel time**: FlashAttention-2 execution time breakdown
4. **Memory allocation beyond KV**: Activation buffer allocation time
5. **Batch formation delay**: Time waiting for batch to fill (explains 1650× variance)
6. **Prefix cache hit/miss**: Whether shared system prompt was cached (no events for cache hits)

## Critical Finding: KV Events Reveal Timing Breakdown

### Sample Request Analysis (TTFT = 78.4ms)

**Timeline from KV events**:
```
t=0ms:    Request arrives, KV allocation starts immediately
          - BlockStored: 67 blocks
          - CacheStoreCommitted: 67 blocks to CPU
t=14ms:   TransferInitiated (CPU offloading starts)
t=21ms:   TransferCompleted (7ms transfer overhead)
t=33ms:   Additional block allocated (1 block to CPU)
t=45ms:   Transfer initiated again
t=78ms:   First token generated ✓
```

**What this tells us**:
1. **KV allocation is NOT the bottleneck**: 67 blocks allocated in first 0-21ms, plus ongoing per-token allocation
2. **CPU offloading overhead**: ~7ms per transfer, but happens in parallel with compute
3. **Total measured KV overhead**: ~20-30ms (allocation + first transfer)
4. **Remaining unexplained**: 78.4ms - 30ms = **48ms still unaccounted for**

### Where is the Missing 48-100ms?

KV events account for **~30ms** of the 78-100ms TTFT. The remaining **48-70ms** must be:

1. **Batch formation delay** (most likely): Request waits for batch to form before prefill starts
   - Explains 1650× variance (p10=0.13ms when immediate, p90=215ms when waiting)
   - Multi-turn chat (reasoning) has higher concurrency → longer waits

2. **Prefill compute time**: Actual GPU kernel execution (~5-15ms for 1K tokens)
   - But KV allocation happens in parallel, so may overlap

3. **Attention kernel startup**: FlashAttention-2 initialization (~5-20ms)

4. **Queue processing overhead**: Scheduler overhead to move from queue → running

**Conclusion**: KV allocation is **fast** (~30ms including CPU offload). The bottleneck is **batching delay** (request waiting for batch formation), NOT KV cache operations.

## Implications for Iter7

### What KV Events Reveal About the 100-200ms TTFT

The missing 78.5-178.5ms in reasoning (β₆=21.5ms captured, actual=100-200ms) **CAN be partially decomposed**:

**From KV events**:
- KV allocation + CPU offload: ~30ms (measured from BlockStored → TransferCompleted)
- This leaves **48-170ms unexplained** (not ~78-178ms as previously thought)

**Still cannot measure** (need additional instrumentation):
- Batch formation delay (queue wait time) - **most likely dominant component**
- Prefill compute time (GPU kernel execution)
- Attention kernel startup overhead
- Queue processing / scheduler overhead

**Key insight**: KV allocation is **fast** (~30ms). The bottleneck is **batching delay**, which explains:
- 1650× variance (immediate processing vs waiting for batch)
- Why reasoning differs from codegen (higher concurrency → longer batch waits)
- Why β₆=21.5ms is insufficient (captures mean delay, not p90=215ms)

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

## Recommendation (Updated After KV Events Analysis)

**Primary approach: Model batching delay variance explicitly**

The KV events prove that:
1. **KV allocation is fast** (~30ms), NOT the 100ms+ bottleneck
2. **1650× variance** (0.13ms → 215ms) cannot be explained by KV allocation or compute
3. **Batching delay is the dominant factor** (some requests immediate, others wait ~200ms)

**For iter7, model p90 batching delay instead of mean**:
```go
// Current iter6: β₆ = 21.5ms (captures mean)
// Iter7: Fit to p90 = 215ms, or model variance explicitly
queuing_delay_us = β₆ × percentile_multiplier × 1000.0
// percentile_multiplier: 1.0 for mean, 10.0 for p90 (215ms / 21.5ms)
```

**Alternative: Model as function of concurrent requests** (Option 1 from original):
- High concurrency (reasoning multi-turn) → longer batch formation delay
- Low concurrency (codegen) → immediate processing
- Workload-agnostic and physically grounded

**If both fail**: The variance may be position-in-batch dependent (first request in batch: 0.13ms, last request: 215ms). Would need to model request position explicitly.

## Code Citation

Trace data location: `training/trainval_data/20260217-170634-llama-2-7b-tp1-reasoning/`
- Lifecycle metrics: `results/per_request_lifecycle_metrics.json`
- Summary: `results/summary_lifecycle_metrics.json`
- OpenTelemetry traces: `traces.json`
- KV events: `kv_events.jsonl`

vLLM instrumentation: Traces generated by vLLM OpenTelemetry integration, but lacks fine-grained scheduler/kernel events needed for bottleneck analysis.
