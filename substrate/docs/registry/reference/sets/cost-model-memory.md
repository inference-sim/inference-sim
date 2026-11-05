---
hide:
  - navigation
  - toc
---

# cost-model-memory

The per-rank memory a deployment holds besides weights and KV cache. These numbers set
how many sequences fit, not how long a step takes.

The terms add, per rank:

> weights + activation + `nccl_communicator_bytes_<N>rank` + `engine_workspace_bytes` + `cudagraph_capture_bytes_<mode>`

with

> activation = max(`activation_buffer_count_<family>_<N>rank` · max_num_batched_tokens · h · 2, `activation_scratch_floor_bytes`)

where *family* is `dense` or `moe`, *N* is min(tp, 8), and *h* is the attention width.
The communicator and workspace terms are in
[cost-model-primitives](cost-model-primitives.md), because NVIDIA's descriptors state
them per part.

**Two provenances.** The activation multiples and floor are `vendor_spec`: NVIDIA's own
memory model, as AISimulate's vLLM backend uses it. The CUDA-graph capture sizes are
`assumed`: their magnitude is the median of vLLM's own start-up logs, collected from
vLLM's issue tracker into a committed CSV
([`vllm-cudagraph-capture-samples.csv`](../../vllm-cudagraph-capture-samples.csv)).
`FULL` and `FULL_DECODE_ONLY` have too few samples and are charged the
`FULL_AND_PIECEWISE` figure as an upper bound. `NONE` is a declared zero.

The kernel does not read this set yet; see
[blis-latency-kernel#22](https://github.com/inference-sim/blis-latency-kernel/issues/22).
[Methodology §11](../../methodology.md#11-memory-occupancy-what-composes-and-where-each-term-comes-from)
gives the composition and each scoping decision.

<!-- registry:set cost-model-memory -->

**Regenerate:** `emit_memory.py`; `--check` is the drift gate.
