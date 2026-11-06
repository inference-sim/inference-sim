---
hide:
  - navigation
  - toc
---

# cost-model-collectives-inferencex

DeepEP dispatch plus combine, priced as one MoE layer's all-to-all:

> latency = `floor` + tokens · hidden · 2 · top_k / `rate`

with per-rank tokens. Two variants are here, both as InferenceX CollectiveX times them:

- `deepep_low_latency_graph`: DeepEP's legacy low-latency kernels under CUDA-graph replay,
  which vLLM v0.31.0 uses for a captured decode step. The eager pair
  (`moe_a2a_*_deepep_low_latency_<N>rank_<part>` in
  [cost-model-collectives](cost-model-collectives.md)) prices a step with no graph.
- `deepep_v2_graph`: DeepEP v2's ElasticBuffer with vLLM `deepep_v2`'s CUDA-graphs-on flags
  (no CPU sync, no expand). CollectiveX replays decode as a graph and runs prefill eager on the
  same path.

**Where the values come from.** InferenceX CollectiveX, backend deepep-v2, bf16, shape hidden
7168, top-6 of 384, uniform routing, the roundtrip p50 per tokens-per-rank point. The h100
and h200 8-rank pair (one node, NVLink) fits tokens 1/4/16/64/256 and holds out 2/8/32/128.
GB200-NVL72 at 8 and 16 ranks (2 and 4 trays of one MNNVL rack) and H200 at 16 ranks (two
nodes over 400G InfiniBand, scoped `fabric: [ib-400g]`) use the same split for the
low-latency mode. The v2 mode adds the prefill ladder: it fits 1/4/16/64/256/1024/4096 and
holds out 2/8/32/128/512/2048/8192. Each entry states its run, its body's sha256 and its
held-out error. The H200 16-rank entries measure a virtualised pool's path, not the
InfiniBand fabric, and say so.

**Why a separate set.** cost-model-collectives comes from a single source, NVIDIA
AISimulate. A set that mixed the two methodologies would confound comparisons across
parts. A scenario lists this set next to cost-model-collectives to take these prices.
Without it, the kernel prices as before.

<!-- registry:set cost-model-collectives-inferencex -->

**Regenerate:** `fit_deepep_ll_graph.py` (h100 and h200, 8 ranks) and
`fit_deepep_collectivex.py` (GB200-NVL72 8 and 16 ranks, H200 16 ranks); `--check` is the
drift gate for each.
