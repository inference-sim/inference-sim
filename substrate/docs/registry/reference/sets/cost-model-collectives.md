---
hide:
  - navigation
  - toc
---

# cost-model-collectives

The constants for the four collectives the kernel prices: all-gather, all-reduce,
all-to-all and reduce-scatter. Each configuration (operation, dtype, rank count, part)
has three:

> latency = max(`floor` + size / `transition_rate`, size / `peak_rate`)

The floor is the latency of a small message, where size no longer matters. The peak
rate is the bandwidth a large message reaches. The transition rate governs the region in
between, roughly 64 KiB to 8 MiB, where a two-parameter model is optimistic and where a
prefill batch's collectives fall.

Operation, dtype and rank count are not scope keys, so they are part of the name, and so
is the part, because the kernel builds the key from the chip name:
`collective_floor_all_reduce_fp16_8rank_h200`. The tables below fold the part into the
columns.

**Where the values come from.** All-reduce at fp16 is measured on vLLM's own custom
all-reduce kernel on most parts; the other operations, and the int8 all-reduce, are
measured on NCCL, which is what vLLM runs for them. The source table on
[Where the numbers come from](../../concepts/evidence.md#fit-the-engine-you-predict)
gives the current split. Rank widths a part was never swept at are
`assumed`, extrapolated from that part's own measured widths with a holdout-validated
linear model, and each such entry states its predictor and holdout error.
[Methodology §8.5](../../methodology.md#85-rack-scale-rank-widths-and-the-one-place-this-registry-extrapolates)
gives the argument, including the model that was tried first and refuted.

The set also holds the MoE all-to-all that DeepEP dispatch plus combine costs, as
`floor` + bytes / `rate` per backend, rank count and part
(`moe_a2a_floor_deepep_low_latency_8rank_h100`).
All of them come from AISimulate. The low-latency pair measured by CollectiveX under
CUDA-graph replay is in
[cost-model-collectives-inferencex](cost-model-collectives-inferencex.md).

<!-- registry:set cost-model-collectives section=^(collective_(?:floor|peak_rate|transition_rate)_(?:all_gather|all_reduce|alltoall|reduce_scatter)|moe_a2a_(?:floor|rate))_ sort=natural -->

**Regenerate:** `emit_collectives.py` and `fit_collectives_vllm.py` for measured widths;
`insert_collectives_part.py` to add a part; `insert_collectives_wide.py` and
`extrapolate_collective_width.py` for wide groups. Do not regenerate the whole file with
`emit_collectives.py`; the
[reproduction guide](../../reproducing-coefficients.md#adding-a-part-the-gb300-worked-example)
explains why.
