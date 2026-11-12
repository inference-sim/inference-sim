# Recurrent state: Mamba-2, KDA and GDN layers in the KV cache

A model with linear-attention or state-space layers (Nemotron-3-Ultra's Mamba-2, Kimi-K3's KDA,
Qwen3.5's GDN) keeps a fixed-size state per request in each such layer. vLLM v0.32.0 stores it in
the KV cache as a `MambaSpec` and manages it with the `MambaManager`. This page says what the
engine builds from a model graph, how vLLM's rules are followed, how to reproduce the figures, and
what is not covered.

Contents: [Method](#method) - [What the engine builds](#what-the-engine-builds) -
[Worked examples](#worked-examples) - [Reproduce](#reproduce) - [Cost](#cost) -
[What this does not show](#what-this-does-not-show) - [Sources](#sources)

## Method

1. **State bytes.** `compat/kernelmodel/mamba_state.go` turns each `RecurrentUpdate` node of the
   catalog graph into the state tensors vLLM's calculators give
   (`vllm/model_executor/layers/mamba/mamba_utils.py:195-327@5df63a72`), divided over the
   tensor-parallel ranks, and sums their bytes at the dtype that runs (`mamba_cache_dtype` for the
   convolution state, `mamba_ssm_cache_dtype` for the temporal state; `auto` as vLLM resolves it).
   The convolution state is as wide as the kernel less one, plus the speculative tokens. A node
   that lacks a field the calculator needs is refused naming the field; nothing is guessed.
2. **Cache mode.** With prefix caching on (vLLM's default) the model hook sets
   `mamba_cache_mode` to `align`, and to `none` with it off, whatever was stated
   (`vllm/model_executor/models/config.py:685-706@5df63a72`). `all` is refused.
3. **Block size.** For a hybrid model vLLM raises the attention block size until the attention page
   covers the state, rounded to the attention backend's kernel block sizes, and pads the state to
   exactly that page (`vllm/platforms/interface.py:1036-1126@5df63a72`,
   `core/vllm/hybridblock.go`). A stated `block_size` below that is replaced. In `align` mode the
   Mamba block size is that block size; in `none` mode it is `max_model_len`.
4. **Groups and pool.** Attention layers and state layers have the same page, so
   `get_kv_cache_groups` splits them by type into groups of equal size
   (`vllm/v1/core/kv_cache_utils.py:1362-1502@5df63a72`, `core/kv/layers.go`). A pool block is the
   widest group's pages; the start-up check counts a state group as `1 + speculative` pages in
   `none` mode and `2 + speculative + checkpoint` in `align` mode
   (`vllm/v1/kv_cache_interface.py:1077-1092@5df63a72`, `core/kv/startup_check.go`).
5. **Manager and scheduler.** `core/kv/mamba_manager_align.go` ports the `MambaManager` for `align`
   mode: a hit is the last cached state at a block boundary, earlier slots hold null blocks, the
   running state sits in the last block, a prefill checkpoint takes one more block, and a state
   cached in this step is not hit by another request until the next step. `core/sched/mamba_split.go`
   ports the scheduler's aligned split, which ends a prefill chunk where a state must be cached
   (`vllm/v1/core/sched/scheduler.py:443-545@5df63a72`). Retention is vLLM's default interval of 0:
   only the prompt's last block-aligned state is kept.

## What the engine builds

For a graph with recurrent nodes `KVLayers()` returns, in model order, runs of Spec `mamba` beside
the attention runs (an MLA layer beside them is the single-latent `mla` spec, not keys and
values). Each Mamba run carries `StateBytes`, `PageSizePadded`, `BlockSize`, `MambaCacheMode`,
`NumSpeculativeBlocks` and the prefill checkpoint fields. The engine block's `block_size` becomes
the aligned size, and `mamba_model: true` marks the model for the hook.

## Worked examples

Both at tensor-parallel width 8, prefix caching on, no speculation, from the catalog graphs of
the pinned blis-substrate release (see `go.mod`).

| | Kimi-K3 (KDA) | Nemotron-3-Ultra (Mamba-2) |
|---|---|---|
| convolution state per layer per rank | 36864 / 8 x 3 = 13824 elements, bf16: 27648 B | (16384 + 2 x 8 x 128) / 8 x 3 = 6912 elements, bf16: 13824 B |
| temporal state per layer per rank | 12 x 128 x 128 = 196608 elements, fp32: 786432 B | 32 x 64 x 128 = 262144 elements, fp32: 1048576 B |
| state bytes | 814080 | 1062400 |
| attention page for one token | 576 x 2 = 1152 B (MLA, bf16) | 512 B (GQA, bf16) |
| aligned block size | 768 (1536 at fp8) | 2080 (fp8 cache and one MTP draft token: 4176) |
| padded state page | 884736 B | 1064960 B |
| prefill checkpoint | 1 block, alignment 16 (FlashKDA) | 1 block, alignment 1 |

`vllmengine/recurrent_test.go` builds Nemotron-3-Ultra (three H100 nodes, tp 8, MTP 1) and
Kimi-K3 (two H200 nodes, tp 8, dp 2, expert parallel) at default settings and runs requests to
completion; before this change both were refused with "recurrent state is not mapped to a KV cache
spec".

## Reproduce

Inputs: this repository at the commit under review; vLLM v0.32.0 at
`5df63a7203cdb9328d824a27b682e77c7b9928be`; the blis-substrate release `go.mod` pins (catalog graphs,
registry coefficients). Commands, from the repository root, with Go from the toolchain `go.mod` names:

```sh
go test ./compat/kernelmodel -run 'Mamba|Recurrent|Hybrid'   # state bytes, spec fields, refusals
go test ./core/vllm -run 'Hybrid|Mamba|KVPool'               # block-size alignment, pool bytes
go test ./core/kv -run 'Mamba'                               # groups, pages, start-up check, manager
go test ./core/sched -run 'Mamba|TestDifferentialAgainstVLLM/mamba|TestKV.*AgainstVLLM'
go test ./vllmengine -run TestRecurrentModelsBuildAndRun     # the two models build and run
```

The scenarios and the golden traces are produced by vLLM's own scheduler on Linux (the
`oracle-traces` workflow, `oracle/README.md`, "Hybrid Mamba scenarios"); regenerate the scenarios
with `python -I oracle/gen_mamba_scenarios.py oracle/scenarios`. Expected: every `mamba-*` scenario
matches its golden step by step, and `mamba-*-startup-refused` and `mamba-*-pool-refused` match
vLLM's refusal text.

## Cost

The Go tests above take seconds. Building one recurrent model costs the same as another hybrid
model: the graph is walked once. The oracle run on CI is about 20 minutes for the whole workflow.

## What this does not show

- The scenarios' state sizes are examples shaped like the two models, not read from checkpoints;
  the unit tests carry the catalog numbers.
- Recorded against vLLM's scheduler with its GPU mocked: no step time or accuracy claim.
- Not modelled, and refused by name: `mamba_cache_mode: all`, `use_replayssm`, `use_kda_recoverssm`,
  `enable_mamba_shared_prefix_checkpoint`, partial hash hits (a hash block smaller than the Mamba
  block, which needs data or context parallelism), a KV connector beside a state group, a nonzero
  `prefix_cache_retention_interval`, a recurrent drafter, context parallelism, and a stated
  `kda_prefill_backend` the graph cannot decide (the engine assumes FlashKDA, which needs
  `gate_lower_bound` in the checkpoint, stated as an assumption in the report).
- The attention block size of a hybrid model whose attention backend is chosen by a rule the block
  size table does not cover (head size above 128) is refused, not guessed.

## Sources

vLLM v0.32.0 (`5df63a72`): `vllm/v1/kv_cache_interface.py:1035-1148`,
`vllm/v1/core/single_type_kv_cache_manager.py:1415-2112`, `vllm/v1/core/sched/scheduler.py:375-545`,
`vllm/platforms/interface.py:759-1126`, `vllm/model_executor/layers/mamba/mamba_utils.py:55-327`,
`vllm/model_executor/models/config.py:672-740`, `vllm/models/kimi_k3/nvidia/kda.py:463,548-600,832`.
