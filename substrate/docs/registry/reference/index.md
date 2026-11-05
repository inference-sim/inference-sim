# Coefficient sets

Everything on the pages in this section is generated from the committed files when the
site is built, at revision <!-- registry:ref -->. Nothing is typed in by hand, so a count
here cannot disagree with the data.

## The sets

<!-- registry:census -->

*Entries* counts every (name, scope) pair. *Coefficients* counts distinct quantities,
after folding the part out of names that carry it, such as
`collective_floor_all_reduce_fp16_8rank_h200`. *Parts* is the number of GPU parts with
at least one entry.

The speculative acceptance-profile sets under `acceptance/` are not coefficient sets and
are not counted here; they have [their own page](acceptance.md).

## What each set rests on

The share of each set by method, with the number of entries at the right:

<!-- registry:chart -->

## Which sets the kernel reads

A scenario names the sets it loads. The five `cost-model-*` sets for step time are the
ones every committed scenario in blis-latency-kernel lists. Within them, a few families
are committed but not looked up by the kernel. As of blis-latency-kernel
[`e9cf217`](https://github.com/inference-sim/blis-latency-kernel/commit/e9cf217):

| Set | Not read | Reader |
|---|---|---|
| `cost-model-primitives` | every `gemm_*_fp8_block` entry: the kernel's dtype selection resolves to `bf16`, `fp8` or `nvfp4` only | selection by weight format and chip, [blis-latency-kernel#34](https://github.com/inference-sim/blis-latency-kernel/issues/34) |
| `cost-model-primitives` | `gemm_*_w4a16_mxfp4_grouped` and `gemm_*_w8a16_mxfp8`, the Marlin weight-only ramps | the same, [blis-latency-kernel#34](https://github.com/inference-sim/blis-latency-kernel/issues/34); the gb300 grouped ramp additionally needs a deployment that states a Marlin MoE backend, since stock v0.31.0 runs CUTLASS there |
| `cost-model-primitives` | `host_gather_rate_uva` | a host-resident gather op, [blis-schemas#48](https://github.com/inference-sim/blis-schemas/issues/48) and [blis-latency-kernel#38](https://github.com/inference-sim/blis-latency-kernel/issues/38) |
| `cost-model-attention` | `attention_decode_{floor,rate}_block_index_scores` | a lookup keyed on the attention node's `role`; until then the kernel prices an indexer node with its layer kind's pair |
| `cost-model-memory` | the whole set. It is committed ahead of its reader | [blis-latency-kernel#22](https://github.com/inference-sim/blis-latency-kernel/issues/22) |

`attention_decode_rate_sparse_mla` IS read, for the one model it is scoped to; that model is
not yet priceable end to end ([methodology §10.5](../methodology.md#105-deepseek-v41-flash-on-h200-what-vllm-v0310-runs-and-what-is-measured-of-it)).

Three families that had no reader were removed rather than listed here:
`hbm_constant_latency`, `p2p_latency` and `moe_routing_imbalance_p90`
([methodology §1](../methodology.md#1-what-a-coefficient-is)).

An unread entry still passes every check, so this table is the place to look before
assuming a change to one will move an estimate.

## Reading a set page

Each set page shows the set as a matrix: one row per coefficient, one column per part.
A glyph before each value gives its method, and its shape alone identifies it:

| Glyph | Method |
|---|---|
| <span class="reg-m reg-m-measured">■</span> | `measured` |
| <span class="reg-m reg-m-vendor_spec">◆</span> | `vendor_spec` |
| <span class="reg-m reg-m-assumed">○</span> | `assumed` |
| <span class="reg-m reg-m-not_charged">–</span> | `not_charged` |

Every value is a link to its entry in the file, where the sources, rationale and scope
are recorded. A dot (·) marks a part the set has no entry for. Whether that is an
omission or a deliberate absence is stated in the set's header and on its page.
