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

A scenario names the sets it loads. Step time reads the five `cost-model-*` sets the cost
model's scenario fixtures list, and memory occupancy reads `cost-model-memory`. Within
them, a few families are committed but not looked up by the cost model in `kernel/` at
this revision:

| Set | Not read | Why |
|---|---|---|
| `cost-model-primitives` | every `gemm_*_fp8_block` entry | the kernel's dtype selection resolves an FP8 GEMM to `fp8`, never to a block-scaled ramp ([blis-latency-kernel#34](https://github.com/inference-sim/blis-latency-kernel/issues/34)) |
| `cost-model-primitives` | `gemm_*_w4a16_mxfp4_grouped` on b200, b300 and gb300 | the weight-only Marlin ramps are read only on a part with no native four-bit peak (h100, h200); a part with one is priced on its FP4 class, as stock vLLM v0.31.0 runs it |
| `cost-model-attention` | `attention_decode_{launch_floor,split_floor,serial_rate,per_token,throughput_rate}_sparse_mla` | the two-branch FlashMLA decode form of [methodology §10.7](../methodology.md#107-deepseek-v41-flash-decode-a-two-branch-flashmla-form-and-what-stays-assumed) has no reader; the kernel prices sparse-MLA decode with `attention_decode_rate_sparse_mla` |

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
