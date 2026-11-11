---
hide:
  - navigation
  - toc
---

# cost-model-attention

The constants for attention, by phase and by kind.

**Decode** is priced as a fixed setup cost plus a read of the KV cache:

> latency = `attention_decode_floor` + KV bytes / `attention_decode_rate`

The floor is why a pure-bandwidth model is wrong at short context: an attention kernel
costs on the order of 10 µs whatever the context. The rate is the bandwidth the kernel
actually reaches, which on these parts is well short of the datasheet figure and varies
too much between parts for one derate to serve.

Three kinds of attention carry their own pair, because each reads a different number of
bytes:

| Suffix | Kind | Read per step, per request |
|---|---|---|
| none | full attention (GQA) | K and V for every cached token |
| `_swa` | sliding window | K and V for at most the window's tokens |
| `_mla` | multi-head latent attention | one compressed latent vector per cached token |

The `_mla` floors are `assumed`: they are set from the part's own measured full-attention
floor, because the tables they were first fitted on measured the whole MLA block,
projections included, and the kernel prices those projections separately.
[Methodology §3](../../methodology.md#3-the-three-level-data-separation) records how that
was found. Sparse MLA has no pair of its own;
[§10](../../methodology.md#10-sparse-mla-the-byte-count-is-the-fix-and-the-rate-is-not-fittable)
explains why none is fittable from this data.

**Prefill** is priced from the GEMM efficiency ramp, scaled by
`attention_prefill_work_scale`, above a floor `attention_prefill_floor`. The scale is a
fraction *of the bf16 ramp in cost-model-primitives*, so refitting that ramp means
refitting this.

**Prefill on the trtllm-gen kernel.** The prefill pair of b200, b300, gb200-nvl72 and gb300
is fitted on every full-attention row of the vLLM 0.25.0 context sweep, and that sweep holds
three kernels: 24,720 rows of FlashInfer's trtllm-gen prefill, 2,038 one-token batches vLLM
classed as decodes, and 816 Triton rows. FlashInfer's prefill runs the first one when
`attention_config.use_trtllm_attention` is true, when a stated block size is 128 or more, and,
unset, when the cache dtype is `auto`. The pair refitted on those rows alone, once for each of
the sweep's two cache dtypes (they differ: on b200 the floor is 18.5 us at bf16 and 21.0 at fp8,
against 20.0 pooled over both), is `attention_prefill_floor_trtllm_gen_kv_{bf16,fp8}` and
`attention_prefill_work_scale_trtllm_gen_kv_{bf16,fp8}`. The kernel reads them for the
non-latent layers when the engine reports that kernel as the implementation of the prefill
primitive and an entry lists it (VAL-8), and not otherwise. They are `transferred`, not `measured`: no kernel hash shows the
0.25.0 build runs the kernel of the v0.32.0 pin, and a `measured` entry needs one (D5, VAL-8;
the hashes are tracked in inference-sim/blis-registry#73). So a lane price grades one rank lower
than the pooled `measured` pair fitted on the same sweep; that is the label's cost, not the
data's. Each entry records the range its rows covered, and every row has no cached prefix, so a
prefill with cached context is outside the data. The lane replaces the pooled pair for every
non-latent kind, sliding-window and sparse layers included, as the pooled pair did. Decode has
no lane, because v0.32.0 decodes with trtllm-gen or XQA wherever the chip supports it.

<!-- registry:set cost-model-attention -->

**Regenerate:** `relane_attention_decode.py`, `relane_attention_prefill.py`,
`relane_attention_swa.py`, `relane_attention_mla.py --check-rate`, `correct_mla_floor.py`
and, for the trtllm-gen lane, `emit_attention_trtllm_gen.py`.
