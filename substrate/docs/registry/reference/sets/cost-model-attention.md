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

<!-- registry:set cost-model-attention -->

**Regenerate:** `relane_attention_decode.py`, `relane_attention_prefill.py`,
`relane_attention_swa.py`, `relane_attention_mla.py --check-rate` and
`correct_mla_floor.py`.
