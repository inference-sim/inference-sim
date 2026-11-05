---
hide:
  - navigation
  - toc
---

# cost-model-primitives

The constants for dense and grouped matrix multiplies, plus the per-part memory and link
figures from NVIDIA's own system descriptors.

**GEMM efficiency ramp** (`measured`). The kernel prices a GEMM at a fraction of the
part's dense peak for the dtype, rising with the token count M:

> efficiency(M) = `gemm_eps_max_<dtype>` · M / (M + `gemm_m_half_<dtype>`)

Both constants are fitted per part and per dtype on AISimulate's GEMM sweeps. They have
to be per part: H100 and H200 share a die and their asymptotes agree, but their knees do
not, because the knee follows memory bandwidth rather than compute.

**MoE routing imbalance** (`measured`). The ratio of grouped-GEMM latency under a skewed
expert distribution to the same shape under a balanced one, as the median and 90th
percentile over many paired shapes. Not a curve fit, so `fitted: false`.

**Descriptor figures** (`vendor_spec`). HBM derate and latency, the host-link bandwidth,
peer-to-peer latency, NCCL communicator buffer sizes and the engine's non-weight
workspace, transcribed from AISimulate's `systems/<sku>.yaml`.

Not every family here is read by the kernel today; the
[reference overview](../index.md#which-sets-the-kernel-reads) lists which.
A100-80 and A100-SXM are the same silicon, so their entries share one scope.

<!-- registry:set cost-model-primitives -->

**Regenerate:** `relane_gemm_envelope.py` (GEMM), `relane_moe_imbalance.py` (MoE),
`emit_primitives.py` (descriptors). See the
[writer table](../../reproducing-coefficients.md#every-committed-value-and-the-one-command-that-regenerates-it).
