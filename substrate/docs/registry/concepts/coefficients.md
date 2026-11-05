# Coefficients

## A constant in a law

The kernel prices each primitive with a closed-form law. The registry supplies the
constants in those laws, and nothing else. Three examples:

| Primitive | Law the kernel evaluates | Constants the registry supplies |
|---|---|---|
| Dense GEMM | efficiency(M) = ε<sub>max</sub> · M / (M + M<sub>½</sub>) | `gemm_eps_max_<dtype>`, `gemm_m_half_<dtype>` |
| Decode attention | latency = floor + KV bytes / rate | `attention_decode_floor`, `attention_decode_rate` |
| Collective | latency = max(floor + size / r<sub>t</sub>, size / r<sub>peak</sub>) | `collective_floor_…`, `collective_transition_rate_…`, `collective_peak_rate_…` |

The division matters. A law is code, reviewed as code, and the same for every part. A
constant is data, specific to a part, and carries the evidence for its value. A constant
is never a correction for one model: if a model is mispriced, the remedy is in the law or
in the evidence, not in a per-model fudge factor.

It also means a coefficient is only meaningful for the law it was fitted against. Two
constants with similar names in models of different form are different quantities. This
is why sets do not inherit from one another.

## What an entry records

Each coefficient is one entry, keyed by its name:

| Field | Required | Records |
|---|---|---|
| `value` | yes | The number. Finite. |
| `units` | yes | Its dimension, from a closed list (`us_per_transfer`, `bytes_per_us`, `tokens`, …). |
| `method` | yes | How the value was obtained. See below. |
| `fitted` | yes | Whether a curve fit produced it. |
| `scope` | yes | Where it applies. See below. |
| `sources` | by method | Citations: the exact file a value came from, and its role. |
| `rationale` | by method | Why this value: what it replaced, what would replace it, where it is weak. |
| `ci95` | no | An interval `[low, high]` that must contain the value. |

The [file format](../reference/format.md) lists every field and rule exactly.

## Method: how the value was obtained

The method is the field that keeps a measurement distinguishable from a guess. It is
never inferred; the author states it.

`measured`
:   Obtained from a measurement this project can re-run. Cites the data it came from.

`vendor_spec`
:   Taken from a vendor's own figure: here, the values NVIDIA ships in AISimulate's
    system descriptors and its memory model. NVIDIA observed or tuned them; this project
    did not. Requires a source.

`assumed`
:   An estimate arrived at by reasoning, where no dataset this project reads measures the
    quantity. Requires a rationale; by this registry's convention it also names the
    measurement that would replace the value. A careful estimate is still `assumed`.

`not_charged`
:   A deliberate zero, so that "this costs nothing" is distinguishable from "this cost is
    unknown". A value of zero under any other method is rejected.

The schema also defines `literature` (a published result not reproduced here),
`copied` (carried over from another scope, which `copied_from` names) and, since
blis-schemas `v0.4.0`, `calibrated` (back-solved through the cost model from end-to-end
measured cases, which it cites as `measured_case` sources, with the held-out metrics in
`validated`). No committed entry uses any of the three.

`fitted` is separate from `method` because the two answer different questions. A
`vendor_spec` value is never fitted. A `measured` value may be fitted (the GEMM ramp) or
not (an order statistic of measured ratios, like the MoE routing imbalance). The schema
allows `fitted: true` only with `method: measured` or `calibrated`.

## Scope: where the value holds

A scope lists the deployments an entry applies to, along five dimensions: `hardware`,
`model`, `tp`, `ep` and `nodes_spanned`. Every committed entry is scoped by hardware
alone, using the catalog's chip names.

Three rules govern it:

- **A stated dimension restricts; an unstated one does not.** An entry scoped to
  `hardware: [h200]` applies to every model and parallelism on an H200, and to nothing
  else.
- **An empty scope is an error, not a wildcard.** A coefficient whose applicability is
  not stated cannot be applied safely, so the schema refuses it.
- **An entry's identity is its name and its scope.** One set may hold
  `gemm_eps_max_bf16` for `h100` and again for `h200`; the kernel picks by scope. Two
  entries with the same name and the same scope are rejected, because only one could be
  used.

Some quantities vary along dimensions that are not scope keys, such as data type,
collective operation and rank count. Those ride in the name:
`collective_floor_all_reduce_fp16_8rank_h200`. The collective names also carry the part,
because the kernel constructs those keys from the chip name.

## Sets

A coefficient set is one YAML file under `coefficients/`, naming itself
(`name: cost-model-attention`) and listing its entries. Sets are standalone: no file
extends another, and a scenario that wants several lists them all. If two listed sets
hold the same name for the same part, the one listed later is used.

How entries are grouped into sets follows the evidence as much as the primitive. The
seven host overheads are all `assumed` and are kept in their own set so that no reader
takes the provenance of a mostly measured set to be uniform. The memory-occupancy
magnitudes are kept apart from the primitives because they rest on a different source.

Most sets are written by a script, not by hand. Their headers say so, and the committed
file is whatever the script prints; [Contributing a change](../contributing/index.md)
explains why that matters.

Next: [where the numbers come from](evidence.md).
