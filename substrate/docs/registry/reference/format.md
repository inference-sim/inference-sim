# File format

A coefficient set is a YAML file under `coefficients/`. The rules on this page are
enforced by [blis-schemas](https://github.com/inference-sim/blis-schemas), at the
release `v0.4.0`, in
[`spec/coefficient`](https://github.com/inference-sim/blis-schemas/tree/v0.4.0/spec/coefficient)
and [`vocab`](https://github.com/inference-sim/blis-schemas/blob/v0.4.0/vocab/vocab.go),
the release CI pins; it also validates `acceptance/` and `routing/`.
Where this page and the schema disagree, the schema is right.

## Shape

```yaml
kind: CoefficientSet          # exactly this
name: cost-model-primitives   # the set's identity; scenarios refer to it by this name
coefficients:                 # a list; each item is a map with one key, the coefficient's name
  - <name>:
      <fields>
```

One entry from the committed sets, quoted from its file:

<!-- registry:entry cost-model-primitives gemm_eps_max_bf16 h100 -->

Unknown keys are errors at every level: top level, entry, scope and source. A misspelled
field is rejected, not ignored.

## Entry fields

| Field | Type | Required | Rule |
|---|---|---|---|
| `value` | number | yes | Finite. Zero only with `method: not_charged`. |
| `units` | term | yes | One of the units below. |
| `method` | term | yes | One of the methods below. |
| `fitted` | bool | yes | Must be present, even when false. `true` requires `method: measured` or `calibrated`. |
| `scope` | map | yes | At least one dimension. |
| `sources` | list | by method | If present, non-empty. Required for `literature` and `vendor_spec`; `calibrated` requires at least one of kind `measured_case`. |
| `rationale` | string | by method | Required for every method except `measured`. |
| `ci95` | `[low, high]` | no | Finite, `low ≤ high`, and the value must lie inside. |
| `copied_from` | string | for `copied` | The scope the value was copied from. |
| `supersedes` | string | no | The entry this one replaces. |
| `validated` | metric or list | for `calibrated` | Metrics the value has been checked against. |
| `unsupported` | metric or list | no | Metrics it has explicitly not been checked against. |

Within one set, no two entries may share both a name and a scope. Scope comparison
ignores the order of values, so `[h100, h200]` and `[h200, h100]` are the same scope.

The registry's own tests add two requirements the schema leaves out: every `measured`
entry cites a source, and every `assumed` entry carries a substantial rationale.

## Vocabularies

Each vocabulary is closed. The tables show every term the schema accepts, and how often
the committed sets use each one, counted when this page was built.

### Units

`dimensionless`, `us_per_layer`, `us_per_request`, `us_per_step`, `us_per_hop`,
`us_per_transfer`, `us_per_load`, `us_per_token`, `bytes_per_us`, `bytes_per_rank`,
`tokens`, `sm_count`. In use:

<!-- registry:usage units -->

### Methods

`measured`, `vendor_spec`, `literature`, `copied`, `assumed`, `not_charged`, `calibrated`.
[Coefficients](../concepts/coefficients.md#method-how-the-value-was-obtained) explains
each. In use:

<!-- registry:usage method -->

### Scope keys

`hardware` (catalog chip names), `model` (catalog model names), `tp`, `ep`,
`nodes_spanned` (integers). Data type and engine are deliberately not scope keys; a
quantity that varies with them carries them in its name. In use:

<!-- registry:usage scope -->

### Sources

A source is `{kind, cite, role}`, all three required.

- `kind`: `discussion`, `publication`, `datasheet`, `model`, `vendor_doc`, `measured_case`
- `role`: `primary`, `supporting`, `upper_bound`
- `cite`: free text, specific enough to find the exact file and rows

In use, by kind and by role:

<!-- registry:usage source-kind -->

<!-- registry:usage source-role -->

### Optional fields in use

<!-- registry:usage optional -->

## Naming

Names are `snake_case`. A name states the quantity, then the axes that vary it and are
not scope keys, in a fixed order per family:

- `gemm_eps_max_<dtype>`
- `attention_decode_floor_<kind>`, with no suffix for full attention
- `collective_<param>_<op>_<dtype>_<N>rank_<chip>`, where `<chip>` is the catalog name
  with `-` replaced by `_`
- `nccl_communicator_bytes_<N>rank`, `cudagraph_capture_bytes_<mode>`

The kernel constructs these names when it looks a value up, so a name is an interface.
Renaming one is a change to the kernel too.
