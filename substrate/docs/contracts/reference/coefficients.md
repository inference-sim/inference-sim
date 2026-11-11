# Coefficient set

<p class="lede">A set of coefficients for a cost model. Each entry records its value and
unit, how it was obtained, the evidence for it, and the hardware and layouts over which it
holds. The registry owns the sets; every consumer loads them through this schema.</p>

The set below is illustrative; its values are not real coefficients. The registry's sets
are in
[`coefficients/`](https://github.com/inference-sim/blis-substrate/tree/main/registry/coefficients).

```yaml
--8<-- "docs/contracts/examples/coefficient-set.yaml"
```

Each entry in `coefficients` is a map with a single key, the entry's name. A `name` key
inside an entry is accepted and replaced by that map key. A set is
complete in itself: it does not inherit from another set. A scenario may still name several
sets, and how a cost model combines them is described under
[Selecting an entry](#selecting-an-entry).

## Keys

<!-- fields: coefficient.Set -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `CoefficientSet`. |
| `name` | string | yes | The set's identity. By convention, not checked, it equals the file name, which is what a scenario's `coefficients` uses. |
| `coefficients` | list | yes | At least one entry, each `- <entry name>: {...}`. |

### Entries

<!-- fields: coefficient.Entry -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `value` | number | unless `method` is `not_charged` | The coefficient. Zero exactly when `method` is `not_charged`. |
| `units` | string | yes | One of the [units](vocabularies.md#units). |
| `method` | string | yes | How the value was obtained. One of the [methods](vocabularies.md#methods). |
| `fitted` | boolean | yes | Whether the value comes from fitting a curve. Must be present even when false. |
| `scope` | mapping | yes | Where the value holds, below. At least one dimension. |
| `rate_basis` | string | no | What a `bytes_per_us` rate measures: `memory` (the default) or `work`. A memory rate cannot exceed the chip's datasheet memory bandwidth (SUB-5). A `work` rate counts bytes of work, such as a record counted once per query and key pair, and may exceed it; it requires a `rationale`. One of the [rate bases](vocabularies.md#rate-bases). |
| `kernel_impls` | list | no | The kernel implementations the value is valid for ([below](#kernel_impls)): the lookup key. A lookup matches the entry only when one of them is what runs. Allowed on `measured`, `calibrated`, `transferred` and `literature`; not on `vendor_spec`, `roofline`, `assumed` or `not_charged`. |
| `measured_on` | list | on `transferred` that states `kernel_impls` | The kernel implementations a transferred value was measured on, same form as `kernel_impls`. Evidence only; a lookup never matches it. Only on `transferred`. |
| `measures` | mapping | no | What the number includes ([below](#measures)). Only on `measured` and `calibrated`. |
| `fit_range` | list | no | The range of inputs the fit's data covered ([below](#fit_range)). Only on `measured`, `calibrated` and `transferred` (the rows a transferred value came from). A price that reads the entry outside it is extrapolated and ranks with `transferred`. |
| `ci95` | `[low, high]` | no | 95% confidence interval. Must contain `value`. |
| `sources` | list | depends on `method` | Citations, below. If present, not empty. |
| `rationale` | string | depends on `method` | The reasoning behind a value that was not measured. |
| `transferred_from` | string | with method `transferred` | The scope the value was measured in, as the scope renders, such as `hardware=h200,model=deepseek-v4.1-flash`. |
| `transfer_rule` | string | with method `transferred` | The stated rule that moved the value to this scope. |
| `supersedes` | string | no | An entry this one replaces. |
| `validated` | string or list | no | Names of metrics the value has been checked against, such as `itl`. Free text. |
| `unsupported` | string or list | no | Names of metrics it has been found not to hold for. Free text. |

### `scope`

Each dimension lists the values the coefficient holds for. An empty scope is not
"everywhere"; it means the range is unknown, and it is rejected.

<!-- fields: coefficient.Scope -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `hardware` | list of strings | no | Chips. |
| `model` | list of strings | no | Models. |
| `tp` | list of integers | no | Tensor-parallel widths. |
| `ep` | list of integers | no | Expert-parallel widths. |
| `nodes_spanned` | list of integers | no | How many nodes a collective crosses. |
| `fabric` | list of strings | no | Inter-node interconnect classes, by catalog identity: the stem of a fabric under `networks/`, such as `roce-200g` or `ib-400g`. `nodes_spanned` says whether a collective leaves a node; `fabric` says over which link. A link inside one chip's NVLink domain, a whole NVL72 rack included, belongs to the chip and is scoped by `hardware`. Each name must be a bare identity, not empty, not a path, and listed once. |

Dtype and backend are not scope dimensions. Both are part of the entry's name, as in
`gemm_eps_max_bf16`.

### `sources`

<!-- fields: coefficient.Source -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | One of the [source kinds](vocabularies.md#source-kinds). |
| `cite` | string | yes | The reference: an issue, a paper, a datasheet URL. |
| `role` | string | yes | One of the [source roles](vocabularies.md#source-roles). |

### `kernel_impls`

An implementation is one exact kernel: a library, the library's own name for the kernel, and the
hash of its source. The lookup (design D5, VAL-8) matches an entry that lists implementations only
for the primitive its name prices (the GEMMs, the decode attention, a collective; host overheads and
other terms that price no lowered kernel cannot be keyed), and only when every implementation the
pool runs for that primitive is one the entry lists, with library, variant and code hash all equal. A different
hash of the same kernel does not match, and neither does the same hash of another kernel: there is
no fallback to a sibling. What the lookup does then is the next entry by method rank (design 9.2),
at worst the family's `roofline` or `assumed` entry, which state no implementation; the price
reports that method.

- An entry that lists none is **unkeyed**: it applies to whatever runs, and nothing shows it was
  measured on the kernel that runs. It prices as it always did, and a reader deciding whether a
  price rests on the running kernel (the G1 hash condition, design 11.3) does not count it.
- With no implementation stated for the primitive, only unkeyed entries match.
- The two lists of an entry say different things. `kernel_impls` is where the value is valid
  and is the key. `measured_on` is where its data came from. For `measured` and `calibrated` they
  are the same by D5, so only `kernel_impls` is stated. A `transferred` value is measured on one
  kernel and used for another: it states the target in `kernel_impls`, the source in `measured_on`,
  and the rule in `transfer_rule`. The same kernel on both is a transfer between chips.
- Two entries of one name, scope and method that differ only in `kernel_impls` are two
  coefficients: a kernel whose source changed has its own value.

<!-- fields: coefficient.KernelImpl -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `library` | string | yes | The library that runs the kernel. |
| `variant` | string | yes | The library's own exported kernel name. |
| `code_hash` | string | yes | The hash of the kernel's source and target GPU architecture, or of a binary library: lower-case hexadecimal, at least 12 digits. |

### `measures`

<!-- fields: coefficient.Measures -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `time` | string | yes | One of the [measurement times](vocabularies.md#measurement-times): `device`, or `device_and_host`. |
| `execution` | string | yes | One of the [measurement executions](vocabularies.md#measurement-executions): `captured` or `eager`. |
| `harness` | string | yes | The harness that collected the data, with its pin. |

### `fit_range`

<!-- fields: coefficient.Bound -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `dim` | string | yes | The input dimension, such as `tokens` or `ranks`. Listed once. |
| `low` | number | yes | The smallest value the fit's data took. |
| `high` | number | yes | The largest value the fit's data took. Not below `low`. |

An interval `ci95` that is unbounded, or wider than half the value, marks the entry
*unidentified*: its data do not determine it, and a price that used it ranks with `assumed`.

## Identity is name and scope

Two entries may share a name if their scopes differ. The set above has two
`gemm_eps_max_bf16` entries, one for H100 and one for H200. Two entries with the same name
*and* the same scope are an error, because a cost model would use whichever came last.
Scopes are compared without regard to order: `[h100, h200]` and `[h200, h100]` are the same
scope.

## Selecting an entry

The schema defines what an entry says. Choosing among entries is the cost model's job, and
the cost model in `kernel/` does it as follows. An entry applies to a deployment if, for every
dimension its scope states, the list includes the deployment's value; hardware, model and
fabric names are compared without regard to case, and a dimension the scope omits does not
constrain it. The kernel reads the scenario's sets in order and, within each set, its
entries in order, and the last applicable entry for each name is the one it uses. A later
set can therefore override part of an earlier one. Entries that do not apply are skipped.
Construction fails if no entry applies at all, or if the cost model needs a coefficient no
applicable entry supplies.

## What a method requires

| Method | `rationale` | `sources` | Other |
|---|---|---|---|
| `measured` | — | — | Allows `fitted: true`. A source is expected but not required. |
| `calibrated` | required | at least one of kind `measured_case` | Allows `fitted: true`. `validated` must name the held-out metrics. |
| `literature` | required | at least one | |
| `vendor_spec` | required | at least one | |
| `transferred` | required | — | `transferred_from` and `transfer_rule` required. |
| `roofline` | required | at least one, naming the catalog facts it was computed from | |
| `assumed` | required | — | |
| `not_charged` | required | — | `value` must be 0. |

## Checks

A set fails to load if an entry is not a single-key map, an entry has no `fitted` key,
`sources` is present but empty, `ci95` is not a two-element list, `validated` or
`unsupported` is neither a string nor a list of strings, or any mapping has a key its type
does not declare.

A loaded set is invalid if

- `kind` is not `CoefficientSet`, `name` is empty, or there are no entries;
- an entry has no name, or repeats another entry's name and scope;
- `value` is not finite, `units` or `method` is not recognized, or `scope` is empty;
- `value` is 0 with a method other than `not_charged`, or nonzero with `not_charged`;
- `fitted` is true with a method other than `measured` or `calibrated`;
- a companion field the method requires, per the table above, is missing;
- a source has an unrecognized `kind` or `role`, or an empty `cite`;
- `ci95` has a non-finite end, has `low` above `high`, or does not contain `value`.

Source: [`spec/coefficient`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/coefficient).
