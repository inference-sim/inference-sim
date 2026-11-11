# Field list

<p class="lede">An engine repository's classification of every configuration field and
environment variable one release of its engine reads: whether the port models it, has proven
it inert, or does not model it, with the evidence for each, and the rule a scoring harness
applies when a run did not record the field (design VLM-3, §11.2).</p>

Each engine repository publishes its list as `fields.json` at its root. The simulator prints
it with `blis fields --engine <kind>`, and the accuracy dashboard and the configuration search
read it there, so neither keeps a table of its own (COV-3, CFG-3). The list is JSON.
`fields.Parse` (and `fields.Load`) reads it strictly: a key the schema does not declare, a key
that appears twice in one object, anything after the document, or a number that does not
convert exactly is an error naming where it is. `fields.Marshal` writes the canonical bytes,
and a repository checks its committed file by comparing the two.

```json
--8<-- "docs/contracts/examples/field-list.json"
```

## Keys

<!-- fields: fields.FieldList -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `FieldList`. |
| `engine` | mapping | yes | The engine and release the list classifies, below. |
| `fields` | list | yes | One entry per field, sorted by `name` and then `source`, in byte order; a field appears once. |

### `engine`

<!-- fields: fields.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | The engine kind, for example `vllm`. |
| `version` | string | yes | The upstream release, for example `0.32.0`. |

### `fields`

<!-- fields: fields.Field -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | The upstream's own name for the field or the variable. |
| `source` | string | yes | `config` for a configuration field, `env` for an environment variable. |
| `status` | string | yes | One of the statuses below. |
| `evidence` | mapping | yes | The one key the status requires, below. |
| `value_classes` | mapping | for `modelled` and `proven_inert` | The classes of values the evidence covers, below. Not allowed for the other statuses. |
| `estimation` | mapping | yes | The estimation rule, below. |

Statuses:

--8<-- "docs/contracts/generated/enum/fields.Status.md"

`modelled`
:   The port reads the field and it changes a decision.
`proven_inert`
:   The field changes no decision, by a passing test or a reviewed argument.
`not_modelled`
:   The port refuses the field when it is set away from its default.
`inert_unproven`
:   Believed inert without a proof. Allowed only until M2 (release plan §3.3); a point that
    sets such a field is graded G2.

### `evidence`

<!-- fields: fields.Evidence -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `traces` | list | for `modelled` | Paths, from the engine repository's root, of recorded traces whose scenarios set the field. At least one, none empty, none twice. |
| `proof` | string | for `proven_inert` | The test, or the argument, that shows the field inert. |
| `issue` | string | for `not_modelled` | The issue tracking the field, as `<owner>/<repo>#<n>`. |
| `note` | string | for `inert_unproven` | Why the field is believed inert. |

Exactly the key its status requires is set; any other is an error.

### `value_classes`

<!-- fields: fields.ValueClasses -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `type` | string | yes | One of the value types below. |
| `values` | list of strings | for a proven-inert enumerated field | Every value an enumerated field accepts. Optional for a modelled one. |
| `covered` | list of strings | for `boolean` and `enumerated` | The values the evidence sets: `"true"` and `"false"` for a boolean. At least one, none twice. |
| `thresholds` | list | for `numeric` | The values the upstream compares the field with, strictly increasing, below. |
| `covered_classes` | list of integers | for `numeric` | The numbered classes the evidence sets a value in, increasing. At least one. |

A modelled field records the classes of values its evidence covers (design VLM-3), because
evidence at one value says nothing about a value the upstream treats differently. A scoring
harness grades a run G1 only when each modelled field it sets has a value in a covered class,
and G2 when it sets one outside every covered class (design §11.3). `ValueClasses.Covers`
answers that for one value.

Value types:

--8<-- "docs/contracts/generated/enum/fields.ValueType.md"

`boolean`
:   Two classes, `true` and `false`.
`enumerated`
:   One class per value.
`numeric`
:   An ordered number. `n` thresholds make `n+1` classes, numbered from 0 for the values
    below the first threshold. A value is in class `k` when it is above `k` thresholds.

A proven-inert field records the classes its probes cover, and they must be all of them: a
probe in one class shows nothing about another. A proven-inert enumerated field lists every
value in `values`, so the check can tell that none was left out. A list the engine cannot
write in full (a free-text field) is not provably inert by this record.

#### `thresholds`

<!-- fields: fields.Threshold -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `value` | number | yes | The number the upstream compares the field with. Finite. |
| `closed` | string | yes | Which class a value equal to `value` falls in. |

Sides:

--8<-- "docs/contracts/generated/enum/fields.Side.md"

`above`
:   A value equal to the threshold is in the class above it: the upstream compares with `x >= value`.
`below`
:   A value equal to the threshold is in the class below it: the upstream compares with `x > value`.

The thresholds are generated from the upstream's comparisons of the field, so a comparison
the engine adds is a class the list must record. `closed` says which class takes the value
equal to the threshold; it is not interval notation for the class itself. Thresholds are
strictly increasing, so two comparisons at one value (`x > 8` in one place and `x >= 8` in
another) cannot both be written. For an integer field, state them at 8 and 9; for a float
field the list cannot say it, and the engine marks the field not modelled.

`ValueClasses.Covers` takes any integer or float width, a `json.Number` or a numeric string,
and refuses NaN and the infinities. A consumer in another language restates the rule, so these
vectors, which a test here checks, fix it at a threshold. With thresholds `[8 above]`: 7.999 is
class 0, 8 is class 1, 8.001 is class 1. With `[8 below]`: 7.999 and 8 are class 0, 8.001 is
class 1. With `[1 above, 256 below]`: 0 is class 0, 1 and 256 are class 1, 257 is class 2.

### `estimation`

<!-- fields: fields.Estimation -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `rule` | string | yes | One of the rules below. |
| `setting` | mapping | for `substitute` | Configuration field names and the values that stand in for this field. Not empty. |
| `reason` | string | for `substitute` | Why the setting is a fair stand-in. |

Rules:

--8<-- "docs/contracts/generated/enum/fields.Rule.md"

`leave_unset`
:   Leave the field unset, so the engine resolves the upstream's default (X19).
`substitute`
:   State `setting` in place of the field; the point names the estimate and is graded G2.
`none`
:   No estimate exists; a point that needs one is not scored (G5).

`setting` and `reason` are allowed only with `substitute`.
