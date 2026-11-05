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
