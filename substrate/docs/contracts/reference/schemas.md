# JSON Schemas

<p class="lede">Each substrate release publishes the JSON Schema of every document that
crosses a repository boundary (SUB-19), generated from the Go types.</p>

| Document | Schema | Go type |
|---|---|---|
| `scenario` | `contracts/schema/scenario.schema.json` | [`scenario.Scenario`](scenario.md) |
| `deployment` | `contracts/schema/deployment.schema.json` | [`deployment.Deployment`](deployment.md) |
| `result` | `contracts/schema/result.schema.json` | [`result.Result`](result.md) |
| `point-record` | `contracts/schema/point-record.schema.json` | [`pointrecord.Record`](point-record.md) |
| `workload` | `contracts/schema/workload.schema.json` | [`workload.Spec`](workload.md#generator-specification) |
| `trace` | `contracts/schema/trace.schema.json` | [`trace.Header`](trace.md), with the data row as `$defs.trace.Record` |

`go run ./tools/schemagen` writes them; `TestPublishedSchemasMatchTheTypes` fails when a
committed schema differs from the types. A key is required in a schema exactly when its
reference table marks it required, every object refuses keys its type does not declare,
and every closed vocabulary is an `enum` read from the Go constants. The package
`contracts/schema` embeds the files, so `blis schema <document>` prints the schema of the
substrate release the build pins.

A schema states structure: keys, types and vocabularies. The Go validators state the
rest, such as that the fates sum to `generated`. `blis schema validate <document> <file>`
runs them, and an application validates every document it reads or writes with it, not with
the schema alone.
