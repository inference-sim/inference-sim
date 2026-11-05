# Contributing

<p class="lede">Most changes to blis-schemas are one of three kinds. Each kind has its own
place in the tree, and only one of them changes what a document may contain.</p>

| Change | Where it goes | Changes what a document may contain |
|---|---|---|
| A new engine release | a new package under `rules/`, registered in `register.go`. See [A new engine release](engine-release.md). | No. It changes which values the new release accepts. |
| A new vocabulary term | `vocab/`, or the constants in a `spec/` package, with a test asserting the new member | Yes, by one value |
| A new key or primitive | the type in `spec/`, its validator, and its tests | Yes |

Every change reaches consumers through a release, because they pin versions. Publishing a
release is described in [Documentation and releases](documentation.md#making-a-release).

A change that also touches the catalog, the registry, the kernel or the simulator follows
[Changes across the repositories](cross-repo-changes.md), which sets the order of pushes and
pin bumps, and the evidence a pull request carries.

Once a rules pack is released, what it accepts and refuses changes only to correct it, so
that it describes its engine release accurately. Any other change would alter, after the
fact, what a scenario pinned to that engine version means. A release that behaves
differently gets its own pack.

## Repository layout

| Path | Contents |
|---|---|
| `vocab/` | Closed vocabularies: units, methods, scope keys, provenance, sources |
| `spec/model/` | A model as a graph of cost primitives, and its identity manifest |
| `spec/hardware/` | Chip, fabric and storage class |
| `spec/coefficient/` | Coefficient sets |
| `spec/scenario/` | The fixed problem: model, cluster, traffic, references |
| `spec/deployment/` | The choices: pools, parallelism, engine settings, offload, PD transfer |
| `spec/workload/` | A workload shape, and a scenario's workload binding |
| `spec/evaluation/` | A measured run |
| `kernel/` | The interface a cost model implements |
| `embed/` | The contract between a simulator and the router it hosts; `embed/embedtest` checks an implementation |
| `rules/` | The rules-pack mechanism, with one package per engine release beneath it |
| `internal/validate/` | The problem list every validator shares |
| `internal/registry/` | Finding coefficient-set files in a registry checkout |
| `internal/docgen/` | The generator for `docs/generated/` |
| `cmd/` | `validate-catalog` and `validate-registry` |
| `testdata/` | A pinned copy of the catalog, used as test input |
| `*.go` at the root | Loaders, `Validate`, `ValidateAgainstCatalog`, and rules-pack registration |

## Tests

```sh
go test ./...
```

This runs every schema, validator and rule test; the example tests the documentation
quotes; and four documentation checks, described in
[Documentation and releases](documentation.md#what-keeps-the-pages-correct).

Every validator has a negative test for each way a document can be wrong. A validator that
cannot fail proves nothing.

A rules pack's constants are transcribed by hand from the engine's source. A second command
checks most of them against a checkout of the release:

```sh
VLLM_SOURCE=/path/to/vllm go test -run MatchSource ./rules/...
```

Without `VLLM_SOURCE` those tests skip and say which constants went unchecked. CI runs them
against the tagged release in a separate job.

## When a key changes

Adding a key to a document type takes:

1. the field, with a YAML key that names its unit if it has one;
2. its check in the type's `Validate`, and a negative test for each way it can be wrong;
3. a row in the key table on its reference page; `go test` fails until the row exists;
4. a release.

Before changing a type the catalog or registry already uses, run `validate-catalog` and
`validate-registry` against current checkouts of both. A new required key invalidates every
committed file that lacks it.
