# Contributing

<p class="lede">A change to the contracts is one of three kinds. Each has its own place in
the tree, and two of them change what a document may contain. Every change lands in the
same pull request as the data and code that depend on it.</p>

| Change | Where it goes | Changes what a document may contain |
|---|---|---|
| A new vocabulary term | `contracts/vocab/`, or the constants in a `contracts/spec/` package, with a test asserting the new member | Yes, by one value |
| A new key or primitive | the type in `contracts/spec/`, its validator, and its tests | Yes |
| A new interface method or type in `engine`, `kernel` or `embed` | a separate interface for an optional ability, so existing implementations still compile | No |

The packages under `contracts/` change only compatibly between releases (SUB-1): an
interface never gains a method, optional abilities are separate interfaces, and a new field
keeps its zero-value meaning. An incompatible change is made as one review set that releases
every implementer against it together.

An engine release's rules are not here. Each engine model owns the rules packs for the
release it models; for vLLM they are in blis-vllm. The contracts define only the interface,
`rules.EngineRules`, that a caller passes to a validator.

A change that also touches another BLIS repository follows
[Changes across repositories](../../contributing/changes.md).

## Layout

| Path | Contents |
|---|---|
| `contracts/*.go` | Package `contracts`: loaders, `Validate`, `ValidateAgainstCatalog`, `LoadBundle` |
| `contracts/vocab/` | Closed vocabularies: units, methods, scope keys, provenance, sources |
| `contracts/spec/model/` | A model as a graph of cost primitives, and its identity manifest |
| `contracts/spec/hardware/` | Chip, fabric and storage class |
| `contracts/spec/coefficient/` | Coefficient sets |
| `contracts/spec/acceptance/`, `contracts/spec/routing/` | Acceptance and routing profile sets |
| `contracts/spec/scenario/` | The fixed problem: model, cluster, traffic, references |
| `contracts/spec/deployment/` | The choices: pools, parallelism, engine settings, offload, PD transfer |
| `contracts/spec/workload/` | A workload shape, a replay workload, and a scenario's workload binding |
| `contracts/spec/evaluation/`, `contracts/spec/measurement/` | An evaluation run, and a measured run with its licence registry |
| `contracts/spec/result/` | The Result document |
| `contracts/spec/connector/`, `contracts/spec/router/` | The connector registry, and the router's configuration types |
| `contracts/kernel/` | The interface a cost model implements |
| `contracts/engine/` | The interface an engine model implements |
| `contracts/embed/` | The contract between a simulator and the router it hosts |
| `contracts/rules/` | The interface an engine release's rules implement |
| `contracts/check/` | The problem list and strict decoding every validator shares |
| `contracts/conformance/` | The suites implementers run: `enginetest`, `kerneltest`, `embedtest` |
| `contracts/testdata/` | Contract fixtures, such as `connectors/` |
| `validate/` | The catalog, registry and measurement validators, and their commands under `validate/cmd/` |
| `tools/docgen/` | The generator for `docs/contracts/generated/` |

## Tests

```sh
go test ./contracts/... ./validate/... ./tools/...
```

This runs every type, validator and conformance test; the example tests the documentation
quotes; and the documentation checks described in
[Documentation and releases](../../contributing/documentation.md#what-keeps-the-pages-correct).
CI runs the same set, with the race detector, in the `contracts` job of
`.github/workflows/ci.yml`.

Every validator has a negative test for each way a document can be wrong. A validator that
cannot fail proves nothing.

## When a key changes

Adding a key to a document type takes, in one pull request (SUB-2):

1. the field, with a YAML key that names its unit if it has one;
2. its check in the type's `Validate`, and a negative test for each way it can be wrong;
3. a row in the key table on its reference page; `go test` fails until the row exists;
4. `go run ./tools/docgen` if the change adds a vocabulary member;
5. the catalog or registry entries that use it, and the cost model change that reads it, if
   any.

The tests and CI run the validators over the repository's own `catalog/` and `registry/`, so
a new required key that a committed file lacks fails the same pull request:

```sh
go run ./validate/cmd/validate-catalog catalog
go run ./validate/cmd/validate-registry registry
```
