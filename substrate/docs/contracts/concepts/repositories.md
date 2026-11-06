# The BLIS repositories

<p class="lede">BLIS is seven repositories. blis-substrate holds what is true whatever
engine runs: the contracts, the catalog, the registry and the cost model. The contracts,
the packages under <code>contracts/</code>, hold no data and run no simulation. They define
the documents the repositories exchange, the checks those documents must pass, and the
interfaces between the Go repositories.</p>

--8<-- "docs/contracts/figures/repositories.html"

| Repository or directory | Owns | What it takes from the contracts |
|---|---|---|
| blis-substrate `catalog/` | Declared facts: model configurations and graphs, chips, fabrics, storage classes, workload shapes, measured runs | CI runs `validate-catalog` over every committed file |
| blis-substrate `registry/` | Coefficient sets: measured, cited or assumed values, each with its method, sources and scope; acceptance and routing profiles | CI runs `validate-registry` over every committed set |
| blis-substrate `kernel/` | The cost model: memory and step time for one pool of a deployment | Imports the document types and loaders, implements `kernel.Kernel`, and validates its inputs before it constructs a kernel |
| blis-substrate `validate/` | The catalog, registry and measurement validators and their commands | Runs the validators of `contracts/spec` over the data |
| [blis-vllm](https://github.com/inference-sim/blis-vllm) | vLLM's engine behaviour: its scheduler, KV cache and connectors as an `engine.Engine`, its rules packs, and its connector registry | Implements `engine.Engine` and `rules.EngineRules`, and checks its engine with `conformance/enginetest` |
| [blis-sglang](https://github.com/inference-sim/blis-sglang) | The same for SGLang | The same |
| [llm-d router fork](https://github.com/inference-sim/llm-d-router) | llm-d's own router code, hosted in simulated time | Implements the `embed` contract (`embed.Constructor`) and runs `conformance/embedtest` |
| [inference-sim](https://github.com/inference-sim/inference-sim) | The discrete-event simulator: the event loop, the cluster, workload and replay, engine selection | Imports the whole substrate module: the contracts, the cost model, the data loader and the embedded data |
| [blis-config-search](https://github.com/inference-sim/blis-config-search) | Experiments that search configurations, and their reports | Validates the documents it reads and writes with `inference-sim schema validate` |
| [accuracy-dashboard](https://github.com/inference-sim/accuracy-dashboard) | Accuracy and coverage against measured benchmarks, and the scoring harness | The same |
| blis-substrate `contracts/` | The shapes of all of the above, and the checks on them | — |

## Shapes apart from contents

The catalog and the registry contain no Go. They are YAML and JSON, edited by people who
read datasheets and fit curves. The cost model and the simulator are Go, edited by people
who write cost models and schedulers. If the type for a chip lived in the cost model, the
catalog could not be checked without it, and every engine that reads a deployment would
import pricing code.

The shapes therefore live in `contracts/`. A catalog contributor checks a new chip file
against these types without reading the cost model. The cost model reads the same file
through the same type, so the two cannot disagree about what `MemoryGiB` means. All of it
is one repository, so a new field, its validator, the catalog entries that use it and the
cost model change that reads it land in one pull request, are checked together, and ship in
one release.

## What each part does with the contracts

**The catalog** is checked by `validate-catalog` in CI, at the same commit. The command
loads every model graph, model identity, vendor `config.json`, chip, fabric, storage class,
workload shape and measured run, and fails the pull request if any is malformed. The
catalog keeps its own derivation checks: that each committed `graph.yaml` re-derives from
its vendor `config.json`, and the deriver's test suite. The contracts do not cover
derivation.

**The registry** is checked by `validate-registry` in CI, at the same commit. Each set goes
through the loader and validator a Go consumer uses, so a set the cost model could not load
fails on the pull request that introduces it. The registry's own tests check something
else: that each committed value re-derives from public data.

**The cost model** loads documents with these loaders, implements the `Kernel` interface
defined here, and runs `contracts.Validate` over its inputs before it constructs a kernel.
A field-layer error stops construction; rule-layer problems do not.

**blis-vllm** is vLLM's engine model. It implements the `engine.Engine` interface defined
here, so a simulator steps it without knowing which engine it is, and checks itself with
`conformance/enginetest`, the suite every engine runs. It also owns the vLLM rules packs,
which a caller such as the cost model reads through `rules.EngineRules`, and the connector
registry, whose shape is `spec/connector`. blis-sglang does the same for SGLang.

**inference-sim** takes its step times from the cost model through `kernel.Kernel`, and its
documents through these loaders. It computes no cost of its own.

**The llm-d router fork** implements the [router contract](../reference/embed.md), package
`embed`: inference-sim hosts the router on its discrete-event runtime (`embed.Runtime`) and
calls it only through `embed.Router`. The fork checks itself against
`conformance/embedtest`.

**blis-config-search and accuracy-dashboard** use no Go from the substrate. They read the
JSON Schema each substrate release publishes for the scenario, the deployment and the
Result, and validate every document with `inference-sim schema validate`, which runs these
validators.

## Dependencies run one way

The packages under `contracts/` import only the Go standard library, `gopkg.in/yaml.v3` and
each other (SUB-1). The engine repositories and the router fork import only these packages,
so a coefficient or pricing change needs no engine or router release. inference-sim imports
the whole module. Nothing in blis-substrate imports a repository above it.

## Data the tests read

The tests under `contracts/` load and validate the repository's own `catalog/`, read as
`../catalog`, so a change to a type is tested against real catalog files rather than ones
written to fit. A type change that the committed data does not satisfy fails in the same
pull request.

## Compatible changes

Every consumer pins a release of blis-substrate, `vX.Y.Z`. Between releases the packages
under `contracts/` change only compatibly: an interface never gains a method, optional
abilities are separate interfaces, and a new field keeps its zero-value meaning (SUB-1). An
incompatible change is made as one review set that releases every implementer against it
together.
