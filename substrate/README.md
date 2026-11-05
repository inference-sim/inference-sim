# blis-substrate

The part of [BLIS](https://github.com/inference-sim/inference-sim) that is true whatever engine
runs: the contracts every BLIS repository implements or calls, the published facts BLIS reads,
the fitted coefficients, and the cost model that prices each engine step.

| Directory | What it holds | Who imports it |
|---|---|---|
| `contracts/` | interfaces (`engine.Engine`, the kernel interface, the router contract), the document types of `contracts/spec`, their validators, and the conformance suites | every Go repository in BLIS |
| `catalog/` | declared facts: chips, networks, storage devices, workloads, and models, each with its vendor `config.json` and the graph derived from it | inference-sim, through the cost model |
| `registry/` | coefficient sets, each coefficient with its method, sources and data units, and the scripts that fit them | inference-sim, through the cost model |
| `kernel/` | the cost model, in Go | inference-sim only |
| `validate/` | the catalog, registry and measurement validators and their commands | CI, and Python tools through the commands |
| `tools/` | repository tools, such as the documentation generator | this repository |
| `docs/` | one documentation site, with a section for each directory above | |

The engine repositories (blis-vllm, blis-sglang) and the llm-d router fork import only packages
under `contracts/`. A change to a coefficient or to pricing therefore needs no engine or router
release.

## Build and test

```sh
go test ./...                                  # every Go package
go run ./validate/cmd/validate-catalog catalog # every catalog entry
go run ./validate/cmd/validate-registry registry
(cd catalog && python3 scripts/derive_graph.py --check)
```

The docs site builds with `pip install -r docs/requirements.txt && DOCS_RELEASE=main mkdocs build --strict`.

## History

This repository was created from four repositories, at these releases:

| Repository | Release | Commit | Moved into |
|---|---|---|---|
| [blis-schemas](https://github.com/inference-sim/blis-schemas) | v0.10.0 | `a649e0f` | `contracts/`, `validate/`, `tools/docgen` |
| [blis-catalog](https://github.com/inference-sim/blis-catalog) | 0.5.0 | `b5550f7` | `catalog/` |
| [blis-registry](https://github.com/inference-sim/blis-registry) | v0.3.0 | `96bead6` | `registry/` |
| [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel) | v0.5.0 | `560bda6` | `kernel/` |

Their history stays in those repositories, which are archived. To trace a file to its history,
take the source repository and release from this table and look the file up there.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
