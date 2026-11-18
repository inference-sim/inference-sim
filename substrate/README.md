# blis-substrate

The part of [BLIS](https://github.com/inference-sim/inference-sim) that is true whatever engine
runs. It owns the contracts every BLIS repository implements or calls, the published facts BLIS
reads (the catalog), the fitted coefficients (the registry) and the cost model that prices each
engine step (the kernel). It never holds engine behaviour, simulator event handling or a router's
scoring: those live in the engine repositories, inference-sim and the router fork.

The design is the reference for what each part must do and why:
<https://inference-sim.github.io/blis-design/>. This repository documents itself here and in the
Go doc comments, the CLI output and the JSON Schemas under `contracts/schema`.

## What is in it

| Directory | What it holds |
|---|---|
| `contracts/` | the interfaces (`engine.Engine`, the kernel interface, the router contract), the document types of `contracts/spec`, their validators, the JSON Schemas, the invariant registry and the conformance suites |
| `catalog/` | declared facts: chips, networks, storage devices, workloads, the router table, and models, each with its vendor `config.json` and the graph derived from it |
| `registry/` | coefficient sets, each coefficient with its method, sources and data units, and the scripts that fit them |
| `kernel/` | the cost model, in Go |
| `ledger/` | the published role lists and the checks over them |
| `load/` | the loader that reads the catalog and the registry from a path or from the embedded copy |
| `validate/` | the catalog, registry and measurement validators and their commands |
| `tools/` | repository tools (the pull-request checks, the invariant renderer, the schema generator) |

## Connections

- **Pins nothing from BLIS.** It depends only on the Go standard library, the YAML parser and its own packages.
- **Is pinned by** inference-sim (through the cost model and the loaders), blis-vllm and blis-sglang (they import only
  packages under `contracts/`), the llm-d router fork (the router contract), blis-config-search and the
  accuracy dashboard (the document types and validators).
- **Implements and consumes:** it defines the contracts above; the engine repositories implement
  `engine.Engine` and are tested by the conformance suites here; inference-sim supplies the
  `Environment` an engine is built with.
- A change to a coefficient or to pricing needs no engine or router release.

## Build, test and run

All commands use the Go toolchain pinned in `contracts/toolchain`.

```sh
go test ./...                                  # every Go package
go run ./validate/cmd/validate-catalog catalog # every catalog entry
go run ./validate/cmd/validate-registry registry
(cd catalog && python3 scripts/derive_graph.py --check)   # graphs match their vendor configs
go run ./tools/schemagen                       # regenerate the JSON Schemas
go run ./tools/invariants render               # regenerate the invariant block of AGENTS.md
```

Pull requests run `go run ./tools/prcheck` (description, commits, workflows, versions). `AGENTS.md`
lists the commands to run for each directory you touch.

## History

This repository was created from four repositories, at these releases:

| Repository | Release | Commit | Moved into |
|---|---|---|---|
| [blis-schemas](https://github.com/inference-sim/blis-schemas) | v0.10.0 | `a649e0f` | `contracts/`, `validate/` |
| [blis-catalog](https://github.com/inference-sim/blis-catalog) | 0.5.0 | `b5550f7` | `catalog/` |
| [blis-registry](https://github.com/inference-sim/blis-registry) | v0.3.0 | `96bead6` | `registry/` |
| [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel) | v0.5.0 | `560bda6` | `kernel/` |

Their history stays in those repositories, which are archived.
