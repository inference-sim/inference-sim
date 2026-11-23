# substrate

The part of [BLIS](https://github.com/inference-sim/inference-sim) that is true whatever engine
runs, kept as a directory of the inference-sim module (`github.com/inference-sim/inference-sim/substrate/...`).
It owns the contracts the simulator, the engines and the router fork implement or call, the
published facts BLIS reads (the catalog), the fitted coefficients (the registry) and the cost
model that prices each engine step (the kernel). It never holds engine behaviour (`engines/`),
simulator event handling (`sim/`, `cmd/`) or a router's scoring (the llm-d router fork).

The design is the reference for what each part must do and why:
<https://inference-sim.github.io/blis-design/>. This directory documents itself here and in the
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

- **Imports nothing from the rest of the module.** It depends only on the Go standard library, the YAML parser and its own packages (`internal/repocheck` enforces the direction: substrate, then the engines, then `sim/` and `cmd/`).
- **Is imported by** the engines, which use only packages under `contracts/` and, for the cost model, `kernel/`; by `sim/` and `cmd/` (the loaders and the cost model); by the llm-d router fork (the router contract); and by the dashboard and config-search repositories (the document types and validators), which pin the inference-sim release.
- **Embedded data.** `data.go` embeds `catalog/` and `registry/` into the binary, so a run reads them without a flag; `--catalog` and `--registry` replace them whole.
- A change to a coefficient or to pricing is a change to files here, in the same pull request as any test or golden it moves.

## Build, test and run

The commands below run from `substrate/`, with the Go toolchain named in `contracts/toolchain`.

```sh
go test ./...                                  # every Go package of the substrate
go run ./validate/cmd/validate-catalog catalog # every catalog entry
go run ./validate/cmd/validate-registry registry
(cd catalog && python3 scripts/derive_graph.py --check)   # graphs match their vendor configs
go run ./tools/schemagen                       # regenerate the JSON Schemas
go run ./tools/invariants render               # regenerate the invariant block of AGENTS.md
```

CI runs these as the `substrate-*` jobs of `.github/workflows/ci.yml`, together with the cost
model's mutation check, and `prcheck` (description, commits, workflows, versions, documentation
references) on every pull request. `AGENTS.md` lists the commands to run for each directory you
touch.
