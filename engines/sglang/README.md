# SGLang engine

A Go model of the [SGLang](https://github.com/sgl-project/sglang) inference engine's scheduling,
KV-cache, hierarchical-cache, disaggregation and metrics behaviour, for the BLIS simulator. It
lives in the `engines/sglang` directory of the one `github.com/inference-sim/inference-sim`
module and implements the engine-neutral contract in `substrate/contracts/` (`engine.Engine`);
the simulator builds it at one site, `cmd/sglang_engine.go`. One build models one SGLang
release, named in `provenance.go` (now 0.5.23, which `stack.json` also pins); its decisions are
checked against that release's own Python classes, run on CPU with only the GPU, the clock,
transports and the platform replaced (`oracle/`).

## Layout

| Path | What |
|---|---|
| `core/sched`, `core/cache`, `core/hicache`, `core/metrics`, `core/sglang` | the scheduler, the radix and pool caches, the hierarchical cache, the metrics, and the engine that ties them together |
| `config/` | `ServerArgs` resolution and validation, with the per-release refusal rules (`rules/`) |
| `lower/` | lowering a model to kernel implementations |
| `sglangengine/` | how a simulator gets the engine: the factory registration |
| `fields.json`, `fields/` | the field list, its generator and hand-written statuses (`fields/README.md`) |
| `surface.json`, `engines/sglang/tools/surface/` | every feature the release can select, with its status; generated, never edited |
| `oracle/` | the Python side: SGLang's classes with the GPU mocked, goldens and runners (`oracle/README.md`) |

## Build, test and run

Run from `engines/sglang/`:

```sh
go build ./... && go vet ./...
go test ./...
golangci-lint run ./...         # v2.14.0
```

The oracle checks run as the jobs of `.github/workflows/oracle-sglang.yml`, which
`.github/workflows/ci.yml` calls when a change touches `engines/sglang` or `substrate`; the header
of `scripts/ci-changes.sh` says how a change selects jobs. `AGENTS.md` gives the rules for a change.
