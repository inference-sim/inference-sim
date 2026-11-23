# vLLM engine

A Go model of the [vLLM](https://github.com/vllm-project/vllm) inference engine's scheduling,
KV-cache, connector and metrics behaviour, for a discrete-event simulator. It lives in the
`engines/vllm` directory of the one `github.com/inference-sim/inference-sim` module and implements
the engine-neutral contract in `substrate/contracts/` (`engine.Engine`), so the simulator builds it
once (`cmd/engine_build.go`) and then holds only that interface. The design, which states every
rule below and why, is at <https://inference-sim.github.io/blis-design/>.

## What it owns, and what it never holds

It owns everything about vLLM's behaviour: defaults, how a step is formed, which KV blocks move,
which kernel runs, and the per-release rules that validate a vLLM engine block. It never holds
the cost model, a hardware fact, a coefficient or a router; those are the substrate's
(`substrate/`) and the simulator's (`sim/`, `cmd/`).

- **One release per build.** A build models one vLLM commit (`provenance.go`, `engines/vllm/internal/upstream/upstream.go`; now
  `0.34.0+0fda86a35419`, the commit `0fda86a35419c9c70dbfada224c838c10fbb62ad`, which `stack.json`
  also pins). The committed oracle recordings were taken at v0.33.0 and are evidence of that
  release until a maintainer re-records them (`oracle/README.md`). Every statement about vLLM cites
  `vllm/<path>:<lines>@<commit8>` at that release. A ported file records its source and keeps
  vLLM's licence.
- **Time is a value.** Nothing reads a clock or starts a goroutine. The simulator passes the
  simulated instant into every call.
- **Checked against vLLM itself.** CI feeds the same inputs to vLLM's own classes (Python, CPU
  only, the GPU mocked) and to this module and compares every step (`oracle/`). The oracle may
  patch only the GPU, the clock, transports and external services, never a decision; each patch
  is listed in `oracle/mocks.json`. A golden trace's decisions change only with a
  `golden-reviewed: <trace>: <reason>` trailer.
- **Every configuration field has a status.** `fields.json` lists each field and environment
  variable the release reads, with its status (`modelled` with evidence, `inert`, or
  `not_modelled` citing an open issue), written from `fields/status.json` and vLLM's own source.
  `surface.json` lists every connector, tier, speculative method, policy and KV-cache spec type
  with its status, and also the tuned-MoE selection rule and device files, the engine defaults,
  the KV-event emitters and report modes, the async-scheduling and step-input mechanisms, the
  step order and the platform preferences. An item's `value` is the comparable fact read from the
  source; when it differs between two commits the upgrade report lists the item under `changed`
  and the earlier status does not carry over. A backport may change only model code
  (`upstream.BackportScope`: model directories, the architecture registry, config classes,
  config parsers and the model-architecture config convertor). `kernel_impls.json` lists the kernel implementations the release can run, with
  the code hash the coefficient registry keys its entries by. All three are generated and CI
  checks they are current.
- **Upgrade reports.** `engines/upgrade/upgrade_report.py` reports, for one move of the modelled
  commit (design 12.4), the surface, field, source, model and coefficient changes; its docstring
  gives the command.

## Connections

- **Imports** only `substrate/` (the `contracts/` packages and the cost model). The substrate
  imports nothing from this module, and no engine imports another (`internal/repocheck`).
- **Is built by** the simulator at one site, `cmd/engine_build.go`, which holds only `engine.Engine`.
  The module has one version for everything in it, so there is no separate engine tag.
- **Is read by** the field and surface commands (`blis fields`, `blis schema`) and by the
  scenario validator.

## Layout

| Path | What |
|---|---|
| `core/` | the engine, diff-tested against vLLM's own classes, and the configuration, cost-model glue and run types inference-sim builds it from (`core/build`, `core/kernelmodel`, `core/runconfig`) |
| `vllmengine/` | how a simulator gets the engine: it registers the factory, prices steps with the cost model and sizes the KV cache |
| `core/rulepack/`, `core/engineconfig/` | the per-release validation rules and the engine block's configuration |
| `fields/`, `fields.json` | the field list, its generator and hand-written statuses |
| `kernels/`, `kernel_impls.json`, `kernelimpls.go` | the kernel-implementation table, its generator, and the lookup the lowering uses |
| `oracle/` | the Python side: vLLM's classes with the GPU mocked, goldens, fuzz corpora, metric manifest (`oracle/metrics/`); `oracle/README.md` gives each runner's command |
| `kerneltests/` | the cost model's tests that open a kernel from a vLLM engine block |
| `engines/vllm/docs/feature-matrix.md` | every feature the sources use, its vLLM citation, owner, oracle tier and status |

## Build, test and run

Run from `engines/vllm/` (the checks are the `vllm-checks`, `vllm-race` and `oracle-vllm` jobs of
`.github/workflows/ci.yml` and `oracle-vllm.yml`):

```sh
go build ./... && go vet ./...
go test ./...                   # every package; the oracle jobs below need Python
golangci-lint run ./...         # v2.14.0
```

The generated lists are regenerated from a checkout of vLLM at the pinned commit (`vllm-source`
below, which CI fetches), and CI compares each with the committed file (`--check`); the docstring
of each generator gives its full usage:

```sh
python3 -I fields/gen_fields.py vllm-source fields/status.json > fields.json
python3 -I fields/gen_classes.py vllm-source fields.json > fields/value_classes.json
python3 -I kernels/regen.py            # kernel_impls.json; needs the network
go run ./cmd/metricsmanifest -vllm vllm-source
python3 -I tools/surface/gen_surface.py vllm-source tools/surface/status.json > surface.json
```

The oracle tiers run from a Python with `oracle/requirements.lock` installed
(`pip install --require-hashes --no-deps -r oracle/requirements-hashed.txt`, Linux x86_64) as the
jobs of `.github/workflows/oracle-vllm.yml`. `.github/workflows/ci.yml` calls it when a change
touches `engines/vllm` or `substrate`; the header of `scripts/ci-changes.sh` says how a change
selects jobs. `AGENTS.md` holds the rules an AI-assisted change follows.
