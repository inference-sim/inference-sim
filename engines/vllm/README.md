# blis-vllm

A Go model of the [vLLM](https://github.com/vllm-project/vllm) inference engine's scheduling,
KV-cache, connector and metrics behaviour, for a discrete-event simulator. It implements the
engine-neutral contract in [blis-substrate](https://github.com/inference-sim/blis-substrate)
`contracts/` (`engine.Engine`), so a simulator such as
[inference-sim](https://github.com/inference-sim/inference-sim) builds it once and then holds only
that interface. The design, which states every rule below and why, is at
<https://inference-sim.github.io/blis-design/>.

## What it owns, and what it never holds

It owns everything about vLLM's behaviour: defaults, how a step is formed, which KV blocks move,
which kernel runs, and the per-release rules that validate a vLLM engine block. It never holds
the cost model, a hardware fact, a coefficient or a router; those are blis-substrate's and
inference-sim's.

- **One release per build.** A build models one vLLM release (`provenance.go`; now v0.33.0,
  `4ba787ec9959e566d3b587a796dff07673324da5`). Every statement about vLLM cites
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
- **Upgrade reports.** `upgrades/<from>-to-<commit8>.json` is the output of
  `engines/upgrade/upgrade_report.py` for one move of the modelled commit (design 12.4): the surface,
  field, source, model and coefficient changes. The first, `v0.33.0-to-0fda86a35419.json`, was run
  with the lists committed at v0.33.0 as the old side, the lists now committed as the new side, and
  the catalog and coefficient registry at the commit that added the report.

## Connections

- **Imports** only blis-substrate's `contracts/` packages and, until the engine boundary moves,
  its cost model. The cost model imports nothing from this module.
- **Is pinned by** inference-sim, which builds the engine at one site and holds only
  `engine.Engine`; the release that runs a build pins it in its `go.mod`. Builds are tagged
  `v0.33.0-blis.N`.
- **Is read by** accuracy-dashboard and blis-config-search through `inference-sim fields`
  (the field list) and the surface list.

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

```sh
go build ./... && go vet ./...
go test ./...                   # every package; the oracle jobs below need Python
golangci-lint run ./...         # v2.14.0
```

The oracle tiers run from a Python with `oracle/requirements.lock` installed
(`pip install --require-hashes --no-deps -r oracle/requirements-hashed.txt`, Linux x86_64) as the
`oracle-scenarios`, `oracle-inertness` and `oracle-fuzz` jobs of `.github/workflows/ci.yml`.
That file's header says how a change selects the jobs it runs and how the slow suites are shared
out over runners; `substrate/tools/changedjobs/workflow_test.go` checks the selection. `AGENTS.md` holds
the rules an AI-assisted change follows.
