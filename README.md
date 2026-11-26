# inference-sim (BLIS, the Blackbox Inference Simulator)

BLIS is a discrete-event simulator of an LLM serving stack: arrivals and workloads, a cluster of
engine instances, the embedded llm-d router (admission, flow control, routing), KV caching and
offload, prefill/decode disaggregation and autoscaling. It is CPU-only and deterministic: the
same seed gives byte-identical results. The design is at
**https://inference-sim.github.io/blis-design/**, and it is the place to read everything not on
this page.

## What this repository is, owns, and never holds

- It **is** the simulator: the clock and event loop, workload generation and trace replay, the
  cluster and its placement, the control loops that run beside the router (autoscaler, LoRA
  placement), metrics, and the `blis` command.
- It **owns** the simulation of the whole stack and the Result document of a run. It drives each
  instance's engine only through the engine interface of `substrate/contracts`, and asks the
  engine for every step.
- It **never holds** an engine's scheduling logic (`engines/`), a price (the latency kernel in
  `substrate/kernel`), a catalog fact or a fitted coefficient (`substrate/catalog`,
  `substrate/registry`), or a router algorithm (the llm-d router fork). `sim/` and `cmd/` have no
  routing, admission or flow-control logic of their own. Nothing is fetched at run time.

## How it connects

This is one Go module, `github.com/inference-sim/inference-sim`. `stack.json` records what this
release pins, and the build selects exactly those versions.

- **Contains** the substrate (`substrate/`: contracts, the cost model, the catalog and the
  coefficient registry), the vLLM engine (`engines/vllm/`) and the SGLang engine
  (`engines/sglang/`). Imports run one way: the substrate, then the engines, then `sim/` and
  `cmd/` (`internal/repocheck` holds the rule).
- **Pins** the llm-d router fork `inference-sim/llm-d-router`, which replaces
  `github.com/llm-d/llm-d-router` at a `-blis.N` tag.
- **Pinned by** blis-config-search and accuracy-dashboard, which pin the inference-sim release
  tag. No other repository pins it.
- **Implements and consumes** the contracts in `substrate/contracts`: the Scenario and Deployment
  documents it reads, the `engine.Engine` interface it drives, the router `embed` contract it
  hosts, and the Result document it writes. Non-test code imports the vLLM engine at one build
  site, `cmd/engine_build.go`, and the SGLang engine at another, `cmd/sglang_engine.go`.

The reference is printed from the data and the code, not written: `blis schema`, `blis fields`,
`blis metrics`, `blis catalog` and `blis router-configs`, with `--help` on
every command, the JSON Schemas and the Go doc comments.

## Layout

| Path | What it holds |
|---|---|
| `cmd/` | the `blis` command (`run`, `replay`, `convert`, `schema`, `fields`, `catalog`, ...) and the one site that builds each engine |
| `sim/` | the simulator library: clock, event loop, cluster (`sim/cluster`), workloads (`sim/workload`), the embedded router host (`sim/llmd`), saturation detection |
| `internal/` | repository machinery: `stack` (the `stack.json` record), `repocheck` (layering and path checks), `bytegate` (the byte-identity gate), `substratedata` |
| `substrate/` | `contracts/`, `kernel/` (the cost model), `catalog/`, `registry/`, `ledger/`, `validate/`, `tools/`; its [README](./substrate/README.md) says what each holds |
| `engines/vllm`, `engines/sglang` | the two engine models; see their READMEs |
| `examplescenarios/`, `testdata/` | ready-to-run scenarios; test inputs, the byte-gate goldens |
| `tools/`, `scripts/` | scoring and validation harnesses; CI and delivery scripts |
| `docs/` | [support-matrix.md](./docs/support-matrix.md) (generated, below) and [deferred.md](./docs/deferred.md) |

The router fork is the one external module that is substituted: `go.mod` carries a single
`replace` of `github.com/llm-d/llm-d-router` by `github.com/inference-sim/llm-d-router` at the
tag `stack.json` records. `blis version` prints every pin.

## Testing and CI

`go test ./...` runs everything that needs no Python or network; `golangci-lint run ./...`
(v2.14.0) lints. CI (`.github/workflows/ci.yml`) runs the jobs a change needs: `scripts/ci-changes.sh`
maps each changed path to path classes (`core`, `cluster`, `kernel`, `cmd`, `substrate`, `vllm`,
`sglang`, ...) and a path no class names runs everything. A change under `substrate/` runs every
class, one under `engines/vllm` or `engines/sglang` every class but the other engine's and
`substrate`, and a test file or a `testdata/` file only its own class. The substrate's catalog and
registry validators, the cost model's mutation check, and the engines' oracle jobs (each
engine's decisions compared with the upstream project's own classes, GPU mocked) are jobs of the
same workflow; each engine's README gives its commands. `docs/support-matrix.md` is generated:
`go test ./cmd -run ClusterMatrix -update-matrix` rewrites it and `docs/support-matrix.tsv` by
running every cell; never edit either by hand.

## Engines

A scenario names its engine kind: `engine: {kind: vllm}` or `engine: {kind: sglang}`, and states no
release (a `version` field is refused as unknown). `stack.json` names the engine commits this build runs,
`blis version` prints them (`--json` for a caller that parses), and every Result records them under
`engine`. Per engine kind, `blis version` reports the release whose serving behaviour the build
prices — the release a benchmarked figure is comparable with, and the one every Result's kernel
assumptions cite — beside the upstream tree its rules were ported from (provenance: it may be a commit
with no release tag) and the deployment roles the build can run it in. An SGLang
pool's `engine:` block states SGLang's own `ServerArgs` field names (`mem_fraction_static`,
`chunked_prefill_size`, ...) and a name that is not one is refused naming it; `blis fields` prints each
engine's field list. Two SGLang scenarios run in CI, `testdata/kernel/sglang/glm-5-h200-fp8-sglang-tp8.yaml`
and `deepseek-r1-h200-fp8-sglang-tp8.yaml`. A feature an engine does not model is refused, naming the
issue that holds it; [docs/support-matrix.md](./docs/support-matrix.md) lists every model, chip and
serving mode that runs or is refused, and [docs/deferred.md](./docs/deferred.md) lists what M3.2
leaves out on purpose.

One named substitution stands on every SGLang point: the host-side facts (model, hardware, parallelism,
KV store, host overheads) are read through the vLLM engine's adapter from a copy of the scenario in vLLM form,
so the Result lists `host_side_facts` and `host_overheads` in its kernel substitutions; SGLang's own
scheduler, KV pool and step pricing decide every step (`adoptSGLangTwin`, `cmd/sglang_engine.go`).

## Build, test and run

<!-- BEGIN GENERATED from stack.json: toolchain -->
Requires Go 1.27.2, the toolchain substrate/contracts/toolchain names.
<!-- END GENERATED -->

```bash
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
go build -o blis main.go
go test ./...                      # CI runs the full set; run the packages you touch
./blis run --scenarios examplescenarios \
  --scenario deepseek-v4.1-flash-h100-tp8-dspark5.yaml \
  --rate 10 --num-requests 100
```

Every `blis run` and `blis replay` needs a scenario directory (`--scenarios <dir>`, any directory
of scenario YAMLs; `examplescenarios/` ships ready-to-run ones) and a scenario (`--scenario <file>`
within it). The catalog and the coefficient registry default to the copies embedded from
`substrate/catalog` and `substrate/registry`; `--catalog`/`BLIS_CATALOG` and
`--registry`/`BLIS_REGISTRY` replace them whole (the flag wins). The Result records which data the
run read under `data`. To read a catalog of your own:

<!-- BEGIN GENERATED from stack.json: data-checkout -->
```bash
export BLIS_CATALOG=$PWD/substrate/catalog   # optional: replaces the embedded catalog (or pass --catalog)
```
<!-- END GENERATED -->

Contributing: [CONTRIBUTING.md](./CONTRIBUTING.md) holds the engineering standards and the pull
request workflow; [CLAUDE.md](./CLAUDE.md) is the operating context for agents.

## Citation

If you use BLIS in your research, please cite our paper (accepted at ACM SIGMETRICS,
to appear). Preprint: [arXiv:2610.14255](https://arxiv.org/abs/2610.14255).

```bibtex
@article{guhathakurta2026blis,
  title     = {Simulate Before You Scale: High-Fidelity LLM Inference Planning Without GPUs},
  author    = {Guhathakurta, Dipanwita and Toslali, Mert and Parthasarathy, Srinivasan and Chen, Jing and Kalantar, Michael and Masluk, Nicholas A. and Ramani, Vishakha and Tantawi, Asser and Oliveira, Fabio},
  journal   = {Proc. ACM Meas. Anal. Comput. Syst.},
  publisher = {Association for Computing Machinery},
  year      = {2026},
  note      = {To appear},
  eprint    = {2610.14255},
  archivePrefix = {arXiv},
  primaryClass  = {cs.PF}
}
```

## License

Apache License, Version 2.0. See [LICENSE](./LICENSE).
