# Installation

## Prerequisites

- **Go** ([Download Go](https://go.dev/dl/)):

<!-- BEGIN GENERATED from stack.json: toolchain -->
Requires Go 1.27.2, the toolchain blis-substrate's contracts/toolchain names at the pinned release.
<!-- END GENERATED -->

- **Git** — for cloning the repository

## Build from Source

```bash
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
go build -o blis main.go
```

## Environment Setup

BLIS reads model architecture configs from the catalog it carries and makes no HuggingFace requests, so running a simulation needs no token and no network access. Set `HF_TOKEN` only when downloading a new model's `config.json` by hand to add a catalog entry, e.g. for a gated model such as [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf):

```bash
export HF_TOKEN=your_token_here
```

Public models (e.g., Qwen3) work without a token. See [HuggingFace access tokens](https://huggingface.co/docs/hub/en/security-tokens) to create a token.

!!! note "Air-gapped / offline environments"
    Nothing to do — BLIS never reaches the network to run a simulation. Every model's architecture comes from the catalog in use (the embedded copy, or the directory `--catalog` / `BLIS_CATALOG` names), and a model with no catalog entry is refused rather than fetched.

    To simulate a model that is not yet catalogued, obtain its `config.json` on a machine with internet access, then commit it at `<catalog>/models/<model>/config.json` in a catalog directory of your own, and point `--catalog` / `BLIS_CATALOG` at that directory.

## The Model Catalog and Registry

`blis run`, `blis replay`, `blis convert preset` and `blis catalog resolve` read the catalog
and the registry embedded in the blis-substrate module this build pins. No flag and no
environment variable is needed.

`--catalog <dir>` (or `BLIS_CATALOG`) names a directory with the layout of blis-substrate's
`catalog/` that replaces the embedded catalog whole. `--registry <dir>` (or `BLIS_REGISTRY`)
does the same for the registry. The flag wins over the environment variable, and the choice is
announced on stderr. `--catalog-overlay <dir>` and `--registry-overlay <dir>` (repeatable) add
or replace entries by identity on top of the catalog or registry in use; at this pin they are
refused until the engine reads overlays. BLIS never fetches or writes a catalog file (NS-6).
The Result records the data the run read under `data.catalog` and `data.registry`.

To read a checkout of blis-substrate instead of the embedded copy, check it out at the commit
this build pins (see [Data compatibility](#catalog-compatibility) below) and point BLIS at its
directories:

<!-- BEGIN GENERATED from stack.json: data-checkout -->
```bash
git clone https://github.com/inference-sim/blis-substrate.git
git -C blis-substrate checkout 4a3becf2d09aeced794e96bb26babfaab36cc8e9
export BLIS_CATALOG=$PWD/blis-substrate/catalog   # optional: replaces the embedded catalog (or pass --catalog)
```
<!-- END GENERATED -->

## Data compatibility { #catalog-compatibility }

The catalog, the registry and the cost model are one module, blis-substrate, and a build of
BLIS selects exactly one version of it. `stack.json` at the repository root records that
version and every other BLIS module and upstream release the build runs (design SIM-3); the
table below is rendered from it (`go run ./internal/stack/cmd/versions-doc`), and CI fails if
it is stale.

<!-- BEGIN GENERATED from stack.json: versions -->
| Component | Version | Commit |
|---|---|---|
| `github.com/inference-sim/blis-substrate` | `v0.0.0-20261105193227-4a3becf2d09a` | `4a3becf2d09a` |
| `github.com/inference-sim/blis-vllm` | `v0.32.1-0.20261105200233-44fb7c506b6c` | `44fb7c506b6c` |
| `github.com/inference-sim/llm-d-router` | `v1.0.0-blis.2.0.20261105040605-fd1830412b30` | `fd1830412b30` |
| vllm (engine) | `0.32.0` | `5df63a7203cd` |
| llm-d (router) | `1.0.0` | `b9393d1a9f1d` |
| llm-d guides | | `4427f55cb626` |
| Go toolchain | `go1.27.2` | |
<!-- END GENERATED -->

Every loader parses its files strictly, so data from another substrate version can be refused
at load. The embedded catalog and registry are those of the substrate module `go.mod` selects,
and the test suite reads them, so they and the checkout above are the data the build is tested
with. A directory of your own works as long as the substrate's validators accept it.

## Scenarios

Every `blis run` and `blis replay` names a **scenario**: a
[`blis-schemas`](https://github.com/inference-sim/blis-schemas) Scenario + Deployment YAML
(two documents) that states the model, the cluster's hardware, fabric, nodes and GPUs per
node, the coefficient sets that price it, the engine version, and each pool's role,
parallelism and engine settings. `--scenarios <dir>` is any directory of scenario YAMLs and
`--scenario <file name>` picks one in it. This repository ships ready-to-run scenarios in
`examplescenarios/` (speculative ones pin their acceptance, so no `--speculative-*` flag is
needed): `deepseek-v4.1-flash-h100-tp8-dspark5.yaml` (the quickstart), an MoE
`gpt-oss-120b-h100-fp4-vllm-tp2.yaml`, a `deepseek-v4-pro-b300-fp4-vllm-tp4-mtp2.yaml`, a P/D
`glm-5-h200-3p1d-ib.yaml`, and a multi-node `kimi-k3-h200-tp8dp4ep32-balanced.yaml`. See
[`examplescenarios/README.md`](https://github.com/inference-sim/inference-sim/tree/main/examplescenarios)
for details.

## Verify the Build

```bash
./blis run --scenarios examplescenarios \
  --scenario deepseek-v4.1-flash-h100-tp8-dspark5.yaml --num-requests 10
```

`blis` carries the repository's `defaults.yaml` (the LoRA cost constants) compiled in, so it runs
from any directory. A `defaults.yaml` in the working directory, or `--defaults-filepath <file>`,
takes precedence over the compiled-in copy.

You should see a Result document on stdout (`"kind": "Result"`) whose `metrics` include `ttft_mean_ms`, `e2e_mean_ms` and `request_throughput`. This confirms BLIS is working correctly.

## Optional: Local Documentation

To preview the documentation site locally:

```bash
pip install mkdocs-material==9.7.3
mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000).

## Optional: Linter

For contributors, install the linter used in CI:

```bash
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.14.0
golangci-lint run ./...
```

## What's Next

- **[Quick Start](quickstart.md)** — Run your first simulation and understand the output
- **[Tutorial: Capacity Planning](tutorial.md)** — Complete walkthrough of a capacity planning exercise
