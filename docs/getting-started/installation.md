# Installation

## Prerequisites

- **Go 1.21+** — [Download Go](https://go.dev/dl/)
- **Git** — for cloning the repository

## Build from Source

```bash
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
go build -o blis main.go
```

## Environment Setup

BLIS reads model architecture configs from a local catalog checkout and makes no HuggingFace requests, so running a simulation needs no token and no network access. Set `HF_TOKEN` only when downloading a new model's `config.json` by hand to add a catalog entry, e.g. for a gated model such as [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf):

```bash
export HF_TOKEN=your_token_here
```

Public models (e.g., Qwen3) work without a token. See [HuggingFace access tokens](https://huggingface.co/docs/hub/en/security-tokens) to create a token.

!!! note "Air-gapped / offline environments"
    Nothing to do — BLIS never reaches the network to run a simulation. Every model's `config.json` comes from your local catalog checkout, and a model with no catalog entry is refused rather than fetched.

    To simulate a model that is not yet catalogued, obtain its `config.json` on a machine with internet access, then commit it at `<catalog>/models/<model>/config.json` — where `<catalog>` is whatever directory you point `--catalog` / `BLIS_CATALOG` at.

## Locate the Model Catalog

`blis run` and `blis replay` must be told where the model catalog is — there is no default
and no search path, so a run with neither `--catalog` nor `BLIS_CATALOG` is refused naming
both forms. Clone the authoritative [`blis-catalog`](https://github.com/inference-sim/blis-catalog)
repository and point BLIS at the clone root:

```bash
git clone https://github.com/inference-sim/blis-catalog.git
export BLIS_CATALOG=$PWD/blis-catalog   # or pass --catalog on every command
```

## Verify the Build

```bash
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1 --num-requests 10 --catalog blis-catalog
```

You should see JSON output on stdout containing fields like `ttft_mean_ms`, `e2e_mean_ms`, and `responses_per_sec`. This confirms BLIS is working correctly.

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
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.9.0
golangci-lint run ./...
```

## What's Next

- **[Quick Start](quickstart.md)** — Run your first simulation and understand the output
- **[Tutorial: Capacity Planning](tutorial.md)** — Complete walkthrough of a capacity planning exercise
