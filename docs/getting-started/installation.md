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
repository **at the pinned release tag** (see [Catalog compatibility](#catalog-compatibility)
below) and point BLIS at the clone root:

```bash
git clone --branch 0.1.1 --depth 1 https://github.com/inference-sim/blis-catalog.git
export BLIS_CATALOG=$PWD/blis-catalog   # or pass --catalog on every command
```

## Catalog compatibility

`blis-catalog` versions independently of BLIS and keeps releasing, so the documented clone
names an explicit tag rather than whatever `main` happens to be. This is the one place the
compatible version is stated; every other page's clone command pins the same tag.

<!-- Canonical declaration. cmd.TestDocExamplesPinTheCatalogVersion reads the tag from the
     line below and requires every documented blis-catalog clone to pin exactly it, so keep
     the `**Compatible blis-catalog release: `<tag>`**` shape intact when bumping. -->

**Compatible blis-catalog release: `0.1.1`** — [release notes](https://github.com/inference-sim/blis-catalog/releases/tag/0.1.1).

Why pin at all: BLIS parses every catalog file strictly (`KnownFields(true)`), and strict
parsing is one-way — an added or renamed key in a future catalog schema is a **hard load
error**, not a silently ignored field. An unpinned clone therefore lets a catalog release
turn a `blis run` that worked yesterday into a startup failure with no change to BLIS
itself. Pinning also makes results attributable: two people running the documented commands
months apart read the same catalog. (`blis run --metrics-path` records the catalog's git
revision and whether it was dirty — see [Results](../guide/results.md); a `--depth 1`
tag clone still carries the revision, so that provenance is unaffected.)

**Upgrading to a newer catalog release** is deliberately opt-in: clone the newer tag, run
your workload against it, and if it works, bump the tag on the line above — the guard test
then requires every documented clone command to match, so no page is left behind. Nothing
stops you pointing `--catalog` at a `main` checkout or a scratch clone of your own; the pin
is what the *documentation* promises, not a restriction the binary enforces.

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
