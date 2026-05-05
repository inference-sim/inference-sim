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

BLIS uses roofline mode by default, which auto-fetches model architecture configs from HuggingFace. Set `HF_TOKEN` to access gated models (e.g., [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)) and avoid rate limits:

```bash
export HF_TOKEN=your_token_here
```

Public models (e.g., Qwen3) work without a token. See [HuggingFace access tokens](https://huggingface.co/docs/hub/en/security-tokens) to create a token.

!!! note "Air-gapped / offline environments"
    The default roofline mode requires network access to HuggingFace on first run (configs are cached in `model_configs/` after that). For environments without internet access:

    - **Pre-populate** `model_configs/<model>/config.json` from a machine with internet access and use `--model-config-folder`
    - Or use `--latency-model roofline` with explicit `--hardware` and `--tp` flags (bypasses model config requirements)

    For CI pipelines, set `HF_TOKEN` in your environment secrets to avoid rate limits on gated models.

## Verify the Build

```bash
./blis run --model qwen/qwen3-14b --num-requests 10
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
