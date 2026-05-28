# scripts/

Reproducible analysis scripts for BLIS. Each script runs `blis` end-to-end and
emits a single CSV summary alongside per-run raw outputs, so anyone can
re-validate a published claim without manually reconstructing the command set.

## find-saturation.sh — Rate-sweep saturation finder

Drives `blis run` across a configurable rate sweep against a chosen
`(model, hardware, TP, workload)` configuration. For each rate it:

1. Runs `blis run` with the **drain-ratio** classifier (the default since #1392).
2. Runs the same workload/seed again with the **slope-based** classifier so both
   verdicts are recorded side-by-side.
3. Extracts throughput, latency, and classifier verdicts into a single CSV row.

The output reproduces the validation table from PR #1395 against the bundled
`model_configs/llama-3.1-70b-instruct/`. Pointing it at any other configuration
should produce a comparable table with the same column shape.

### Quick start

```bash
# Default: Llama-3.1-70B / TP=8 / H100 / chatbot, sweeps 0.5..100 req/s
./scripts/find-saturation.sh

# Llama-2-7B / TP=1, narrower sweep
MODEL=meta-llama/Llama-2-7b-hf \
  MODEL_CONFIG_FOLDER=model_configs/llama-2-7b-hf \
  TP=1 RATES="2 4 6 8 10 12 16 20" \
  ./scripts/find-saturation.sh

# Custom workload, slower coarse sweep
WORKLOAD=summarization NUM_REQUESTS=2000 RATES="4 6 8 10 12" \
  ./scripts/find-saturation.sh

# No bundled config — let blis fetch from HuggingFace
MODEL=qwen/qwen3-14b MODEL_CONFIG_FOLDER="" TP=1 \
  ./scripts/find-saturation.sh
```

### Inputs (all environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `meta-llama/Llama-3.1-70B-Instruct` | HuggingFace-style model name |
| `MODEL_CONFIG_FOLDER` | `model_configs/llama-3.1-70b-instruct` | Path to bundled `config.json`; set to `""` to force HF auto-fetch |
| `HARDWARE` | `H100` | GPU type passed to `--hardware` |
| `TP` | `8` | Tensor parallelism degree |
| `WORKLOAD` | `chatbot` | Built-in preset (chatbot/summarization/contentgen/multidoc) |
| `LATENCY_MODEL` | `trained-physics` | `--latency-model` backend |
| `NUM_REQUESTS` | `6000` | `--num-requests` per rate |
| `HORIZON_US` | `600000000` (600s) | `--horizon` per rate |
| `SATURATION_WINDOW_S` | `10` | `--saturation-window` (seconds) |
| `RATES` | `0.5 1 2 4 6 8 10 12 14 16 20 30 40 50 60 80 100` | Space-separated rate sweep |
| `SEED` | `42` | RNG seed (held constant across both classifier runs) |
| `OUT_DIR` | `results/saturation-<ts>-<pid>` | Output directory |

### Outputs

```
$OUT_DIR/
├── summary.csv                          # one row per rate (12 columns)
├── rate-{R}.json                        # blis run stdout (metrics)
├── rate-{R}.stderr                      # blis run stderr (progress logs)
├── rate-{R}.drain-ratio.json            # BacklogDriftReport (drain-ratio classifier)
└── rate-{R}.slope-based.json            # BacklogDriftReport (slope-based classifier)
```

`summary.csv` columns:

| Column | Source | Meaning |
|---|---|---|
| `intended_rate` | input flag | What `--rate` was set to |
| `sustained_throughput` | `injected_requests / vllm_estimated_duration_s` | Actual req/s injected over total sim time |
| `goodput_rps` | `responses_per_sec` | Completed req/s |
| `goodput_vs_intended` | `goodput_rps / intended_rate` | Ratio; <100% indicates the engine couldn't sustain intended load |
| `timeout_frac` | `timed_out_requests / injected_requests` | Fraction culled by client timeout |
| `e2e_p99_ms` / `ttft_p99_ms` | metrics | Tail latencies |
| `still_queued` / `still_running` | metrics | End-state residue |
| `drain_ratio_verdict` | `--saturation-classifier drain-ratio` report | UNSATURATED / TRANSIENT_BACKLOG / PERSISTENTLY_SATURATED |
| `drain_ratio_rho` | drain-ratio note (parsed) | Quantified `ρ ≈ 1/DrainRatio` from steady-state windows |
| `slope_based_verdict` | `--saturation-classifier slope-based` report | Same three verdicts via OLS regression |

### Reading the output

A clean read of "where does this configuration saturate?" looks like:

```
intended_rate  goodput_rps  ratio  ρ      drain-ratio              slope-based
0.5            0.50         100%   1.00   UNSATURATED              UNSATURATED
…
60             56.29        94%    1.00   UNSATURATED              PERSISTENTLY_SATURATED
80             64.49        81%    1.16   PERSISTENTLY_SATURATED   PERSISTENTLY_SATURATED   ← knee
100            64.76        65%    1.45   PERSISTENTLY_SATURATED   PERSISTENTLY_SATURATED
```

The saturation knee is the first rate where `ratio` falls below ~100%, `ρ`
crosses 1.05, OR a classifier flips to PERSISTENTLY_SATURATED. Drain-ratio is
the cleanest signal — it gives a quantified ρ. Slope-based is more sensitive
and may surface TRANSIENT_BACKLOG at moderate load (peak/mean burstiness even
when average ρ < 1) — that's a feature when planning for tail latency.

### Tips

- **Build once.** The script auto-builds `./blis` if absent. Subsequent runs
  reuse it; remove the binary to force a rebuild.
- **Pin the seed.** Two seeds at the same rate land in different parts of
  Poisson variance and look noisy. Default `SEED=42` keeps every step of the
  sweep on the same noise realization.
- **Don't trust a single rate.** Saturation curves are smoother than they look;
  the knee is a transition zone (~3-5 rate steps wide). Look at three rates
  before and after the suspected knee.
- **Fine sweep after coarse.** First pass with the default 17 rates spanning
  200×; identify the knee zone (e.g., between 60 and 80); then re-run with
  `RATES="62 64 66 68 70 72 74 76 78"` to pin the exact transition.

### Running on a custom configuration

The script's defaults match the PR validation experiment so anyone can
reproduce that exact table. To validate any other configuration, override the
relevant variables:

```bash
# Example: probe Mixtral-8x7B FP8 on 4×H100 TP=4 with summarization workload
MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1 \
  MODEL_CONFIG_FOLDER=model_configs/mixtral-8x7b-instruct \
  TP=4 WORKLOAD=summarization \
  RATES="2 4 8 16 24 32 40 48" \
  ./scripts/find-saturation.sh
```

If your model isn't in `model_configs/`, either:
- Drop a `config.json` into `model_configs/<your-model-slug>/` and point
  `MODEL_CONFIG_FOLDER` at it, or
- Set `MODEL_CONFIG_FOLDER=""` to let `blis` fetch from HuggingFace at startup.

### Dependencies

- `bash` 4+
- `jq` (for JSON parsing)
- `bc` (for ratio arithmetic)
- `column` (for the final pretty-print; falls back gracefully if missing)
- `go` (auto-builds `./blis` on first run)
