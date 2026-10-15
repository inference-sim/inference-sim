# Latency Models

BLIS takes its latency model from one source: **blis-latency-kernel**
(Go module `github.com/inference-sim/blis-latency-kernel`, pinned at `v0.2.0`). The adapter
in `sim/kernelmodel` translates a BLIS batch into the kernel's request shapes and converts
the kernel's answers to BLIS ticks; it computes no cost of its own. There is no other
backend and no `--latency-model` flag: the roofline and trained-physics backends, the
alpha/beta coefficients, `hardware_config.json` calibration, and the
`trained_physics_coefficients` block of `defaults.yaml` were removed.

## Who does what

inference-sim simulates the cluster: arrivals, admission, routing, scheduling, KV block
accounting, placement and metrics. blis-latency-kernel prices: step time, KV and fixed
memory, P/D KV transfer, offload tier transfer and host overheads. It reads model graphs,
chips, fabrics and storage devices from **blis-catalog** and fitted coefficients from
**blis-registry**; every file format involved, the scenario included, is defined by
**blis-schemas**. See [The BLIS repositories](../concepts/architecture.md#the-blis-repositories).

## Running

```bash
git clone --branch 0.2.2 --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.1.2 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_CATALOG=$PWD/blis-catalog
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --rate 10 --num-requests 100
```

The pinned releases are explained in
[Catalog compatibility](../getting-started/installation.md#catalog-compatibility).

## Inputs

Four roots, all required on `blis run` and `blis replay`:

| input | what it names |
|---|---|
| `--scenario` | a scenario **file name** inside `--scenarios` |
| `--scenarios` | a directory of scenario YAML files |
| `--registry` | a `blis-registry` clone root, holding the coefficient sets a scenario names |
| `--catalog` / `BLIS_CATALOG` | a `blis-catalog` clone root |

A scenario directory can be any directory of scenario YAMLs. The kernel module ships the
scenarios it was scored on, at
`$(go env GOMODCACHE)/github.com/inference-sim/blis-latency-kernel@v0.2.0/testdata/aisimulate`
(after `go mod download`). This repo's own fixtures are in `testdata/scenarios/`:
`glm-5-h200-3p1d-ib.yaml` (P/D) and `glm-5-h200-tp8-mtp3.yaml` (MTP speculation).

### Scenario structure

A scenario is a blis-schemas file of two YAML documents, a `Scenario` and a `Deployment`:

```yaml
kind: Scenario
name: llama-3.1-70b-instruct-h200-fp8-vllm-tp4
engine_version: "0.29.0"
model: llama-3.1-70b-instruct
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads,
               cost-model-attention, cost-model-recurrent, cost-model-memory]
cluster:
  hardware: h200
  nodes: 1
  gpus_per_node: 8
  # fabric: ib-400g      # the inter-node fabric, needed when a handoff crosses nodes
---
kind: Deployment
name: llama-3.1-70b-instruct-h200-fp8-vllm-tp4
pools:
  - role: colocated        # colocated | prefill | decode
    nodes: 1
    parallel: {tp: 4, pp: 1, dp: 1, enable_expert_parallel: false}
    engine:
      quantization: fp8
      cache_dtype: fp8
      block_size: 16
      max_num_batched_tokens: 8192
      max_num_seqs: 256
      max_model_len: 32768   # required
      cudagraph_mode: PIECEWISE
      gpu_memory_utilization: 0.9
      # enable_prefix_caching: false
      # speculative: {method: mtp, num_spec_tokens: 3}
# offload: ...               # optional
# pd_transfer: {connector: NixlConnector}   # optional, P/D scenarios
```

### What comes from the scenario, and what stays on the CLI

The scenario and the kernel supply, with no flag to restate them: the model, the hardware,
TP/DP/EP, the KV block budget (from the kernel's memory methods, per data-parallel rank),
block size, `max_num_seqs`, `max_num_batched_tokens`, `max_model_len` (a count is kept; unstated takes the model
graph's context window; `auto` or `-1` takes that window shrunk to what one rank's KV cache
holds, as vLLM v0.31.0 starts; a window one rank cannot hold, or none at all, is refused), prefix caching, the KV cache dtype, and the speculative draft length and
method. The flags that used to state these (`--model`, `--hardware`, `--tp`, `--dp`,
`--total-kv-blocks`, `--max-num-seqs`, `--max-model-len`, `--kv-cache-dtype`,
`--num-speculative-tokens`, ...) no longer exist; to change one, edit the scenario.

Everything a scenario does not state stays on the command line: the workload, rate and
concurrency, routing, admission, scheduling, flow control, saturation detectors, the
instance count and P/D topology, LoRA, KV offload tiers, and seed/horizon/output flags. See
the [Configuration Reference](../reference/configuration.md).

## What the kernel prices

- **Step time.** One forward pass over the scheduled batch, composed from per-operator
  costs (GEMMs, attention, grouped expert GEMMs, recurrent state updates, collectives,
  elementwise and host terms) with coefficients fitted against vendor kernel sweeps.
  BLIS reads the kernel's no-overlap edge of the band it reports. Expert geometry comes
  from the model graph in the catalog, not from an HF config.
- **Memory.** Weights, fixed overheads and the per-block KV cost, from which BLIS takes the
  KV block budget for each data-parallel rank.
- **Host overheads.** Per-request queueing (tokenization and preprocessing), per-output-token
  processing, and the fixed per-request completion overhead.
- **P/D KV transfer** and **offload tier transfer**, below.

A scenario that cannot be resolved -- absent file, unreadable coefficient set, unknown
engine version, a deployment that does not fit -- aborts the run naming what failed. There
is no fallback.

## Prefill/decode disaggregation

A scenario with a `prefill` pool and a `decode` pool, run with `--prefill-instances` and
`--decode-instances`. BLIS builds one kernel per pool, so each pool is priced at its own
parallelism and engine settings. The KV handoff from a prefill instance to a decode
instance is priced by the kernel's `PDTransferTime` over the scenario's fabric, between the
two instances' placements (NVLink when they share a node). Each instance count must not
exceed its pool's rank capacity: pool nodes × `gpus_per_node` / (pp × tp × pcp).

Refused: a P/D topology over a colocated scenario; a disaggregated scenario run without
`--prefill-instances`/`--decode-instances`; pools that differ in block size, dp or draft
configuration; KV offload combined with P/D; shared (`--prefill-decode-instances`) or
encode instances.

## Data and expert parallelism (MoE)

With `dp > 1` BLIS runs one replica per data-parallel rank, each sized as one EngineCore
(the kernel's per-rank KV budget and caps). `enable_expert_parallel` is priced by the
kernel. A dense model with `dp > 1` is refused.

## Speculative Decoding / MTP (#1528)

A pool that states `engine.speculative: {method, num_spec_tokens}` drafts `K` tokens per
step. Pass `--speculative-acceptance-rate α` (required when the scenario drafts): BLIS does
not predict acceptance, since it runs no draft model.

- **Cost: verify width `w = K+1`.** Every decoding request's forward pass is priced by the
  kernel at the full verify width, whatever is later accepted. Step time is independent of
  `α` and non-decreasing in `K`.
- **Progress: `g = 1 + α·K` tokens per step** (mean), applied deterministically through a
  per-request fractional carry (no RNG), so runs stay byte-identical for a seed (INV-6).

Progress stops at the completion boundary: the final step is granted only the tokens the
request still needs, as vLLM trims the accepted tail once `check_stop` fires. A request's
output-token count, and a closed-loop `accumulate` session's context growth, are therefore
identical to a `K=0` run. Under speculation raw ITL percentiles are per verification step;
use TPOT for per-token latency.

**Known limitation ([#1627](https://github.com/inference-sim/inference-sim/issues/1627)).**
KV and token-budget occupancy are accounted by the accepted count `g`, not the speculative
footprint. vLLM reserves `K` lookahead KV slots per running request per step and consumes
about `K+1` of `max_num_batched_tokens`, so above KV saturation BLIS over-predicts batch size
and under-predicts preemption. Below saturation the two agree.

## KV offload

`--kv-offload-config` tiers must each name a catalog `device_class`; the kernel prices every
transfer with `TierTime` for that device at the tier's in-service queue depth. Explicit
`read_bandwidth`, `write_bandwidth` and `base_latency` in the tier config are refused.
The legacy `--kv-cpu-blocks` tier is priced by the kernel from the catalog `cpu_dram`
device. See [KV Offload](kv-offload-calibration.md).

## LoRA

The per-step adapter compute overhead is applied to the kernel's step time, and the static
adapter HBM reservation is set aside from the kernel's KV budget before blocks are counted.
The LoRA cost coefficients come from `--defaults-filepath` (the `lora:` block of
`defaults.yaml`) or `--lora-config`.

## What it does not yet do

- Predict speculative acceptance (it is an input).
- Account the speculative KV/token-budget footprint (#1627, above).
- Price shared prefill-decode or encode instances.
- Make concurrent P/D handoffs share fabric bandwidth: each is priced as if alone (#1922).
- Run a model or chip absent from the catalog, or a coefficient set absent from the
  registry. Adding either is a change to blis-catalog or blis-registry, not to this repo.

## The published accuracy figures

`cmd/metricscore` is the scorer behind the evaluation tables. It simulates nothing itself. For
every measured point it writes two files and runs `blis replay` on them, exactly as a user replays
a trace:

- a scenario: the sweep's own, with every run setting the measured engine passed;
- a trace of the InferenceX benchmark client's main phase: the `10*CONC` target lengths numpy drew
  at the client's seed 0, in request order, which are also the lengths the client sent (verified
  per point; see below).

The trace runs in pool mode at concurrency `CONC`. On a point whose engine caches prefixes, the
main phase runs after copies of request 0 have completed. That leaves the one state the
benchmark's drained warm-up leaves which the simulator reads: request 0's prompt cached, on an
idle engine.

TTFT and TPOT are the client's own `mean_ttft` and `mean_tpot` over the main-phase requests. Its
figures are therefore the CLI's. What is and is not reproduced, with the tests that check each
claim, is in [What the accuracy comparison reproduces](../kernel-exclusive/APPLES-TO-APPLES.md).

At the tagged releases above, with the vendored catalog and registry and the measurement corpora in
`BLIS_MEASUREMENTS` (see [Reproducing](../kernel-exclusive/REPRODUCE.md#the-measurement-data)):

```bash
go build -o blis main.go
go run ./cmd/metricscore -blis ./blis -framework vllm
```

Its flags are `-blis`, `-gaps`, `-corpus`, `-scenarios`, `-catalog`, `-registry`,
`-engine-settings`, `-absolutes`, `-framework`, `-config-tier`, `-simulated-only` and `-seed`.
`-seed` seeds only the synthetic prompts' token IDs, which move no latency.

`-gaps <file>` (default: stderr) lists every point that cannot be scored as the measured run was,
with the repository whose change would close the gap. Possible gaps:

- a model, chip or fabric missing from blis-catalog;
- coefficients missing from blis-registry;
- a run setting blis-schemas cannot express;
- a setting the measured run did not record;
- sent prompt lengths not verified equal to the targets;
- an engine BLIS does not model (sglang, TensorRT-LLM).

The report ends with a count of points per cause. The scorer reads those gaps to put every sweep
into one of three tiers, and prints every arm on each:

- **STRICT**: every point has an engine log, and every setting it passed is carried. The admission
  settings and `max_model_len` were measured. The sent prompt lengths are verified. No other gap.
- **MEASURED-OR-RESOLVED**: as STRICT, but an unpassed admission setting may take the value vLLM
  resolves for it.
- **ALL (approximate)**: every scoreable sweep. These are not the measured runs as they ran.

Mean absolute error, BLIS / AISimulate / AIC, on the same points:

| tier | sweeps | n (shape / mape) | TPOT shape | TPOT mape | TTFT shape | TTFT mape |
|---|---|---|---|---|---|---|
| STRICT | 0 | 0 | -- | -- | -- | -- |
| MEASURED-OR-RESOLVED | 10 | 52 / 62 | 11.38 / 8.73 / 9.15 | 16.79 / 19.85 / 19.08 | 13.96 / 23.22 / 22.34 | 23.47 / 42.96 / 23.77 |
| ALL (approximate) | 113 | 460 / 573 | 9.64 / 10.43 / 10.56 | 15.04 / 15.39 / 15.71 | 17.85 / 23.05 / 30.12 | 25.30 / 37.86 / 36.66 |

No sweep is STRICT. Every logged vLLM sweep either passed `max_cudagraph_capture_size`, which
blis-schemas cannot express, or left an admission setting unpassed.

On the MEASURED-OR-RESOLVED tier BLIS is **behind on TPOT shape** (11.38% against 8.73% and 9.15%) and
ahead on the other three metrics. The TPOT-shape deficit comes from two sweeps of one deployment,
minimax-m2.5-h200-fp8-vllm-tp4:

- **8k1k:** BLIS over-predicts the rise of TPOT with concurrency by +14.7% to +22.6% from
  concurrency 8 to 256, scoring 16.42% against AISimulate's 5.10%.
- **1k8k:** BLIS over-predicts it by +12.8% to +38.4% from concurrency 8 to 64, scoring 27.83%
  against AISimulate's 15.86%.

Without those two sweeps the two models are level, 8.73% against 8.75%. On the same tier BLIS's
TTFT is biased low: its signed mean TTFT mape is **-21.53%**.

A figure quoted without its tier and `-framework` cannot be checked. `-config-tier` (sweeps with
or without an engine log) changes the ALL row as well.

## The interface

The simulator consumes latency through the `sim.LatencyModel` interface
(`sim/latency_model.go`): `StepTime(batch)`, `QueueingTime(req)`,
`OutputTokenProcessingTime()` and `PostDecodeFixedOverhead()`, all in microseconds (ticks).
`sim/kernelmodel` is its only production implementation. Improving or extending latency
pricing is done in blis-latency-kernel (the cost model) and blis-registry (its
coefficients), not here; see [Extension Recipes](../contributing/extension-recipes.md).

## Further Reading

- [Configuration Reference](../reference/configuration.md) -- the scenario flags and what the scenario supplies
- [Supported Models](../reference/models.md) -- which models and chips a run can use
- [Core Engine](../concepts/core-engine.md) -- where step time enters the event loop
