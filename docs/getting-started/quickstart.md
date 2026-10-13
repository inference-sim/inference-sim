# Quick Start

Run your first BLIS simulation in 30 seconds.

No credentials or network access are needed to *run* a simulation: BLIS reads each model's architecture from a local checkout of the model catalog and makes no HuggingFace requests. (`HF_TOKEN` matters only if you are downloading a new — possibly gated — model's `config.json` by hand to add a catalog entry.)

## Set up the catalog, registry and scenarios

Every `blis run` / `blis replay` needs three things besides the binary:

- the **catalog** — a clone of [`blis-catalog`](https://github.com/inference-sim/blis-catalog)
  (models, chips, fabrics, storage devices, workload presets), located by `--catalog <path>`
  or `BLIS_CATALOG` (the flag wins; there is no default and no search path);
- the **registry** — a clone of [`blis-registry`](https://github.com/inference-sim/blis-registry)
  (the fitted coefficients), passed as `--registry`;
- a **scenario** — a [`blis-schemas`](https://github.com/inference-sim/blis-schemas) Scenario +
  Deployment YAML naming the model, hardware, parallelism and engine settings, picked with
  `--scenario <file name>` from `--scenarios <dir>`.

```bash
git clone --branch 0.2.1 --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.1.1 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_CATALOG=$PWD/blis-catalog   # or pass --catalog on every command
export SCENARIOS=$(go env GOMODCACHE)/github.com/inference-sim/blis-latency-kernel@v0.1.0/testdata/aisimulate
```

A scenario directory can be any directory of scenario YAMLs. `$SCENARIOS` above is the kernel
module's own set (present after `go build`); `testdata/scenarios/` in this repository adds a
P/D and an MTP example. The clones pin the releases BLIS is tested against — see
[Catalog compatibility](installation.md#catalog-compatibility).

## Single-Instance Simulation

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios $SCENARIOS --registry $PWD/blis-registry \
  --rate 10 --num-requests 100
```

This runs 100 requests at 10 requests/second through one Llama-3.1-70B-Instruct engine on
H200 (FP8, vLLM, TP4), priced by `blis-latency-kernel`.

!!! note "The scenario states the deployment"
    There are no `--model`, `--hardware` or `--tp` flags. Model, hardware, parallelism, block
    size, batch limits, max model length, prefix caching, cache dtype and speculative decoding
    all come from the scenario; the KV block budget comes from the kernel. A model runs only
    if it is in the catalog — nothing is fetched at run time.

### Reading the Output

BLIS prints diagnostic logs to stderr and results to stdout. You'll see log lines (prefixed with `INFO` or `WARN`) followed by a `=== Simulation Metrics ===` header and pretty-printed JSON:

**Latency metrics** (all in milliseconds, reported as mean/p90/p95/p99):

| Field | What It Measures |
|-------|-----------------|
| `ttft_mean_ms`, `ttft_p99_ms` | **Time to First Token** — how long until the first output token is generated. Lower is better for interactive use. |
| `e2e_mean_ms`, `e2e_p99_ms` | **End-to-End latency** — total time from request arrival to final output token. |
| `itl_mean_ms`, `itl_p99_ms` | **Inter-Token Latency** — time between consecutive output tokens. Lower means smoother streaming. |
| `scheduling_delay_p99_ms` | Wait time from request arrival until processing begins (includes any queueing). |

**Throughput:**

| Field | What It Measures |
|-------|-----------------|
| `responses_per_sec` | Completed requests per second. |
| `tokens_per_sec` | Output tokens generated per second. |
| `completed_requests` | How many requests finished within the simulation window. |
| `total_input_tokens`, `total_output_tokens` | Total tokens processed across all completed requests. |

**Health indicators:**

| Field | What It Measures |
|-------|-----------------|
| `preemption_count` | Number of times a running request was evicted to make room for others. Non-zero suggests the system is overloaded. |
| `dropped_unservable` | Requests rejected because they were too large for the configured memory or context length. |
| `still_queued`, `still_running` | Requests not yet completed when the simulation ended. Non-zero means the workload outlasted the simulation window. |

## Cluster Mode

Scale to 4 instances with routing. `--num-instances` cannot exceed the scenario pool's rank
capacity (pool nodes × gpus_per_node / (pp × tp × pcp)); the single-node scenario above holds
two TP4 engines, so this uses the committed four-node fixture in `testdata/scenarios/` (up to
eight instances):

```bash
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 \
  --routing-policy weighted \
  --rate 100 --num-requests 500
```

This simulates a 4-instance cluster receiving 100 requests/second. The `weighted` routing policy uses the default scorer profile (`precise-prefix-cache:2, queue-depth:1, kv-utilization:1`) to distribute requests across instances.

!!! note "Multi-instance output format"
    In cluster mode, BLIS prints one JSON block per instance plus a cluster-level summary (5 blocks total for 4 instances). The cluster summary has `"instance_id": "cluster"`. If piping to `jq`, use `--slurp` to handle multiple JSON objects: `./blis run ... 2>/dev/null | jq --slurp '.[] | select(.instance_id == "cluster")'` to extract the cluster summary.

## Try Different Configurations

```bash
# Higher traffic rate
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 8 --rate 500 --num-requests 2000

# With decision tracing (see where each request was routed)
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --rate 100 --num-requests 500 \
  --trace-level decisions --summarize-trace

# A different deployment: pick another scenario (one node, TP4: at most 2 instances)
./blis run --scenario gpt-oss-120b-h200-fp4-vllm-tp4.yaml \
  --scenarios $SCENARIOS --registry $PWD/blis-registry \
  --num-instances 2 --rate 100 --num-requests 500
```

## What's Next

- **[Tutorial: Capacity Planning](tutorial.md)** — Full walkthrough: find the right instance count for your workload
- **[Routing Policies](../guide/routing.md)** — Understand and compare routing strategies
- **[Latency Model](../guide/latency-models.md)** — What the kernel prices and how scenarios are structured
- **[Configuration Reference](../reference/configuration.md)** — Complete CLI flag reference
