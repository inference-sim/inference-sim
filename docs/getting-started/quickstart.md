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
git clone --branch 0.3.0 --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.2.0 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_CATALOG=$PWD/blis-catalog   # or pass --catalog on every command
export SCENARIOS=examplescenarios
```

A scenario directory can be any directory of scenario YAMLs. `$SCENARIOS` above is this
repository's ready-to-run set (`examplescenarios/`). The clones pin the releases BLIS is tested
against — see [Catalog compatibility](installation.md#catalog-compatibility).

## Single-Instance Simulation

```bash
./blis run --scenarios $SCENARIOS \
  --scenario deepseek-v4.1-flash-h100-tp8-dspark5.yaml --registry $PWD/blis-registry \
  --rate 10 --num-requests 100
```

This runs 100 requests at 10 requests/second through one DeepSeek-V4.1-Flash engine on an
8-GPU H100 node (FP8, vLLM, TP8 with expert parallelism and DSpark speculative decoding),
priced by `blis-latency-kernel`.

!!! note "The scenario states the deployment"
    There are no `--model`, `--hardware` or `--tp` flags. Model, hardware, parallelism, block
    size, batch limits, max model length, prefix caching, cache dtype and speculative decoding
    all come from the scenario; the KV block budget comes from the kernel. A model runs only
    if it is in the catalog — nothing is fetched at run time.

### Reading the Output

BLIS prints diagnostic logs to stderr and one **Result** document to stdout: JSON by default,
YAML with `--format yaml`; `--output result.json` writes the same bytes to a file, and
`--view summary` renders a readable summary to stderr. The metrics are over the measured
window, under blis-schemas' standard names ([Metrics & Results](../guide/results.md)):

| Path | What It Measures |
|------|-----------------|
| `metrics.ttft_mean_ms`, `metrics.ttft_p99_ms` | **Time to First Token**: how long until the first output token. |
| `metrics.e2e_mean_ms`, `metrics.e2e_p99_ms` | **End-to-End latency**: request arrival to the last output token. |
| `metrics.itl_mean_ms`, `metrics.itl_p99_ms` | **Inter-Token Latency**, per token: (E2E - TTFT) / (output tokens - 1). |
| `metrics.output_tok_s`, `metrics.request_throughput` | Output tokens and completed requests per second. |
| `requests.completed`, `requests.injected` | The request ledger (INV-1): every injected request ends in one bucket. |
| `requests.still_queued`, `requests.still_running` | Not done when the run ended: the workload outlasted the horizon. |
| `engine.cluster.preemptions` | Running requests evicted to make room. Non-zero suggests overload. |

```bash
./blis run --scenarios $SCENARIOS \
  --scenario deepseek-v4.1-flash-h100-tp8-dspark5.yaml --registry $PWD/blis-registry \
  --rate 10 --num-requests 100 2>/dev/null | jq '.metrics.ttft_p99_ms'
```

## Cluster Mode

Scale to 4 instances behind the router. `--num-instances` cannot exceed the scenario pool's rank
capacity (pool nodes × gpus_per_node / (pp × tp × pcp)); the single-node scenario above is one
rank, so this uses the two-node `gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml` (2 × 8 ÷ TP2 = 8
ranks, up to eight instances):

```bash
./blis run \
  --scenarios examplescenarios \
  --scenario gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml --registry $PWD/blis-registry \
  --num-instances 4 \
  --rate 100 --num-requests 500
```

This simulates a 4-instance cluster receiving 100 requests/second. Every cluster run routes with the llm-d router (the Endpoint Picker) embedded in the simulator; with no router configuration it uses the router's built-in default configuration (queue, KV-cache utilization and prefix-cache scorers). A real deployment's router configuration is stated by the scenario's `router:` block and pool labels; see [Routing](../guide/routing.md).

!!! note "Multi-instance output"
    The Result's `metrics` and `requests` are the whole cluster's. Each instance's own figures
    are under `engine.instances`, keyed by instance id:
    pipe the run above to `jq '.engine.instances.instance_0.completed_requests'`.

## Try Different Configurations

```bash
# Higher traffic rate
./blis run --scenarios examplescenarios \
  --scenario gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml --registry $PWD/blis-registry \
  --num-instances 8 --rate 500 --num-requests 2000

# With decision tracing (see where each request was routed)
./blis run --scenarios examplescenarios \
  --scenario gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml --registry $PWD/blis-registry \
  --num-instances 4 --rate 100 --num-requests 500 \
  --trace-level decisions --summarize-trace --view summary

# A different deployment: pick another scenario (single-node TP2: up to 4 instances)
./blis run --scenarios examplescenarios \
  --scenario gpt-oss-120b-h100-fp4-vllm-tp2.yaml --registry $PWD/blis-registry \
  --num-instances 2 --rate 100 --num-requests 500
```

## What's Next

- **[Tutorial: Capacity Planning](tutorial.md)** — Full walkthrough: find the right instance count for your workload
- **[Routing with the llm-d Router](../guide/routing.md)** — Configure the router and compare configurations
- **[Latency Model](../guide/latency-models.md)** — What the kernel prices and how scenarios are structured
- **[Configuration Reference](../reference/configuration.md)** — Complete CLI flag reference
