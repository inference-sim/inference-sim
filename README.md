# Blackbox Inference Simulator (BLIS)

BLIS is a discrete-event simulator of an LLM serving stack. Its goal is **llm-d stack-level
parity**: simulating an [llm-d](https://github.com/llm-d/llm-d) deployment end to end — the
router (EPP), admission and flow control, prefill/decode disaggregation, KV caching and
offload, and autoscaling — initially with vLLM as the engine, and eventually a wider range of
engines.

It is built for three things, done robustly and with high fidelity:

- **Configuration search and optimization** — which parallelism, engine limits, routing
  profile and P/D split serve a workload best.
- **Capacity planning** — how many GPUs and instances a target rate and SLO need.
- **Policy discovery** — designing and comparing routing, admission, scheduling and
  autoscaling policies before they ship.

BLIS is CPU-only and deterministic: the same seed produces byte-identical results, and no GPU
is needed.

---

## How BLIS fits together

A BLIS run draws on five repositories. Each owns one kind of thing.

| Repository | Role | Holds |
|---|---|---|
| **inference-sim** (this one) | *simulates* | The end-to-end simulation: arrivals, admission, routing, scheduling, KV block accounting, placement, metrics. |
| [`blis-latency-kernel`](https://github.com/inference-sim/blis-latency-kernel) | *prices* | Step time, KV and fixed memory, P/D KV transfer, offload tier transfer, host overheads. The only latency backend. |
| [`blis-catalog`](https://github.com/inference-sim/blis-catalog) | *states the facts* | Model graphs, chips, fabrics, storage devices, workload presets — declared facts, nothing fitted. |
| [`blis-registry`](https://github.com/inference-sim/blis-registry) | *holds the fitted numbers* | Coefficient sets the kernel prices with, each with its provenance and scope. |
| [`blis-schemas`](https://github.com/inference-sim/blis-schemas) | *defines the formats* | The source of truth for every shared format: scenario, deployment, engine and catalog files. |

The key relationships:

- A **scenario** (a blis-schemas Scenario + Deployment YAML) names a model and a chip from
  the catalog, coefficient sets from the registry, and the deployment: cluster nodes and
  fabric, and each pool's role, parallelism and engine settings.
- **inference-sim** reads the scenario, builds one kernel per pool, and simulates traffic
  through the stack. Every time it needs a cost — a batch step, a KV budget, a KV handoff, an
  offload transfer — it asks the kernel.
- **blis-latency-kernel** computes that cost from the catalog's facts and the registry's
  coefficients. It holds no data of its own.

Deployment choices live in the scenario; facts live in the catalog; fitted numbers live in the
registry. Nothing is fetched at run time.

---

## Install and run

Requires Go 1.24+.

```bash
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
go build -o blis main.go

git clone --branch 0.2.1 --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.1.1 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_CATALOG=$PWD/blis-catalog

./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios $(go env GOMODCACHE)/github.com/inference-sim/blis-latency-kernel@v0.1.0/testdata/aisimulate \
  --registry $PWD/blis-registry \
  --rate 10 --num-requests 100
```

Every `blis run` and `blis replay` needs four inputs:

- the **catalog**, via `--catalog <clone root>` or `BLIS_CATALOG` (no default; the flag wins);
- the **registry**, via `--registry <clone root>`;
- a **scenario directory**, via `--scenarios <dir>` — any directory of scenario YAMLs. The
  kernel module ships its own at
  `$(go env GOMODCACHE)/github.com/inference-sim/blis-latency-kernel@v0.1.0/testdata/aisimulate`
  (present after `go build`), and `testdata/scenarios/` in this repository adds a P/D and an
  MTP example;
- a **scenario**, via `--scenario <file name>` within that directory.

The clones are pinned to the releases BLIS is tested against; see
[Catalog compatibility](docs/getting-started/installation.md#catalog-compatibility).

The run prints JSON metrics on stdout — TTFT, ITL and E2E distributions
(`ttft_mean_ms`, `itl_p99_ms`, `e2e_p99_ms`, ...), throughput (`responses_per_sec`,
`tokens_per_sec`), `completed_requests` and `preemption_count`. See
[Interpreting Results](docs/guide/results.md).

The examples below use `testdata/scenarios/llama-3.1-70b-instruct-h200-tp4-4node.yaml`, the
same deployment on four nodes, so they can run several instances.

---

## Features

### Latency pricing

The scenario fixes the model, hardware, TP/PP/DP/EP, block size, batch limits, max model
length, prefix caching, cache dtype and speculative decoding; the kernel derives the KV block
budget from them. There are no `--model`, `--hardware` or `--tp` flags.
→ [Latency model guide](docs/guide/latency-models.md)

### Workloads

Named presets from the catalog, token distributions, multi-client YAML specs, closed-loop
sessions.

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry \
  --workload chatbot --rate 20 --num-requests 500
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry \
  --workload-spec examples/servegen-language.yaml
```

→ [Workloads](docs/guide/workloads.md)

### Routing

Multi-instance clusters with llm-d-style weighted scoring (default profile
`precise-prefix-cache:2,queue-depth:1,kv-utilization:1`).

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry \
  --num-instances 8 --routing-policy weighted --rate 20 --num-requests 80
```

`--num-instances` cannot exceed the scenario pool's rank capacity (pool nodes × gpus_per_node /
(pp × tp × pcp)); the fixture above is four H200 nodes at TP4, so up to eight. For other shapes,
copy a scenario, raise `cluster.nodes` and the pool's `nodes`, and add a `cluster.fabric`.

→ [Routing](docs/guide/routing.md), [Cluster simulation](docs/guide/cluster.md)

### Admission and flow control

Token-bucket and tier-shedding admission, a gateway queue with saturation-gated dispatch, and
SLO goodput.

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry \
  --num-instances 4 --flow-control --rate 200 --num-requests 1000
```

→ [Admission](docs/guide/admission.md)

### Prefill/decode disaggregation

A scenario with `prefill` and `decode` pools gets one kernel per pool; the kernel prices each
KV handoff over the scenario's fabric.

```bash
./blis run --scenario glm-5-h200-3p1d-ib.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry \
  --num-instances 4 --prefill-instances 3 --decode-instances 1 --pd-decider always \
  --rate 10 --num-requests 200
```

→ [Cluster simulation](docs/guide/cluster.md)

### KV caching and offload

Block-level KV accounting with prefix caching, and tiered offload whose tiers name catalog
storage devices (`--kv-offload-config`).
→ [KV cache](docs/guide/kv-cache.md), [KV offload](docs/guide/kv-offload-calibration.md)

### Autoscaling

Horizontal autoscaling over node pools with provisioning delays, configured in a
`--policy-config` YAML (each pool's `gpu_type` must equal the scenario's hardware).
→ [Cluster simulation](docs/guide/cluster.md)

### Trace replay

Replay any TraceV2 — exported by `blis run --trace-output`, or converted with
`blis convert weka|otel|inference-perf|servegen` — through the same kernel and deployment
path. A run's exported trace replayed with identical flags (including `--horizon`) gives
byte-identical stdout.

```bash
./blis replay --trace-header t.yaml --trace-data d.csv \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry
```

→ [Observe / Replay / Calibrate](docs/guide/observe-replay-calibrate.md).
`blis observe` and `blis calibrate` are deprecated
([#1901](https://github.com/inference-sim/inference-sim/issues/1901)).

### Analysis

Saturation detection (`--detectors`), decision tracing with counterfactual regret, per-SLO-class
metrics and fitness scoring.
→ [Interpreting Results](docs/guide/results.md)

---

## Documentation

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting-started/index.md) | Installation, quick start, capacity planning tutorial |
| [User Guide](docs/guide/index.md) | Task-oriented guides for every feature above |
| [Concepts](docs/concepts/index.md) | Architecture, core engine, glossary |
| [Reference](docs/reference/index.md) | CLI flags, supported models, workload spec schema, [project structure](docs/reference/project-structure.md) |
| [Methodology](docs/methodology/index.md) | Strategy Evolution methodology, discovered principles |
| [Contributing](docs/contributing/index.md) | Extension recipes, PR workflow, design process, standards |

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the engineering standards, development workflow,
and guides for adding components.

## License

Apache License, Version 2.0. See [LICENSE](./LICENSE).
