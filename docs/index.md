# BLIS — Blackbox Inference Simulator

BLIS is a discrete-event simulator of an LLM serving stack. Its goal is **llm-d stack-level
parity**: simulating an [llm-d](https://github.com/llm-d/llm-d) deployment end to end — the
router (EPP), admission and flow control, prefill/decode disaggregation, KV caching and
offload, and autoscaling — initially with vLLM as the engine, and eventually a wider range of
engines.

Its primary uses are **configuration search and optimization**, **capacity planning**, and
**policy discovery**, done robustly and with high fidelity. BLIS is CPU-only and
deterministic: the same seed produces byte-identical results, and no GPU is needed.

---

## Quick Start

```bash
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
go build -o blis main.go
git clone --branch 0.2.1 --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.1.1 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_CATALOG=$PWD/blis-catalog   # or pass --catalog on every run/replay
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios $(go env GOMODCACHE)/github.com/inference-sim/blis-latency-kernel@v0.1.0/testdata/aisimulate \
  --registry $PWD/blis-registry \
  --rate 10 --num-requests 100
```

`blis run` and `blis replay` need a catalog (`--catalog` or `BLIS_CATALOG`, no default), a
registry (`--registry`), a scenario directory (`--scenarios`, any directory of scenario YAMLs)
and a scenario file name within it (`--scenario`). The kernel module's own scenarios are at
`$(go env GOMODCACHE)/github.com/inference-sim/blis-latency-kernel@v0.1.0/testdata/aisimulate`;
`testdata/scenarios/` in this repository adds a P/D and an MTP example. The clones pin the
releases BLIS is tested against — see
[Catalog compatibility](getting-started/installation.md#catalog-compatibility).

---

## How BLIS fits together

| Repository | Role | Holds |
|---|---|---|
| **inference-sim** | *simulates* | Arrivals, admission, routing, scheduling, KV block accounting, placement, metrics. |
| [`blis-latency-kernel`](https://github.com/inference-sim/blis-latency-kernel) | *prices* | Step time, KV and fixed memory, P/D KV transfer, offload tier transfer, host overheads. |
| [`blis-catalog`](https://github.com/inference-sim/blis-catalog) | *states the facts* | Model graphs, chips, fabrics, storage devices, workload presets. |
| [`blis-registry`](https://github.com/inference-sim/blis-registry) | *holds the fitted numbers* | The coefficient sets the kernel prices with. |
| [`blis-schemas`](https://github.com/inference-sim/blis-schemas) | *defines the formats* | Scenario, deployment, engine and catalog file formats. |

A **scenario** names a catalog model and chip, registry coefficient sets, and the deployment
(nodes, fabric, and each pool's parallelism and engine settings). inference-sim simulates
traffic through the stack and asks the kernel for every cost; the kernel computes it from the
catalog's facts and the registry's coefficients.

```
Request Arrival → Admission → Routing → WaitQueue → Batch Formation → Step Execution → Completion
                                            ↓              ↓
                                      KV Allocation   Step pricing (blis-latency-kernel)
```

Admission and Routing apply in cluster mode (multi-instance). Single-instance mode skips
directly to WaitQueue. See [Architecture](concepts/architecture.md).

---

## Features

| Feature | In one line | Guide |
|---|---|---|
| Latency pricing | The scenario fixes the deployment; the kernel prices every step, memory budget and transfer. | [Latency model](guide/latency-models.md) |
| Workloads | Catalog presets, token distributions, multi-client YAML specs, closed-loop sessions. | [Workloads](guide/workloads.md) |
| Routing | llm-d-style weighted scoring across instances (`--routing-policy weighted`). | [Routing](guide/routing.md) |
| Admission and flow control | Token bucket, tier shedding, gateway queue (`--flow-control`), SLO goodput. | [Admission](guide/admission.md) |
| Scheduling and preemption | Per-instance schedulers and preemption policies. | [Scheduling](guide/scheduling.md) |
| P/D disaggregation | Prefill and decode pools, KV handoff priced over the scenario's fabric. | [Cluster](guide/cluster.md) |
| KV caching and offload | Prefix caching and tiered offload over catalog storage devices. | [KV cache](guide/kv-cache.md) |
| Autoscaling | Node pools with provisioning delays (`--policy-config`). | [Cluster](guide/cluster.md) |
| Trace replay | Any TraceV2 through the same deployment path; run/replay byte-identical. | [Replay](guide/observe-replay-calibrate.md) |
| Results and analysis | TTFT/ITL/E2E, saturation detectors, decision traces, fitness. | [Results](guide/results.md) |

---

## Documentation Guide

| Section | What You'll Find |
|---------|-----------------|
| [Getting Started](getting-started/index.md) | What is BLIS, installation, quick start, capacity planning tutorial |
| [Concepts](concepts/index.md) | System architecture, core engine, glossary |
| [User Guide](guide/index.md) | Task-oriented guides for each feature |
| [Reference](reference/index.md) | Configuration reference, supported models, workload spec schema |
| [Contributing](contributing/index.md) | Extension recipes, PR workflow, standards, templates |

Newcomers: [What is BLIS?](getting-started/index.md) →
[Quick Start](getting-started/quickstart.md) →
[Tutorial](getting-started/tutorial.md) → [Glossary](concepts/glossary.md).

---

## License

Apache License, Version 2.0. See [LICENSE](https://github.com/inference-sim/inference-sim/blob/main/LICENSE).
