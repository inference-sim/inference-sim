# Blackbox Inference Simulator (BLIS)

BLIS is a discrete-event simulator of an LLM serving stack. Its goal is **llm-d stack-level
parity**: simulating an [llm-d](https://github.com/llm-d/llm-d) deployment end to end — the
router (EPP), admission and flow control, prefill/decode disaggregation, KV caching and
offload, and autoscaling — initially with vLLM as the engine, and eventually a wider range of
engines.

It is built for three things, done robustly and with high fidelity:

- **Configuration search and optimization** — which parallelism, engine limits, router
  configuration and P/D split serve a workload best.
- **Capacity planning** — how many GPUs and instances a target rate and SLO need.
- **Policy discovery** — comparing llm-d router configurations, scheduling and autoscaling
  policies before they ship.

BLIS is CPU-only and deterministic: the same seed produces byte-identical results, and no GPU
is needed.

---

## How BLIS fits together

A BLIS run draws on six repositories. Each owns one kind of thing.

| Repository | Role | Holds |
|---|---|---|
| **inference-sim** (this one) | *simulates* | The end-to-end simulation: the clock and event loop, arrivals and workloads, the cluster and placement, the embedded llm-d router (admission, flow control, routing), metrics and reports. It holds no vLLM code. |
| [`blis-vllm`](https://github.com/inference-sim/blis-vllm) | *runs the engine* | The vLLM engine: vLLM v0.32.0's scheduler, KV cache, MTP, DP and connectors, translated to Go and diff-tested against real vLLM; the vLLM engine configuration and rules; the kernel adapter. |
| [`blis-latency-kernel`](https://github.com/inference-sim/blis-latency-kernel) | *prices* | Step time, KV and fixed memory, P/D KV transfer, offload tier transfer, host overheads. The only latency backend. |
| [`blis-substrate` `catalog/`](https://github.com/inference-sim/blis-substrate/tree/main/catalog) | *states the facts* | Model graphs, chips, fabrics, storage devices, workload presets — declared facts, nothing fitted. |
| [`blis-substrate` `registry/`](https://github.com/inference-sim/blis-substrate/tree/main/registry) | *holds the fitted numbers* | Coefficient sets the kernel prices with, each with its provenance and scope. |
| [`blis-schemas`](https://github.com/inference-sim/blis-schemas) | *defines the formats* | The source of truth for every shared format: scenario, deployment, engine and catalog files. |

The key relationships:

- A **scenario** (a blis-schemas Scenario + Deployment YAML) names a model and a chip from
  the catalog, coefficient sets from the registry, and the deployment: cluster nodes and
  fabric, and each pool's role, parallelism and engine settings.
- **inference-sim** reads the scenario and simulates traffic through the stack. It drives each
  instance's engine only through blis-schemas' `engine.Engine` interface, and asks the engine
  for every step: what runs, how long it takes, and what it emits.
- **blis-vllm** is that engine. It decides each batch as vLLM does and asks the kernel what the
  step costs.
- **blis-latency-kernel** computes that cost from the catalog's facts and the registry's
  coefficients. It holds no data of its own.

The dependencies run one way: blis-schemas ← blis-latency-kernel ← blis-vllm ← inference-sim.
The kernel imports only blis-schemas. Non-test code under `sim/` and `cmd/` imports blis-vllm at
one build site, `cmd/engine_build.go`.

Deployment choices live in the scenario; facts live in the catalog; fitted numbers live in the
registry. Nothing is fetched at run time.

---

## Install and run

<!-- BEGIN GENERATED from stack.json: toolchain -->
Requires Go 1.27.2, the toolchain blis-substrate's contracts/toolchain names at the pinned release.
<!-- END GENERATED -->

```bash
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
go build -o blis main.go
```

```bash
./blis run --scenarios examplescenarios \
  --scenario deepseek-v4.1-flash-h100-tp8-dspark5.yaml \
  --rate 10 --num-requests 100
```

The run reads the catalog and registry embedded in the blis-substrate module this build pins
([Data compatibility](docs/getting-started/installation.md#catalog-compatibility)). No flag or
environment variable is needed for them. To read a catalog of your own instead, check out
blis-substrate (or keep a directory with the layout of its `catalog/`) and name it with
`--catalog` or `BLIS_CATALOG`:

<!-- BEGIN GENERATED from stack.json: data-checkout -->
```bash
git clone https://github.com/inference-sim/blis-substrate.git
git -C blis-substrate checkout 565db4d00f46927f1ca3cb863da2100be4675f44
export BLIS_CATALOG=$PWD/blis-substrate/catalog   # optional: replaces the embedded catalog (or pass --catalog)
```
<!-- END GENERATED -->

Every `blis run` and `blis replay` needs two inputs:

- a **scenario directory**, via `--scenarios <dir>` — any directory of scenario YAMLs. This
  repository ships ready-to-run ones in [`examplescenarios/`](examplescenarios/);
- a **scenario**, via `--scenario <file name>` within that directory.

Two more inputs have a default:

- the **catalog**: the embedded copy, or a directory named by `--catalog <dir>` or
  `BLIS_CATALOG` that replaces it whole (the flag wins);
- the **registry**: the embedded copy, or a directory named by `--registry <dir>` or
  `BLIS_REGISTRY` that replaces it whole (the flag wins).

The Result records which data the run read under `data`. See
[Catalog compatibility](docs/getting-started/installation.md#catalog-compatibility).

**vLLM versions.** A scenario names its engine as `engine: {kind: vllm, version: "0.32.0"}`, or
with the deprecated `engine_version: "0.32.0"`, accepted with a warning until M4. This build runs
blis-vllm's translated vLLM v0.32.0 engine (the latency kernel's 0.31.0 tuning prices it, and the
run says so on stderr). A scenario naming another release is refused; for vLLM 0.29.0 and 0.31.0
the refusal names v0.15.0, the last inference-sim release that runs them. Priority
scheduling (`scheduling_policy: priority`, `--scheduler priority-fcfs` or `reverse-priority`,
`--preemption-policy priority`, SLO priority overrides) is refused for now. The engine block must
state `max_model_len` as an explicit number (unstated or `auto` is refused), plus `max_num_seqs`
and `max_num_batched_tokens`.

The quickstart's `--rate 10 --num-requests 100` is a synthetic workload for a quick first run.
[`examplescenarios/`](examplescenarios/) also ships MoE deployments to try the same commands
against — `deepseek-v4-pro-b300-fp4-vllm-tp4-mtp2.yaml` (B300, MTP); each file's header
records its provenance. Kimi-K3 is not among them: its linear-attention (KDA) layers have no KV
mapping in the engine yet, so a Kimi-K3 scenario is refused at startup.

The run prints one Result document on stdout (JSON; `--format yaml` for YAML, `--output` for a
file): its input, metrics over the measured window (`ttft_p99_ms`, `itl_p99_ms` per token,
`e2e_p99_ms`, `output_tok_s`, `request_throughput`, ...) and the request ledger (`requests`).
`--view summary` prints a readable summary to stderr. See
[Interpreting Results](docs/guide/results.md).

The examples below use `examplescenarios/gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml`, a two-node
deployment (2 nodes × 8 GPUs ÷ TP2 = 8 ranks), so they can run up to eight instances.

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
./blis run --scenarios examplescenarios --scenario gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml \
  --workload chatbot --rate 20 --num-requests 500
./blis run --scenarios examplescenarios --scenario gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml \
  --workload-spec examples/servegen-language.yaml
```

→ [Workloads](docs/guide/workloads.md)

### Routing

Every multi-instance cluster routes with the real llm-d router (the Endpoint Picker), embedded
in-process. It reads a deployment's own EndpointPickerConfig, EPP flags and `InferencePool` /
`InferenceObjective` manifests, named by the scenario's `router:` block (and pod labels from
`pools[].labels`); with none, it uses the router's built-in default configuration.
Kernel-backed scenarios expose host and declared-rack identity to llm-d topology plugins. Explicit simulated Node manifests can add zone and region; the [routing guide](docs/guide/routing.md#topology-identity-in-kernel-backed-scenarios) explains the stamp handler and limits.

```bash
./blis run --scenarios examples/llm-d/scenarios \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node-prefix-affinity.yaml \
  --num-instances 8 --rate 20 --num-requests 80
```

`--num-instances` cannot exceed the scenario pool's rank capacity (pool nodes × gpus_per_node /
(pp × tp × pcp)); the fixture above is four H200 nodes at TP4, so up to eight. For other shapes,
copy a scenario, raise `cluster.nodes` and the pool's `nodes`, and add a `cluster.fabric`.

→ [Routing](docs/guide/routing.md), [Cluster simulation](docs/guide/cluster.md)

### Admission and flow control

The llm-d router's admission and flow control: its legacy admission controller sheds sheddable
requests from a saturated pool, and with `featureGates: [flowControl]` in the
EndpointPickerConfig requests wait in priority bands with capacity limits and queue-wait budgets.
SLO classes reach the router as `InferenceObjective` names. SLO goodput is reported per class.

```bash
./blis run --scenarios examplescenarios --scenario gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml \
  --num-instances 4 --rate 200 --num-requests 1000
```

→ [Admission](docs/guide/admission.md)

### Prefill/decode disaggregation

A scenario with `prefill` and `decode` pools gets one kernel per pool; the kernel prices each
KV handoff over the scenario's fabric.

```bash
./blis run --scenarios examplescenarios --scenario glm-5-h200-3p1d-ib.yaml \
  --num-instances 4 --prefill-instances 3 --decode-instances 1 \
  --rate 10 --num-requests 200
```

Which requests are disaggregated is the router's decision: the EndpointPickerConfig's
`disagg-profile-handler` and its P/D decider. This scenario's `router:` block names one that
disaggregates every request.

→ [Cluster simulation](docs/guide/cluster.md)

### KV caching and offload

Block-level KV accounting with prefix caching, and tiered offload whose tiers name catalog
storage devices (`--kv-offload-config`).
→ [KV cache](docs/guide/kv-cache.md)

### Autoscaling

Horizontal autoscaling over node pools with provisioning delays, configured in a
`--policy-config` YAML (each pool's `gpu_type` must equal the scenario's hardware).
→ [Cluster simulation](docs/guide/cluster.md)

### Trace replay

Replay any trace (trace format version 4) — exported by `blis run --trace-output`, or
converted with `blis convert weka|otel` — through the same kernel and deployment path. A
version 3 trace from v0.15.0 is converted with `blis convert trace`. A run's exported trace replayed with identical flags (including `--horizon`) gives
byte-identical stdout.

```bash
./blis replay --trace-header t.yaml --trace-data d.csv \
  --scenarios examplescenarios --scenario gpt-oss-120b-h100-fp4-vllm-tp2-2node.yaml
```

→ [Trace replay](docs/guide/replay.md)

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
| [Roadmap](docs/roadmap.md) | Where BLIS is going: llm-d parity, day-0 models, multi-node, networking, accuracy, config search |

---

## Roadmap

We plan to bring BLIS to
[llm-d-perf-simulator](https://github.com/llm-d-incubation/llm-d-perf-simulator) as the
community-maintained **llm-d performance simulator**. The direction, guided by *llm-d parity by
design*, spans six themes: llm-d parity, day-0 model support, multi-node accuracy, networking
configs, accuracy transparency, and config search. See the full [Roadmap](docs/roadmap.md).

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the engineering standards, development workflow,
and guides for adding components.

## Citation

If you use BLIS in your research, please cite our paper (accepted at ACM SIGMETRICS,
to appear). Preprint: [arXiv:2610.14255](https://arxiv.org/abs/2610.14255).

```bibtex
@article{guhathakurta2026blis,
  title     = {Simulate Before You Scale: High-Fidelity LLM Inference Planning Without GPUs},
  author    = {Guhathakurta, Dipanwita and Toslali, Mert and Parthasarathy, Srinivasan and Chen, Jing and Kalantar, Michael and Masluk, Nicholas A. and Ramani, Vishakha and Tantawi, Asser and Oliveira, Fabio},
  journal   = {Proc. ACM Meas. Anal. Comput. Syst.},
  publisher = {Association for Computing Machinery},
  year      = {2026},
  note      = {To appear},
  eprint    = {2610.14255},
  archivePrefix = {arXiv},
  primaryClass  = {cs.PF}
}
```

## License

Apache License, Version 2.0. See [LICENSE](./LICENSE).
