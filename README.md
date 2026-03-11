# Blackbox Inference Simulator (BLIS)

A discrete-event simulator for LLM inference serving systems. BLIS models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients, analytical roofline estimates, or physics-informed cross-model prediction.

The simulator is CPU-only, deterministic, and designed for capacity planning, policy optimization research, and performance prediction across model/GPU/TP configurations without requiring real GPUs.

---

## Features

- **Discrete-event simulation** for prefill, decode, and request scheduling
- **KV-cache modeling** (blocks, prefix caching, prefill chunking, tiered GPU+CPU offload)
- **CPU-only inference cost model** via analytical roofline estimation or learned α/β coefficients
- **HuggingFace config.json support** for model architecture
- **Dense and MoE model support** (Mixtral, DeepSeek-MoE, etc.)
- **vLLM deployment configuration** (TP, PP, EP, batch limits)
- **Four latency estimation modes**: roofline (default, analytical), blackbox (data-driven), cross-model (physics-informed, MoE-aware), and trained-roofline (roofline × learned corrections)
- **Multiple workload types**: preset (`chatbot`, `contentgen`, `summarization`, `multidoc`) or custom distributions
- **Trace replay**: replay recorded request traces for deterministic testing
- **Multi-instance cluster simulation** with shared-clock event loop
- **Pluggable routing policies**: round-robin, least-loaded, weighted-scoring
- **Priority policies**: constant, slo-based (request prioritization)
- **Instance schedulers**: fcfs, priority-fcfs, sjf (batch formation policies)
- **Admission control**: always-admit or token-bucket rate limiting
- **YAML policy configuration**: define all policies in a single config file (`--policy-config`)
- **ServeGen-informed workload generation**: multi-client specs with Poisson/Gamma/Weibull/Constant arrivals (`--workload-spec`)
- **Decision tracing and counterfactual analysis**: record routing decisions and evaluate alternative choices
- **Fitness evaluation**: weighted multi-objective scoring with configurable metric weights
- **Per-SLO-class metrics**: breakdown by SLO class with Jain fairness index

---

## Installation

**Requirements:**
- Go ≥ **1.21**

**Build the binary:**

```bash
git clone git@github.com:inference-sim/inference-sim.git
cd inference-sim
go build -o blis main.go
```

**Environment setup (optional):**

Set `HF_TOKEN` to access gated models (e.g., [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)) and avoid HuggingFace rate limits. Public models like Qwen3-14B work without a token.

```bash
export HF_TOKEN=your_token_here
```

See [HuggingFace access tokens](https://huggingface.co/docs/hub/en/security-tokens) to create a token.

---

## Supported Models

BLIS supports any dense transformer model with a HuggingFace `config.json` — the default roofline mode auto-fetches configs on first use. MoE (Mixture-of-Experts) models are also supported. The following models have been validated end-to-end:

- [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) / [Llama-2-70B](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
- [Mixtral-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) (MoE)
- [CodeLlama-34B](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)

For blackbox mode (`--latency-model blackbox`), which uses pre-trained per-model coefficients, see the [supported models catalog](docs/reference/models.md#blackbox-coefficient-catalog).

---

## Quick Start

Run BLIS for `qwen/qwen3-14b` with default configs (roofline mode auto-fetches model config from HuggingFace):

```bash
./blis run --model qwen/qwen3-14b
```

You should see JSON output on stdout with key fields:

| Field | Description |
|-------|-------------|
| `ttft_mean_ms`, `ttft_p99_ms` | **Time to First Token** — how long until the first token is generated |
| `e2e_mean_ms`, `e2e_p99_ms` | **End-to-End latency** — total time from request arrival to final token |
| `itl_mean_ms`, `itl_p99_ms` | **Inter-Token Latency** — time between consecutive output tokens |
| `responses_per_sec` | Completed requests per simulated second |
| `tokens_per_sec` | Output tokens generated per simulated second |
| `completed_requests` | Number of requests that finished within the simulation window |
| `preemption_count` | Number of times a running request was evicted to make room for others (0 = healthy) |

---

## Usage

### Multi-client workload specification

```bash
./blis run --model qwen/qwen3-14b --workload-spec examples/servegen-language.yaml
```

### Cluster simulation with weighted routing

```bash
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --routing-policy weighted \
  --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"
```

### Blackbox mode (explicit trained coefficients)

```bash
./blis run --model qwen/qwen3-14b --latency-model blackbox
```

> **Tip:** Roofline, trained-roofline, and cross-model modes auto-fetch model configs from HuggingFace. Set `HF_TOKEN` to access gated models (e.g., LLaMA) and avoid rate limits:
>
> ```bash
> export HF_TOKEN=your_token_here
> ```
>
> See [HuggingFace access tokens](https://huggingface.co/docs/hub/en/security-tokens) to create a token.

> **Note:** Since roofline is the default latency model, `HF_TOKEN` may be needed for the default run mode when using gated models. For public models like Qwen3, no token is required.

### Convert workload formats

```bash
blis convert preset --name chatbot --rate 10 --num-requests 100
blis convert csv-trace --file trace.csv
blis convert servegen --path data/
```

### Compose multiple workload specs

```bash
blis compose --from spec1.yaml --from spec2.yaml
```

For comprehensive usage guides, see the [Documentation](#documentation) section below.

---

## Documentation

BLIS has a comprehensive documentation site built with MkDocs Material:

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting-started/index.md) | Installation, quick start, capacity planning tutorial |
| [User Guide](docs/guide/index.md) | Routing policies, KV cache, roofline mode, workloads, cluster simulation, interpreting results |
| [Concepts](docs/concepts/index.md) | Architecture, core engine, roofline estimation, glossary |
| [Reference](docs/reference/index.md) | CLI flag reference, supported models, workload spec YAML schema |
| [Methodology](docs/methodology/index.md) | Strategy Evolution methodology, discovered principles |
| [Contributing](docs/contributing/index.md) | Extension recipes, PR workflow, design process, standards |

---

## Project Structure

> For the authoritative file-level architecture documentation with interface names, method signatures, and module descriptions, see [`CLAUDE.md`](./CLAUDE.md).

```
inference-sim/
├── main.go                 # CLI entry point
├── cmd/                    # CLI commands
│   ├── root.go             # CLI flags (--policy-config, --routing-policy, --workload-spec, --latency-model, etc.)
│   ├── observe.go          # Real-mode HTTP client for observe-predict-calibrate
│   ├── convert.go          # `blis convert` subcommands (servegen, csv-trace, preset, inference-perf)
│   ├── compose.go          # `blis compose` for merging v2 specs
│   ├── hfconfig.go         # HuggingFace config resolution (--latency-model auto-fetch into model_configs/)
│   └── default_config.go   # defaults.yaml loading (includes GetHFRepo for HF repo mapping)
├── sim/                    # Core simulation engine
│   ├── config.go           # Module-scoped sub-config types (R16)
│   ├── doc.go              # Package reading guide
│   ├── simulator.go        # Discrete-event simulation loop
│   ├── admission.go        # Admission policy interface and templates
│   ├── routing.go          # Routing policy interface and templates
│   ├── routing_scorers.go  # ScorerConfig, stateless scorers, ParseScorerConfigs
│   ├── routing_prefix_scorer.go # Prefix-affinity scorer + observer
│   ├── prefix_cache_index.go # PrefixCacheIndex: per-instance LRU of block hashes
│   ├── priority.go         # Priority policy interface and templates
│   ├── scheduler.go        # Instance scheduler interface and templates
│   ├── latency_model.go    # LatencyModel interface and registration
│   ├── router_state.go     # RouterState bridge type for cluster-level policies
│   ├── bundle.go           # PolicyBundle YAML configuration
│   ├── event.go            # Event types (Arrival, Queued, Step, Scheduled, Preemption, RequestLeft)
│   ├── kv_store.go         # KVStore interface and registration variables
│   ├── batch.go            # Batch struct
│   ├── batch_formation.go  # BatchFormation interface, VLLMBatchFormation
│   ├── queue.go            # FIFO wait queue
│   ├── request.go          # Request lifecycle
│   ├── metrics.go          # TTFT, TPOT, E2E collection
│   ├── metrics_utils.go    # MetricsOutput JSON struct, percentile calculations
│   ├── rng.go              # PartitionedRNG for deterministic simulation
│   ├── model_hardware_config.go  # ModelConfig, HardwareCalib structs
│   └── internal/           # Shared internal packages (hash, testutil, util)
├── sim/kv/                 # KV cache implementations
│   ├── cache.go            # KVCacheState (single-tier GPU)
│   ├── tiered.go           # TieredKVCache (GPU+CPU)
│   └── register.go         # NewKVStore factory + init()-based registration into sim/
├── sim/latency/            # Latency model implementations
│   ├── latency.go          # BlackboxLatencyModel, RooflineLatencyModel, CrossModelLatencyModel, NewLatencyModel factory
│   ├── crossmodel.go       # CrossModelLatencyModel: physics-informed step time from architecture features
│   ├── roofline.go         # Analytical FLOPs/bandwidth latency estimation
│   ├── config.go           # HFConfig, GetHWConfig, GetModelConfig, ValidateRooflineConfig
│   ├── kv_capacity.go      # KV cache block auto-calculation from model architecture + GPU memory
│   └── register.go         # init()-based registration into sim/
├── sim/cluster/            # Multi-replica cluster simulation
│   ├── cluster.go          # Shared-clock event loop, online routing
│   ├── instance.go         # Per-instance simulator wrapper
│   ├── cluster_event.go    # Cluster-level event types
│   ├── snapshot.go         # Instance observability snapshots
│   ├── metrics.go          # RawMetrics, FitnessResult, anomaly detection, per-SLO-class metrics
│   ├── counterfactual.go   # Top-k candidate ranking and regret computation
│   ├── deployment.go       # DeploymentConfig (embeds SimConfig + cluster fields)
│   └── evaluation.go       # EvaluationResult wrapper (metrics + trace + summary)
├── sim/workload/           # ServeGen-informed workload generation
│   ├── spec.go             # WorkloadSpec, ClientSpec, ArrivalSpec, DistSpec, YAML loading
│   ├── arrival.go          # ArrivalSampler: Poisson, Gamma, Weibull, Constant
│   ├── distribution.go     # LengthSampler: Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF, Constant
│   ├── client.go           # Rate normalization, prefix group management
│   ├── generator.go        # GenerateRequests pipeline with client decomposition
│   ├── servegen.go         # Native ServeGen data file loading
│   ├── tracev2.go          # Trace v2 format (YAML header + CSV data)
│   ├── replay.go           # Trace v2 → sim.Request with synthetic token IDs
│   ├── calibrate.go        # CalibrationReport, MAPE, Pearson r
│   ├── multimodal.go       # Multimodal token generation (text+image+audio+video)
│   ├── reasoning.go        # Reasoning multi-turn with context accumulation
│   ├── network.go          # Client-perspective latency (RTT + bandwidth)
│   ├── inference_perf.go   # inference-perf format loading and validation
│   ├── scenarios.go        # Built-in presets (bursty, unfair, prefix-heavy, mixed-slo)
│   ├── cohort.go           # CohortSpec expansion: diurnal, spike, drain patterns
│   ├── convert.go          # Format converters: ConvertServeGen, ConvertCSVTrace, ConvertPreset
│   └── synthesis.go        # Flag-to-spec synthesis: SynthesizeFromDistribution, SynthesizeFromPreset
├── sim/trace/              # Decision trace recording
│   ├── trace.go            # TraceLevel, TraceConfig, SimulationTrace
│   ├── record.go           # AdmissionRecord, RoutingRecord, CandidateScore
│   └── summary.go          # TraceSummary, Summarize()
├── examples/               # Example configuration files
│   ├── policy-config.yaml
│   ├── weighted-routing.yaml
│   ├── routing-comparison.sh
│   ├── servegen-language.yaml
│   ├── prefix-affinity-demo.yaml
│   ├── multiturn-chat-demo.yaml
│   ├── epp-estimate-prefix.yaml
│   ├── epp-precise-prefix.yaml
│   ├── inference-perf-shared-prefix.yaml
│   ├── regression_workload_cache_warmup.yaml
│   ├── regression_workload_load_spikes.yaml
│   └── regression_workload_multiturn.yaml
├── model_configs/          # Auto-fetched HuggingFace config.json files (gitignored)
├── defaults.yaml           # Pre-trained coefficients, model defaults
├── hardware_config.json    # GPU hardware specifications
├── docs/                   # Documentation (MkDocs Material site)
│   ├── getting-started/    # New user onboarding
│   ├── guide/              # Task-oriented user guides
│   ├── concepts/           # Architecture and design documentation
│   ├── reference/          # Configuration and model reference
│   ├── methodology/        # Research methodology
│   ├── contributing/       # Contributor documentation
│   └── plans/              # Active implementation plans
└── mkdocs.yml              # MkDocs Material site configuration
```

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for the engineering standards, development workflow, and step-by-step guides for adding new components. For ongoing work and architectural decisions, see `docs/plans/`.

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE) for details.
