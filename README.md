# Blackbox Inference Simulator (BLIS)

A discrete-event simulator for LLM inference serving systems. BLIS models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients, analytical roofline estimates, or physics-informed cross-model prediction.

The simulator is CPU-only, deterministic, and designed for capacity planning, policy optimization research, and performance prediction across model/GPU/TP configurations without requiring real GPUs.

---

## Features

### Core

- **Discrete-event simulation** for prefill, decode, and request scheduling
- **KV-cache modeling** (blocks, prefix caching, prefill chunking, tiered GPU+CPU offload)
- **CPU-only inference cost model** via analytical roofline estimation or learned α/β coefficients
- **Two latency estimation modes**: roofline (analytical) and trained-physics (physics-informed basis functions with architecture-aware MoE scaling). The deprecated `blackbox`, `crossmodel`, and `trained-roofline` backends have been removed; use `trained-physics` for modern physics-informed estimation.
- **Multi-instance cluster simulation** with shared-clock event loop and pluggable routing (round-robin, least-loaded, weighted-scoring)
- **Multiple workload types**: preset (`chatbot`, `contentgen`, `summarization`, `multidoc`), custom distributions, or trace replay

### Advanced

- **Any HuggingFace model**: dense (Llama-2, Qwen3, etc.) and MoE (Mixtral, etc.) — auto-fetches model config on first run
- **vLLM deployment configuration** (TP, chunk size, batch limits)
- **Priority policies and instance schedulers**: constant, slo-based; fcfs, priority-fcfs, sjf
- **Preemption policies**: fcfs (tail-of-batch), priority (least-urgent SLO tier evicted first)
- **Admission control**: always-admit or token-bucket rate limiting
- **YAML policy configuration**: define all policies in a single config file (`--policy-config`)
- **ServeGen-informed workload generation**: multi-client specs with Poisson/Gamma/Weibull/Constant arrivals (`--workload-spec`)
- **Decision tracing and counterfactual analysis**: record routing decisions and evaluate alternative choices
- **Fitness evaluation**: weighted multi-objective scoring with configurable metric weights
- **Per-SLO-class metrics**: breakdown by SLO class with Jain fairness index
- **Post-hoc saturation detection**: automated classification of simulation runs (STABLE/BACKLOGGED/OVERLOADED) using composite or threshold detectors

---

## Installation

**Requirements:**
- Go ≥ **1.21**

**Build the binary:**

```bash
git clone <REPOSITORY_URL>
cd blis
go build -o blis main.go
```

**Note:** On first run, BLIS auto-fetches the model's `config.json` from HuggingFace (~1 second for public models like Qwen3). Subsequent runs use the cached config in `model_configs/`. For offline use with cached configs, both roofline and trained-physics modes work without network access.

**Environment setup (optional):**

Set `HF_TOKEN` to access gated models (e.g., [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)) and avoid HuggingFace rate limits.

```bash
export HF_TOKEN=your_token_here
```

See [HuggingFace access tokens](https://huggingface.co/docs/hub/en/security-tokens) to create a token.

---

## Quick Start

Run BLIS for `qwen/qwen3-14b` with default configs (auto-fetches model config from HuggingFace):

```bash
./blis run --model qwen/qwen3-14b
```

**Hardware/TP defaults:** Omitting `--hardware` and `--tp` flags will default to H100 and TP=1 with warnings. Specify explicitly for other configurations.

You should see JSON output on stdout with key fields:

| Field | Description |
|-------|-------------|
| `ttft_mean_ms`, `ttft_p99_ms` | **Time to First Token** — how long until the first token is generated |
| `e2e_mean_ms`, `e2e_p99_ms` | **End-to-End latency** — total time from request arrival to final token |
| `itl_mean_ms`, `itl_p99_ms` | **Inter-Token Latency** — time between consecutive output tokens |
| `responses_per_sec` | Completed requests per second |
| `tokens_per_sec` | Output tokens generated per second |
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
  --routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1" \
  --rate 100 --num-requests 500
```

### Trained-physics mode (architecture-aware, no per-model calibration)

```bash
./blis run --model qwen/qwen3-14b --latency-model trained-physics 
```

Accurate across most model architectures (dense, uniform MoE, interleaved MoE) using physics-informed basis functions with learned corrections. See the [latency models guide](docs/guide/latency-models.md) for details.

### Observe real server latency

Record timing from a real inference server into a TraceV2 file:

```bash
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload-spec workload.yaml \
  --trace-header trace.yaml --trace-data trace.csv
```

For servers exposing `/v1/chat/completions` (most production vLLM/SGLang deployments), use `--api-format chat` and optionally account for network round-trip time:

```bash
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --api-format chat --rtt-ms 2.5 \
  --workload-spec workload.yaml \
  --trace-header trace.yaml --trace-data trace.csv
```

To capture per-chunk timestamps for ITL (inter-token latency) calibration, add `--record-itl`:

```bash
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 \
  --record-itl --itl-output trace.itl.csv \
  --trace-header trace.yaml --trace-data trace.csv
```

See [Workload Specifications](docs/guide/workloads.md) for the workload spec YAML schema.

### Replay traces through simulator

Replay a captured TraceV2 file through the discrete-event simulator:

```bash
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b
```

To produce per-request results for calibration, add `--results-path`:

```bash
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b \
  --results-path results.json
```

### Calibrate simulator accuracy

Compare real observed latencies against simulator predictions (using the per-request results from `blis replay --results-path`):

```bash
./blis calibrate --trace-header t.yaml --trace-data d.csv \
  --sim-results results.json --report calibration.json
```

To include ITL (inter-token latency) metric in the calibration report, add `--itl-data` (requires `blis observe --record-itl`):

```bash
./blis calibrate --trace-header t.yaml --trace-data d.csv \
  --sim-results results.json --itl-data trace.itl.csv \
  --report calibration.json
```

### Convert workload formats

```bash
# Generate a v2 workload spec YAML from a built-in preset
./blis convert preset --name chatbot --rate 10 --num-requests 100

# Import a ServeGen dataset directory (requires your own ServeGen data/)
./blis convert servegen --path data/

# Import an inference-perf workload spec
./blis convert inference-perf --spec spec.yaml
```

### Compose multiple workload specs

Merge workload spec YAMLs produced by `blis convert` or written by hand (see [Workload Specifications](docs/guide/workloads.md)):

```bash
./blis compose --from spec1.yaml --from spec2.yaml
```

For comprehensive usage guides, see the [Documentation](#documentation) section below.

### PD disaggregation (prefill/decode pool separation)

Separate prefill (prompt processing) and decode (token generation) onto dedicated instance pools, connected by a simulated KV cache transfer network. Useful for long-context workloads where prefill computation dominates, or when co-location interference is significant.

```bash
# Baseline: 4 co-located instances (no disaggregation)
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 4 --rate 100 --num-requests 1000

# PD disaggregation: 2 prefill + 2 decode, always disaggregate
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 4 \
  --prefill-instances 2 --decode-instances 2 \
  --pd-decider always \
  --pd-transfer-bandwidth 25 --pd-transfer-base-latency 0.05 \
  --rate 100 --num-requests 1000

# Selective disaggregation: only disaggregate when non-cached token count exceeds threshold
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 4 \
  --prefill-instances 2 --decode-instances 2 \
  --pd-decider prefix-threshold --pd-prefix-threshold 16 \
  --rate 100 --num-requests 1000

# Heterogeneous pools: prefill on A100-80 (high compute), decode on H100 (high memory)
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 4 \
  --prefill-instances 2 --decode-instances 2 \
  --hardware A100-80 --decode-hardware H100 \
  --pd-decider always \
  --rate 100 --num-requests 1000
```

See `examples/pd-disaggregation-demo.yaml` for a full policy configuration.

### Autoscaling

Simulate horizontal pod autoscaling (HPA) with configurable node pools, provisioning delays, and scale-up/down thresholds. Use a `--policy-config` YAML to configure the autoscaler:

```yaml
# autoscaler-demo.yaml
autoscaler:
  interval_us: 10000000        # tick every 10 seconds
  scale_up_stabilization_window_us: 0
  scale_down_stabilization_window_us: 300000000  # 5-minute cooldown (Kubernetes HPA default)
  hpa_scrape_delay:
    mean: 15.0    # 15-second scrape delay (seconds)
    stddev: 2.0
  analyzer:
    scale_up_threshold: 0.8    # scale up when KV utilization > 80%
    scale_down_boundary: 0.3   # scale down when KV utilization < 30%
    avg_input_tokens: 512

node_pools:
  - name: standard
    gpu_type: H100
    gpus_per_node: 8
    gpu_memory_gib: 80
    initial_nodes: 1
    min_nodes: 1
    max_nodes: 8
    provisioning_delay:
      mean: 120.0    # 2-minute provisioning delay (seconds)
      stddev: 30.0
    cost_per_hour: 32.77

admission:
  policy: always-admit

routing:
  policy: weighted
  scorers:
    - name: queue-depth
      weight: 1.0
    - name: kv-utilization
      weight: 1.0
```

```bash
# Run with autoscaler enabled via policy config
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 1 \
  --policy-config autoscaler-demo.yaml \
  --workload-spec examples/regression_workload_load_spikes.yaml

# Override the autoscaler tick interval from the CLI
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 1 \
  --policy-config autoscaler-demo.yaml \
  --model-autoscaler-interval-us 5000000 \
  --workload-spec examples/regression_workload_load_spikes.yaml
```

The standard output fields (`responses_per_sec`, `e2e_mean_ms`, `e2e_p99_ms`, etc.) reflect the autoscaler's effect: watch throughput and latency change as instances are added or removed during the simulation horizon.

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

> For the authoritative file-level architecture documentation with interface names, method signatures, and module descriptions, see [`docs/reference/project-structure.md`](docs/reference/project-structure.md).

<details>
<summary>Click to expand full directory tree</summary>

```
inference-sim/
├── main.go                 # CLI entry point
├── cmd/                    # CLI commands
│   ├── root.go             # CLI commands and flags (--num-instances, --policy-config, --routing-scorers, --workload-spec, --latency-model, etc.)
│   ├── replay.go           # `blis replay` command: replays TraceV2 file through DES
│   ├── calibrate.go        # `blis calibrate` command: compares real vs simulated latencies
│   ├── observe.go          # Real-mode HTTP client (RealClient with functional options); Recorder for TraceV2 output
│   ├── observe_cmd.go      # `blis observe` command: flags, prefix string generation, dispatch orchestrator
│   ├── convert.go          # `blis convert` subcommands (servegen, preset, inference-perf)
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
│   ├── latency.go          # RooflineLatencyModel, TrainedPhysicsLatencyModel, NewLatencyModel factory
│   ├── trained_physics_model.go # TrainedPhysicsLatencyModel: physics-informed basis functions with architecture-aware scaling
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
│   ├── session.go          # SessionManager: closed-loop session tracking, follow-up round generation
│   ├── network.go          # Client-perspective latency (RTT + bandwidth)
│   ├── inference_perf.go   # inference-perf format loading and validation
│   ├── scenarios.go        # Built-in presets (bursty, unfair, prefix-heavy, mixed-slo)
│   ├── cohort.go           # CohortSpec expansion: diurnal, spike, drain patterns
│   ├── convert.go          # Format converters: ConvertServeGen, ConvertPreset, ComposeSpecs
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

</details>

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for the engineering standards, development workflow, and step-by-step guides for adding new components. For ongoing work and architectural decisions, see `docs/plans/`.

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE) for details.
