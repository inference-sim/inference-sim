# Project Structure

The simulator uses a discrete-event architecture with a min-heap event queue.

```
inference-sim/
├── .claude/commands/          # Claude Code slash commands (speckit.specify, speckit.plan, speckit.tasks, speckit.implement, speckit.clarify, speckit.checklist, speckit.analyze, speckit.constitution, speckit.taskstoissues)
├── .specify/                  # Speckit feature-development toolkit
│   ├── memory/constitution.md # BLIS project constitution (principles, invariants, rules)
│   ├── templates/             # Spec, plan, tasks, checklist, agent-file templates
│   └── scripts/bash/          # Bash scripts: create-new-feature.sh, setup-plan.sh, update-agent-context.sh, check-prerequisites.sh, common.sh
├── .github/workflows/         # CI configuration (build, lint, test)
├── main.go                    # CLI entry point (Cobra)
├── cmd/
│   ├── root.go                # CLI commands and flags (--num-instances, --policy-config, --routing-scorers, --workload-spec, --trace-level, --fitness-weights, --kv-cpu-blocks, --kv-offload-threshold, --kv-transfer-bandwidth, --kv-transfer-base-latency, --snapshot-refresh-interval, --latency-model, --max-model-len, --trace-output)
│   ├── replay.go              # `blis replay` command: replays TraceV2 file through DES; flags: --trace-header, --trace-data (required), all sim config flags shared via registerSimConfigFlags(); --results-path writes []workload.SimResult (integer request_id, ttft_us/e2e_us in µs); SimResult type lives in sim/workload/calibrate.go
│   ├── calibrate.go           # `blis calibrate` command: compares real observed latencies (TraceV2 from blis observe) against sim predictions ([]SimResult JSON from blis replay --results-path); flags: --trace-header, --trace-data, --sim-results, --report (required), --warmup-requests (default: from header, sentinel -1), --network-rtt-us (default: from header, sentinel -1), --network-bandwidth-mbps; writes CalibrationReport JSON with MAPE/PearsonR/percentiles per metric
│   ├── observe.go             # Real mode HTTP client (RealClient with functional options: WithAPIFormat for completions/chat, stream_options for streaming usage, finish_reason extraction, configurable max_tokens); Recorder for TraceV2 output
│   ├── observe_cmd.go         # `blis observe` command: flags --server-url, --model, --api-format (completions/chat), --unconstrained-output, --rtt-ms, --workload-spec/--rate; prefix string generation (buildPrefixStrings with FNV-seeded vocabulary); dispatch orchestrator with session support
│   ├── convert.go             # `blis convert` subcommands (servegen, preset, inference-perf)
│   ├── compose.go             # `blis compose` for merging v2 specs
│   ├── hfconfig.go            # HuggingFace config resolution chain (--latency-model auto-fetch, caching)
│   └── default_config.go      # defaults.yaml loading (includes GetHFRepo for HF repo name mapping)
├── sim/                       # Core single-instance simulator
│   ├── config.go              # Module-scoped sub-config types (KVCacheConfig, BatchConfig, LatencyCoeffs, ModelHardwareConfig, PolicyConfig, WorkloadConfig) — composed into SimConfig via embedding (R16)
│   ├── doc.go                 # Package reading guide: start with request.go, event.go, simulator.go
│   ├── simulator.go           # SimConfig struct (composed of embedded sub-configs + Horizon/Seed), NewSimulator(SimConfig) (*Simulator, error) constructor (validates MaxModelLen vs KV capacity), event loop (Run()), batch formation (delegated to BatchFormation interface), step execution with phased metric recording, EnqueueRequest (MaxModelLen + KV capacity guards), processCompletions (proactive MaxModelLen cap at maxModelLen-1 boundary), observation methods (QueueDepth(), BatchSize(), CurrentClock(), SimHorizon()). All workload generation external via InjectArrival().
│   ├── admission.go           # AdmissionPolicy interface (accepts *RouterState), AlwaysAdmit, TokenBucket, RejectAll, NewAdmissionPolicy factory
│   ├── routing.go             # RoutingPolicy interface (accepts *RouterState), RoutingSnapshot (with EffectiveLoad() for canonical load calculation), RoutingDecision (with Priority hint), RoundRobin, LeastLoaded, WeightedScoring (composable scorer pipeline), AlwaysBusiest templates, NewRoutingPolicy factory
│   ├── routing_scorers.go     # ScorerConfig, scorer implementations (queue-depth, kv-utilization, load-balance), ParseScorerConfigs, IsValidScorer, DefaultScorerConfigs, newScorerWithObserver factory
│   ├── routing_prefix_scorer.go # Prefix-affinity scorer + observer (proportional prefix matching)
│   ├── prefix_cache_index.go  # PrefixCacheIndex: per-instance LRU of hierarchical block hashes
│   ├── priority.go            # PriorityPolicy interface with ConstantPriority, SLOBasedPriority, and InvertedSLO templates, NewPriorityPolicy factory
│   ├── scheduler.go           # InstanceScheduler interface with FCFSScheduler, PriorityFCFSScheduler, SJFScheduler, and ReversePriority templates, NewScheduler factory
│   ├── latency_model.go       # LatencyModel interface (3 methods), NewLatencyModelFunc registration variable, MustNewLatencyModel nil-guarded wrapper
│   ├── router_state.go        # RouterState bridge type (Snapshots + Clock) for cluster-level policies
│   ├── bundle.go              # PolicyBundle YAML loading, LoadPolicyBundle, Validate
│   ├── event.go               # Event types (Arrival, Queued, Step, Scheduled, RequestLeft, Timeout) with (timestamp, priority, seqID) ordering
│   ├── request.go             # RequestState typed constants (StateQueued, StateRunning, StateCompleted, StateTimedOut), Request lifecycle and state machine, Deadline field for client timeout, Priority field for scheduler-aware ordering, AssignedInstance for cluster routing provenance (#181), workload metadata (TenantID, SLOClass, etc.), MaxOutputLen (client output budget for enqueue guard)
│   ├── kv_store.go            # KVStore interface (12 methods: +SetClock, +ConsumePendingTransferLatency, +MirrorToCPU), NewKVStoreFromConfig registration variable, MustNewKVCacheState/MustNewKVStoreFromConfig nil-guarded wrappers
│   ├── batch.go               # Batch struct
│   ├── batch_formation.go     # BatchFormation interface, BatchContext/BatchResult types, VLLMBatchFormation (FCFS + chunked-prefill + preemption), NewBatchFormation() factory
│   ├── queue.go               # FIFO wait queue
│   ├── metrics.go             # TTFT, TPOT, E2E collection and SaveResults()
│   ├── metrics_utils.go       # Percentile/mean calculation, MetricsOutput JSON struct, NewRequestMetrics canonical constructor
│   ├── rng.go                 # PartitionedRNG for deterministic multi-subsystem simulation
│   ├── model_hardware_config.go # ModelConfig, HardwareCalib structs (config types stay in sim/); HardwareCalib includes MemoryGiB (used by KV capacity auto-calculation in roofline/crossmodel modes). ModelConfig.WeightBytesPerParam (0=fallback to BytesPerParam) with EffectiveWeightBytesPerParam() accessor decouples weight storage precision from compute/KV dtype. Note: MaxModelLen is int64 (aligned with ProgressIndex, TotalKVBlocks, BlockSizeTokens).
│   └── internal/              # Shared internal packages
│       ├── hash/              # Block-level hashing for prefix cache
│       ├── testutil/          # Shared test infrastructure (golden dataset loading)
│       └── util/              # General utility functions
├── sim/kv/                    # KV cache implementations (PKG-1)
│   ├── cache.go               # KVCacheState (single-tier GPU)
│   ├── tiered.go              # TieredKVCache (GPU+CPU mirror/reload, vLLM v1 model)
│   └── register.go            # NewKVStore factory + init()-based registration into sim/
├── sim/latency/               # Latency model implementations (PKG-2)
│   ├── latency.go             # RooflineLatencyModel (default, analytical FLOPs/bandwidth), BlackboxLatencyModel (alpha/beta regression), CrossModelLatencyModel (physics-informed cross-model), NewLatencyModel(LatencyCoeffs, ModelHardwareConfig) factory
│   ├── trained_roofline.go    # TrainedRooflineLatencyModel: roofline basis functions × learned corrections (7β + 3α from training pipeline)
│   ├── crossmodel.go          # CrossModelLatencyModel: physics-informed step time from architecture features (MoE-aware)
│   ├── roofline.go            # rooflineStepTime(), calculateTransformerFlops(), calculateMemoryAccessBytes(), StepConfig/PrefillRequestConfig/DecodeRequestConfig types
│   ├── kv_capacity.go         # CalculateKVBlocks: auto-derive total KV cache blocks from model architecture + GPU memory; KVCapacityParams, ExtractKVCapacityParams, computeModelWeightBytes
│   ├── config.go              # HFConfig, GetHWConfig(), GetModelConfig(), ValidateRooflineConfig(), parseHWConfig(), ParseHFConfig()
│   └── register.go            # init()-based registration of NewLatencyModelFunc into sim/
├── sim/cluster/               # Multi-replica cluster simulation
│   ├── instance.go            # InstanceSimulator wraps sim.Simulator via NewInstanceSimulator(id, SimConfig) with run-once guard; delegates to Simulator observation methods (QueueDepth(), BatchSize(), etc.)
│   ├── cluster.go             # ClusterSimulator orchestrates N instances with shared-clock event loop, online routing pipeline, and metrics aggregation; Run() returns error
│   ├── cluster_event.go       # ClusterArrivalEvent, AdmissionDecisionEvent, RoutingDecisionEvent
│   ├── counterfactual.go      # computeCounterfactual() for top-k candidate ranking and regret computation
│   ├── snapshot.go            # CachedSnapshotProvider (returns sim.RoutingSnapshot), ObservabilityConfig
│   ├── metrics.go             # RawMetrics, Distribution, FitnessResult, CollectRawMetrics (accepts priorityPolicy), ComputeFitness (returns (FitnessResult, error)), anomaly detection, ParseFitnessWeights with NaN/Inf validation, per-SLO-class metrics, JainFairnessIndex
│   ├── deployment.go          # DeploymentConfig embeds sim.SimConfig + cluster-only fields; ToSimConfig() returns the embedded config
│   └── evaluation.go          # EvaluationResult wrapper (RawMetrics + FitnessResult + trace + summary)
├── sim/workload/              # ServeGen-informed workload generation
│   ├── spec.go                # WorkloadSpec v2, ClientSpec (with Model field), ArrivalSpec, DistSpec, YAML loading, v1→v2 auto-upgrade (UpgradeV1ToV2), IsValidSLOClass accessor
│   ├── arrival.go             # ArrivalSampler: Poisson, Gamma (Marsaglia-Tsang), Weibull (bisection), Constant (fixed-interval)
│   ├── distribution.go        # LengthSampler: Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF, Constant
│   ├── client.go              # Rate normalization, prefix group management
│   ├── generator.go           # GenerateRequests pipeline with client decomposition
│   ├── servegen.go            # Native ServeGen data file loading (chunk-*-trace.csv + dataset.json)
│   ├── tracev2.go             # Trace v2 format (YAML header + CSV data); 26-column schema including finish_reason (backward-compat with 25-column pre-finish_reason traces)
│   ├── replay.go              # Trace v2 → sim.Request with synthetic token IDs
│   ├── calibrate.go           # CalibrationReport, PrepareCalibrationPairs, MAPE/Pearson r
│   ├── multimodal.go          # Multimodal token generation (text+image+audio+video)
│   ├── reasoning.go           # Reasoning multi-turn with context accumulation
│   ├── session.go             # SessionManager: closed-loop session tracking, follow-up round generation on completion
│   ├── network.go             # Client-perspective latency (RTT + bandwidth)
│   ├── inference_perf.go      # inference-perf format: InferencePerfSpec, expansion, validation
│   ├── scenarios.go           # Built-in presets (bursty, unfair, prefix-heavy, mixed-slo)
│   ├── convert.go             # Format converters: ConvertServeGen, ConvertPreset, ComposeSpecs
│   ├── cohort.go              # CohortSpec expansion: diurnal, spike, drain patterns → lifecycle windows
│   └── synthesis.go           # Flag-to-spec synthesis: SynthesizeFromDistribution, SynthesizeFromPreset
├── sim/trace/                 # Decision trace recording
│   ├── trace.go               # TraceLevel, TraceConfig, SimulationTrace, NewSimulationTrace, recording methods
│   ├── record.go              # AdmissionRecord, RoutingRecord, CandidateScore (pure data types, no sim/ dependency)
│   └── summary.go             # TraceSummary, Summarize()
├── model_configs/             # Auto-fetched HuggingFace config.json files (gitignored)
├── defaults.yaml              # Pre-trained coefficients, default GPU/TP/vLLM mappings, workload presets
├── hardware_config.json       # GPU specifications
├── examples/                  # Example configuration files
├── testdata/goldendataset.json # Golden dataset for regression tests
├── docs/
│   ├── getting-started/       # New user onboarding
│   │   ├── index.md           # What is BLIS?
│   │   ├── installation.md    # Build from source
│   │   ├── quickstart.md      # First simulation
│   │   └── tutorial.md        # Capacity planning walkthrough
│   ├── guide/                 # Task-oriented user guides
│   │   ├── index.md           # Guide overview
│   │   ├── routing.md         # Routing policies
│   │   ├── admission.md       # Admission control
│   │   ├── scheduling.md      # Scheduling & priority
│   │   ├── latency-models.md  # Latency models (roofline + blackbox)
│   │   ├── kv-cache.md        # KV cache & memory management
│   │   ├── workloads.md       # Workload specifications
│   │   ├── cluster.md         # Cluster simulation
│   │   ├── results.md         # Metrics & results
│   │   ├── experimentation.md # Hypothesis-driven experimentation
│   │   └── skills-and-plugins.md # Claude Code skills & plugins
│   ├── concepts/              # Architecture and design documentation
│   │   ├── index.md           # Concepts overview
│   │   ├── glossary.md        # Concepts glossary
│   │   ├── architecture.md    # Cluster architecture
│   │   ├── core-engine.md     # Core DES engine
│   │   └── roofline.md        # Roofline step time estimation
│   ├── reference/             # Configuration and model reference
│   │   ├── index.md           # Reference overview
│   │   ├── project-structure.md # Project file organization (this file)
│   │   ├── configuration.md   # Configuration reference
│   │   ├── models.md          # Supported models catalog
│   │   └── workload-spec.md   # Workload spec YAML schema
│   ├── methodology/           # Research methodology documentation
│   │   ├── index.md           # Methodology overview
│   │   ├── strategy-evolution.md # Strategy Evolution methodology guide
│   │   ├── hypothesis-bundles.md # Hypothesis bundle examples and writing guide
│   │   └── principles.md     # Discovered principles catalog (30 principles)
│   ├── contributing/          # Contributor documentation
│   │   ├── index.md           # Contributing landing page
│   │   ├── extension-recipes.md # Step-by-step extension guides
│   │   ├── pr-workflow.md     # PR development workflow
│   │   ├── design-process.md  # Design document process
│   │   ├── macro-planning.md  # Macro-level planning process
│   │   ├── hypothesis.md      # Hypothesis experiment process
│   │   ├── convergence.md     # Universal Convergence Protocol
│   │   ├── standards/         # Canonical rules, invariants, principles, experiment standards
│   │   └── templates/         # Artifact templates + agent prompts
│   │       ├── design-guidelines.md  # DES foundations, module architecture
│   │       ├── macro-plan.md         # Multi-PR template (human-readable)
│   │       ├── macro-plan-prompt.md  # Agent preamble for macro planning
│   │       ├── micro-plan.md         # Single-PR template (human-readable)
│   │       ├── micro-plan-prompt.md  # Agent preamble for writing-plans skill
│   │       └── hypothesis.md         # Experiment FINDINGS.md template
│   └── plans/                 # Active implementation plans (excluded from MkDocs)
│       └── archive/           # Completed design docs (architectural reference)
├── CONTRIBUTING.md            # Contributor guide (references docs/contributing/standards/)
└── mkdocs.yml                 # MkDocs Material site configuration
```
