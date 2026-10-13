# Project Structure

The simulator uses a discrete-event architecture with a min-heap event queue.

```
inference-sim/
├── .claude/skills/            # Claude Code skills (blis-pr-review, issue-review)
├── .github/workflows/         # CI configuration (build, lint, test)
├── main.go                    # CLI entry point (Cobra)
├── cmd/
│   ├── root.go                # CLI commands and flags (--scenario, --scenarios, --registry, --num-instances, --policy-config, --routing-scorers, --workload-spec, --trace-level, --fitness-weights, --kv-cpu-blocks, --kv-offload-threshold, --snapshot-refresh-interval, --trace-output, --preemption-policy)
│   ├── kernel_pools.go        # Builds each pool's deployment from the scenario: one blis-latency-kernel per pool, engine settings and KV budget from the scenario + kernel, P/D handoff priced by the kernel
│   ├── kv_offload.go          # --kv-offload-config loading; each tier's device_class resolves against the catalog and the kernel prices the transfer
│   ├── replay.go              # `blis replay` command: replays TraceV2 file through DES; flags: --trace-header, --trace-data (required), all sim config flags (including --scenario/--scenarios/--registry) shared via registerSimConfigFlags(); --results-path writes []workload.SimResult (integer request_id, ttft_us/e2e_us in µs); SimResult type lives in sim/workload/calibrate.go
│   ├── network_topology.go    # crossNodeSpanForTrace: normalizes a cluster's widest instance node span for the TraceV2 header (#1530). A span of 0 (no node pools) or 1 (every instance on one node) records nothing, so a run without multi-node placement writes a byte-identical header; only a real multi-node fleet is recorded, and only that makes `blis replay` refuse the trace.
│   ├── calibrate.go           # `blis calibrate` command (deprecated, #1901): compares real observed latencies (TraceV2 from blis observe) against sim predictions ([]SimResult JSON from blis replay --results-path); flags: --trace-header, --trace-data, --sim-results, --report (required), --warmup-requests (default: from header, sentinel -1), --network-rtt-us (default: from header, sentinel -1), --network-bandwidth-mbps; writes CalibrationReport JSON with MAPE/PearsonR/percentiles per metric
│   ├── observe.go             # Real mode HTTP client (RealClient with functional options: WithAPIFormat for completions/chat, stream_options for streaming usage, finish_reason extraction, configurable max_tokens); Recorder for TraceV2 output
│   ├── observe_cmd.go         # `blis observe` command (deprecated, #1901): flags --server-url, --model, --api-format (completions/chat), --unconstrained-output, --rtt-ms, --workload-spec/--rate; prefix string generation (buildPrefixStrings with FNV-seeded vocabulary); dispatch orchestrator with session support
│   ├── convert.go             # `blis convert` subcommands (servegen, preset, inference-perf)
│   ├── compose.go             # `blis compose` for merging v2 specs
│   ├── hfconfig.go            # Catalog LOCATION (--catalog / BLIS_CATALOG, no default and no search path — #1731) and read-only lookup of a model's config.json within it (refuses an uncatalogued model)
│   ├── catalog_provenance.go  # Catalog provenance for the results file (#1732): captures the resolved catalog path, its git revision (`rev-parse HEAD`) and a dirty flag (`status --porcelain` scoped to the catalog subtree), degrading to revision "unknown" for a non-git or absent catalog. catalogProvenanceEmitOptions is the single shared helper both `blis run` and `blis replay` pass to sim.EmitOutput (file-only, INV-6/INV-13)
│   ├── catalog_devices.go     # Catalog device namespace reader (#1770): the KV-offload storage-device physics table at `<catalog>/devices/storage.yaml` (a sibling of `models/`), which `--kv-offload-config`'s per-tier `device_class` resolves against. Read LAZILY (only when a tier names a class) and strictly (R10); an absent/malformed/empty table is refused naming the path
│   ├── catalog_workloads.go   # Named workload presets read from the catalog: <catalog>/workloads/<name>.yaml (#1769). The single reader and PresetConfig construction site behind `blis run --workload`, `blis convert preset` and `blis observe --workload`; strict KnownFields parsing (R10), refusals name the path and list the presets the catalog defines
│   ├── catalog_load.go        # Strict WHOLE-catalog load behind the R1/C6 CI gate (#1750): loads every entry through the code path a run uses (models, presets via readCatalogPresetWorkload, devices via loadCatalogStorageDevices), plus strict readers for the two namespaces no run reads yet (`models/*/model.yaml`, `hardware/*.yaml`). Requires both halves of a model entry, refuses a GPU/tensor-parallel key in any catalog-authored YAML, accumulates and sorts problems (INV-6). Driven by scripts/catalog-load-gate.sh; there is deliberately NO validate subcommand
│   └── default_config.go      # defaults.yaml loading: the LoRA cost coefficients (and `version`) only
├── internal/                  # Repo-root internal packages (test tooling; not importable outside this module)
│   └── invariantscan/         # AST scan for hand-rolled INV-1 conservation sums (#1720) — used by the guard tests in sim/, sim/cluster/. Root-level rather than sim/internal so cmd/ can import it too.
├── sim/                       # Core single-instance simulator
│   ├── config.go              # Module-scoped sub-config types (KVCacheConfig, BatchConfig, PolicyConfig, WorkloadConfig, ...) — composed into SimConfig via embedding (R16).
│   ├── doc.go                 # Package reading guide: start with request.go, event.go, simulator.go
│   ├── simulator.go           # SimConfig struct (composed of embedded sub-configs + Horizon/Seed), NewSimulator(SimConfig) (*Simulator, error) constructor (validates MaxModelLen vs KV capacity), event loop (Run()), batch formation (delegated to BatchFormation interface), step execution with phased metric recording, EnqueueRequest (MaxModelLen + KV capacity guards), processCompletions (proactive MaxModelLen cap at maxModelLen-1 boundary), observation methods (QueueDepth(), BatchSize(), CurrentClock(), SimHorizon()). All workload generation external via InjectArrival().
│   ├── admission.go           # AdmissionPolicy interface (accepts *RouterState), AlwaysAdmit, TokenBucket, RejectAll, NewAdmissionPolicy factory
│   ├── routing.go             # RoutingPolicy interface (accepts *RouterState), RoutingSnapshot (with EffectiveLoad() used by load-balance, least-loaded, and admission), RoutingDecision (TargetInstance + Reason + Scores), RoundRobin, LeastLoaded, WeightedScoring (composable scorer pipeline), AlwaysBusiest templates, NewRoutingPolicy factory
│   ├── routing_scorers.go     # ScorerConfig, scorer implementations (queue-depth, kv-utilization, load-balance), ParseScorerConfigs, IsValidScorer, DefaultScorerConfigs, newScorerWithObserver factory
│   ├── routing_prefix_scorer.go # Prefix-affinity scorer + observer (proportional prefix matching)
│   ├── prefix_cache_index.go  # PrefixCacheIndex: per-instance LRU of hierarchical block hashes
│   ├── slo_priority.go        # SLOPriorityMap: BLIS/llm-d convention (higher=urgent) for cluster layer; InvertForVLLM() converts to vLLM convention (lower=urgent) at instance entry (EnqueueRequest)
│   ├── scheduler.go           # InstanceScheduler interface with FCFSScheduler, PriorityFCFSScheduler, SJFScheduler, and ReversePriority templates, NewScheduler factory
│   ├── latency_model.go       # LatencyModel interface (3 methods), NewLatencyModelFunc registration variable, MustNewLatencyModel nil-guarded wrapper
│   ├── router_state.go        # RouterState bridge type (Snapshots + Clock) for cluster-level policies
│   ├── bundle.go              # PolicyBundle YAML loading, LoadPolicyBundle, Validate
│   ├── event.go               # Event types (Arrival, Queued, Step, Scheduled, RequestLeft, Timeout) with (timestamp, priority, seqID) ordering
│   ├── request.go             # RequestState typed constants (StateQueued, StateRunning, StateCompleted, StateTimedOut), Request lifecycle and state machine, Deadline field for client timeout, Priority field for scheduler-aware ordering, AssignedInstance for cluster routing provenance (#181), workload metadata (TenantID, SLOClass, etc.), MaxOutputLen (client output budget for enqueue guard)
│   ├── kv_store.go            # KVStore interface (12 methods: +SetClock, +ConsumePendingTransferLatency, +MirrorToCPU), NewKVStoreFromConfig registration variable, MustNewKVCacheState/MustNewKVStoreFromConfig nil-guarded wrappers
│   ├── batch.go               # Batch struct
│   ├── batch_formation.go     # BatchFormation interface, BatchContext/BatchResult types, VLLMBatchFormation (fcfs/priority preemption + chunked-prefill), NewBatchFormation(preemptionPolicy string) factory
│   ├── queue.go               # FIFO wait queue
│   ├── metrics.go             # TTFT, TPOT, E2E collection and SaveResults()
│   ├── metrics_utils.go       # Percentile/mean calculation, MetricsOutput JSON struct, NewRequestMetrics canonical constructor
│   ├── rng.go                 # PartitionedRNG for deterministic multi-subsystem simulation
│   └── internal/              # Shared internal packages
│       ├── hash/              # Block-level hashing for prefix cache (the sole hash source)
│       ├── kvkey/             # KV block keying (#1589): DeriveChunkKeys (chunk-stride keys, BC-K4) + Interner (BlockKey→KeyID, BC-K3); delegates all hashing to internal/hash (BC-K1)
│       ├── testutil/          # Shared test infrastructure (golden dataset loading)
│       ├── tokenid/           # TokenID (int32) type — defined-type compact token ID shared with sim/internal/hash
│       └── util/              # General utility functions
├── sim/kv/                    # KV cache implementations (PKG-1)
│   ├── cache.go               # KVCacheState (single-tier GPU)
│   ├── tiered.go              # TieredKVCache (GPU+CPU mirror/reload, vLLM v1 model)
│   └── register.go            # NewKVStore factory + init()-based registration into sim/
├── sim/kernelmodel/           # Adapter from blis-latency-kernel (github.com/inference-sim/blis-latency-kernel v0.1.0) to sim.LatencyModel. Translation only: units (ns → µs ticks), the StepTime >= 1 floor, and BLIS batch → kernel request shapes; every latency number comes from the kernel
│   ├── kernelmodel.go         # The LatencyModel adapter
│   ├── capacity.go            # KV block budget from the kernel's memory methods, per data-parallel rank
│   ├── settings.go            # Settings: everything one engine is sized from (block size, batch limits, max_model_len, KV budget), per data-parallel rank, answered by the scenario + kernel
│   ├── roots.go               # The artifact roots (scenarios, catalog, registry) a kernel-path command or test reads
│   └── harness/               # Closed-loop runs against the AISimulate accuracy corpus (used by cmd/metricscore)
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
│   ├── tracev2.go             # Trace v2 format (YAML header + CSV data); 27-column schema including finish_reason (backward-compat with 26-column pre-finish_reason traces)
│   ├── replay.go              # Trace v2 → sim.Request with synthetic token IDs
│   ├── reexport.go            # ReExportClosedLoopRecords: faithful closed-loop/pool --trace-output re-export (#1630) — captures all rounds, re-derives accumulate deltas+resets, propagates think/prefix
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
│                               # (The model catalog is NOT in this repo: it is the external blis-catalog
│                               #  repository, located at run time via --catalog / BLIS_CATALOG (#1731, #1771).
│                               #  It holds model graphs, chips, fabrics, storage devices and workload
│                               #  presets. A verbatim
│                               #  subset of a tagged release is vendored at testdata/catalog/ for tests,
│                               #  blis-registry's coefficient sets at testdata/registry/, and this repo's
│                               #  own scenario fixtures (P/D, MTP) at testdata/scenarios/.)
├── defaults.yaml              # LoRA cost coefficients only (latency coefficients live in blis-registry)
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
│   │   ├── latency-models.md  # Latency pricing (blis-latency-kernel)
│   │   ├── kv-cache.md        # KV cache & memory management
│   │   ├── workloads.md       # Workload specifications
│   │   ├── cluster.md         # Cluster simulation
│   │   ├── results.md         # Metrics & results
│   │   └── skills-and-plugins.md # Claude Code skills & plugins
│   ├── concepts/              # Architecture and design documentation
│   │   ├── index.md           # Concepts overview
│   │   ├── glossary.md        # Concepts glossary
│   │   ├── architecture.md    # Cluster architecture
│   │   └── core-engine.md     # Core DES engine
│   ├── reference/             # Configuration and model reference
│   │   ├── index.md           # Reference overview
│   │   ├── project-structure.md # Project file organization (this file)
│   │   ├── configuration.md   # Configuration reference
│   │   ├── models.md          # Model compatibility and validation
│   │   └── workload-spec.md   # Workload spec YAML schema
│   │   ├── index.md           # Methodology overview
│   │   └── principles.md     # Discovered principles catalog (30 principles)
│   ├── contributing/          # Contributor documentation
│   │   ├── index.md           # Contributing landing page
│   │   ├── extension-recipes.md # Step-by-step extension guides
│   │   ├── pr-workflow.md     # PR development workflow
│   │   ├── rfc.md             # RFC template for large features
│   │   ├── standards/         # Canonical rules, invariants, principles, experiment standards
│   │   └── templates/         # Artifact templates
│   │       ├── design-guidelines.md  # DES foundations, module architecture
│   ├── templates/             # Claude prompt templates
│   │   └── rfc-to-plan.md    # Prompt: encode .archon + create sub-issues
│   └── plans/                 # Active implementation plans (excluded from MkDocs)
│       └── archive/           # Completed design docs (architectural reference)
├── CONTRIBUTING.md            # Contributor guide (references docs/contributing/standards/)
└── mkdocs.yml                 # MkDocs Material site configuration
```
