# Project Structure

The simulator uses a discrete-event architecture with a min-heap event queue. Its engine is
[blis-vllm](https://github.com/inference-sim/blis-vllm), reached through blis-schemas'
`engine.Engine` interface; the paths below are this repository's (`git ls-files`).

```
inference-sim/
├── .claude/skills/            # Claude Code skills (blis-pr-review, issue-review)
├── .github/workflows/         # CI configuration (build, lint, test)
├── main.go                    # CLI entry point (Cobra)
├── cmd/
│   ├── root.go                # CLI commands and flags (--scenario, --scenarios, --registry, --num-instances, --policy-config, --workload-spec, --trace-level, --counterfactual-k, --fitness-weights, --kv-cpu-blocks, --kv-offload-threshold, --trace-output, --preemption-policy)
│   ├── router.go              # The llm-d router's inputs: the scenario's router block and pool labels, plus --gateway-latency, --epp-scrape-jitter, --kv-event-delay on run and replay; records them in the TraceV2 header and rebuilds them on replay (INV-13); builds the sim/llmd gateway
│   ├── engine_build.go        # The one place cmd builds blis-vllm (#2021): hands the scenario and the engine flags, unread, to blis-vllm compat/build, which opens, checks and prices the pools, resolves the KV offload (scenario block + --kv-offload-config), the engine knobs, routing imbalance and speculative acceptance, and reads the catalog's storage-device table
│   ├── kv_offload.go          # --kv-offload-config flag and the KV-offload trace-header round trip for replay (the resolution itself is blis-vllm compat/build's, via engine_build.go)
│   ├── replay.go              # `blis replay` command: replays TraceV2 file through DES; flags: --trace-header, --trace-data (required), all sim config flags (including --scenario/--scenarios/--registry) shared via registerSimConfigFlags(); --results-path writes []workload.SimResult (integer request_id, ttft_us/e2e_us in µs); SimResult type lives in sim/workload/calibrate.go
│   ├── network_topology.go    # crossNodeSpanForTrace: normalizes a cluster's widest instance node span for the TraceV2 header (#1530). A span of 0 (no node pools) or 1 (every instance on one node) records nothing, so a run without multi-node placement writes a byte-identical header; only a real multi-node fleet is recorded, and only that makes `blis replay` refuse the trace.
│   ├── calibrate.go           # `blis calibrate` command (deprecated, #1901): compares real observed latencies (TraceV2 from blis observe) against sim predictions ([]SimResult JSON from blis replay --results-path); flags: --trace-header, --trace-data, --sim-results, --report (required), --warmup-requests (default: from header, sentinel -1), --network-rtt-us (default: from header, sentinel -1), --network-bandwidth-mbps; writes CalibrationReport JSON with MAPE/PearsonR/percentiles per metric
│   ├── observe.go             # Real mode HTTP client (RealClient with functional options: WithAPIFormat for completions/chat, stream_options for streaming usage, finish_reason extraction, configurable max_tokens); Recorder for TraceV2 output
│   ├── observe_cmd.go         # `blis observe` command (deprecated, #1901): flags --server-url, --model, --api-format (completions/chat), --unconstrained-output, --rtt-ms, --workload-spec/--rate; prefix string generation (buildPrefixStrings with FNV-seeded vocabulary); dispatch orchestrator with session support
│   ├── convert.go             # `blis convert` subcommands (servegen, preset, inference-perf)
│   ├── compose.go             # `blis compose` for merging v2 specs
│   ├── catalog_root.go        # Catalog LOCATION (--catalog / BLIS_CATALOG, no default and no search path — #1731); a model's config.json is read within it
│   ├── catalog_provenance.go  # Catalog provenance for the results file (#1732): captures the resolved catalog path, its git revision (`rev-parse HEAD`) and a dirty flag (`status --porcelain` scoped to the catalog subtree), degrading to revision "unknown" for a non-git or absent catalog. catalogProvenanceEmitOptions is the single shared helper both `blis run` and `blis replay` pass to sim.EmitOutput (file-only, INV-6/INV-13)
│   ├── catalog_workloads.go   # Named workload presets read from the catalog: <catalog>/workloads/<name>.yaml (#1769). The single reader and PresetConfig construction site behind `blis run --workload`, `blis convert preset` and `blis observe --workload`; strict KnownFields parsing (R10), refusals name the path and list the presets the catalog defines
│   ├── catalog_load.go        # Strict WHOLE-catalog load behind the R1/C6 CI gate (#1750): loads every entry through the code path a run uses (models, presets via readCatalogPresetWorkload, devices via loadCatalogStorageDevices), plus strict readers for the two namespaces no run reads yet (`models/*/model.yaml`, `hardware/*.yaml`). Requires both halves of a model entry, refuses a GPU/tensor-parallel key in any catalog-authored YAML, accumulates and sorts problems (INV-6). Driven by scripts/catalog-load-gate.sh; there is deliberately NO validate subcommand
│   └── default_config.go      # defaults.yaml loading: the LoRA cost coefficients (and `version`) only
├── internal/                  # Repo-root internal packages (test tooling; not importable outside this module)
│   └── invariantscan/         # AST scan for hand-rolled INV-1 conservation sums (#1720) — used by the guard tests in sim/, sim/cluster/. Root-level rather than sim/internal so cmd/ can import it too.
├── sim/                       # The host of one instance: clock, event queue, timeouts and metrics; the engine is blis-vllm's, through engine.Engine
│   ├── doc.go                 # Package overview
│   ├── engine_driver.go       # EngineDriver: drives one engine.Engine from the host's event loop (Add, Schedule, StepTime, Complete, Finish, NextWake) and records the report
│   ├── engine_driver_group.go # A data-parallel group's ranks stepped as one engine
│   ├── engine_driver_handoff.go # P/D: the decode side's remote prefill and KV handoff through the engine
│   ├── engine_records.go      # The engine's results as the host's records
│   ├── event_queue.go         # The host's event queue: (timestamp, priority, seqID) ordering
│   ├── instance_config.go     # InstanceConfig: the host's settings for one instance, and the engine's configuration held unopened
│   ├── request.go             # Request: the host's view of one request (states, timestamps, SLO class)
│   ├── renderer.go            # The API-server front end: rendering and tokenization before a request reaches the engine
│   ├── progress_hook.go       # ProgressHook: periodic read-only state snapshots during a run
│   ├── recorder.go            # The *Metrics side of the engine's recorder seam
│   ├── routing.go             # RoutingSnapshot: a lightweight view of one instance's live state (EffectiveLoad()), read by the model autoscaler; routing itself is the llm-d router's
│   ├── router_state.go        # RouterState bridge type (Snapshots + Clock) built by the cluster for its own policies
│   ├── bundle.go              # PolicyBundle YAML loading (priority, scheduler, preemption, node_pools, autoscaler, instance_lifecycle), LoadPolicyBundle, Validate
│   ├── metrics.go             # TTFT, TPOT, E2E collection and SaveResults()
│   ├── metrics_utils.go       # Percentile/mean calculation, MetricsOutput JSON struct, NewRequestMetrics canonical constructor
│   ├── vendored/              # Names this repository as the kernel adapter's vendoring root (testdata/catalog, testdata/registry)
│   └── internal/              # Shared internal packages
│       └── testutil/          # Shared test infrastructure (golden dataset loading, the fake latency model)
│                              # (The engine itself -- request scheduling, batch formation, the KV cache,
│                              #  LoRA, the kernel adapter -- is blis-vllm's: compat/ for 0.29/0.31,
│                              #  core/vllm for 0.32.0; see github.com/inference-sim/blis-vllm.)
├── sim/cluster/               # Multi-replica cluster simulation
│   ├── instance.go            # InstanceSimulator: one instance's engine (a sim.EngineDriver, built by NewEngineInstance) with the cluster's lifecycle and placement state for it
│   ├── cluster.go             # ClusterSimulator orchestrates N instances with shared-clock event loop, hands arrivals to the Gateway, and aggregates metrics; Run() returns error
│   ├── cluster_event.go       # ClusterArrivalEvent, ScalingTickEvent, ScaleActuationEvent; cluster event queue
│   ├── gateway.go             # Gateway interface (everything between arrival and dispatch: admission, flow control, endpoint picking), GatewayHost, Placement, Rejection (INV-1 buckets), IdleFastForwarder; sim/llmd is the production implementation
│   ├── metrics.go             # RawMetrics, Distribution, FitnessResult, CollectRawMetrics (accepts priorityPolicy), ComputeFitness (returns (FitnessResult, error)), anomaly detection, ParseFitnessWeights with NaN/Inf validation, per-SLO-class metrics, JainFairnessIndex
│   ├── deployment.go          # DeploymentConfig embeds sim.SimConfig + cluster-only fields; ToSimConfig() returns the embedded config
│   └── evaluation.go          # EvaluationResult wrapper (RawMetrics + FitnessResult + trace + summary)
├── sim/llmd/                  # The embedded llm-d router (Endpoint Picker), from the inference-sim fork of llm-d-router; see docs/guide/routing.md and docs/contributing/llm-d-router-fork.md
│   ├── gateway.go             # Gateway (implements cluster.Gateway): builds the router (blis-schemas embed) from EPP flags, EndpointPickerConfig and manifests, with BLIS's pre-build checks (Config.Check); translates arrivals into router requests, decisions into dispatches, rejection kinds into INV-1 buckets; DefaultEPPFlags
│   ├── router.go              # newRouter = the fork's pkg/embed.New, the only identifier BLIS takes from the fork (imports_test.go guards it)
│   ├── runtime.go             # embed.Runtime on the cluster event loop: fixed Epoch + simulation clock, timers and deadlines as (daemon) cluster events, seeded named random streams
│   ├── clock.go               # the runtime's timers and tickers on simulated time
│   ├── coroutine.go           # Coroutine scheduler: the router's goroutines run one at a time under a single baton, parked until their ready predicate holds; watchdog for unconverted blocking calls
│   ├── server.go              # The simulated model servers' HTTP surface for the router: /metrics (published vLLM series) and /v1/completions/render (prompt token IDs)
│   └── kvevents.go            # the engines' KV-cache events to the router (embed.KVBatch: ordered, delayed, canonical tokens)
│   └── prompt.go              # canonical prompt tokens the router sees (prefix-chain blocks), so a replay shows it the run's prompts
│   └── node_topology.go       # zone/region from explicit Node manifests, overlaid on the simulated placement
│   └── support.go             # BLIS's signal policy (embed.Config.Allow) and the checks of the router's Requirements against what the simulated engines produce
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
│   ├── record.go              # AdmissionRecord, RoutingRecord, CandidateScore, P/D and KV-transfer records (pure data types, no sim/ dependency)
│   └── summary.go             # TraceSummary, Summarize()
│                               # (The model catalog is NOT in this repo: it is the external blis-catalog
│                               #  repository, located at run time via --catalog / BLIS_CATALOG (#1731, #1771).
│                               #  It holds model graphs, chips, fabrics, storage devices and workload
│                               #  presets. A verbatim
│                               #  subset of a tagged release is vendored at testdata/catalog/ for tests,
│                               #  blis-registry's coefficient sets at testdata/registry/, and this repo's
│                               #  own scenario fixtures (P/D, MTP) at testdata/scenarios/.)
├── tools/                     # Offline scoring tools, outside the simulator (sim/, cmd/ link blis-vllm only through cmd/engine_build.go)
│   ├── harness/               # Closed-loop runs against the AISimulate accuracy corpus (the blis-latency-kernel adapter itself is blis-vllm compat/kernelmodel)
│   │   └── internal/artifacts/ # Gates the harness tests on the kernel artifact roots; the measurement-corpus tests run under -tags scoring
│   ├── metricscore/           # The scorer behind the evaluation tables: every estimator on every metric, on three tiers of points
│   ├── kernelscore/           # Scores the concurrency-response shape against the AISimulate snapshot
│   ├── sensitivity/           # How much each assumed engine setting (and the length draw) moves the score
│   ├── signedprobe/           # One CSV row per direct-corpus point, to find where the error is one-sided (temporary)
│   └── validation/            # render_docs.py: renders docs/validation/ from the validation results
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
│   │   ├── routing.md         # Routing with the embedded llm-d router
│   │   ├── admission.md       # Admission and flow control (the llm-d router's)
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
│   ├── methodology/           # Research methodology
│   │   ├── index.md           # Methodology overview
│   │   └── principles.md      # Discovered principles catalog
│   ├── contributing/          # Contributor documentation
│   │   ├── index.md           # Contributing landing page
│   │   ├── extension-recipes.md # Step-by-step extension guides
│   │   ├── llm-d-router-fork.md # The llm-d router fork: seams, release updates, hosting lint, patch budget
│   │   ├── pr-workflow.md     # PR development workflow
│   │   ├── rfc.md             # RFC template for large features
│   │   ├── standards/         # Canonical rules, invariants, principles, experiment standards
│   │   └── templates/         # Artifact templates
│   │       ├── design-guidelines.md  # DES foundations, module architecture
│   └── plans/                 # Active implementation plans (excluded from MkDocs)
├── CONTRIBUTING.md            # Contributor guide (references docs/contributing/standards/)
└── mkdocs.yml                 # MkDocs Material site configuration
```
