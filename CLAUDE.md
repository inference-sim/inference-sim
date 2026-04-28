# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients (alpha/beta), analytical roofline estimates, or physics-informed cross-model prediction.

The simulator is CPU-only, deterministic, and designed for capacity planning, policy optimization research, and performance prediction across model/GPU/TP configurations without requiring real GPUs.

## Build and Run Commands

```bash
# Build
go build -o blis main.go

# Run with default model
./blis run --model qwen/qwen3-14b

# Run and export workload as TraceV2 (prefix auto-appends .yaml/.csv)
./blis run --model qwen/qwen3-14b --trace-output traces/run1

# Replay a captured TraceV2 file through the DES (fixed timing from trace)
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b

# Replay and re-export trace with simulation-computed timing (mode: replayed)
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b \
  --trace-output out

# Replay with closed-loop session mode (follow-ups arrive at completion + think time)
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b \
  --session-mode closed-loop

# Replay closed-loop with explicit think-time override (500ms between rounds)
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b \
  --session-mode closed-loop --think-time-ms 500

# Observe real server latency and record timing into TraceV2
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload-spec workload.yaml --trace-header trace.yaml --trace-data trace.csv

# Observe with chat completions endpoint and network RTT
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --api-format chat --rtt-ms 2.5 --workload-spec workload.yaml \
  --trace-header trace.yaml --trace-data trace.csv

# Observe with named workload preset (chatbot, summarization, contentgen, multidoc)
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 \
  --trace-header trace.yaml --trace-data trace.csv

# Observe with rate-mode distribution synthesis and optional flags
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --api-format chat --rate 10 --num-requests 100 \
  --prompt-tokens 512 --output-tokens 128 --prefix-tokens 64 \
  --warmup-requests 5 --no-streaming --api-key $API_KEY \
  --max-concurrency 32 --unconstrained-output \
  --trace-header trace.yaml --trace-data trace.csv

# Observe with exact output length control (min_tokens=output_tokens defers EOS to target length; vLLM/compatible servers only)
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --rate 10 --num-requests 100 --output-tokens 2048 --min-tokens 2048 \
  --trace-header trace.yaml --trace-data trace.csv

# Observe closed-loop with lognormal think-time distribution (requires --concurrency)
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --concurrency 10 --num-requests 100 \
  --think-time-dist "lognormal:mu=2.0,sigma=0.6,min=3s,max=30s" \
  --trace-header trace.yaml --trace-data trace.csv

# Observe with ITL (inter-token latency) recording for streaming requests
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 \
  --record-itl --itl-output trace.itl.csv \
  --trace-header trace.yaml --trace-data trace.csv

# Compare real observed latencies against simulator predictions
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json --report calibration.json

# Compare with ITL metric included (requires observe --record-itl)
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json \
  --itl-data trace.itl.csv --report calibration.json

# Convert workload formats
./blis convert preset --name chatbot --rate 10 --num-requests 100
./blis convert servegen --path data/
./blis convert inference-perf --spec spec.yaml
./blis compose --from spec1.yaml --from spec2.yaml

# Run with gateway queue flow control (utilization-based saturation gating)
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
  --queue-depth-threshold 5 --kv-cache-util-threshold 0.8

# Run with concurrency-based flow control and priority dispatch ordering
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector concurrency \
  --max-concurrency 64 --dispatch-order priority --max-gateway-queue-depth 1000
```

## Testing

```bash
# Run all tests
go test ./...

# Run tests in a specific package
go test ./sim/...

# Run a single test by name
go test ./sim/... -run TestKVCache

# Run tests with verbose output
go test -v ./...

# Run tests with coverage
go test -cover ./...
```

## Development Guidelines

### Design Principles

BLIS follows a layered design document hierarchy. Each tier has a specific abstraction level and audience:

- **Design guidelines** (`docs/contributing/templates/design-guidelines.md`): Target architecture, DES foundations, module contracts, extension framework. Read this first when designing a new feature or extending BLIS.
- **Design docs** (per-feature): Behavioral specifications written per the guidelines. Describe what modules do and why, never how they're implemented. Four species: decision record, specification, problem analysis, system overview.
- **Macro plans** (multi-PR features): PR decomposition with module contracts and extension types. Written per `docs/contributing/templates/macro-plan.md` (human template; agent prompt: `macro-plan-prompt.md`). May include frozen interface signatures (facts about merged code) but never method implementations (aspirations about unwritten code).
- **Micro plans** (single PR): Full implementation detail with behavioral contracts, TDD tasks, exact code. Written per `docs/contributing/templates/micro-plan.md` (human template; agent prompt: `micro-plan-prompt.md`).

**The abstraction rule:** Design docs describe *what a module does and what it guarantees*. Macro plans describe *what to build and in what order*. Micro plans describe *how to implement each piece*. Go struct definitions, method implementations, and file:line references belong only in micro plans.

**Module architecture:** BLIS has a two-layer architecture — a domain-agnostic simulation kernel (event queue, clock, RNG, statistics) and domain-specific modules (router, scheduler, KV cache manager, latency model, autoscaler, batch formation). Each module is defined by a behavioral contract with six aspects: what it observes, what it controls, what state it owns, what invariants it maintains, what events it produces/consumes, and its extension friction (how many files to add one more variant). See design guidelines Section 4 for the full module map and contract template.

**Extending BLIS:** Four extension types, each with a different recipe — policy template (new algorithm behind existing interface), subsystem module (new module with its own interface), backend swap (alternative implementation requiring interface extraction), tier composition (delegation wrapper). See design guidelines Section 5.

### BDD/TDD Development

> **Canonical source:** [`docs/contributing/standards/principles.md`](docs/contributing/standards/principles.md) (BDD/TDD section). If this section diverges, principles.md is authoritative.

This project follows BDD/TDD practices. When implementing features:

1. **Write behavioral contracts first**: Define invariants and expected behavior in Gherkin-style scenarios
2. **Implement tests before code**: Tests verify contracts hold
3. **Use table-driven tests**: Go's table-driven test pattern for comprehensive coverage
4. **Test laws, not just values**: Golden tests answer "did the output change?" but not "is the output correct?" Every golden test should have a companion invariant test that verifies a law the system must satisfy (conservation, causality, monotonicity)
5. **Refactor survival test**: Before accepting a test, ask: "Would this test still pass if the implementation were completely rewritten but the behavior preserved?" If no, the test is structural — rewrite it to assert observable behavior instead of internal structure. See `docs/contributing/standards/principles.md` BDD/TDD section for prohibited/required assertion patterns.
6. **THEN clauses drive test quality**: A structural THEN clause produces a structural test. If a contract's THEN clause contains a concrete type name or internal field name, rewrite the THEN clause to describe observable behavior before writing the test.

### PR Workflow

Diligently follow the workflow in docs/contributing/pr-workflow.md. Before I approve any plan, validate it: 1) Check every task's dependencies — can each task actually start given what comes before it? 2) Verify all sections from the template are present and non-empty. 3) Read the executive summary as if you're a new team member — is it clear and human-readable? 4) Flag any tasks that seem under-specified for implementation. List all issues found.

For new features that introduce module boundaries or modify the architecture, a design doc (per the design guidelines) should exist before micro-planning begins. For smaller changes (bug fixes, new policy templates behind existing interfaces), a design doc is optional — proceed directly to micro-planning.

### Code Review Standards

During PR reviews, check all Antipattern Prevention rules (R1-R23) in [`docs/contributing/standards/rules.md`](docs/contributing/standards/rules.md). Pay special attention to rules 8-10 (exported mutable maps, YAML pointer types, strict YAML parsing) which are easy to miss in new code. Always run `go test ./...` and lint after fixes.

### Key Invariants to Maintain

> **Canonical source:** [`docs/contributing/standards/invariants.md`](docs/contributing/standards/invariants.md). If this section diverges, invariants.md is authoritative.

Full details (verification strategies, evidence): see [`docs/contributing/standards/invariants.md`](docs/contributing/standards/invariants.md).

- **INV-1 Request conservation**: `injected_requests == completed_requests + still_queued + still_running + dropped_unservable + timed_out + routing_rejections + gateway_queue_depth + gateway_queue_shed + gateway_queue_rejected` at simulation end. Full pipeline: `num_requests == injected_requests + rejected_requests`.
- **INV-2 Request lifecycle**: Requests transition queued → running → completed; not completed before horizon remain in current state
- **INV-3 Clock monotonicity**: Simulation clock never decreases
- **INV-4 KV cache conservation**: `allocated_blocks + free_blocks = total_blocks` at all times
- **INV-5 Causality**: `arrival_time <= enqueue_time <= schedule_time <= completion_time`
- **INV-6 Determinism**: Same seed must produce byte-identical stdout across runs. Wall-clock timing goes to stderr.
- **INV-7 Signal freshness**: Routing snapshot signals have tiered freshness — InFlightRequests (synchronous) vs QueueDepth/BatchSize/KVUtilization (Periodic when `--snapshot-refresh-interval > 0`, Immediate when 0). See `docs/contributing/standards/invariants.md` for the full hierarchy.
- **INV-8 Work-conserving**: After every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. The simulator must not idle while work is waiting.
- **INV-9 Oracle knowledge boundary**: Servability decisions (enqueue guard, admission, routing, priority) must not read `Request.OutputTokens`. The control plane uses `MaxOutputLen` (client budget) or input-only checks. Only the execution engine may access `OutputTokens` for token generation and completion detection. See `docs/contributing/standards/invariants.md`.
- **INV-10 Session causality**: For all rounds N in a closed-loop session: `round[N+1].ArrivalTime >= round[N].CompletionTime + ThinkTimeUs`. See `docs/contributing/standards/invariants.md`.
- **INV-11 Session completeness**: Every session reaches exactly one terminal state: completed, cancelled, horizon-interrupted, or budget-exhausted (concurrency mode: global request cap reached). No session is silently abandoned. See `docs/contributing/standards/invariants.md`.
- **INV-12 Phase 1 Completeness**: After Phase 1 of `FormBatch`, every non-preempted running request in decode phase has `NumNewTokens > 0`. No request silently skipped due to index drift from non-tail eviction. Trivially satisfied for FCFS. See `docs/contributing/standards/invariants.md`.

### Engineering Principles

> **Canonical source:** [`docs/contributing/standards/principles.md`](docs/contributing/standards/principles.md). If this section diverges, principles.md is authoritative.

Full details: see [`docs/contributing/standards/principles.md`](docs/contributing/standards/principles.md).

**Separation of concerns:** `sim/` is a library (never terminates). Cluster-level policies see global state via `*RouterState`. Instance-level policies see only local data. Dependency direction: `cmd/ → sim/cluster/ → sim/`.

**Interface design:** Single-method interfaces. Pure query methods. Factory validation. Behavioral contracts, not implementation-specific (R13). Single-module methods (R14).

**Configuration design:** Group by module (R16). `SimConfig` composed of 6 embedded sub-configs. Factory signatures accept the narrowest sub-config: `NewKVStore(KVCacheConfig)`, `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)`. Each module's config independently validatable.

**Canonical constructors:** Struct literals in exactly one place (R4). Grep for ALL construction sites before adding fields.

**Output channel separation:** stdout (deterministic results), stderr (diagnostics via logrus).

**Error handling boundaries:** CLI → `logrus.Fatalf`. Library → `error` or `panic`. Never silent `continue` (R1).

### Antipattern Prevention

> **Canonical source:** [`docs/contributing/standards/rules.md`](docs/contributing/standards/rules.md). If this section diverges, rules.md is authoritative.

23 rules (R1-R23), each tracing to a real bug. See [`docs/contributing/standards/rules.md`](docs/contributing/standards/rules.md) for the full table with evidence, checks, and enforcement locations.

### Current Implementation Focus

Composable Scorer Framework completed: PR17 (scorer framework + stateless scorers) and PR18 (prefix-affinity scorer + router-side cache). Default weighted routing profile: `precise-prefix-cache:2,queue-depth:1,kv-utilization:1` (llm-d parity). Precise prefix scoring (#883): `precise-prefix-cache` scorer queries actual instance KV cache state with min-max normalization (llm-d production parity); `no-hit-lru` scorer distributes cold requests to least-recently-used endpoints. Valid scorer names: `prefix-affinity`, `precise-prefix-cache`, `no-hit-lru`, `queue-depth`, `kv-utilization`, `load-balance`, `active-requests`, `running-requests`, `load-aware`, `vllm-dp`.

Phase 0 workload unification complete (see issue #420): W0-1 (spec v2 schema + SLO tiers), W0-2 (binary rename + converters), W0-3 (cohort population dynamics), W0-4 (legacy retirement). All workload generation now flows through `sim/workload/GenerateRequests()`. SLO tiers: critical, standard, sheddable, batch, background. Arrival processes: poisson, gamma, weibull, constant. CLI binary renamed from `simulation_worker` to `blis`.

Observe/replay/calibrate pipeline complete: `blis observe` (#659) dispatches workload to real servers with closed-loop session support, `blis replay` (#689) replays through DES, `blis calibrate` (#701) compares real vs simulated latencies. Observe fidelity (#660): chat completions endpoint (`--api-format chat`), `stream_options` for streaming token counts, `finish_reason` extraction, configurable `max_tokens` (`--unconstrained-output`), deterministic prefix strings for KV cache activation, `--rtt-ms` for network RTT.

Recent work: MkDocs documentation site (#450), roofline auto-fetch flag (#435), metrics substrate fixes (#458), cross-cutting documentation audit (#460).

### Extension Recipes

Step-by-step guides for adding policies, scorers, latency model backends, KV tiers, trace records, and per-request metrics: see `docs/contributing/extension-recipes.md`.

### Code Style

- Use composition over inheritance (e.g., `InstanceSimulator` wraps existing `sim` components)
- Timestamp-based event ordering via min-heap; both cluster and per-instance event queues use `(timestamp, priority, seqID)` ordering; cluster-level instance ties broken by lowest instance index
- Partitioned RNG per subsystem to isolate randomness

### CI/CD

GitHub Actions CI runs on all PRs to main:

- `.github/workflows/ci.yml` — Build verification (`go build ./...`), static analysis (`golangci-lint run ./...`, v2.9.0), test suite (`go test ./...`)
- `.github/workflows/docs.yml` — MkDocs site: PR validation (build-only), deploy on push to main, versioned on tag

Run lint locally before pushing: `golangci-lint run ./...`

## Agent Behavioral Instructions

The following instructions are for Claude Code and other AI assistants working on this codebase. Human contributors can skip this section.

### Context Management

When running multi-agent PR reviews, keep individual agent scopes narrow and summarize results concisely. Never try to synthesize all parallel agent outputs into one massive prompt. If hitting context limits, deliver incremental summaries per agent rather than a consolidated report.

### Task Agent Guidelines

When using Task agents: 1) Do NOT poll TaskList repeatedly — check at reasonable intervals (every 30-60 seconds, not continuously). 2) If a sub-agent goes idle or fails, fall back to doing the work directly rather than retrying indefinitely. 3) Keep sub-agent scopes focused to avoid context overflow.

### Macro Plan Updates

When asked to update the macro implementation plan, directly edit the document. Do NOT spend time re-reading all source documents or dispatching sub-agents to gather information you already have in context. Start writing immediately.

### Issue Filing

<!-- Keep in sync with .github/ISSUE_TEMPLATE/ — update when templates change -->

When filing a GitHub issue, pick the template that matches your situation:

1. **Found a bug or wrong simulation result?** → `Bug report` (`.github/ISSUE_TEMPLATE/bug_report.md`)
2. **Porting a feature from an external repo (llmd, gaie, vllm, sglang)?** → `Cross-repo feature` (`.github/ISSUE_TEMPLATE/cross_repo_feature.md`) — requires GitHub permalinks to source code
3. **Proposing a new BLIS-native capability?** → `Feature request` (`.github/ISSUE_TEMPLATE/feature_request.md`)
4. **Testing a hypothesis or running an experiment?** → `Hypothesis Proposal` (`.github/ISSUE_TEMPLATE/hypothesis.md`)
5. **Fixing an antipattern, hardening, or refactoring?** → `Hardening / refactoring` (`.github/ISSUE_TEMPLATE/custom.md`)

Every issue must have at least one label. Use `gh issue create --template "Template name"` to pre-fill the template.

## Speckit Feature-Development Toolkit

`.specify/` and `.claude/commands/` contain the speckit tooling for structured feature development:

- **Slash commands**: `/speckit.specify`, `/speckit.clarify`, `/speckit.plan`, `/speckit.tasks`, `/speckit.implement`, `/speckit.checklist`, `/speckit.analyze`, `/speckit.constitution`, `/speckit.taskstoissues`
- **Constitution**: `.specify/memory/constitution.md` — project principles, invariants, and rules distilled for AI agents
- **Templates**: `.specify/templates/` — spec, plan, tasks, checklist, and agent-file templates
- **Scripts**: `.specify/scripts/bash/` — feature branch creation, plan setup, and agent context update automation

Speckit does not affect Go build, test, or lint. All `.specify/` artifacts are opt-in for AI-assisted workflows.

## File Organization

For the full annotated file tree, see [`docs/reference/project-structure.md`](docs/reference/project-structure.md).

### Latency Estimation

Two latency model modes (roofline, trained-physics), selected via `--latency-model` flag. **Trained-physics** is the recommended default for new models.

**Migration note:** The deprecated `blackbox`, `crossmodel`, and `trained-roofline` backends have been removed. Use `--latency-model trained-physics` for modern physics-informed estimation with MoE-aware overhead modeling.

**Trained-physics model**: Roofline basis functions with learned correction coefficients. Generalizes across model architectures, workloads, and TP configurations. No per-model calibration needed.

See [`docs/guide/latency-models.md`](docs/guide/latency-models.md) for details.

**Quantized model support**: Three-tier auto-detection of weight precision: (1) `quantization_config` in HF `config.json` — GPTQ/AWQ (`bits`), FP8 (implicit), compressed-tensors (`config_groups.*.weights.num_bits`); (2) model name conventions (`w4a16` → 0.5, `FP8` → 1.0 via `InferWeightBytesFromModelName`); (3) fallback to `BytesPerParam` from `torch_dtype`. Uses quantized weight precision for weight bandwidth and KV capacity calculations while keeping compute dtype for KV cache and activations. `ModelConfig.WeightBytesPerParam` (0=fallback to `BytesPerParam`) with `EffectiveWeightBytesPerParam()` accessor decouples weight storage precision from compute/KV dtype.

### Key Data Flow

Request processing pipeline: Arrival → Admission → Routing → WaitQueue → Batch Formation → Step Execution → Completion. Admission and Routing apply in cluster mode only; single-instance skips directly to WaitQueue. See [`docs/concepts/architecture.md`](docs/concepts/architecture.md) for the full diagram.

## Project Governance Documents

### Standards (what rules apply)

- `docs/contributing/standards/rules.md`: **23 antipattern rules** (R1-R23) — each with evidence, checks, enforcement locations
- `docs/contributing/standards/invariants.md`: **12 system invariants** (INV-1 through INV-12) — with verification strategies
- `docs/contributing/standards/principles.md`: **Engineering principles** — separation of concerns, interface design, BDD/TDD
- `docs/contributing/standards/experiments.md`: **Experiment standards** — hypothesis families (6 families × type classification), rigor requirements, root cause verification (RCV-1 through RCV-6), iterative review protocol (summary; see `docs/contributing/convergence.md`), findings classification
- `docs/contributing/standards/agent-trust.md`: **Agent trust boundaries** — three trust tiers (Trusted, Verify-after, Never-trust) for agent operations, with known failure modes

### Process (how to do each activity)

- `docs/contributing/pr-workflow.md`: End-to-end PR workflow (worktree → plan → review → implement → audit → commit)
- `docs/contributing/design-process.md`: Design document creation process
- `docs/contributing/macro-planning.md`: Macro-level (multi-PR) planning process
- `docs/contributing/hypothesis.md`: End-to-end hypothesis experiment process (Steps 0-10, three review gates)
- `docs/contributing/convergence.md`: Universal Convergence Protocol (used by all review gates across PR, hypothesis, design, and macro-plan workflows)

### Templates (what to produce)

- `docs/contributing/templates/design-guidelines.md`: **BLIS Design Guidelines** — DES foundations, module architecture, extension framework. **Start here when designing anything new.**
- `docs/contributing/templates/macro-plan.md`: Human-readable template for macro-level planning (multi-PR features). **Agent prompt:** `macro-plan-prompt.md`
- `docs/contributing/templates/micro-plan.md`: Human-readable template for micro-level (per-PR) planning with TDD tasks and behavioral contracts. **Agent prompt:** `micro-plan-prompt.md`
- `docs/contributing/templates/hypothesis.md`: Template for hypothesis experiment artifacts

### Per-Feature Plans

- **Active plans:** `docs/plans/` (implementation plans for in-progress work)
- **Archived design docs:** `docs/plans/archive/` (completed design docs for architectural reference)
- **PR history:** Use `git log --oneline main` for the definitive commit history

## Active Technologies
- Go 1.22+ + `gopkg.in/yaml.v3` (strict parsing), `gonum` (stats), `cobra`, `logrus`
- In-memory node/GPU inventory maps; no external storage

## Recent Changes
- Criticality-based gateway queue shedding (#1190): `GatewayQueue.Enqueue()` now only evicts sheddable requests (`priority < 0`) on overflow. Non-sheddable requests (priority >= 0) are never evicted. When queue is full with no sheddable candidate, incoming request is rejected (`EnqueueOutcome.Rejected`). New `GatewayQueueRejected` counter added to INV-1. `TenantBudgetAdmission` policy wraps any inner `AdmissionPolicy` with per-tenant budget enforcement (decorator pattern). Inline tenant budget check removed from `cluster_event.go`. Closes #1020.
- SLO priority override threading (#1170): `SLOPriorityOverrides` field moved from `DeploymentConfig` to `SimConfig`, enabling policy bundle `slo_priorities` to reach `NewBatchFormation` for priority preemption. Previously custom SLO priorities only affected admission (tier-shed, GAIE-legacy); now they also affect `--preemption-policy priority` victim selection. `simulator.go` changed from `NewBatchFormation(cfg.PreemptionPolicy, nil)` to `NewBatchFormation(cfg.PreemptionPolicy, NewSLOPriorityMap(cfg.SLOPriorityOverrides))`. Field promoted via Go struct embedding — cluster.go:181 access unchanged.
- **BREAKING**: ServeGen temporal parity (#1124): `blis convert servegen` now produces `aggregate_rate: 0` (absolute rate mode) instead of a scalar peak value. Per-window `trace_rate` values are used directly as arrival rates, preserving ServeGen's temporal variation. Workloads generated before this change used proportional allocation with fixed aggregate rate, causing 13-18% over-generation at non-peak periods. New format: each window's `trace_rate` is the absolute rate in requests/second for that window. Validation now allows `aggregate_rate=0` when all rate-based clients have explicit per-window `trace_rate`. Added `--time` flag for temporal snapshot extraction (midnight/morning/afternoon). Lognormal distribution fitting reduces file size 100x vs empirical PMFs.
- Priority preemption mode (#1169): `--preemption-policy priority` evicts the least-urgent running request (min SLOPriorityMap value) under KV pressure instead of the batch tail. `selectPriorityVictim()`: `min(SLOPriority)` with `max(ArrivalTime)` tiebreak — analog of vLLM `scheduler.py:827-829` `max(priority, arrival_time)` with inverted convention. Phase 1 index adjustment (`reqIndex -= adj`) prevents element skipping after non-tail eviction (INV-12, analog of vLLM `scheduler.py:853` `req_index -= 1`). `NewBatchFormation(preemptionPolicy, sloMap)` 2-arg constructor; `nil` sloMap → `DefaultSLOPriorityMap()`.
- --preemption-policy flag (#1168): `--preemption-policy` (default `fcfs`) selects preemption victim strategy. Valid: `fcfs` (tail-of-batch, matches vLLM FCFS), `priority` (least-urgent SLO tier; wired in #1169). Registered on both `blis run` and `blis replay`. Bundle `preemption.policy` field overrides CLI default when flag not explicitly set. `PreemptionPolicy` type (`sim/batch_formation.go`) + `PreemptionConfig` (`sim/bundle.go`); `PolicyConfig.PreemptionPolicy` field + 3-arg `NewPolicyConfig()` in `sim/config.go`.
- blis run --timeout flag (#1127): `--timeout` (seconds, default `300`) overrides per-request deadline for all synthesized clients. Default 300s matches the session-client default in `computeDeadline` (preserves `UnlimitedRounds` termination). Negative value disables the deadline (explicit `*0` suppresses the 300s fallback); `0` is rejected as invalid — consistent with `blis observe` which also rejects `<= 0`. File-loaded `--workload-spec` clients are only overridden when `--timeout` is explicitly set. `blis observe` has its own `--timeout` (HTTP client timeout, default 300s, configurable via #1118; rejects `<= 0`). `blis replay` has no HTTP client and no `--timeout` flag.
- fix(observe): streaming timeout status + configurable timeout (#1118): `blis observe` now correctly sets `status=timeout` (not silent `ok`) when HTTP client timeout fires during streaming, non-streaming body read, or HTTP round-trip. New `--timeout` flag (seconds, default 300) configures per-request HTTP timeout. `isTimeoutError()` helper checks both `os.IsTimeout()` and `context.DeadlineExceeded` for robust detection. Three error paths fixed: `httpClient.Do`, `handleStreamingResponse` scanner error, `handleNonStreamingResponse` body read.
- Workload-level aggregate metrics in calibration (#1084): `MetricComparison` extended with `RealMean`, `SimMean`, `RealMedian`, `SimMedian`, `MeanError`, `MeanPercentError`, `MedianError`, `MedianPercentError`. CLI summary logs include `MeanError=±Xµs (±Y%)` alongside existing MAPE/PearsonR/quality. JSON report auto-includes new fields. Mean computed via single-pass sum; median aliased from P50. Division-by-zero guarded for degenerate inputs (R11, R20).
- GAIE-legacy saturation-based admission (#1014): `gaie-legacy` admission policy replicates production llm-d/GAIE admission behavior. Saturation formula: `avg(max(qd/qdThreshold, kvUtil/kvThreshold))` across instances. Non-sheddable requests (priority >= 0) always pass; sheddable requests (priority < 0) rejected when saturation >= 1.0. Defaults: `gaie_qd_threshold=5`, `gaie_kv_threshold=0.8`. Empty pool → saturation=1.0 (conservative). Configured via policy bundle YAML only. Per-tier shed counter (`shedByTier`) now tracks all tier-aware admission rejections unconditionally.
- Configurable SLO tier priorities (#1013): `SLOPriorityMap` type replaces hardcoded `SLOTierPriority()`. GAIE-compatible defaults: critical=4, standard=3, batch=-1, sheddable=-2, background=-3. `IsSheddable(class)` returns `priority < 0` (matches llm-d `sheddable.go`). Configurable via policy bundle YAML `slo_priorities` in `AdmissionConfig` (e.g., `admission: { slo_priorities: { batch: 0 } }` to make batch non-sheddable). Tenant budget enforcement uses `IsSheddable()` instead of hardcoded priority threshold. `TierShedMinPriority` validation removed (GAIE priorities are unbounded integers).
- TraceV2 ITL timestamps (#992): `blis observe --record-itl` captures per-chunk timestamps for ITL (inter-token latency) calibration. `blis calibrate --itl-data` computes ITL metric (mean chunk-to-chunk delta) alongside TTFT and E2E. ITL data stored in separate CSV (`request_id,chunk_index,timestamp_us`). Backward compatible: ITL is opt-in, TraceV2 format unchanged.
- fix(workload): inference_perf SLOClass regression (#965): Changed `SLOClass` from `"batch"` to `"standard"` in `ExpandInferencePerfSpec`. `model_configs/*/config.json` is now checked in for testing and documentation. Golden dataset `testdata/trained_physics_iter29.json` added with 15 iter29 experiments and `TestTrainedPhysics_GoldenDataset` in `sim/cluster/`.
- Unified cache signal staleness (#1060): `StaleCacheIndex` replaced by `CachedSnapshotProvider` cache management — all signal staleness now configured via `ObservabilityConfig`. `--cache-signal-delay` default changed from 2s to 50ms (eliminates ~13s warm-up artifact where PPC scorer was blind to cached prefixes). `ObservabilityConfig.CacheBlocks` field controls cache block hash map refresh; independent of `--snapshot-refresh-interval`. Set `--cache-signal-delay 0` for oracle mode (live cache state).
- Cache signal propagation delay (#919): `--cache-signal-delay` flag adds configurable staleness to `precise-prefix-cache` and `no-hit-lru` scorers. When > 0, scorers query periodically-refreshed stale snapshots of each instance's `HashToBlock` map via `CachedSnapshotProvider`, modeling asynchronous KV event propagation from production llm-d. Default 50ms (50,000 µs). Set to 0 for oracle mode (live cache state). INV-7 table updated with cacheQueryFn signal freshness tier.
- Precise prefix cache scoring (#883): `precise-prefix-cache` and `no-hit-lru` scorers query actual instance KV cache state via `CacheQueryFn` threading through `NewRoutingPolicy`. `GetCachedBlockCount` accessor on `InstanceSimulator`. Cluster layer builds `cacheQueryFn` from instances, including deferred NodePool instances.
- Gateway queue with saturation-gated dispatch (#882): `SaturationDetector` interface (NeverSaturated, UtilizationDetector, ConcurrencyDetector), `GatewayQueue` with FIFO/Priority dispatch, completion-triggered dispatch, per-request `GatewayQueueDelay` metric, INV-1 conservation extended with `gateway_queue_depth` + `gateway_queue_shed`
- fix(cluster): pool-authoritative hardware calibration for all backends (#888/#893): `NodePool.gpu_type` overrides `--gpu` for both GPU label and roofline `HWConfig` (TFlopsPeak/BwPeakTBs). `DeploymentConfig.HWConfigByGPU` map supplies per-pool `HardwareCalib`; lookup applied in both sync and deferred (`NodeReadyEvent`) construction paths. `CachedSnapshotProvider.AddInstance` added for dynamic instance registration.
- Phase 1B-2b: Per-tenant Jain fairness index in simulation output (#812, PR #881): `ComputePerTenantMetrics` + `printPerTenantMetrics` wired into `blis run` and `blis replay`; section absent for untenanted/legacy workloads
- Phase 1B-1b: Per-tenant fair-share tracking and admission enforcement (#811)
- Phase 1A: Added node/GPU placement, instance lifecycle, multi-model routing, per-model metrics
