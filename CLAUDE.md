# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients (alpha/beta), analytical roofline estimates, or physics-informed cross-model prediction.

The simulator is CPU-only, deterministic, and designed for capacity planning, policy optimization research, and performance prediction across model/GPU/TP configurations without requiring real GPUs.

## Build and Run Commands

```bash
# Build
go build -o blis main.go
```

**Catalog (required for `run`/`replay`/`observe`/`convert preset`).** BLIS reads model configs and workload presets from a `blis-catalog` clone, located by `--catalog <path>` or the `BLIS_CATALOG` env var (`--catalog` wins; no default, no search path, no remote fetch). Set it once:

```bash
git clone https://github.com/inference-sim/blis-catalog.git
export BLIS_CATALOG=$PWD/blis-catalog
```

Catalog rules: a model runs **iff** it is catalogued at `<catalog>/models/<short-name>/config.json` (runs never create or modify a catalog file). `--hardware` and `--tp` are **required** on `run` and `replay`. Named workload presets live under `<catalog>/workloads/`. See `docs/getting-started/quickstart.md` and `docs/reference/configuration.md`. The examples below assume `BLIS_CATALOG` is set; for the full flag reference of any command see `docs/guide/` or `--help`.

### run

```bash
# Basic run
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1

# Goodput SLO targets (per-class Go durations)
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1 \
  --slo-ttft "critical=100ms,standard=500ms" --slo-itl "critical=50ms" --slo-e2e "critical=5s"

# Speculative decoding / MTP (K=0 = off; acceptance rate required when K>0)
./blis run --model zai-org/GLM-5.2-FP8 --hardware H100 --tp 16 \
  --num-speculative-tokens 5 --speculative-acceptance-rate 0.7 --speculative-method mtp

# MoE data + expert parallelism
./blis run --model zai-org/GLM-5.2-FP8 --hardware H100 --tp 16 --dp 2 --enable-expert-parallel

# Independent KV-cache dtype
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1 --kv-cache-dtype fp8

# Multi-tier KV offload (strict-YAML config; absent = inert)
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1 --kv-offload-config offload.yaml

# Export the workload as a TraceV2 (prefix auto-appends .yaml/.csv)
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1 --trace-output traces/run1

# Gateway queue flow control (utilization- or concurrency-based saturation gating)
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1 --flow-control \
  --saturation-detector utilization --queue-depth-threshold 5 --kv-cache-util-threshold 0.8

# Lazy request generation (alpha; streams requests instead of pre-generating)
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1 --lazy-generation

# Post-hoc saturation detection (see the Post-Hoc Saturation Detection section)
./blis run --model qwen/qwen3-14b --hardware H100 --tp 1 --detectors all --saturation-report sat.json
```

### replay

```bash
# Replay a captured TraceV2 through the DES (use the same deployment flags as the run)
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b --hardware H100 --tp 1

# Closed-loop session replay (follow-ups arrive at completion + think time)
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b --hardware H100 --tp 1 \
  --session-mode closed-loop --think-time-ms 500

# Fixed pool of N concurrent closed-loop sessions from a corpus
./blis replay --trace-header corpus.yaml --trace-data corpus.csv --model qwen/qwen3-14b --hardware H100 --tp 1 \
  --concurrent-sessions 8 --total-sessions 200 --max-model-len 1000000

# Faithful high-concurrency agentic replay (recorded arrivals + accumulate-delta input)
./blis replay --trace-header corpus.yaml --trace-data corpus.csv --model qwen/qwen3-30b-a3b --hardware H100 --tp 2 \
  --dp 2 --enable-expert-parallel --session-mode fixed-accumulate --max-model-len 1000000
```

For INV-13 parity, pass `--horizon` on both `run` and `replay`. `--kv-cache-dtype`, `--enable-expert-parallel`, spec-decode, and the offload config are model-level inputs re-supplied on `replay` (not round-tripped through the trace header).

### observe

```bash
# Observe a real server and record timing into a TraceV2
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 --trace-header trace.yaml --trace-data trace.csv

# Chat-completions endpoint, network RTT, ITL recording, system prewarm
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b --api-format chat --rtt-ms 2.5 \
  --record-itl --itl-output trace.itl.csv --prewarm-duration 60s \
  --workload chatbot --rate 10 --num-requests 100 --trace-header trace.yaml --trace-data trace.csv

# Scrape KV cache hit-rate from the server's Prometheus /metrics
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --scrape-kv-metrics --vllm-commit 63a9a5010a \
  --workload chatbot --rate 10 --num-requests 100 --trace-header trace.yaml --trace-data trace.csv
```

`observe` is a black-box dispatcher: it takes neither `--hardware`/`--tp` nor `--kv-offload-config` (it resolves no model config). See `docs/guide/observe-replay-calibrate.md`.

### calibrate

```bash
# Compare real observed latencies against simulator predictions
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json --report calibration.json

# Add KV hit-rate, ITL, goodput, throughput, or per-GPU comparisons via the matching flags
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json \
  --sim-metrics simagg.json --itl-data trace.itl.csv --num-gpus 4 --report calibration.json
```

### convert

```bash
# Named preset from the catalog, or external workload formats
./blis convert preset --name chatbot --rate 10 --num-requests 100
./blis convert servegen --path data/
./blis convert inference-perf --spec spec.yaml
./blis compose --from spec1.yaml --from spec2.yaml

# Agentic traces -> closed-loop TraceV2 corpus (replay with --session-mode closed-loop)
./blis convert otel --input traces.jsonl --trace-output corpus --context-growth accumulate
./blis convert weka --input traces.jsonl --trace-output corpus --context-growth accumulate --max-think-time 0
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
- **RFC + .archon plan** (multi-PR features): Tracking issue with holes/surfaces/contracts (see `docs/contributing/rfc.md`), encoded into a machine-checkable `.archon` plan (see `docs/contributing/templates/rfc-to-plan.md`). Sub-issues created per hole, each delivered as a PR.
- **Implementation plans** (single PR): Behavioral contracts, TDD tasks. Follow `docs/contributing/pr-workflow.md`.

**The abstraction rule:** Design docs and RFCs describe *what a module does and what it guarantees*. The `.archon` plan describes *what structure to build and in what order*. Implementation plans describe *how to implement each piece*.

**Module architecture:** BLIS has a two-layer architecture — a domain-agnostic simulation kernel (event queue, clock, RNG, statistics) and domain-specific modules (router, scheduler, KV cache manager, latency model, autoscaler, batch formation). Each module is defined by a behavioral contract with six aspects: what it observes, what it controls, what state it owns, what invariants it maintains, what events it produces/consumes, and its extension friction (how many files to add one more variant). See design guidelines Section 4 for the full module map and contract template.

**Extending BLIS:** Four extension types, each with a different recipe — policy template (new algorithm behind existing interface), subsystem module (new module with its own interface), backend swap (alternative implementation requiring interface extraction), tier composition (delegation wrapper). See design guidelines Section 5.

### BDD/TDD Development

> **Canonical source:** [`docs/contributing/standards/principles.md`](docs/contributing/standards/principles.md) (BDD/TDD section). If this section diverges, principles.md is authoritative.

Write behavioral contracts (Gherkin GIVEN/WHEN/THEN) first, then tests, then code. Use table-driven tests. Test laws (conservation, causality, monotonicity), not just golden values — every golden test needs a companion invariant test. Assert observable behavior, not internal structure (the "refactor survival" test). See principles.md for the full prohibited/required assertion patterns.

### PR Workflow

Diligently follow the workflow in docs/contributing/pr-workflow.md. Before I approve any plan, validate it: 1) Check every task's dependencies — can each task actually start given what comes before it? 2) Verify all sections from the template are present and non-empty. 3) Read the executive summary as if you're a new team member — is it clear and human-readable? 4) Flag any tasks that seem under-specified for implementation. List all issues found.

For new features that introduce module boundaries or modify the architecture, an RFC (per `docs/contributing/rfc.md`) should exist before implementation planning begins. For smaller changes (bug fixes, new policy templates behind existing interfaces), an RFC is optional — proceed directly to `docs/contributing/pr-workflow.md`.

### Code Review Standards

During PR reviews, check all Antipattern Prevention rules (R1-R23) in [`docs/contributing/standards/rules.md`](docs/contributing/standards/rules.md). Pay special attention to rules 8-10 (exported mutable maps, YAML pointer types, strict config parsing — YAML and JSON) which are easy to miss in new code. Always run `go test ./...` and lint after fixes.

### Key Invariants to Maintain

> **Canonical source:** [`docs/contributing/standards/invariants.md`](docs/contributing/standards/invariants.md). If this section diverges, invariants.md is authoritative.

Full details (verification strategies, evidence): see [`docs/contributing/standards/invariants.md`](docs/contributing/standards/invariants.md).

- **INV-1 Request conservation**: 12-term cluster equation `injected == completed + still_queued + still_running + dropped_unservable + timed_out + routing_rejections + gateway_queue_depth + gateway_queue_shed + gateway_queue_rejected + gateway_evicted + gateway_expired + encode_routing_rejections` at end; single-instance reduces to 5 terms. Pipeline: `num_requests == injected + rejected`.
- **INV-2 Request lifecycle**: queued → running → completed; requests not completed before horizon stay in their current state.
- **INV-3 Clock monotonicity**: sim clock never decreases; every *processed* event's timestamp ≥ the previous processed event's (assert on processed events, not a raw clock field).
- **INV-4 KV cache conservation**: `allocated_blocks + free_blocks == total_blocks` at all times.
- **INV-5 Causality**: `arrival_time ≤ enqueue_time ≤ schedule_time ≤ completion_time`.
- **INV-6 Determinism**: same seed ⇒ byte-identical stdout; wall-clock timing goes to stderr.
- **INV-7 Signal freshness**: routing signals are tiered — InFlightRequests synchronous; QueueDepth/BatchSize/KVUtilization periodic (50ms default, immediate at `--snapshot-refresh-interval 0`).
- **INV-8 Work-conserving**: after a step, if `WaitQ.Len() > 0` a `StepEvent` must exist — never idle while work waits.
- **INV-9 Oracle knowledge boundary**: servability decisions must not read `Request.OutputTokens` (use `MaxOutputLen`/input-only); only the execution engine reads OutputTokens.
- **INV-10 Session causality**: closed-loop `round[N+1].ArrivalTime ≥ round[N].CompletionTime + ThinkTimeUs`.
- **INV-11 Session completeness**: every session reaches exactly one terminal state; none silently abandoned.
- **INV-12 Phase 1 completeness**: after `FormBatch` Phase 1, every non-preempted decode-phase request has `NumNewTokens > 0`.
- **INV-13 Run/Replay parity**: same config ⇒ identical per-request metrics; at the CLI assert byte-identical stdout (pass `--horizon` on both legs). Unsupported replay features must `logrus.Fatalf`, never degrade.
- **INV-14 Instance lifecycle transitions**: `State` changes only along `validInstanceTransitions`; `TransitionTo` panics otherwise (monotone; first transition from `""` seeds unvalidated).
- **INV-15 Arrival ordering**: the fresh-arrival stream into the cluster is non-decreasing in arrival time (precondition on input; fresh arrivals only).
- **INV-16 Gateway queue counter consistency**: `totalLen` equals entries across all bands/flows; a positive counter must yield a dequeue (panic sites).
- **INV-17 Shed victim is the flow tail**: the shed victim is always the tail, so removal is a truncation and the head is preserved.
- **INV-18 Prefix-cache LRU structural consistency**: the intrusive list and `lookup` map describe the same block set; `tail == nil` iff empty.
- **INV-19 Scale decisions carry a non-zero delta**: every emitted `ScaleDecision` has `Delta != 0`; a zero delta is warned and skipped, not fatal.
- **Additional families** (see `invariants.md`): INV-A / INV-A2 (GPU conservation; placement-failure visibility), INV-W3 (cohort-expansion purity), INV-BC-DP1 (dense DP=1 step-time byte-identity), INV-L1–INV-L7 (LoRA control plane), INV-PD-* (PD disaggregation), INV-P2-* (pool/transfer), and NS-6 (catalog is authoritative and read-only; `--hardware`/`--tp` required at the CLI).

### Engineering Principles

> **Canonical source:** [`docs/contributing/standards/principles.md`](docs/contributing/standards/principles.md) — authoritative; see it for full details.

**Separation of concerns:** `sim/` is a library (never terminates). Cluster-level policies see global state via `*RouterState`. Instance-level policies see only local data. Dependency direction: `cmd/ → sim/cluster/ → sim/`.

**Interface design:** Single-method interfaces. Pure query methods. Factory validation. Behavioral contracts, not implementation-specific (R13). Single-module methods (R14).

**Configuration design:** Group by module (R16). `SimConfig` composed of 6 embedded sub-configs. Factory signatures accept the narrowest sub-config: `NewKVStore(KVCacheConfig)`, `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)`. Each module's config independently validatable.

**Canonical constructors:** Struct literals in exactly one place (R4). Grep for ALL construction sites before adding fields.

**Output channel separation:** stdout (deterministic results), stderr (diagnostics via logrus).

**Error handling boundaries:** CLI → `logrus.Fatalf`. Library → `error` or `panic`. Never silent `continue` (R1).

### Antipattern Prevention

> **Canonical source:** [`docs/contributing/standards/rules.md`](docs/contributing/standards/rules.md). If this section diverges, rules.md is authoritative.

23 rules (R1-R23), each tracing to a real bug. See [`docs/contributing/standards/rules.md`](docs/contributing/standards/rules.md) for the full table with evidence, checks, and enforcement locations.

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

### GitHub Action: PR Reviews

When triggered via `@claude /blis-pr-review` on a PR, follow the blis-pr-review skill exactly. For all other triggers (questions, debugging, etc.), respond normally without creating a PR unless explicitly asked.

The `/blis-pr-review` path runs with a **read-only** token (`contents: read`, #1697): it can post the review comment, but cannot push (the commit status is published by a separate job). Report findings — do not fix them, and do not attempt a commit or push, which fails with a 403. Every other `@claude` trigger keeps `contents: write`. See [`docs/contributing/standards/agent-trust.md`](docs/contributing/standards/agent-trust.md).

### Context Management

When running multi-agent PR reviews, keep individual agent scopes narrow and summarize results concisely. Never try to synthesize all parallel agent outputs into one massive prompt. If hitting context limits, deliver incremental summaries per agent rather than a consolidated report.

### Task Agent Guidelines

When using Task agents: 1) Do NOT poll TaskList repeatedly — check at reasonable intervals (every 30-60 seconds, not continuously). 2) If a sub-agent goes idle or fails, fall back to doing the work directly rather than retrying indefinitely. 3) Keep sub-agent scopes focused to avoid context overflow.


### Issue Filing

<!-- Keep in sync with .github/ISSUE_TEMPLATE/ — update when templates change -->

When filing a GitHub issue, pick the template that matches your situation:

1. **Found a bug or wrong simulation result?** → `Bug report` (`.github/ISSUE_TEMPLATE/bug_report.md`)
2. **Porting a feature from an external repo (llmd, gaie, vllm, sglang)?** → `Cross-repo feature` (`.github/ISSUE_TEMPLATE/cross_repo_feature.md`) — requires GitHub permalinks to source code
3. **Proposing a new BLIS-native capability?** → `Feature request` (`.github/ISSUE_TEMPLATE/feature_request.md`)
4. **Testing a hypothesis or running an experiment?** → `Hypothesis Proposal` (`.github/ISSUE_TEMPLATE/hypothesis.md`)
5. **Fixing an antipattern, hardening, or refactoring?** → `Hardening / refactoring` (`.github/ISSUE_TEMPLATE/custom.md`)

Every issue must have at least one label. To file an issue: read the relevant template file under `.github/ISSUE_TEMPLATE/`, reproduce its structure in your issue body, and use `gh issue create --title "..." --body "..." --label "..."`. Apply the template's front-matter labels yourself.


## Post-Hoc Saturation Detection

BLIS can classify a completed run's saturation state (distinct from the real-time flow-control detector used for admission control). Package: `sim/saturation/`.

Four streaming detectors, each with a false-alarm calibration knob:
- **composite** — rate deficit + quartile-filtered latency trend vs a noise floor (`composite.sensitivity`).
- **threshold** — mean E2E latency vs a fixed target (`threshold.threshold_ms`, default 5000ms).
- **backlog-drift** — online OLS slope of in-flight over a trailing window (`backlog_drift.slope_k`).
- **peak-rate** — `R_t = Peak_t/t`, the backlog high-water mark over elapsed time; needs no latency target or capacity estimate (`peak_rate.threshold` and friends).

CLI flags (on `run`, `replay`, `observe`):
- `--detectors <selection>` — empty = off; a single name; `all`; or a comma-list. Unknown name is a hard error.
- `--saturation-config <path>` — strict-YAML tuning file with per-detector blocks (only the selected detectors' blocks are allowed).
- `--saturation-report <path>` — writes `{"final":{...},"trace":[...]}` (final label + per-event trace).
- `--saturation-final-window <duration>` — trailing window for the stdout final-label plurality vote (default 30s).

A run with `--detectors` adds a `"saturation"` map (detector → final label) to the metrics JSON on stdout; without it, stdout is byte-identical to the no-feature output. run→replay of the same trace is byte-identical; observe reflects real-server latencies.

For detector internals, the config schema, the report format, the migration table from the pre-#1516 flags, and how to add a new detector, see [`docs/contributing/saturation-analyzer-extension.md`](docs/contributing/saturation-analyzer-extension.md).

## File Organization

For the full annotated file tree, see [`docs/reference/project-structure.md`](docs/reference/project-structure.md).

### Latency Estimation

Two latency backends, selected by `--latency-model`:
- **trained-physics** (default) — roofline basis functions with learned correction coefficients; generalizes across models, workloads, and TP without per-model calibration.
- **roofline** — pure analytical FLOPs/bandwidth model (`--latency-model roofline`).

(The former `blackbox`, `crossmodel`, and `trained-roofline` backends were removed.) See [`docs/guide/latency-models.md`](docs/guide/latency-models.md) for the model, coefficients, MoE/EP/inter-node terms, and calibration.

The models also account for:
- **Quantized weights** — three-tier auto-detection of weight precision (HF `quantization_config`, model-name conventions, then `torch_dtype` fallback); weight precision drives weight bandwidth/footprint while activations keep the compute dtype.
- **Independent KV-cache dtype** — `--kv-cache-dtype` (`auto`/`fp8`/`fp8_e4m3`/`fp8_e5m2`) sets KV storage precision independent of compute/weight dtype; `auto` (default) ⇒ KV dtype == compute dtype ⇒ byte-identical (INV-6). Re-supply it on `replay` (not round-tripped through the trace header).
- **MoE / expert parallelism** — `--dp N` places N real engine replicas; `--enable-expert-parallel` shards routed-expert weights over the EP group and is step-time-live. Supported on both `run` and `replay`.
- **Inter-node network cost** (trained-physics only) — priced from placement (node span) and the hardware interconnect calibration in `hardware_config.json`; inert/byte-identical when a group fits one node or the fabric is uncalibrated.
- **MLA / hybrid-attention model shape** — compressed-KV latent (`kv_lora_rank`), explicit `head_dim`, dense-prefix MoE, and hybrid full/linear-attention layer splits size KV capacity and step time. See [`docs/reference/models.md`](docs/reference/models.md).
- **Speculative decoding / MTP** — `--num-speculative-tokens K` with `--speculative-acceptance-rate` models the decode throughput of MTP/EAGLE/Medusa; `K=0` (default) ⇒ byte-identical (INV-6). Re-supply on `replay`.

### Key Data Flow

Request processing pipeline: Arrival → Admission → Routing → WaitQueue → Batch Formation → Step Execution → Completion. Admission and Routing apply in cluster mode only; single-instance skips directly to WaitQueue. See [`docs/concepts/architecture.md`](docs/concepts/architecture.md) for the full diagram.

## Project Governance Documents

### Standards (what rules apply)

- `docs/contributing/standards/rules.md`: **23 antipattern rules** (R1-R23) — each with evidence, checks, enforcement locations
- `docs/contributing/standards/invariants.md`: the invariant registry — **19 core invariants** (INV-1 through INV-19; INV-14 … INV-19 are the enforcement-anchored set promoted by #1772) plus INV-A, INV-A2, INV-W3, INV-BC-DP1, the LoRA family (INV-L1–INV-L7, by pointer), PD disaggregation (INV-PD-*) and pool/transfer (INV-P2-*). Grouped by scope (run-level, subsystem) plus a cross-cutting code-boundary group. **Every `INV-*` ID cited in `sim/` or `cmd/` is resolvable from it** (#1721) — with verification strategies
- `docs/contributing/standards/principles.md`: **Engineering principles** — separation of concerns, interface design, BDD/TDD
- `docs/contributing/standards/agent-trust.md`: **Agent trust boundaries** — three trust tiers (Trusted, Verify-after, Never-trust) for agent operations, with known failure modes

### Process (how to do each activity)

- `docs/contributing/pr-workflow.md`: End-to-end PR workflow (worktree → plan → review → implement → audit → commit). **Step 1.5 defines the source document as the issue body PLUS the design refinements in its comment thread** — a body is written once and the design is then refined in comments, so a plan made from the body alone builds an out-of-date spec faithfully (#1782). `scripts/deliver-issue-refinements.sh <issue>` prints the comments that carry authority: a comment counts iff its author holds `admin`/`write`/`maintain` (`authorAssociation` is NOT the signal — this repository's maintainer reports `CONTRIBUTOR`), a refinement overrides the body on any point it addresses, later refinements win, and the target branch / `archon-plan:` / `Depends on:` stay body-only
- `docs/contributing/issue-comment-authority.md`: Decision record for that rule — the rejected alternative and the measurements behind it
- `docs/contributing/rfc.md`: RFC template for large features (tracking issue with holes/surfaces/contracts)
- `docs/contributing/templates/rfc-to-plan.md`: Claude prompt for encoding RFC into .archon plan + creating sub-issues

### Templates (what to produce)

- `docs/contributing/templates/design-guidelines.md`: **BLIS Design Guidelines** — DES foundations, module architecture, extension framework. **Start here when designing anything new.**
- `docs/contributing/templates/rfc-to-plan.md`: Claude prompt for .archon encoding + sub-issue creation

### Per-Feature Plans

- **Active plans:** `docs/plans/` (implementation plans for in-progress work)
- **Archived design docs:** `docs/plans/archive/` (completed design docs for architectural reference)
- **PR history:** Use `git log --oneline main` for the definitive commit history

## Active Technologies
- Go 1.22+ + `gopkg.in/yaml.v3` (strict parsing), `gonum` (stats), `cobra`, `logrus`
- In-memory node/GPU inventory maps; no external storage

## Change History

See `git log --oneline main` for the definitive commit history. Durable cross-cutting facts live in the standards docs and topic guides (admission, routing, scheduling, observe-replay-calibrate, workloads, configuration reference).
