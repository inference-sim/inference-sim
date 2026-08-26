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

# Run with goodput SLO targets (#1413). --slo-ttft / --slo-itl / --slo-e2e accept
# class=duration[,class=duration...] using Go duration syntax. Precedence:
# CLI > trace header > workload spec. Distinct from --slo-targets (dispatch ordering).
./blis run --model qwen/qwen3-14b \
  --slo-ttft "critical=100ms,standard=500ms" \
  --slo-itl  "critical=50ms,standard=150ms" \
  --slo-e2e  "critical=5s,standard=30s"

# Run and export workload as TraceV2 (prefix auto-appends .yaml/.csv)
./blis run --model qwen/qwen3-14b --trace-output traces/run1

# Run with a multi-tier KV-offload config (#1587). One strict-YAML flag captures vLLM's
# offload config surface (kv_offload: block — cpu_bytes_to_use, block_size/blocks_per_chunk,
# eviction_policy, offload_prompt_only [vLLM default TRUE], secondary_tiers[] with per-tier
# device_class/direct_io/bandwidth). Absent => offload subsystem inert, byte-identical output
# (BC-G5). Defaults match vLLM knob-for-knob; store_threshold>=2 and non-fs tiers fail loudly.
# device_class names resolve against defaults.yaml kv_offload_devices. The resolved config is
# recorded in the trace header and round-trips through replay (INV-13); on replay the header is
# authoritative (a config it cannot reproduce is a hard error). As of #1590 the config DRIVES the
# N-tier chain mechanism (below) — mutually exclusive with the legacy --kv-cpu-blocks.
./blis run --model qwen/qwen3-14b --kv-offload-config offload.yaml --trace-output traces/run1

# Replay a captured TraceV2 file through the DES (fixed timing from trace)
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b

# Replay with goodput SLO targets (#1413). When the trace header carries
# goodput_slo_targets the CLI flags are not required (header is the fallback).
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b \
  --slo-ttft "critical=100ms" --slo-e2e "critical=5s"

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

# Observe with goodput SLO targets (#1413). Resolved targets are persisted into
# the exported TraceHeader so downstream replay/calibrate inherit them. ITL
# attainment requires --record-itl; otherwise it's skipped with a warning.
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 \
  --slo-ttft "critical=100ms" --slo-e2e "critical=5s" \
  --trace-header trace.yaml --trace-data trace.csv

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

# Observe closed-loop with lognormal think-time distribution (requires --concurrency)
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --concurrency 10 --num-requests 100 \
  --think-time-dist "lognormal:mu=2.0,sigma=0.6,min=3s,max=30s" \
  --trace-header trace.yaml --trace-data trace.csv

# Observe with system prewarm (recommended for cold systems, #1430)
# --prewarm-duration: warms infrastructure (EPP/gateway connections, TCP pools)
#   before measurement. Use for cold systems. Shape-independent.
# --warmup-requests: excludes first N real-workload requests from trace.
#   Use for statistical trimming. Rarely needed if --prewarm-duration is set.
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --prewarm-duration 60s \
  --workload chatbot --rate 10 --num-requests 100 \
  --trace-header trace.yaml --trace-data trace.csv

# Observe with ITL (inter-token latency) recording for streaming requests
# --record-itl forces streaming on non-streaming workloads to capture per-chunk timestamps
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 \
  --record-itl --itl-output trace.itl.csv \
  --trace-header trace.yaml --trace-data trace.csv

# Observe with KV cache hit-rate scraping (#1583). --scrape-kv-metrics scrapes the
# server's Prometheus /metrics at the start and end of the measured window and records
# the observed hit-rate (vllm:kv_offload_tiering_block_hits/_block_queries, or the GPU
# prefix-cache fallback) in the trace header for downstream calibrate. --vllm-commit
# records the pinned (unreleased, PR #48798) vLLM the tiering counters require. A scrape
# miss warns and omits the block (never fatal). observe is a black-box dispatcher — it
# does NOT take --kv-offload-config; the offload config is supplied on the replay step
# (below). See docs/guide/kv-offload-calibration.md.
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 \
  --scrape-kv-metrics --vllm-commit 63a9a5010a \
  --trace-header trace.yaml --trace-data trace.csv

# Compare real observed latencies against simulator predictions
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json --report calibration.json

# Compare KV cache hit-rate too (#1583). The real hit-rate comes from the trace header
# (observe --scrape-kv-metrics); the sim hit-rate comes from --sim-metrics (a
# MetricsOutput from `blis replay --metrics-path`, which carries cache_hit_rate). The
# report gains a hit_rate block with abs_error_pp and a within-tolerance verdict
# (default 5 pp); a TTFT-MAPE verdict uses --ttft-mape-threshold (default 0.15).
# Skipped with a warning when --sim-metrics is absent or lacks cache_hit_rate. For a
# tiered observed hit-rate, supply --kv-offload-config on replay to model the observed
# deployment (observe traces are exempt from the "cannot add offload on replay" rule;
# sim-generated traces stay header-authoritative). Replaying a tiered observation with
# no offload config is a hard error (never a silent GPU-only value).
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b \
  --kv-offload-config offload.yaml \
  --results-path results.json --metrics-path simagg.json
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json \
  --sim-metrics simagg.json --hit-rate-tolerance-pp 5 --ttft-mape-threshold 0.15 \
  --report calibration.json

# Compare with ITL metric included (requires observe --record-itl)
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json \
  --itl-data trace.itl.csv --report calibration.json

# Compare goodput per SLO class (#1413). Targets default to the trace header's
# goodput_slo_targets; CLI flags override per-dimension. Skipped with a warning
# when ITL is configured but absent.
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json \
  --slo-ttft "critical=100ms" --slo-e2e "critical=5s" --report calibration.json

# Compare LoRA adapter-cost fidelity vs a Digital Twin reference (#1470, US5).
# Standalone mode: BLIS aggregate (from blis run --metrics-path) vs a committed DT
# reference (per-config adapter_aware/adapter_blind), per-metric MAPE on TTFT +
# throughput. --sim-metrics-blind enables the delta-normalized (aware/blind)
# diagnostic that isolates the ported adapter physics. Does not use --trace-*.
./blis calibrate --adapter-reference dt-ref.json \
  --sim-metrics aware.json --sim-metrics-blind blind.json \
  --adapter-mape-threshold 0.20 --report adapter-fidelity.json

# Run with lazy request generation (alpha, #1441). Streams requests from the
# workload generator into the cluster instead of pre-generating the full
# slice — reduces peak generator memory from O(total_requests) to the
# concurrent working set: the global heap holds one entry per client;
# single-session reasoning holds at most one session's pending rounds, while
# multi-session reasoning (#1458) holds its live (overlapping) sessions,
# bounded by ~ arrival_rate x session_duration (Little's law) — independent
# of horizon. (Cluster-side memory still scales with in-flight cluster
# requests, which this PR does not change.)
# Supports EVERY workload class — there is NO eager fallback (#1460):
# single-shot; single-session AND multi-session reasoning (SingleSession=false,
# #1458 — per-client live-session merge); concurrency clients (Concurrency > 0,
# #1459 — seeds merged as individual heap entries; the win is modest for
# pure-concurrency specs since the seed set is O(N virtual users)); time-varying
# / per-window workloads (trace_rate/arrival/input_distribution/output_distribution
# overrides, #1460 — per-window batches merged via a live-window heap, so resident
# memory is the concurrent-window working set rather than all windows at once, a
# real win for the many-small-windows layout typical of spike/servegen/diurnal
# schedules; note a single huge window materializes one full batch, so it yields
# no memory win over eager); prefix-group sharing; multi-client / cohort workloads.
# Behavior with the flag off is unchanged.
./blis run --model qwen/qwen3-14b --lazy-generation

# Observe with lazy request generation (alpha, #1443). Same flag, default, and
# semantics as `blis run` — streams requests from the generator into the observe
# dispatch loop instead of pre-generating the full slice. As of #1460 there is no
# eager fallback: every class blis run supports (multi-session reasoning #1458,
# concurrency clients #1459, time-varying / per-window workloads #1460) is
# streamed. Observe already paces
# itself against the real server, so the memory win is smaller than run's; the
# flag mainly makes run and observe share one generation pipeline (#1438). Default
# (flag off) dispatch behavior is unchanged.
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 --lazy-generation \
  --trace-header trace.yaml --trace-data trace.csv

# Convert workload formats
./blis convert preset --name chatbot --rate 10 --num-requests 100
./blis convert servegen --path data/
./blis convert servegen --path data/ --time midnight  # Single period for testing
./blis convert inference-perf --spec spec.yaml
./blis compose --from spec1.yaml --from spec2.yaml

# Convert captured agentic traces to a TraceV2 corpus for closed-loop replay (#1477).
# Both readers feed the shared session→TraceV2 encoder (#1479): each session becomes
# rounds 0..N with per-round input DELTAS, TraceRecord.Model left empty (routing safety —
# the recorded cross-model name would drop every request under --model), and closed-loop
# replay's accumulate buffer reconstructs the growing prompt. Output is delta-encoded, so
# it MUST be replayed with --session-mode closed-loop (or --concurrent-sessions, which
# auto-promotes); the default --session-mode fixed would misread the deltas as absolutes.
#
# convert otel — OpenTelemetry agentic trace JSON (file / dir of *.json / *.jsonl); one
# LLM chat span per round; think derived from arrival gaps (--max-think-time default 15s).
./blis convert otel --input traces.jsonl --trace-output corpus \
  --context-growth accumulate
#
# convert weka — SemiAnalysis WekaTrace JSONL (#1604, PR-F; one proxy session per line).
# Filters requests[] to the linear main-agent stream (type:"subagent" groups skipped,
# deferred to PR-E); recomputes pure client think as max(0, t_i − t_{i-1} − api_time_{i-1})
# between consecutive main turns into the think_time_us column (--max-think-time default 0
# = uncapped, since Weka gaps are genuine away-from-keyboard times). Reads `in` directly
# (never len(hash_ids)×64). Weka ISL is huge (p50 ≈ 110K, p90 ≈ 395K), so replay MUST raise
# --max-model-len (the ~41K default drops every request unservable) and scale --total-kv-blocks.
# Fidelity note (important): real agentic traces compact/trim context heavily — ~30% of
# rounds on the full 051926 dataset (219 sessions, 37.7K rounds) have in_N < in_{N-1}+out_{N-1}.
# Such non-monotone rounds clamp their input delta to 0 (the accumulate buffer can only grow,
# never shrink), so it OVER-counts the true cumulative input by ≈3–4× on real Claude Code
# traffic (+312% on this dataset). Replayed input length / KV pressure / hit-rate is therefore
# a substantial UPPER BOUND — do NOT read it as a faithful reproduction of the recorded ISL.
# (Property of the PR-A/PR-B accumulate delta law, not this converter; the conversion is exact
# per that law. Faithful compaction support is tracked in #1609.) The recorded think time is
# non-lossy (#1608): a genuinely-zero recorded think (an overlapping turn) is a &0 in the
# think_time_us column, distinct from a not-recorded (empty) cell, so an all-overlap session
# uses the recorded zeros rather than degrading to arrival-gap think at replay.
./blis convert weka --input traces.jsonl --trace-output corpus \
  --context-growth accumulate --max-think-time 0
#
# Replay a corpus (from either converter) closed-loop — reconstructs each session's growing prompt.
./blis replay --trace-header corpus.yaml --trace-data corpus.csv \
  --model qwen/qwen3-14b --session-mode closed-loop --max-model-len 1000000
# Or replay a fixed pool of N concurrent closed-loop sessions (#1486, PR-C); --total-sessions
# duplicates the corpus with cache-busting to fill the target (--concurrent-sessions auto-promotes
# to closed-loop). Same huge-ISL caveat — raise --max-model-len and scale --total-kv-blocks.
./blis replay --trace-header corpus.yaml --trace-data corpus.csv \
  --model qwen/qwen3-14b --concurrent-sessions 8 --total-sessions 200 --max-model-len 1000000

# Observe corpus-mode: drive the SAME corpus as a fixed session pool against a
# LIVE server (observe-side twin of `blis replay --concurrent-sessions`), so
# `blis calibrate` can compare real vs simulated over the same session set.
# --corpus-* are the INPUT corpus; --trace-* remain the OUTPUT observed trace.
# Corpus-mode is mutually exclusive with --workload/--workload-spec/--rate/--concurrency.
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --corpus-header corpus.yaml --corpus-data corpus.csv \
  --concurrent-sessions 8 --total-sessions 200 \
  --trace-header observed.yaml --trace-data observed.csv

# Run with gateway queue flow control (utilization-based saturation gating)
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
  --queue-depth-threshold 5 --kv-cache-util-threshold 0.8

# Run with concurrency-based flow control and priority dispatch ordering
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector concurrency \
  --max-concurrency 64 --dispatch-order priority --max-gateway-queue-depth 1000

# Run with flow control and request TTL (expire queued requests after 5 seconds)
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
  --queue-depth-threshold 5 --kv-cache-util-threshold 0.8 --request-ttl 5000000

# Run with SLO-deadline dispatch ordering (tightest SLO target dispatches first)
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
  --queue-depth-threshold 5 --kv-cache-util-threshold 0.8 \
  --dispatch-order slo-deadline --slo-targets "critical=100000,standard=500000"

# Run with flow control and opt-in queue shedding (BLIS-extra, not in llm-d)
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
  --queue-depth-threshold 5 --kv-cache-util-threshold 0.8 \
  --max-gateway-queue-depth 1000 --queue-shedding

# Run with custom dispatch tick interval (default 1000µs = 1ms, llm-d parity)
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
  --queue-depth-threshold 5 --kv-cache-util-threshold 0.8 \
  --dispatch-tick-interval 5000

# Run with opt-in in-flight eviction of sheddable requests (BLIS-extra, not in llm-d)
./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
  --queue-depth-threshold 5 --kv-cache-util-threshold 0.8 --in-flight-eviction

# Run with a single saturation detector (#1516). --detectors takes one of
# composite, threshold, backlog-drift (empty = off). --saturation-report writes a
# {"final":{...},"trace":[...]} JSON file (per-detector final label + one record
# per event). stdout regains a per-detector "saturation" final-label map (#1517),
# derived uniformly from the trace by the last-window plurality reducer.
./blis run --model qwen/qwen3-14b --detectors composite --saturation-report sat.json

# Tune the final-label trailing window (#1517). --saturation-final-window takes a
# Go duration for the last-window plurality vote (default: backlog_drift.window_size_sec
# if configured, else 30s). Same value for every detector; requires --detectors.
./blis run --model qwen/qwen3-14b --detectors all --saturation-final-window 10s

# Run the detector BANK over ONE deterministic replay (#1519). --detectors "all"
# runs the full roster; a comma-list runs exactly the named subset. The bank fans
# each event out to every selected detector, so the trace tags each record by
# detector name. "all" ≡ the full comma-list byte-identically, and a subset
# detector's records are byte-identical to its records under "all" (selection
# filters WHICH detectors run, never HOW they see traffic — INV-6). A single
# <name> still uses #1516's single-detector streaming path for byte-identical
# continuity. Unknown name (single or in a list) → hard error listing valid names.
./blis run --model qwen/qwen3-14b --detectors all --saturation-report sat.json
./blis run --model qwen/qwen3-14b --detectors composite,threshold --saturation-report sat.json

# Tune a detector via a strict-YAML config file (#1516). composite has no params
# (a composite: block errors). threshold has one knob; backlog-drift mirrors
# saturation.BacklogDriftConfig (#1547). The config must carry ONLY the selected detector's
# block — a block for another detector errors (no silent drop). Absent block =
# defaults; partial block overrides only named fields; unknown key / bad value
# errors naming the field.
cat > sat-config.yaml <<'YAML'
backlog_drift:
  window_size_sec: 30
  min_windows: 5
YAML
./blis run --model qwen/qwen3-14b --detectors backlog-drift \
  --saturation-config sat-config.yaml --saturation-report sat.json

# Replay writes the same {"final":{...},"trace":[...]} format and emits the same
# stdout saturation map (run→replay byte-identical, INV-13)
./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b \
  --detectors composite --saturation-report sat.json

# Observe writes the same format over REAL-server latencies (same pipeline,
# different input; #1516)
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload-spec workload.yaml --trace-header trace.yaml --trace-data trace.csv \
  --detectors backlog-drift --saturation-report sat.json
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
- **RFC + .archon plan** (multi-PR features): Tracking issue with holes/surfaces/contracts (see `docs/contributing/rfc.md`), encoded into a machine-checkable `.archon` plan (see `docs/templates/rfc-to-plan.md`). Sub-issues created per hole, each delivered as a PR.
- **Implementation plans** (single PR): Behavioral contracts, TDD tasks. Follow `docs/contributing/pr-workflow.md`.

**The abstraction rule:** Design docs and RFCs describe *what a module does and what it guarantees*. The `.archon` plan describes *what structure to build and in what order*. Implementation plans describe *how to implement each piece*.

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

For new features that introduce module boundaries or modify the architecture, an RFC (per `docs/contributing/rfc.md`) should exist before implementation planning begins. For smaller changes (bug fixes, new policy templates behind existing interfaces), an RFC is optional — proceed directly to `docs/contributing/pr-workflow.md`.

### Code Review Standards

During PR reviews, check all Antipattern Prevention rules (R1-R23) in [`docs/contributing/standards/rules.md`](docs/contributing/standards/rules.md). Pay special attention to rules 8-10 (exported mutable maps, YAML pointer types, strict YAML parsing) which are easy to miss in new code. Always run `go test ./...` and lint after fixes.

### Key Invariants to Maintain

> **Canonical source:** [`docs/contributing/standards/invariants.md`](docs/contributing/standards/invariants.md). If this section diverges, invariants.md is authoritative.

Full details (verification strategies, evidence): see [`docs/contributing/standards/invariants.md`](docs/contributing/standards/invariants.md).

- **INV-1 Request conservation**: `injected_requests == completed_requests + still_queued + still_running + dropped_unservable + timed_out + routing_rejections + gateway_queue_depth + gateway_queue_shed + gateway_queue_rejected + gateway_evicted + gateway_expired + encode_routing_rejections` at simulation end. Full pipeline: `num_requests == injected_requests + rejected_requests`. `encode_routing_rejections` (GAP-4, #1264) is always zero when `--encode-instances 0`.
- **INV-2 Request lifecycle**: Requests transition queued → running → completed; not completed before horizon remain in current state
- **INV-3 Clock monotonicity**: Simulation clock never decreases
- **INV-4 KV cache conservation**: `allocated_blocks + free_blocks = total_blocks` at all times
- **INV-5 Causality**: `arrival_time <= enqueue_time <= schedule_time <= completion_time`
- **INV-6 Determinism**: Same seed must produce byte-identical stdout across runs. Wall-clock timing goes to stderr.
- **INV-7 Signal freshness**: Routing snapshot signals have tiered freshness — InFlightRequests (synchronous) vs QueueDepth/BatchSize/KVUtilization (Periodic by default at 50ms; Immediate when `--snapshot-refresh-interval 0`). See `docs/contributing/standards/invariants.md` for the full hierarchy.
- **INV-8 Work-conserving**: After every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. The simulator must not idle while work is waiting.
- **INV-9 Oracle knowledge boundary**: Servability decisions (enqueue guard, admission, routing, priority) must not read `Request.OutputTokens`. The control plane uses `MaxOutputLen` (client budget) or input-only checks. Only the execution engine may access `OutputTokens` for token generation and completion detection. See `docs/contributing/standards/invariants.md`.
- **INV-10 Session causality**: For all rounds N in a closed-loop session: `round[N+1].ArrivalTime >= round[N].CompletionTime + ThinkTimeUs`. See `docs/contributing/standards/invariants.md`.
- **INV-11 Session completeness**: Every session reaches exactly one terminal state: completed, cancelled, horizon-interrupted, or budget-exhausted (concurrency mode: global request cap reached). No session is silently abandoned. See `docs/contributing/standards/invariants.md`.
- **INV-12 Phase 1 Completeness**: After Phase 1 of `FormBatch`, every non-preempted running request in decode phase has `NumNewTokens > 0`. No request silently skipped due to index drift from non-tail eviction. Trivially satisfied for FCFS. See `docs/contributing/standards/invariants.md`.
- **INV-13 Run/Replay parity**: For any configuration supported by both `blis run` and `blis replay`, a trace exported via `--trace-output` and replayed with identical flags MUST produce identical per-request metrics. Unsupported replay features (autoscaler, node pools) MUST `logrus.Fatalf` at startup — never silent degradation. See `docs/contributing/standards/invariants.md`.
- **INV-BC-DP1 Dense DP=1 step-time byte-identity**: For a dense model at `DP=1` (EP off), `trained-physics` `StepTime` MUST be byte-identical to the pre-#1419 value across the TP matrix (the DP/EP term split is value-preserving for dense). MoE step time intentionally changes (B1 expert-weight scoping + newly-charged MoE-FFN reduction) — a deliberate fidelity gain. See `docs/contributing/standards/invariants.md`.

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

Composable Scorer Framework completed: PR17 (scorer framework + stateless scorers) and PR18 (prefix-affinity scorer + router-side cache). Default weighted routing profile: `precise-prefix-cache:2,queue-depth:1,kv-utilization:1` (llm-d parity). Precise prefix scoring (#883): `precise-prefix-cache` scorer queries actual instance KV cache state with min-max normalization (llm-d production parity); `no-hit-lru` scorer distributes cold requests to least-recently-used endpoints. Valid scorer names: `prefix-affinity`, `precise-prefix-cache`, `no-hit-lru`, `queue-depth`, `kv-utilization`, `load-balance`, `active-requests`, `running-requests`, `load-aware`, `vllm-dp`, `lora-affinity` (#1469, off by default; scores instances with the request's adapter already resident higher, min-max normalized).

LoRA control-plane subsystem (#1464, epic PRs 1–7): adapter identity + pre-declared `id→rank` registry, per-instance resident set (capacity-bounded LRU), three DT-derived cost terms (cold-load latency, per-step compute overhead, static HBM reservation), the `lora-affinity` scorer, and per-adapter metrics. No-op by default (`lora:` absent ⇒ byte-identical output, INV-6). Adapter ids round-trip through TraceV2 (trailing conditional `adapter` column) for run/replay parity (INV-13). Fidelity vs the Agullo Digital Twin (#1470, `blis calibrate --adapter-reference`): the compute-overhead (throughput) term validates ≤20% MAPE for both calibrated configs (Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct); the absolute-TTFT leg is bounded but unsupported (BLIS ports adapter *deltas* onto its own separately-calibrated base, which differs from the DT's H100 base fit) — reported honestly, never silently passed.

Phase 0 workload unification complete (see issue #420): W0-1 (spec v2 schema + SLO tiers), W0-2 (binary rename + converters), W0-3 (cohort population dynamics), W0-4 (legacy retirement). All workload generation now flows through `sim/workload/GenerateRequests()`. SLO tiers: critical, standard, sheddable, batch, background. Arrival processes: poisson, gamma, weibull, constant. CLI binary renamed from `simulation_worker` to `blis`.

Observe/replay/calibrate pipeline complete: `blis observe` (#659) dispatches workload to real servers with closed-loop session support, `blis replay` (#689) replays through DES, `blis calibrate` (#701) compares real vs simulated latencies. Observe fidelity (#660): chat completions endpoint (`--api-format chat`), `stream_options` for streaming token counts, `finish_reason` extraction, configurable `max_tokens` (`--unconstrained-output`), deterministic prefix strings for KV cache activation, `--rtt-ms` for network RTT.

Replay injection-origin normalization (#1606): `blis replay` re-bases each request's DES injection time onto the trace's `arrival_time_us`/`deadline_us` origin. A real `blis observe` trace writes `send_time_us` in Unix-epoch µs but `arrival_time_us`/`deadline_us` on a run-relative clock; using the raw epoch `send_time_us` as an absolute injection tick made every request instantly past-due (0 completions, empty `sim_result.json`). `LoadTraceV2Requests`/`LoadTraceV2SessionBlueprints` subtract a single per-trace `injectionOriginShift = min(injectionTime) − min(arrival_time_us)`, preserving #1304's send-delta (concurrency-slot-wait) spacing. Generated `blis run` traces write `send_time_us == arrival_time_us`, so the shift is exactly 0 and replay stays byte-identical (INV-13/INV-6); the closed-loop preliminary horizon uses `workload.MaxNormalizedInjectionTimeUs` (the normalized injection, not raw `arrival_time_us`).

KV cache hit-rate calibration (#1583, epic #1585 S6): `blis observe --scrape-kv-metrics` scrapes the server's Prometheus `/metrics` KV-offload tiering counters (`vllm:kv_offload_tiering_block_hits`/`_block_queries`; released fallback `vllm:gpu_prefix_cache_*`, tagged distinctly) over the measured window and records the observed hit-rate in a new trace-header block (`observed_kv_metrics`, with `--vllm-commit` pinning the unreleased vLLM PR #48798). `blis replay --metrics-path` writes the sim aggregate `cache_hit_rate` (a `*float64` on `MetricsOutput`, populated file-only in `EmitOutput` so stdout stays byte-identical — INV-6). `blis calibrate --sim-metrics` then compares TTFT, E2E, **and** KV hit-rate (`hit_rate` report block, default ≤5 pp band; TTFT MAPE ≤0.15 verdict). On replay a tiered observation with no reproducible `kv_offload` config is a hard error (BC-10, INV-13, never silent GPU-only degradation); the observed block round-trips verbatim on re-export. The sim exposes one aggregate hit-rate (no per-tier), so the automated comparison is overall-hit-rate; per-tier read/write time is recorded for a documented manual bandwidth cross-check. Pure Prometheus-text parser + hit-rate derivation in `sim/workload/prom_hitrate.go` (no new go.mod dep). Empirical cluster tolerance-pass is an operator step (needs an unreleased-vLLM GPU cluster); see `docs/guide/kv-offload-calibration.md`.

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

### GitHub Action: PR Reviews

When triggered via `@claude /blis-pr-review` on a PR, follow the blis-pr-review skill exactly. For all other triggers (questions, debugging, etc.), respond normally without creating a PR unless explicitly asked.

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

Every issue must have at least one label. Use `gh issue create --template "Template name"` to pre-fill the template.


## Post-Hoc Saturation Detection

BLIS includes post-hoc saturation detection for analyzing completed runs. This is distinct from the real-time flow control saturation detector used for admission control.

**Package**: `sim/saturation/`

**Streaming detectors** (all stream via `Observe`/`Detect`; the batch `Classify` path was removed in #1516):
- **composite**: Combines rate deficit (1 - completions/arrivals) and a quartile-filtered latency trend. Zero parameters (a `composite:` config block errors). STABLE / BACKLOGGED / OVERLOADED by score vs a 1/√arrivals noise floor.
- **threshold**: Mean E2E latency vs a configurable threshold (default 5000ms). STABLE when mean < threshold, OVERLOADED when mean > threshold.
- **backlog-drift**: Online OLS slope of in-flight over a trailing window (became streaming in #1515), banded against the noise floor.

**CLI flags** (`run`, `observe`, `replay`; #1516, #1519, #1517):
- `--detectors <selection>`: empty = off. A single name (`composite`, `threshold`, `backlog-drift`) runs #1516's single-detector streaming path. `all` runs the full roster; a comma-list (e.g. `composite,threshold`) runs exactly the named subset. `all` and comma-lists route through the **detector bank** (#1519). Unknown name — single or inside a list — is a hard error listing valid names (R1).
- `--saturation-config <path>`: strict-YAML tuning file with optional `threshold:` and `backlog_drift:` blocks. composite has no block. For a **single** detector the config must carry only that detector's block — a foreign block errors (no silent drop, R1). For the **bank**, ownership is enforced over the selected SET (`checkBlockOwnershipSet`): a block whose owning detector is not among the selected names is a hard error (R1), same as the single-detector path — `--detectors all` selects every owner so a full shared config is fine, but a subset that omits a detector whose block was supplied errors. A value error inside a *selected* detector's own block also errors. Absent block = defaults; a partial block overrides only named fields; an unknown key or out-of-range value errors naming the field; an empty file = all defaults.
- `--saturation-report <path>`: writes the selected detector(s)' **final label + per-event verdict trace** as one `{"final":{...},"trace":[...]}` JSON object (one trace record per event, tagged by detector name; map keys sorted so repeated runs are byte-identical). Requires `--detectors`. Optional — the stdout final label (#1517) is emitted regardless. `--saturation-config`, `--saturation-report`, and `--saturation-final-window` without `--detectors` are hard errors, as is an unwritable report path (checked up front).
- `--saturation-final-window <duration>` (#1517): Go duration for the trailing window of the stdout final-label plurality vote. Resolution: this flag if set → else `backlog_drift.window_size_sec` from `--saturation-config` → else 30s. Same value for every detector; a non-positive or unparseable value is a hard error. Requires `--detectors`.

**Final label reducer** (`sim/saturation/reduce.go`, #1517): `ReduceOne(records, windowUs) Level` collapses ONE detector's per-event trace into a headline label by a **last-window plurality** rule — keep records within `windowUs` of the max timestamp, take the most-frequent `Level`, break count-ties toward the more severe level (`OVERLOADED > BACKLOGGED > STABLE`), empty group → `STABLE` (R20). `ReduceAll(records, windowUs) map[string]Level` groups by detector name and reduces each group. It is a **pure function, not a `Detector` method** — so every detector is collapsed identically (fair cross-detector comparison) and new detectors get final labeling for free. The rule is order-independent (INV-6) and identical traces yield identical maps (INV-13). `cmd` calls `ReduceAll` for every selection: a single detector yields a one-key map, `all` the full map. There is exactly one stdout shape — a `map[string]Level` (`--detectors composite` → `{"composite":"STABLE"}`, never a bare label).

**Detector bank** (`sim/saturation/bank.go`, #1519): `Bank` holds and drives a roster of streaming detectors over ONE deterministic replay, fanning each event out to every selected detector (`fanout`) so all are scored on a byte-identical event sequence in a single pass. It reimplements no `Detector` method — it only multiplexes the shared `buildSortedEvents` (a #1519 extraction) + #1516's `TraceSink` + `WriteCombinedReport`. Its sole public driver is `Run(requests) error` (the multi-detector analogue of `ReplayOneDetector`); the collected records are reduced to the stdout label by `saturation.ReduceAll` in `cmd` (#1517). `NewBank(names, cfg, sink)` validates/de-dups names and orders the roster canonically (`composite`, `threshold`, `backlog-drift`), so selection order and spelling never change output: `all` ≡ the full comma-list byte-identically, and a subset detector's records are byte-identical to its records under `all` (selection filters WHICH detectors run, never HOW they see traffic — INV-6/INV-13).

**One pipeline, three input adapters**: run/replay/observe all produce the trace through the same shared `resolveSaturation → saturationTracer.run` path (single detector via `ReplayOneDetector`, bank via `Bank.Run`), reducing to the final label via `ReduceAll` and writing the report via `TraceSink → WriteCombinedReport` (R23). The only difference is the `[]RequestMetrics` input — run/replay from the sim (`Metrics.CompletedRequestMetrics()`), observe from the real server (`workload.TraceRecordsToRequestMetrics`). run→replay of the same trace is byte-identical (INV-13); observe's trace reflects real-server latencies by design.

**stdout** (#1517): a run WITH `--detectors` regains a `"saturation"` field — a per-detector `map[string]Level` final label spliced onto the metrics JSON via the goodput build-then-mutate-then-emit pattern (`cmd` sets `MetricsOutput.Saturation`; `sim` stays saturation-agnostic). A run WITHOUT `--detectors` is byte-identical to the historical no-feature output (the field stays dropped by `omitempty`, BC-8). The former `sim.BatchClassifier` seam was retired (`sim/classifier.go` deleted; `BuildOutput`/`SaveResults` lost their detector param) since `cmd` imports both `sim` and `sim/saturation` and wires the reducer directly.

**Trace file example**:
```json
{
  "final": { "composite": "STABLE" },
  "trace": [
    {
      "timestamp": 150000,
      "detector": "composite",
      "result": {
        "level": "STABLE",
        "score": 0.35,
        "confidence": 0.95,
        "signals": { "latency_trend": 0.12, "rate_deficit": 0.0 }
      }
    }
  ]
}
```

**Migration from the pre-#1516 flags**:
- `--post-hoc-detector X` → `--detectors X`
- `--saturation-threshold-ms N` → `--saturation-config` `threshold: {threshold_ms: N}`
- the 10 backlog-drift tuning flags (`--saturation-window`, `--saturation-min-windows`, `--saturation-classifier`, …) → `--saturation-config` `backlog_drift:` block
- the standalone `--saturation-report` (per-window `BacklogDriftReport`) is removed; `--saturation-report` now writes the `{"final":{...},"trace":[...]}` object (final label + per-event trace)
- detector `Classify` is removed from the `Detector` interface (streaming-only: `Name`/`Observe`/`Detect`/`Reset`); the post-hoc batch analysis library (`workload.AnalyzeBacklogDrift*`, the `slope-based`/`drain-ratio` classifiers, `BacklogDriftReport`) was fully removed in #1547 — it had no live-path caller once the streaming detectors (#1515/#1516) and the reducer (#1517) landed. `BacklogDriftConfig` was relocated verbatim into `sim/saturation` in the same PR, fully decoupling `sim/saturation` from `sim/workload` (the config type was the last import edge)

**Use cases**: a one-line end-of-run saturation verdict per detector on stdout (#1517) for quick capacity-planning answers, plus per-event saturation trajectories (the trace file) for detecting queue buildup or throughput saturation over time. The bank (`--detectors all`) additionally lets you compare detectors head-to-head on byte-identical traffic in a single run — both the final labels and the trajectories — instead of separate runs where each detector sees different traffic.

## File Organization

For the full annotated file tree, see [`docs/reference/project-structure.md`](docs/reference/project-structure.md).

### Latency Estimation

Two latency model modes (trained-physics, roofline), selected via `--latency-model` flag. **Trained-physics is the default** — it provides better out-of-box accuracy by combining roofline basis functions with learned correction coefficients.

**Migration note:** The deprecated `blackbox`, `crossmodel`, and `trained-roofline` backends have been removed. Use `--latency-model trained-physics` for modern physics-informed estimation with MoE-aware overhead modeling.

**Trained-physics model** (default): Roofline basis functions with learned correction coefficients. Generalizes across model architectures, workloads, and TP configurations. No per-model calibration needed.

**Roofline model**: Pure analytical model available via explicit `--latency-model roofline` flag. Useful when you want FLOPs/bandwidth-based estimation without learned corrections.

See [`docs/guide/latency-models.md`](docs/guide/latency-models.md) for details.

**Quantized model support**: Three-tier auto-detection of weight precision: (1) `quantization_config` in HF `config.json` — GPTQ/AWQ (`bits`), FP8 (implicit), compressed-tensors (`config_groups.*.weights.num_bits`); (2) model name conventions (`w4a16` → 0.5, `FP8` → 1.0 via `InferWeightBytesFromModelName`); (3) fallback to `BytesPerParam` from `torch_dtype`. Uses quantized weight precision for weight bandwidth and KV capacity calculations while keeping compute dtype for KV cache and activations. `ModelConfig.WeightBytesPerParam` (0=fallback to `BytesPerParam`) with `EffectiveWeightBytesPerParam()` accessor decouples weight storage precision from compute/KV dtype.

**MLA / model-shape KV & weight fidelity (#1527, F1–F3)**: The KV-capacity model (`sim/latency/kv_capacity.go`, `config.go`) represents the modern MLA MoE family (DeepSeek-V2/V3, Kimi-K3, GLM-5.2 `glm_moe_dsa`). **F1 — explicit `head_dim`**: `ModelConfig.HeadDim` (`json:"head_dim"`) + `EffectiveHeadDim()` accessor (returns `HeadDim` when >0, else `HiddenDim/NumHeads`) feed `KVBytesPerToken` and `computeModelWeightBytes`; the step-time models (trained-physics/roofline) intentionally still use `hidden/heads`, so step-time goldens + INV-BC-DP1 are byte-identical. **F2 — MLA compressed-KV**: when `ModelConfig.KVLoraRank > 0`, `KVBytesPerToken` returns `(kv_lora_rank + qk_rope_head_dim) × num_layers × BytesPerParam` — a single latent per token per layer, independent of `num_kv_heads`/`head_dim` and **NOT divided by TP** (the latent is replicated across TP ranks, matching vLLM's MLA cache); both auto KV-block sizing and PD KV-transfer sizing inherit it. **F3 — dense-prefix MoE**: `ModelConfig.FirstKDenseReplace` splits `computeModelWeightBytes` into K dense-MLP layers + (L−K) MoE-MLP layers (K clamped to [0,L]), distinct from the every-Nth `InterleaveMoELayerStep`. All three are **no-ops when the config keys are absent** (INV-6 byte-identity); INV-4/INV-13 preserved (`run`/`replay` share the path; `observe` doesn't derive capacity from shape). Committed fixture: `model_configs/glm-5.2-fp8/config.json`. **Documented known approximations (F4/F5a)**: block-wise FP8 (`weight_block_size`, `modules_to_not_convert`) treated as flat 1.0 byte/param (optimistic); DSA indexer and MLA attention weight projections unmodeled. MTP/spec-decode throughput is out of scope (#1528).

**Per-instance KV capacity for mixed-GPU node pools (#1522)**: When node pools are configured (`--policy-config` with `node_pools`) and `--total-kv-blocks` is NOT explicitly set, each placed instance auto-calculates its KV-block capacity from its ACTUAL placed GPU's `gpu_memory_gib` (plus TP, DP, block size, `--gpu-memory-utilization`, and weight precision) — so an H100 pool and an L40S pool no longer share one global capacity (restores INV-P2-1: an instance's GPU calibration and KV capacity describe the same device). Applied at all three placement sites (startup, deferred `NodeReadyEvent`, autoscaler scale-up) via `cluster.applyPerInstanceKVCapacity`, immediately after the `HWConfigByGPU` execution-calibration override (issue #893) so the placed GPU is authoritative for capacity as well. **Precedence**: an explicit `--total-kv-blocks` disables per-instance recalc (every instance keeps that uniform global capacity); when both node pools and PD per-pool KV overrides are present, the placement-derived per-GPU capacity wins (mirrors how `HWConfigByGPU` overrides the resolved `HWConfig`). A per-GPU capacity smaller than the configured `--max-model-len` auto-caps that instance's `MaxModelLen` to the KV-feasible maximum. Missing pool memory or a capacity-calc error falls back to the inherited global capacity with a warning (never a panic). `blis replay`/`observe` reject node pools, so this is `blis run` only (INV-13 parity N/A). Distinct from #1315 (role-specific capacity correct, latency coefficients wrong) and #633 (per-role overrides that can't express mixed hardware within one role).

**KV block keying (`sim/internal/kvkey`, #1589, hole H4 of epic #1585)**: the single gated home for KV block-key derivation and interning. `DeriveChunkKeys(prevKey, tokens, tokensPerChunk)` produces hierarchical content keys at an arbitrary stride — a strict generalization of `hash.ComputeBlockHashes` (at `tokensPerChunk == blockSize`, `prevKey == ""` it is byte-identical, BC-K1). With `tokensPerChunk = tokensPerBlock × blocks_per_chunk` it yields one key per **chunk** (a group of `blocks_per_chunk` blocks), matching vLLM's per-chunk offload keys — NOT one key per block (BC-K4). `Interner` maps each distinct 64-hex `BlockKey` to a dense integer `KeyID` — injective, idempotent, deterministic given call order (BC-K3) — with a reverse `Key(KeyID)` accessor for boundaries where content identity must stay a string (KV events, traces, the router index). All hashing routes through `sim/internal/hash` (the sole hash source, BC-K1), enforced by a static-analysis test (`static_test.go`) over `sim/kv` + the box: production code must not import `crypto/*`, stdlib `hash`/`hash/*`, or `fnv`. **Chunk keys are a disjoint keyspace** — `DeriveChunkKeys` chain-hashes each whole chunk (BLIS's own SHA256), so a chunk key matches no block hash by value (vLLM instead anchors each chunk key to the chunk's trailing block-hash; the frozen surface here provides only the chunk stride). A consumer with the request tokens derives both keyspaces from this one hash source, so no cross-referencing capability is lost. **Wired into the offload hot path as of #1590**: `sim/kv.OffloadCache.consultAndReload` calls `DeriveChunkKeys` (seeded by the resumed prefix's last hash, so only the uncached tail is keyed) — so the INV-6 byte-identity guarantee is now conditional on offload being **disabled** (`cfg.Offload` absent ⇒ the offload path, and this call, never run ⇒ byte-identical stdout and zero benchmark regression). The `Interner` is not yet on the hot path; when a future hole adopts it, a `KeyID`-keyed probe beats a 64-byte-string probe for the disk-tier / ref-count / in-flight-job probes.

**N-tier KV-offload chain (`sim/kv.OffloadCache`, #1590, hole H1 of epic #1585)**: the multi-tier offload mechanism the #1587 config surface drives. `OffloadCache` implements `sim.KVStore` (still **12 entities**) and composes the GPU tier (`KVCacheState`) with a `ref_cnt`-managed **CPU staging tier** and ordered secondary **`fs` tiers**, plus the #1588 bounded transfer station (`sim/kvtransfer`) for per-job service. Activated **only** when `cfg.Offload.IsEnabled()` (`--kv-offload-config`); the legacy `--kv-cpu-blocks` `TieredKVCache` is untouched and the two are mutually exclusive (loud error if both set). Absent ⇒ byte-identical output (INV-6). **Mechanism (models vLLM `tiering/manager.py` + `cpu/manager.py`, not just its outputs):** the CPU tier is a `ref_cnt` state machine — `-1` allocated-not-ready (HIT_PENDING, unreadable AND unevictable), `0` ready-evictable (HIT), `n>0` pinned (HIT, non-evictable) — with an **O(1) evictable counter** maintained only on zero-crossing transitions (BC-C8). `AllocateKVBlocks` reloads CPU-resident prefix blocks to GPU synchronously and kicks off an **async promotion** (a station Read job) for a secondary-resident run under the **evictable gate** (BC-C5: a promotion of `k` blocks succeeds iff `k ≤ free + evictable`, gating on the *evictable* count, NOT the free count — so a full-but-locked CPU tier degrades promotions to recompute, the non-linear mode a capacity gate misses). `MirrorToCPU` stores each request's newly-completed blocks and **cascades** each to *every* secondary tier (BC-C7a write-through fan-out), pinning the CPU block for each write's duration — the lock that starves the evictable pool. **Which blocks are offered for store is a single offloadable-token clamp (`offload_prompt_only`, S8/#1584), modeling vLLM's `_calc_num_offloadable_tokens` + `storable_chunks`** — NOT a per-block prompt/decode classification: the request's computed KV (its full owned blocks) is truncated to the prompt length when `offload_prompt_only` (vLLM default TRUE), then floor-divided by the chunk stride into whole chunks, so a chunk holding any decode token is never formed (a `1.5×`-chunk prompt offers exactly 1 chunk). With `offload_prompt_only: false` (`promptAndDecode`) full decode blocks are offered too, and — because BLIS already hashes every completed block prefix-consistently (the partial-fill path, `cache.go`), *including* decode-generated ones for `block_size > 1` — they become CPU-resident and are reloaded by a later same-instance request whose input contains those tokens, so hit-rate reflects the policy. Decode offload needs **no** new hashing: `MirrorToCPU` is a pure *consumer* of GPU block hashes (it never writes `HashToBlock`), so switching the policy cannot perturb GPU-tier behavior and default/prompt-only runs stay byte-identical (INV-6). Completions apply only via `SetClock`→`station.Poll` (before same-step lookups). A secondary block reaches GPU strictly via **two hops** (secondary→CPU→GPU, BC-C1), never directly. Timing flows through **token counts** (recompute raises prefill tokens via the existing `StepTime` model) — H1 charges no explicit per-request transfer latency and adds no `sim.Event` types; the dominant step-boundary deferral is H3 (#1591, blocked by H1). The resolved `per_block_bytes` (derived `KVBytesPerToken × block_size`, since `sim/kv` cannot import `sim/latency`) round-trips through the trace header for run/replay parity (INV-13). **H1 scope trims (rejected loudly, follow-ups):** `blocks_per_chunk == 1` only (block-granular; `>1` chunk coalescing deferred); `eviction_policy: lru` only (`arc` deferred). Offload tiers are invisible to the `precise-prefix-cache` router scorer (`GetCachedBlocks`/`SnapshotCachedBlocksFn` stay GPU-only) — a routing-fidelity boundary. Files: `sim/kv/offload_chain.go`, `offload_cputier.go`, `offload_secondary.go` (+ `offload_*_test.go`).

**Step-boundary KV-deferral (`sim/kv.OffloadCache` + `sim.DeferrableKVStore`, #1591, hole H3 of epic #1585 — the dominant TTFT effect)**: the piece H1 deferred. A new prefill admission whose prefix needs KV blocks resident only on a secondary tier is **not** promoted-and-recomputed immediately (the H1 behavior) — it is **set aside and re-examined at each scheduler step** (vLLM `scheduler.py:835-841` `continue`), so the offload-attributable TTFT delay is a **whole multiple of step time, not disk latency** (BC-T2, BC-T7). Because step time grows with batch size while disk latency does not, a bandwidth model drifts in the wrong direction; the discriminating metamorphic test proves ~3× longer steps ⇒ ~3× delay while halving disk bandwidth barely moves it. **Round count (verified vs vLLM):** a COLD secondary hit costs `k ≥ 3` rounds — `RETRY` (the tier's existence check is itself async/step-batched) → promote (secondary→CPU Read submitted, CPU slot `ref_cnt=-1`) → in-flight → `HIT`; a WARM hit (existence resolved, tracked in a persistent per-key `existenceKnown` cache) costs `k ≥ 2`. Both are modeled and distinguished (`cold − warm == 1`, BC-T4). **Realization:** no new `sim.Event` — the delay is the request sitting in the WaitQ, admitted `k` steps later (its `FirstTokenTime` shifts by `(k-1)×step`); `ConsumePendingTransferLatency` stays 0. Deferred requests **stay in the WaitQ**, so INV-1 (`still_queued`) and INV-8 (`scheduleNextStep` schedules a StepEvent while `WaitQ.Len() > 0`, or an `AdapterLoadCompletionEvent` is pending under a co-active LoRA load) hold for free. `DeferrableKVStore` (`PollDeferred`/`IsDeferred`/`ClearDeferred`, implemented only by `OffloadCache`) is type-asserted by `FormBatch`: Phase 2 becomes a **non-blocking skip-scan** — a still-deferred request is skipped and requests behind it are admitted (vLLM `step_skipped_waiting`), while GPU pressure still breaks head-of-line. `PollDeferred` (once per step, top of `FormBatch`, after `SetClock` applies completions — completions-before-lookups, BC-T6) advances the state machines in **sorted `Request.ID` order for its side effects** (station `Submit` assigns JobIDs by call order; `prepareStore` evicts CPU-LRU by call order) so runs are byte-identical (INV-6); cost is **O(deferred), not O(waitq)** (property P). **Gating (correctness):** deferral fires only for NEW admissions (`!running`) — running-request continuations keep the H1 background-promote path (where a `false` return is GPU pressure, not "skip"); the resolved/recompute admit paths never re-enter defer; and a bounded single fetch attempt (plus the station's completion guarantee) means no request defers forever (BC-T3). `ClearDeferred` is wired into the non-admit WaitQ removals (timeout, gateway eviction, drain-redirect) so the deferred map never leaks. The whole mechanism is inert (byte-identical, INV-6) when offload is off or has no secondary tiers, and adds no new package arrow (`DeferrableKVStore` lives in `sim`, implemented in `sim/kv`). Files: `sim/kv/offload_deferral.go` (+ `offload_deferral_test.go`, `offload_deferral_bench_test.go`), `sim/batch_formation.go`, `sim/kv_store.go`; e2e `sim/cluster/offload_deferral_e2e_test.go`.

**Non-linear FS device model (`sim/kvtransfer` + `sim/kv.OffloadCache`, #1581, S7 of epic #1585)**: refines the transfer station's linear `base + bytes/bandwidth` service cost (#1588 BC-S3) into a device curve, all **opt-in** (a device class that declares none of the fields resolves byte-identically to pre-#1581, INV-6). Three additive layers: **(1) queue-depth bandwidth ramp (deterministic, in the station)** — `TierConfig.{SaturationQueueDepth Qsat, SingleTransferFraction f₁}` make effective per-transfer bandwidth ramp linearly from `f₁·bw` at the tier-direction's in-service depth `q=1` up to the peak `bw` at `q=Qsat`, flat beyond; `q` is fixed at service-start (never recomputed mid-flight, so `completeAt`/BC-S4 stay stable); the ramp is disabled (constant `bw`) when `Qsat≤1`, `f₁≥1`, or `f₁≤0`. **(2) O_DIRECT vs buffered regime** — each `kv_offload_devices` device class may declare a buffered `(bandwidth, base_latency[, Qsat, f₁, σ])` set alongside its O_DIRECT set; the per-tier `direct_io` bool (captured by #1587) now **selects** the regime (`resolveDeviceRegime`), buffered falling back per-field to the O_DIRECT value. **(3) optional seeded latency jitter** — a per-device relative stddev `σ` (`latency_jitter_stddev`) makes `OffloadCache` draw a multiplicative factor `max(0.05, 1+N(0,σ))` from a dedicated seeded RNG partition (`SubsystemKVOffload`, derived from the run seed via `NewKVStore(cfg, seed)`) and pass it on the `TransferJob`; the station applies it but **draws no randomness itself** (its no-RNG determinism guarantee, BC-S4, is preserved — it consumes a caller-supplied scalar). `σ=0` (default) draws nothing ⇒ byte-identical. The three resolved fields round-trip through the trace header (`TraceKVOffloadTier`) so run→replay under the same seed is byte-identical (INV-13). Delivered under the single `--kv-offload-config` flag (no new flag); the committed device classes are unchanged (ramp/jitter/buffered all absent). Observability is via completion-timing → hit/miss/recompute → token counts (the service curve moves *when* promotions/cascades complete, hence hit/miss/recompute and step counts; the per-request TTFT deferral itself is H3's mechanism, #1591). Files: `sim/kvtransfer/station.go`, `sim/kv/offload_chain.go`, `sim/kv_offload_config.go`, `cmd/kv_offload.go`, `cmd/default_config.go`, `sim/workload/tracev2.go`, `sim/rng.go`.

### Key Data Flow

Request processing pipeline: Arrival → Admission → Routing → WaitQueue → Batch Formation → Step Execution → Completion. Admission and Routing apply in cluster mode only; single-instance skips directly to WaitQueue. See [`docs/concepts/architecture.md`](docs/concepts/architecture.md) for the full diagram.

## Project Governance Documents

### Standards (what rules apply)

- `docs/contributing/standards/rules.md`: **23 antipattern rules** (R1-R23) — each with evidence, checks, enforcement locations
- `docs/contributing/standards/invariants.md`: **13 system invariants** (INV-1 through INV-13) — with verification strategies
- `docs/contributing/standards/principles.md`: **Engineering principles** — separation of concerns, interface design, BDD/TDD
- `docs/contributing/standards/agent-trust.md`: **Agent trust boundaries** — three trust tiers (Trusted, Verify-after, Never-trust) for agent operations, with known failure modes

### Process (how to do each activity)

- `docs/contributing/pr-workflow.md`: End-to-end PR workflow (worktree → plan → review → implement → audit → commit)
- `docs/contributing/rfc.md`: RFC template for large features (tracking issue with holes/surfaces/contracts)
- `docs/templates/rfc-to-plan.md`: Claude prompt for encoding RFC into .archon plan + creating sub-issues

### Templates (what to produce)

- `docs/contributing/templates/design-guidelines.md`: **BLIS Design Guidelines** — DES foundations, module architecture, extension framework. **Start here when designing anything new.**
- `docs/templates/rfc-to-plan.md`: Claude prompt for .archon encoding + sub-issue creation

### Per-Feature Plans

- **Active plans:** `docs/plans/` (implementation plans for in-progress work)
- **Archived design docs:** `docs/plans/archive/` (completed design docs for architectural reference)
- **PR history:** Use `git log --oneline main` for the definitive commit history

## Active Technologies
- Go 1.22+ + `gopkg.in/yaml.v3` (strict parsing), `gonum` (stats), `cobra`, `logrus`
- In-memory node/GPU inventory maps; no external storage

## Change History

See `git log --oneline main` for the definitive commit history. Durable cross-cutting facts live in the standards docs and topic guides (admission, routing, scheduling, observe-replay-calibrate, workloads, configuration reference).
