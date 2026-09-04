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

# Run with speculative decoding / MTP (#1528). Models the throughput of a model that
# verifies K draft tokens per decode step (MTP/EAGLE/Medusa; GLM-5.2 ships 5-token MTP).
# --num-speculative-tokens K: draft tokens proposed per step (0 = off, the default).
# --speculative-acceptance-rate α: mean fraction of the K drafts accepted, in [0,1];
#   REQUIRED when K>0 (α defaulting to 0 would model pure slowdown). Decode advances by
#   ~1+α·K tokens/step (throughput) while each step pays the K+1-position verify cost
#   (so accepting isn't free). --speculative-method labels the scheme (mtp|eagle|medusa|
#   ngram|draft). K=0 ⇒ byte-identical to a run without the flags (INV-6). α is
#   user-supplied, not predicted. Same flags on `blis replay` (INV-13, model-level so no
#   trace change). Note: under spec-decode the raw ITL percentiles are per-verification-
#   step; use TPOT for per-token latency (SLO-ITL attainment reads TPOT, unaffected).
./blis run --model zai-org/GLM-5.2 \
  --num-speculative-tokens 5 --speculative-acceptance-rate 0.7 --speculative-method mtp

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

# Compare aggregate throughput too (#1647). A `throughput` block is added to the report
# automatically from the already-required inputs (no flag), comparing real vs simulated
# output-token throughput (total output tokens / makespan) and request throughput (matched
# requests / makespan). Both makespans are in the CLIENT frame over the same request_id-matched
# set: real end = last-chunk time; sim end = send + client-frame sim E2E (sr.E2E normalized with
# the same network shift as the latency comparison). Omitted (never Inf/NaN) when the makespan is
# non-derivable. --throughput-tolerance-pct adds a within-tolerance verdict on raw output-token
# throughput; --num-gpus adds per-GPU normalization (output_tokens_per_sec_per_gpu, real+sim) —
# operator-supplied since the trace header records only TP (not DP/PP/instances). Per-GPU
# percent_error equals raw percent_error (one GPU count divides both sides). Validity boundary:
# the reconstructed sim makespan is a physical sim timeline only for FIXED-mode replay; under
# closed-loop replay the sim regenerates the arrival schedule, so treat it as open-loop only.
./blis calibrate --trace-header t.yaml --trace-data d.csv --sim-results results.json \
  --throughput-tolerance-pct 15 --num-gpus 4 --report calibration.json

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
# Context compaction (#1609): real agentic traces compact/trim context heavily — ~30% of
# rounds on the full 051926 dataset (219 sessions, 37.7K rounds) have in_N < in_{N-1}+out_{N-1}.
# The shared encoder now emits a per-round input_tokens_reset marker (the recorded absolute)
# on exactly those non-monotone rounds, and accumulate closed-loop replay RE-SEEDS its growing
# buffer to that absolute at the boundary — so the reconstructed input tracks the recorded
# cumulative input (previously it clamped the delta to 0 and over-counted by ≈3–4× / +312% on
# this dataset). Re-seeding intentionally breaks strict prefix identity across the compaction
# boundary (the summary is not a literal prefix of the pre-compaction buffer), which also
# corrects the prefix-cache hit-rate over-estimate. The marker is a trailing conditional CSV
# column, absent for monotone sessions and for `blis run`, so a trace without any compaction
# round replays byte-identically to before (INV-6). Traces converted by an OLDER build (no
# marker column) still over-count — re-run convert to get compaction-aware output.
# The recorded think time is likewise non-lossy (#1608): a genuinely-zero recorded think (an
# overlapping turn) is a &0 in the think_time_us column, distinct from a not-recorded (empty)
# cell, so an all-overlap session uses the recorded zeros rather than degrading to arrival-gap
# think at replay.
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
# composite, threshold, backlog-drift, peak-rate (empty = off). --saturation-report writes a
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

# Tune a detector via a strict-YAML config file (#1516, #1614). EVERY detector now
# has a false-alarm calibration knob: composite: {sensitivity},
# threshold: {threshold_ms}, backlog_drift: {slope_k, ...} (backlog-drift mirrors
# saturation.BacklogDriftConfig, #1547), peak_rate: {threshold, min_observations,
# warmup_us, consecutive_k, overload_multiple}. The config must carry ONLY the selected detector's
# block — a block for another detector errors (no silent drop). Absent block =
# defaults; partial block overrides only named fields; unknown key / bad value
# errors naming the field.
cat > sat-config.yaml <<'YAML'
backlog_drift:
  window_size_sec: 30
  min_windows: 5
  slope_k: 3.0          # BACKLOGGED/OVERLOADED boundary multiplier (#1614)
YAML
./blis run --model qwen/qwen3-14b --detectors backlog-drift \
  --saturation-config sat-config.yaml --saturation-report sat.json

# Run the peak-rate detector (#1614): R_t = Peak_t/t, the backlog high-water mark
# over elapsed time. Needs NO latency target and NO capacity estimate, unlike
# --detectors threshold whose millisecond target must be re-tuned per model/GPU.
# threshold is in backlog per second, so calibrate it per deployment: run a healthy
# workload and raise it until it stops firing.
cat > pk.yaml <<'YAML'
peak_rate:
  threshold: 0.5           # primary false-alarm dial; larger fires less
  min_observations: 20     # hold the verdict until enough EVENTS have been seen
  warmup_us: 0             # hold the verdict until this much TIME has elapsed
  consecutive_k: 3         # successive breaches before firing
  overload_multiple: 3.0   # OVERLOADED above this multiple of threshold (>= 1)
YAML
./blis run --model qwen/qwen3-14b --detectors peak-rate \
  --saturation-config pk.yaml --saturation-report sat.json

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
- **RFC + .archon plan** (multi-PR features): Tracking issue with holes/surfaces/contracts (see `docs/contributing/rfc.md`), encoded into a machine-checkable `.archon` plan (see `docs/contributing/templates/rfc-to-plan.md`). Sub-issues created per hole, each delivered as a PR.
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
- **INV-13 Run/Replay parity**: For any configuration supported by both `blis run` and `blis replay`, a trace exported via `--trace-output` and replayed with identical flags MUST produce identical per-request metrics; at the CLI boundary assert the stronger **byte-identical stdout** (it subsumes them), with a non-vacuity gate and a conservation companion. "Identical flags" includes `--horizon` — run defaults it to unlimited while replay auto-computes 2x the max arrival time, so a parity comparison must pass it on both legs. Unsupported replay features (autoscaler, node pools) MUST `logrus.Fatalf` at startup — never silent degradation. MoE `--dp>1` as placement was run-only from #1531 and is supported on both paths since #1556 (one shared `cmd.resolveDPPlacement`). See `docs/contributing/standards/invariants.md`.
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

**EP placement + live EP-mode step time (#1548)**: `--enable-expert-parallel` is now **supported alongside MoE `--dp N`** on both `blis run` and `blis replay` (the `planDPPlacement` rejection is gone) and is **step-time-live**. It adds **no** cost term — a second additive comm term would double-charge the dispatch/combine the model already prices (#1530's note) — and **no** new coefficient. It moves two existing things, both read off vLLM's own MoE path: (1) the **MoE-FFN collective gate** partitions on `DP>1 || EP-on` instead of `DP>1`, so EP-on replaces the `tMoEReduce` TP all-reduce with `tMoEDispatch` dispatch/combine (EP-off tensor-shards the experts and reduces the FFN output; EP-on owns whole experts and must route tokens to their owner) — still exactly one of the two, mutually exclusive and exhaustive; (2) routed-expert **weights** shard over a new *expert-shard group* (`ModelHardwareConfig.EffectiveExpertShardGroupSize`) = `EffectiveEP()` when EP is really in force, else `EffectiveMoEGroupSize()`. The WEIGHT divisor additionally obeys the shared `latency.ClampExpertShardToExpertCount` "a loaded rank holds one WHOLE expert" rule (the same clamp `resolveExpertShardSize` applies for capacity, so the two agree on the same experts) — **EP-on only**: under EP-off experts are TENSOR-sharded, so `numExperts/group < 1` is the correct charge and clamping there would be wrong physics *and* an INV-6 break for every pre-#1548 MoE config whose `TP·DP` exceeds its expert count. The dispatch/combine group is deliberately left UNCLAMPED — the all-to-all spans every rank regardless of expert count. **Weights and compute deliberately use different groups**: routed-expert COMPUTE is EP-mode-invariant (with EP on, `G` GPUs jointly process the whole group's `n_dp·T_local` tokens ⇒ `T_local·k/TP` per GPU — the same value tensor-sharding gives), while the per-GPU WEIGHT footprint falls from `numExperts/TP` to `numExperts/G`; dividing compute by `G` too would under-charge it by `DP`. **The #1531 collapse trap, avoided:** DP-as-placement rewrites each replica's `DP` to 1, so a config-bound EP size collapses to `TP`; the CLI therefore carries the LOGICAL DP **width** (not an absolute group size) into each replica via the new `sim.WithExpertParallelGroupDP` `ModelHardwareOption` (carried on `dpPlacementPlan.EPGroupDP` and spread by `dpPlan.EPGroupOptions()` at both `NewModelHardwareConfig` sites, set only when the plan is Active AND EP is on). A width composes with a per-pool TP override (`cluster.ResolvePoolConfig` rewrites `TP`), where an absolute size stamped from the global TP would contradict it. Step time and #1656's KV capacity now shard the same experts over the same group. **Motivating case:** the real GLM/Kimi Wide-EP shape `--tp 1 --dp 16 --enable-expert-parallel` previously charged *every* expert's weights to one GPU (`numExperts/1`); it is now `numExperts/16`. **Per-role all-to-all backend (vLLM `VLLM_ALL2ALL_BACKEND` is per-process):** `--prefill-moe-comm-backend` / `--decode-moe-comm-backend` (empty = inherit `--moe-comm-backend`) on both commands, carried by `cluster.PoolOverrides.MoECommBackend` through `ResolvePoolConfig`; both roles and both commands validate through the single shared `cmd.applyPerRoleMoECommBackends` (R23), and an unrecognized name is a hard error (R1). The selection routes through a **per-mode step-time profile** (`latency.all2AllProfile{commScale}`, one entry per backend in `moeCommBackends`); every backend ships the exact `1.0` nominal placeholder, so DeepEP HT and LL cost the same today — **#1568** fills the table (and may add fields) with no re-plumbing, and a sentinel test fails when it does. **Byte-identity (INV-6/INV-BC-DP1):** EP off, dense models, and a config without the option are bit-for-bit unchanged — `commScale` is an exact multiplicative identity, the two groups coincide for every pre-#1548 shape, and the option's `max` absorbs a width at or below the config's own DP. **INV-13:** the EP-group width is re-supplied from the CLI on both legs (a model-level input like `--kv-cache-dtype`, not a trace field), and `TestINV13_RunReplayParity_MoEDPPlacement`'s `tp2-ep` case asserts byte-identical stdout. **Documented neutral shape:** at `DP=1` with vLLM's default `allgather_reducescatter`, EP-on and EP-off are *numerically equal* — all-gather+reduce-scatter IS the ring-all-reduce decomposition (same wire volume) and β_EP defaults to β₄. That is honest physics, pinned by its own test; the toggle is observable via a modular all-to-all backend, a group wider than TP, or a calibrated β_EP. **Honesty boundary (warned on stderr, R1):** the EP group spans `N` *independently placed* replicas, and cross-node pricing is placement-derived (#1530) while `node_pools`+`--dp>1` remains a #1553 fail-fast — so the inter-replica leg is charged at the on-node rate and multi-node EP step time is optimistic. Also folded in from #1530's deferred comments: `tpAllReduceBasis` now divides only the byte VOLUME by `dp` (a per-collective launch latency is paid in full by each DP rank, not shared), and `cluster.warnIfCrossNodeUnpriced` scores the widest collective group rather than `TP` alone — both byte-identical today. And the latent hole `resolveLatencyConfig`'s roofline gate documented is CLOSED (`cmd.validatePerPoolLatencyBackends`): that gate reads only the GLOBAL backend, so a per-pool `--prefill/--decode-latency-model roofline` could put one pool on DP/EP-blind step time while the cluster ran EP — harmless while EP was inert, a real mis-model once it is live.

DP-as-real-placement (#1531 `blis run`, #1556 `blis replay`): on an MoE model, `--dp N` now spawns `N` real single-node engine replicas per `--num-instances` (`num_instances × N` instances, each conceptually `TP` GPUs) instead of one lumped instance, reusing the existing per-instance placement path; the existing cluster router distributes requests disjointly (vLLM's internal DP load balancer). Each replica is configured per-rank (`DP=1`): its KV budget is the per-rank total (so the aggregate over the `N` replicas equals the pre-#1531 lumped `dp`-multiplied total — no `dp²` double-count) and its step-time uses `moeGroup=TP` (experts replicated per DP rank — the correct expert-parallel-OFF physics; the lumped `/dp` token division is replaced by the router split). Note the cluster's aggregate `max_num_seqs`/KV scales `×N` (N real EngineCores), a deliberate fidelity change from the lumped model. The whole transformation lives in `cmd/` (`planDPPlacement` decides + guards; `applyDPPlacement` expands `numInstances` and divides the auto-KV total to per-rank; `resolveDPPlacement` composes those two and adds the diagnostics + the per-rank `--max-model-len` re-cap that avoids a per-replica `NewSimulator` panic; the per-rank DP is threaded straight through the canonical `NewModelHardwareConfig`); `sim/cluster` is unchanged. **`blis replay` supports it too since #1556**: both command bodies call the single shared `resolveDPPlacement`, so the #1531 run-only `logrus.Fatalf` is gone and a trace exported by `blis run --dp N` replayed with the same flags yields byte-identical stdout (INV-13 — note replay's horizon defaults to a drain-time estimate rather than run's `MaxInt64`, so pass `--horizon` on both legs, exactly as for any other replay comparison). `blis observe` is unaffected (a black-box dispatcher that places no instances). `--dp 1` and dense `--dp>1` (already rejected) are byte-identical to pre-#1531 (INV-6). Guarded (fail-fast) combos: PD disaggregation / autoscaler / node pools + `--dp>1` → #1553 (node pools are where GPU reservation would be literal — deferred until the `N×M` pool placement is audited). `--enable-expert-parallel`+`--dp>1` was a fourth until #1548 lifted it (below).

Phase 0 workload unification complete (see issue #420): W0-1 (spec v2 schema + SLO tiers), W0-2 (binary rename + converters), W0-3 (cohort population dynamics), W0-4 (legacy retirement). All workload generation now flows through `sim/workload/GenerateRequests()`. SLO tiers: critical, standard, sheddable, batch, background. Arrival processes: poisson, gamma, weibull, constant. CLI binary renamed from `simulation_worker` to `blis`.

Observe/replay/calibrate pipeline complete: `blis observe` (#659) dispatches workload to real servers with closed-loop session support, `blis replay` (#689) replays through DES, `blis calibrate` (#701) compares real vs simulated latencies. Observe fidelity (#660): chat completions endpoint (`--api-format chat`), `stream_options` for streaming token counts, `finish_reason` extraction, configurable `max_tokens` (`--unconstrained-output`), deterministic prefix strings for KV cache activation, `--rtt-ms` for network RTT.

Replay injection-origin normalization (#1606): `blis replay` re-bases each request's DES injection time onto the trace's `arrival_time_us`/`deadline_us` origin. A real `blis observe` trace writes `send_time_us` in Unix-epoch µs but `arrival_time_us`/`deadline_us` on a run-relative clock; using the raw epoch `send_time_us` as an absolute injection tick made every request instantly past-due (0 completions, empty `sim_result.json`). `LoadTraceV2Requests`/`LoadTraceV2SessionBlueprints` subtract a single per-trace `injectionOriginShift = min(injectionTime) − min(arrival_time_us)`, preserving #1304's send-delta (concurrency-slot-wait) spacing. Generated `blis run` traces write `send_time_us == arrival_time_us`, so the shift is exactly 0 and replay stays byte-identical (INV-13/INV-6); the closed-loop preliminary horizon uses `workload.MaxNormalizedInjectionTimeUs` (the normalized injection, not raw `arrival_time_us`).

Faithful closed-loop `--trace-output` re-export (#1630, option (a) of #1621): `blis replay --session-mode closed-loop --trace-output` (and pool `--concurrent-sessions`) now re-export a FAITHFUL trace instead of the #1623 fail-fast guards (both removed) or the pre-#1621 silent follow-up drop. `cmd/replay.go` captures every round (`append(requests, followUpRequests...)`) plus each follow-up's think time (`fu.ArrivalTime − completionClock`, captured in the `onRequestDone` wrapper), and `workload.ReExportClosedLoopRecords` (new `sim/workload/reexport.go`) reconstructs the records: for an `accumulate` corpus it re-derives per-round `input_tokens` DELTAS + `input_tokens_reset` compaction markers via the shared `EncodeSessionToTraceRecords` law (#1613) and the header carries `session_context_growth: accumulate`; for a non-accumulate corpus it emits absolute per-round suffix via `RequestsToTraceRecords` with per-follow-up `think_time_us` and round-0 `prefix_group`/`prefix_length` propagated onto follow-ups (no prefix double-count). **The delta law feeds the encoder the ACTUAL accumulated output (`ProgressIndex − InputLen`, `accumulatedOutputLen`) — NOT the oracle `len(OutputTokens)` — so the re-derived deltas are the exact inverse of `SessionManager.OnComplete`'s accumulate growth; the emitted `output_tokens` column keeps the oracle `MaxOutputLen` so re-replay reproduces the same completion → same actual output → the abs sequence reconstructs exactly.** Result: replaying the re-export reproduces the original run's per-request per-round input/metrics (INV-13 round-trip; accumulate reproduces byte-for-byte). Fixed mode (and closed-loop with no session records) keeps the byte-identical `RequestsToTraceRecords(requests)` path (INV-6). Pool re-export is a COMPLETE closed-loop session corpus (all originals + clones, every round; replay with `--session-mode closed-loop`, not re-cloned); its aggregate CONSERVATION metrics (completed/injected requests, total tokens) reproduce, but per-request and cache/latency aggregates are not guaranteed (data-dependent admission timing; and a non-accumulate clone's cache-busting divergence is not preserved when it shares a `prefix_group`). The accumulate branch copies per-request metadata (SLO/deadline/model/adapter/modality) via `copyReExportMetadata` so it has field-coverage parity with the non-accumulate `RequestsToTraceRecords` path (R23); `PrefixGroup`/`PrefixLength` are excluded (accumulate folds the prefix into round-0's absolute). The re-export block runs LAST (after `EmitOutput`/`--results-path`) so a secondary-artifact failure never discards primary metrics. Documented boundary: a length-capped round reproduces its input length but may diverge in prefix-cache token *content* across the cap boundary (huge-ISL corpora use a large `--max-model-len` to avoid capping).

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

Every issue must have at least one label. To file an issue: read the relevant template file under `.github/ISSUE_TEMPLATE/`, reproduce its structure in your issue body, and use `gh issue create --title "..." --body "..." --label "..."`. Apply the template's front-matter labels yourself.


## Post-Hoc Saturation Detection

BLIS includes post-hoc saturation detection for analyzing completed runs. This is distinct from the real-time flow control saturation detector used for admission control.

**Package**: `sim/saturation/`

**Streaming detectors** (all stream via `Observe`/`Detect`; the batch `Classify` path was removed in #1516). Every detector has at least one false-alarm calibration knob — a precondition for comparing them, since scores are only comparable at a matched false-alarm rate:
- **composite**: Combines rate deficit (1 - completions/arrivals) and a quartile-filtered latency trend. STABLE / BACKLOGGED / OVERLOADED by score vs a 1/√arrivals noise floor, scaled by `composite.sensitivity` (#1614; default 1.0 = the historical unscaled floor, so absent config is byte-identical). A larger sensitivity raises the floor and fires less.
- **threshold**: Mean E2E latency vs a configurable threshold (default 5000ms). STABLE when mean < threshold, OVERLOADED when mean > threshold.
- **backlog-drift**: Online OLS slope of in-flight over a trailing window (became streaming in #1515), banded against the noise floor. The BACKLOGGED/OVERLOADED boundary sits at `backlog_drift.slope_k × noiseFloor` (#1614; default 3.0). `slope_k` is read through `BacklogDriftConfig.effectiveSlopeK()`, which falls back to 3.0 for the struct-literal construction paths — a literal zero would otherwise band every rising trace OVERLOADED. Both the band switch and the score denominator read the same hoisted value, so `Score == 1.0` coincides with the OVERLOADED edge at every `slope_k`. All calibration knobs are rejected below `1e-6`, though for two different reasons: for knobs that multiply a noise floor and then divide (`slope_k`, `composite.sensitivity`) a subnormal value underflows the product to zero and decouples Level from Score, whereas `peak_rate`'s knobs compare before dividing and so take the bound purely as a usability floor. `slope_k <= 1` makes the BACKLOGGED band unsatisfiable — accepted as a legitimate "maximally severe" setting, yielding a two-level detector; `peak_rate.overload_multiple` accepts sub-1 identically, so the two detectors can be swept over a common range.
- **peak-rate**: `R_t = Peak_t / t` — the backlog high-water mark over elapsed time. Backlog is a random walk reflected at zero, and the reflection makes the regimes separable *without any capacity estimate*: under positive drift `R_t` converges to a positive constant, under zero drift it decays as `1/√t`, under negative drift as `1/t`. So an overloaded server HOLDS `R_t` while a healthy one lets it decay, and the detector fires when it holds above `peak_rate.threshold`. This is also why it is the natural foil to `backlog-drift`: a straight-line fit to backlog is degenerate at exactly ρ≈1 (backlog grows like √t, so the slope tends to ZERO and the detector reports STABLE at criticality — the worst failure direction), while `R_t` has no such degeneracy. **Horizon-dependent by construction** (the result is asymptotic): measured separation between sub- and super-capacity traffic is 2.3× at n=500, 4.7× at n=2000, 14.6× at n=8000, so `min_observations` is part of the algorithm rather than input validation. O(1) state — four scalars, no per-event retention. Validated by an optimization campaign (5 seeds × 11 load rungs, false-alarm-calibrated first) and reproduced on `blis run`: a clean step at the true capacity cliff, seed-stable across 5 seeds, with BACKLOGGED one rung *before* the cliff (early warning). `peak_rate.overload_multiple` accepts values below 1, which collapse it to two levels exactly as `slope_k <= 1` does for backlog-drift — the band split is the one parameter the selection campaign never varied (its scorer counted BACKLOGGED and OVERLOADED alike), so rejecting a value there would be severity without evidence. **peak-rate's knobs:** `threshold` (the primary FPR dial, in backlog per second, so its calibrated value is deployment-specific), `warmup_us` (a TIME gate, not redundant with the event gate: `R_t`'s numerator counts events while its denominator measures seconds, so a dense burst — 300 arrivals in 3 ms — makes `R_t` enormous on negligible evidence and no observation count can suppress it; default 0 = off), `min_observations` (holds the verdict through the opening transient — it moves the per-event trace and usually not the stdout headline, since the reducer's trailing window normally lies past the gate; it *can* move the headline on a short run, a very large gate, or a long `--saturation-final-window`), `consecutive_k` (anti-flapping), `overload_multiple` (the BACKLOGGED/OVERLOADED split). Recovery after a transient takes about `peak / threshold` **seconds** of elapsed time (`threshold` is backlog per second), since the numerator is an all-time high-water mark — so the detector answers "did this run saturate?", not "is it saturated right now?"

**CLI flags** (`run`, `observe`, `replay`; #1516, #1519, #1517):
- `--detectors <selection>`: empty = off. A single name (`composite`, `threshold`, `backlog-drift`, `peak-rate`) runs #1516's single-detector streaming path. `all` runs the full roster; a comma-list (e.g. `composite,threshold`) runs exactly the named subset. `all` and comma-lists route through the **detector bank** (#1519). Unknown name — single or inside a list — is a hard error listing valid names (R1).
- `--saturation-config <path>`: strict-YAML tuning file with optional `composite:`, `threshold:`, `backlog_drift:`, and `peak_rate:` blocks — one calibration knob per detector (#1614), since a detector with no knob cannot be moved onto a matched false-alarm rate and so cannot be fairly compared. For a **single** detector the config must carry only that detector's block — a foreign block errors (no silent drop, R1). For the **bank**, ownership is enforced over the selected SET (`checkBlockOwnershipSet`): a block whose owning detector is not among the selected names is a hard error (R1), same as the single-detector path — `--detectors all` selects every owner so a full shared config is fine, but a subset that omits a detector whose block was supplied errors. A value error inside a *selected* detector's own block also errors. Absent block = defaults; a partial block overrides only named fields; an unknown key or out-of-range value errors naming the field; an empty file = all defaults.
- `--saturation-report <path>`: writes the selected detector(s)' **final label + per-event verdict trace** as one `{"final":{...},"trace":[...]}` JSON object (one trace record per event, tagged by detector name; map keys sorted so repeated runs are byte-identical). Requires `--detectors`. Optional — the stdout final label (#1517) is emitted regardless. `--saturation-config`, `--saturation-report`, and `--saturation-final-window` without `--detectors` are hard errors, as is an unwritable report path (checked up front).
- `--saturation-final-window <duration>` (#1517): Go duration for the trailing window of the stdout final-label plurality vote. Resolution: this flag if set → else `backlog_drift.window_size_sec` from `--saturation-config` → else 30s. Same value for every detector; a non-positive or unparseable value is a hard error. Requires `--detectors`.

**Final label reducer** (`sim/saturation/reduce.go`, #1517): `ReduceOne(records, windowUs) Level` collapses ONE detector's per-event trace into a headline label by a **last-window plurality** rule — keep records within `windowUs` of the max timestamp, take the most-frequent `Level`, break count-ties toward the more severe level (`OVERLOADED > BACKLOGGED > STABLE`), empty group → `STABLE` (R20). `ReduceAll(records, windowUs) map[string]Level` groups by detector name and reduces each group. It is a **pure function, not a `Detector` method** — so every detector is collapsed identically (fair cross-detector comparison) and new detectors get final labeling for free. The rule is order-independent (INV-6) and identical traces yield identical maps (INV-13). `cmd` calls `ReduceAll` for every selection: a single detector yields a one-key map, `all` the full map. There is exactly one stdout shape — a `map[string]Level` (`--detectors composite` → `{"composite":"STABLE"}`, never a bare label).

**Detector bank** (`sim/saturation/bank.go`, #1519): `Bank` holds and drives a roster of streaming detectors over ONE deterministic replay, fanning each event out to every selected detector (`fanout`) so all are scored on a byte-identical event sequence in a single pass. It reimplements no `Detector` method — it only multiplexes the shared `buildSortedEvents` (a #1519 extraction) + #1516's `TraceSink` + `WriteCombinedReport`. Its sole public driver is `Run(requests) error` (the multi-detector analogue of `ReplayOneDetector`); the collected records are reduced to the stdout label by `saturation.ReduceAll` in `cmd` (#1517). `NewBank(names, cfg, sink)` validates/de-dups names and orders the roster canonically (`composite`, `threshold`, `backlog-drift`, `peak-rate`), so selection order and spelling never change output: `all` ≡ the full comma-list byte-identically, and a subset detector's records are byte-identical to its records under `all` (selection filters WHICH detectors run, never HOW they see traffic — INV-6/INV-13).

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

**Inter-node network cost (#1530, trained-physics only)**: the two trained-physics communication bases divide byte volume by an **effective** link bandwidth rather than always by `bwHbmUs`. When a collective's participant group does not fit inside one node, the divisor becomes `bwHbmUs / spanScale` — a **re-scale in place** of the existing basis, never a separate additive term (an additive term would double-charge a `DP>1` MoE instance whose all-to-all `moeDispatchBasis` already prices). `tpAllReduceBasis` covers all three TP-group collectives (`tTpAttention`, `tTpDenseFFN`, and the `dp==1` `tMoEReduce`); `moeDispatchBasis` covers the expert dispatch/combine. Two penalty shapes, both of the form `1 + (r-1)·crossHops/(G-1)` with `r = IntraNodeBwGBps/InterNodeBwGBps`: a **ring** penalty `1+(r-1)(n-1)/(G-1)` derived from NCCL's hierarchical (two-level) multi-node all-reduce — applied to the TP group *and* to the `allgather_reducescatter` MoE family, whose volume basis is ring-shaped (2 phases × `(G-1)/G`) — and an **all-to-all** penalty `1+(r-1)(G-g)/(G-1)` for a true all-to-all backend (`deepep_*`/pplx/mori/flashinfer), where a rank's data must reach every peer with no reduction on the way, so far more of it leaves the node. Both are exactly `1.0` at `n==1` or `r<=1`, and monotone in `r` (AC-2). **The topology is placement-derived, never a flag**: `PlacementManager.PlacedGPUsPerNode(gpuIDs)` reports the size of the node(s) an instance actually occupies, and `ClusterSimulator.applyPlacementTopology` stamps it onto the per-instance `SimConfig` at all three placement sites (startup / deferred `NodeReadyEvent` / autoscaler scale-up, R23), immediately after the per-instance GPU type (#893), KV capacity (#1522) and cost (#1529). A guard test asserts no CLI flag declares it — a declared node size can contradict real `node_pools` placement. It lives on `ModelHardwareConfig.NetworkTopology` (a latency input like TP/DP/`MoECommBackend` — R16, not a SimConfig-level sub-config), supplied via `sim.WithNetworkTopology` — a new `ModelHardwareOption` variadic, mirroring `WithKVOffload` on `NewKVCacheConfig`, so all 211 `NewModelHardwareConfig` call sites are untouched (R4). Production writes the field directly at the placement sites (placement is not known until after the config exists); the option serves tests/standalone construction and is what the constructor's `validate()` guard covers. **Exported surface is deliberately minimal**: everything crossing a package boundary is exported (`NetworkTopology` + `NewNetworkTopology`/`NodesSpanned`/`MembersPerNode`/`IsKnown` for `sim/latency`+`sim/cluster`; the four `HardwareCalib` interconnect methods; `ModelHardwareOption`+`WithNetworkTopology`; `ClusterSimulator.MaxNodesSpanned` for `cmd`; and `LatencyBackendRoofline`/`LatencyBackendTrainedPhysics`, which *remove* three duplicated string literals). What does not cross one is package-private: `NetworkTopology.validate` and `PlacementManager.placedGPUsPerNode` are unexported. **Fabric speeds are hardware calibration**: `HardwareCalib.IntraNodeBwGBps`/`InterNodeBwGBps` (per-GPU *effective unidirectional* GB/s, committed for H100 450/50, A100 300/25, L40S 32/12.5 in `hardware_config.json` with provenance comments) — only their **ratio** is used, so the absolute scale cancels; both must be set or neither. Validation lives on `HardwareCalib.ValidateInterconnect()` and runs at BOTH the `GetHWConfig` load boundary (so a malformed file fails identically under either backend, R23) and the trained-physics constructor (so a programmatically-supplied calib is covered too): a half-set bandwidth pair or an unusable value is a hard error, never a silent clamp (R1). A committed-file companion invariant (`sim/latency/interconnect_calib_file_test.go`) enumerates the file's own keys and asserts every entry loads, is complete, and has a plausible ratio. **Second, size-independent half (`InterNodeLatencyUs`, adopted from the `exp-issue1530-network-cost` prior art)**: the fixed launch + fabric round-trip of one cross-node collective, charged once per cross-node COLLECTIVE and skipped when a step communicates no tokens. The multiplier is the step's collective count, which differs by shape: a dense TP step launches 2 per layer (attention + FFN all-reduce — a ring all-reduce is ONE call even though its volume has two phases), while an MoE step at `dp>1` launches 3 (attention all-reduce + expert dispatch + combine, two separate calls — `moeDispatchCollectivesPerLayer`). Charging dispatch/combine once would make the latency half disagree 2× with the volume half, which already counts both. For decode-sized messages this plausibly EXCEEDS the bandwidth half — but it is **0 (uncalibrated, not charged) in the bundled config**, deliberately: BLIS has no measured value and a guessed constant would sit in front of every multi-node estimate (#1661 records the calibration bar; a test pins the 0 so any future value arrives with a source). So AC-2's latency clause is exercised only synthetically today — no shipped config makes the term fire. Like the bandwidth half it rides β₄/β_EP, so it is calibrated in that frame (`β·units·latency`), and it stands alone (a latency-dominated fabric is meaningful, so it is not paired with the bandwidths). **Inert at three independent gates** (INV-6/INV-BC-DP1): no node pools ⇒ topology unknown; group fits a node ⇒ `n=1`; no fabric calibration ⇒ `r=1`. In each case the divisor is `bwHbmUs` itself, bit-for-bit. **`blis run` only, fenced on BOTH paths** (INV-13). #1530 offered two parity designs — round-trip the cross-node signal so replay reproduces the run's step times (a), or keep a loud rejection (b) — and this is **(b), extended to cover the trace path**. The *config* path: `blis replay` `logrus.Fatalf`s on `node_pools` and `observe` builds no simulator, so a non-inert topology is unreachable there. The *trace* path is NOT covered by that guard, and — verified empirically against `main`, where the same export-and-replay round-trip returns identical metrics (0 of 27) while with this feature 16 of 27 differ — this feature is what first makes that divergence observable in step time, so it is fenced here rather than inherited. `blis run --trace-output` records the widest instance node span in the TraceV2 header (`MaxNodesSpanned`, `omitempty`; the writer normalizes a span of 1 to 0 via `cmd.crossNodeSpanForTrace`, since `omitempty` drops only 0 — so a run without multi-node placement writes a byte-identical header) and `blis replay` hard-errors on any trace reporting more than one node. **The header field is a REFUSAL signal, not a reconstruction input**: replay reads it only to fail loudly, never to rebuild a fleet (reproducing cross-node timing would need per-instance topology, not one fleet-wide maximum) — so this remains design (b), and the asymmetry is CLOSED rather than merely documented. Both fences have dedicated subprocess tests plus negative controls (field absent, and the boundary value 1), and the header field has its own round-trip / omission / byte-identity tests. **Behavior change on the record:** replaying a newly-exported spanning trace is now a hard error where it previously produced (wrong, single-node-speed) numbers. Backward compatibility is preserved for every trace written before this feature — no key ⇒ 0 ⇒ accepted — verified by replaying a `main`-exported spanning trace under the new binary with 0 of 27 metrics differing. That is also the guard's known limitation: it protects traces exported from this version onward, since a pre-existing trace's span cannot be detected retroactively. **Diagnostics (R1)**: `warnIfCrossNodeUnpriced` latches one warning PER CAUSE (so a mixed fleet reports each distinct reason once) for the three ways a genuinely spanning placement ends up unpriced — a backend with no comm term, an unresolvable node size, or an uncalibrated interconnect — plus a fourth for an implausible (>1000×) ratio, a deliberate unit-error detector rather than a plausibility judgement (real hardware spans ~2× on PCIe nodes to ~75×). The diagnostics score the REAL distinct-node count, not `NodesSpanned`, so they still fire when the node size itself could not be resolved — otherwise they would go quiet exactly when most needed. **Calibration honesty**: magnitude is modest — the TP comm term rises ~1.53× for TP=16 on 2×8 H100 at `r=9`, ≈+10% on total step time for a mid-size dense model — because hierarchical all-reduce moves the same bytes and only the reduced `S/g` chunk crosses the fabric; a **flat** multi-node ring would instead be throttled to ≈`r` (9×), and measured two-node H100 bus-bandwidth degradation (~1.3–1.5×) is why the hierarchical model is the one used. `IntraNodeBwGBps` should be read as "the on-node link speed β₄ was calibrated against", since β₄ already absorbs the NVLink/HBM ratio. Known approximations, each with a follow-up: per-collective launch/RTT cost is unmodeled and is plausibly the *dominant* cross-node effect at decode message sizes (**#1661**); the fabric is keyed by `gpu_type` rather than by pool, correct only while #1529's unique-`gpu_type` rule holds (**#1662**); the roofline backend prices no communication at all (**#1663**); the all-to-all penalty sums rather than overlaps the on/off-node portions and ignores DeepEP's per-node RDMA coalescing (both pessimistic); and the MoE group's span is extrapolated from the placed node size, since BLIS places a TP group. **#1548 made the expert dispatch/combine TERM reachable** (`--enable-expert-parallel` now charges it, at the on-node rate) but its **cross-node pricing** is still inert in every reachable config: a cross-node price needs `node_pools`, which remains a fail-fast alongside `--dp>1` (#1553). `blis run` warns rather than presenting the on-node price as complete — exactly the per-leg split #1530 prescribes.

**Quantized model support**: Three-tier auto-detection of weight precision: (1) `quantization_config` in HF `config.json` — GPTQ/AWQ (`bits`), FP8 (implicit), compressed-tensors (`config_groups.*.weights.num_bits`); (2) model name conventions (`w4a16` → 0.5, `FP8` → 1.0 via `InferWeightBytesFromModelName`); (3) fallback to `BytesPerParam` from `torch_dtype`. Uses quantized weight precision for weight bandwidth and model-weight-bytes calculations while keeping compute dtype for activations. KV-cache storage precision is configured independently via `--kv-cache-dtype` (`ModelConfig.KVBytesPerParam` / `EffectiveKVBytesPerParam()`, #1565) — it is NOT tied to the compute dtype. `ModelConfig.WeightBytesPerParam` (0=fallback to `BytesPerParam`) with `EffectiveWeightBytesPerParam()` accessor decouples weight storage precision from the compute dtype.

**Independent KV-cache dtype (`--kv-cache-dtype`, #1565)**: mirrors vLLM's `CacheConfig.cache_dtype` — KV-cache *storage* precision, independent of both compute/activation dtype and weight quantization (they are separate vLLM engine args). `ModelConfig.KVBytesPerParam` (0 = `auto`, fall back to `BytesPerParam`) with the `EffectiveKVBytesPerParam()` accessor supplies the per-token byte width on **both** `KVBytesPerToken` branches (MLA compressed-latent and standard MHA/GQA); `--kv-cache-dtype fp8` (also `fp8_e4m3`/`fp8_e5m2`) → 1 byte/element while compute stays bf16 (2) ⇒ ~2× KV-block capacity (and smaller PD KV-transfer bytes), the common recipe for large MoE/MLA models (GLM-5.2-FP8, DeepSeek-V3.2). Scope is capacity/byte-width only — the step-time (trained-physics/roofline) models keep `BytesPerParam` for the KV-read term (step-time goldens + INV-BC-DP1 byte-identical). Resolved at the CLI (`cmd`) via `KVCacheDtypeToBytes` + `applyKVCacheDtype`, applied alongside the weight-precision fallback; a value error is a hard error listing valid names (R1). **INV-6**: `auto` (the default / flag absent) leaves `KVBytesPerParam` at 0 ⇒ KV dtype == compute dtype ⇒ byte-identical. **INV-13**: the flag lives on both `run` and `replay` (shared `registerSimConfigFlags`) and both recompute capacity through the shared `resolveLatencyConfig` — the resolved KV dtype is **re-supplied on the replay CLI, not round-tripped through the TraceV2 header** (identical treatment to `--gpu-memory-utilization` and auto-computed `--total-kv-blocks`, which the header records only informationally and replay never reads back). Not on `observe` (a black-box dispatcher that derives no KV capacity — same boundary as `--kv-offload-config`). Independent of the KV-block *count* work (#1545) and weight quant (#1527).

**MLA / model-shape KV & weight fidelity (#1527, F1–F3)**: The KV-capacity model (`sim/latency/kv_capacity.go`, `config.go`) represents the modern MLA MoE family (DeepSeek-V2/V3, Kimi-K3, GLM-5.2 `glm_moe_dsa`). **F1 — explicit `head_dim`**: `ModelConfig.HeadDim` (`json:"head_dim"`) + `EffectiveHeadDim()` accessor (returns `HeadDim` when >0, else `HiddenDim/NumHeads`) feed `KVBytesPerToken` and `computeModelWeightBytes`; the step-time models (trained-physics/roofline) intentionally still use `hidden/heads`, so step-time goldens + INV-BC-DP1 are byte-identical. **F2 — MLA compressed-KV**: when `ModelConfig.KVLoraRank > 0`, `KVBytesPerToken` returns `(kv_lora_rank + qk_rope_head_dim) × EffectiveKVBearingLayers × BytesPerParam` — a single latent per token per layer, independent of `num_kv_heads`/`head_dim` and **NOT divided by TP** (the latent is replicated across TP ranks, matching vLLM's MLA cache); both auto KV-block sizing and PD KV-transfer sizing inherit it. `EffectiveKVBearingLayers` equals `num_layers` for a non-hybrid model, but for a hybrid-attention model (#1635, below) it is the full-attention layer count (Kimi-K3: 24 of 93), so the standard MHA/GQA branch uses it too. **F3 — dense-prefix MoE**: `ModelConfig.FirstKDenseReplace` splits `computeModelWeightBytes` into K dense-MLP layers + (L−K) MoE-MLP layers (K clamped to [0,L]), distinct from the every-Nth `InterleaveMoELayerStep`. All three are **no-ops when the config keys are absent** (INV-6 byte-identity); INV-4/INV-13 preserved (`run`/`replay` share the path; `observe` doesn't derive capacity from shape). Committed fixture: `model_configs/glm-5.2-fp8/config.json`. **Documented known approximations (F4/F5a)**: block-wise FP8 (`weight_block_size`, `modules_to_not_convert`) treated as flat 1.0 byte/param (optimistic); DSA indexer and MLA attention weight projections unmodeled. MTP/spec-decode throughput is out of scope (#1528).

**Hybrid attention (#1635, #1636)**: for models that interleave full-attention layers with linear-attention layers (`linear_attn_config.full_attn_layers`, e.g. Kimi-K3: 24 MLA layers + 69 Kimi-Delta-Attention layers of 93), `ModelConfig.KVBearingLayers` (derived from `len(full_attn_layers)`, clamped to `[0, num_layers]`) is the full-attention layer count, exposed via `EffectiveKVBearingLayers()` (0/absent ⇒ `num_layers` ⇒ byte-identical for every non-hybrid model, INV-6). **KV capacity (#1635)** sizes the KV cache over the full-attention layers only. **Step time (#1636)** now also splits the per-layer attention cost by type in BOTH latency backends (default trained-physics `sim/latency/trained_physics_model.go` + roofline `sim/latency/roofline.go`, for backend parity): the sequence-length-dependent attention cost — the O(context)/O(N²) attention-**score** compute and the growing-KV read/write bandwidth — is charged over the full-attention (`numKVBearingLayers`) layers only; the KDA layers charge a **linear-attention** cost, O(N) in prefill and O(state) per token in decode (context-independent), by substituting the growing context dimension with the fixed KDA state dimension (`head_dim`) in the *existing* FlashAttention FLOP convention and reusing the *existing* β₁ₐ/β₂ₐ compute coefficients — **no new empirical coefficient** (calibration honesty: KDA layers reuse the full-attention head count / head_dim since `linear_attn_config` is not re-parsed, and the small context-independent KDA recurrent-state bandwidth is left folded into the per-layer overhead β₅·L rather than given a separate fabricated coefficient — a documented calibration follow-up). So K3 decode step time / TTFT reflects 24 full + 69 linear layers, materially below the pre-#1636 all-93-full estimate. Every term is a strict no-op for non-hybrid models (`numKVBearingLayers == num_layers`), so step-time goldens + INV-BC-DP1 are byte-identical (INV-6); `run` and `replay` share the step-time model so both change together (INV-13); `observe` computes no step time and is unaffected. **Still scoped out**: KDA layer *weights* stay charged as full attention (#1638, PESSIMISTIC); `blis run` warns when a hybrid model is detected. A hybrid config whose `linear_attn_config` lacks a usable `full_attn_layers` list warns and falls back to all-layers sizing (R1, never silent).

**EP-aware weight-footprint sizing (#1656)**: `CalculateKVBlocks` charges the model's weight bytes against ONE DP rank's TP-GPU budget (`gpu_mem × util × TP`), so the returned total carries an implicit `/TP` — weights modeled as tensor-sharded across the TP group. Under expert parallelism vLLM instead shards the ROUTED (FusedMoE) expert weights across the whole EP group (`ep_size = TP·DP`, each rank holding `num_experts/ep_size` WHOLE experts), which for a Kimi-K3-class MoE is ~99% of the checkpoint — so the pre-#1656 accounting over-counted per-GPU weights by a factor of `DP` and auto-sizing failed outright ("Minimum GPUs required per instance: 21" for a deployment that runs on 32). `computeModelWeightBytes` now isolates the routed-expert subtotal and scales it by `TP/expertShardSize` — **the total is scaled by `TP/EP`, NOT divided by `EP`** (dividing the total by `EP` would charge `R/(TP²·DP)` per GPU, an extra `TP` under-count); every other term (attention, shared experts, dense-prefix MLP, router/gate, embeddings, norms) stays TP-sharded. Delivered as a variadic `latency.WithExpertParallelSize(ep)` option (same seam as `WithAdapterReservedBytes`); `0`/`1` = EP off ⇒ the original expression, bit-identical (INV-6). Valid sizes are `{0,1} ∪ [TP, TP·DP]`; anything `<= TP` resolves to TP (a strict no-op — so the very common `ep == TP` DP=1 EP-on case can never fail validation), `> TP·DP` is a hard error (R1: it would hand back capacity for memory that does not exist), and a group wider than `num_routed_experts` is **CLAMPED to that count with a `logrus.Warnf`** (the loaded ranks still hold one WHOLE expert each; charging the sub-one-expert average would be optimistic, while rejecting would both fail deployments whose block count is unaffected and mask the CLI's own topology diagnostics — wide EP over fewer experts is a real planning input). The canonical group-size formula is the pure `sim.EffectiveEPSize(isMoE, tp, dp, epOn)` (`ModelHardwareConfig.EffectiveEP()` is now its config-bound accessor); **CLI sites must pass the LOGICAL, user-requested group** via `cmd.epSizeForKVCapacity` — a per-replica config is not a safe source because DP-as-placement (#1531) sets `DP=1` per replica, collapsing EP to TP and silently no-opping the sharding. All 6 production call sites are plumbed (`cmd/root.go` global + PD prefill/decode, `cmd/replay.go` PD prefill/decode, `sim/cluster/kv_autocalc.go`) and a static test fails if a new one omits the option (R23). **Observability boundary**: the reduction only bites when `EP > TP`, i.e. `DP>1` — at `DP=1` the EP group IS the TP group (`/EP ≡ /TP`, correct physics, byte-identical). `--dp>1 --enable-expert-parallel` was rejected by `planDPPlacement` when #1656 landed; **#1548 lifted that**, so the EP-sharded deployment now runs end-to-end on both commands and this arithmetic is live rather than library-only. **The EP-OFF baseline is BLIS's own DP model, not vLLM's** — a calibration-honesty point worth knowing before touching either side: vLLM's `FusedMoEParallelConfig.make` calls `flatten_tp_across_dp_and_pcp` **unconditionally**, so at `TP=2/DP=2` with EP OFF the MoE `tp_size` is 4 and per-GPU routed bytes are `R/(TP·DP)` in **both** EP modes. BLIS instead models MoE `--dp N` as N independent engine replicas (#1531), each holding a full tensor-sharded copy of the experts (`R/TP` per GPU), so its EP-off `DP>1` capacity is CONSERVATIVE (over-charges) relative to vLLM. The EP-ON value this PR computes matches vLLM exactly. Since **#1548** the step-time model shards routed-expert WEIGHTS over the same expert group this capacity model uses (`EffectiveExpertShardGroupSize`, clamped by the shared `latency.ClampExpertShardToExpertCount`), so the two now agree on the footprint of the same experts; routed-expert COMPUTE keeps the flattened `moeGroup = TP·DP`, which is EP-mode-invariant. What #1666 still tracks is BLIS's EP-**off** `DP>1` capacity baseline (BLIS's own replica model vs vLLM's unconditional flattening) — unchanged by #1548. Documented optimism (`docs/reference/models.md`): the per-rank expert charge is the continuous average (vLLM sizes from the most-loaded rank ⇒ non-divisible `ep_size`, EPLB/redundant experts are optimistic), and the flat `8.0 GiB` MoE activation + `0.6 GiB/GPU` non-torch terms (no DeepEP/all-to-all workspace) become the dominant residual once the routed term shrinks.

**Speculative decoding / MTP (#1528)**: Models the decode-throughput effect of speculative decoding / Multi-Token Prediction (GLM-5.2's 5-token MTP, DeepSeek-V3, EAGLE, Medusa). Off by default (`--num-speculative-tokens 0`) ⇒ byte-identical output (INV-6). The `SpeculativeConfig` sub-config (8th `SimConfig` module) decouples two quantities: **verify width** `w = K+1` (the target processes K drafts + 1 bonus token per forward pass) drives the per-step **cost** — the decode token-population terms of both latency backends scale by `w` while the once-per-step weight-load/comm/const terms do not, so cost is sublinear in `w` (the physics behind the speedup); **accepted tokens** `g = 1 + α·K` (α = user-supplied mean acceptance rate) drives **progress** — a per-request deterministic fractional carry advances `ProgressIndex` by `g` tokens/step (no RNG, so INV-6 holds and metrics are expectations; capping defers rather than drops tokens, no drift). `--speculative-acceptance-rate` is **required** when `K>0` (α defaulting to 0 would model MTP as pure slowdown). Model-level, supplied identically to `blis run` and `blis replay` (INV-13; no TraceV2 change). **Progress is clamped at the completion boundary (#1657):** `FormBatch` caps the granted token count by the request's distance to `InputLen + max(len(OutputTokens),1) − 1` (the `processCompletions` boundary), so the final verified block lands exactly ON the target instead of 1..K past it — mirroring vLLM's `_update_from_output`, which appends accepted tokens one at a time and trims the tail once `check_stop` fires. Consequences: a request's final `ProgressIndex` (and hence output-token count and a closed-loop accumulate session's context growth) is **identical to a `K=0` run** — spec-decode buys fewer, wider decode steps, never different token accounting; step time and step COUNT are unaffected (both latency backends size a decode step by verify width `w`, not by the granted count, so a clamped final step still pays the full verify pass, as in vLLM). The one timing effect is per-token detokenization (`NumNewTokens × OutputTokenProcessingTime`): a clamped final step charges OTPT for the tokens it actually emitted, so total OTPT is now exactly `(L−1)×OTPT` — identical to `K=0`, where pre-#1657 spec-decode over-charged by up to `K×OTPT`. So spec-decode E2E/ITL shift slightly (downward) with this fix; `K=0` output is untouched. Before #1657 the overshoot made `SessionManager.OnComplete` read `actualOutputLen > len(OutputTokens)` and **cancel the whole closed-loop session** as accounting corruption (INV-11). The completion-time output-token clamp in `recordRequestCompletion` remains as defense-in-depth (INV-1). A PD decode sub-request whose remaining budget is 0 (a 1-output-token sub-request, admitted already at its boundary) is still granted 1 token — a 0-token grant would strand it in the wait queue forever (INV-8/INV-11). Raw ITL percentiles (`AllITLs`) are per-verification-step under spec-decode; TPOT (`RequestITLs`) is the amortized per-token metric and is what SLO-ITL goodput reads. `K` capped at `MaxSpeculativeTokens=1024`. **Known limitation — speculative occupancy contention (#1627):** BLIS reserves KV and debits the batched-token budget by the *accepted* count `g`, whereas vLLM reserves `K` lookahead slots per running request per step *unconditionally*; so under KV saturation BLIS over-predicts max batch size and under-predicts preemption (the MTP speedup without its contention penalty). Faithful below saturation — the stated capacity-planning use case. #1657's boundary clamp sharpens this in one place: BLIS caps the *scheduler grant*, while vLLM grants `K+1`, reserves the lookahead KV, then trims post-execution — so on a request's FINAL step BLIS's KV reservation and batched-token debit are further below vLLM's than before (same divergence class, same tracking issue). Deferred (design §9): per-step acceptance distribution, per-request heterogeneity, acceptance-rate prediction, a separate draft-model engine.

**Per-instance KV capacity for mixed-GPU node pools (#1522)**: When node pools are configured (`--policy-config` with `node_pools`) and `--total-kv-blocks` is NOT explicitly set, each placed instance auto-calculates its KV-block capacity from its ACTUAL placed GPU's `gpu_memory_gib` (plus TP, DP, block size, `--gpu-memory-utilization`, and weight precision) — so an H100 pool and an L40S pool no longer share one global capacity (restores INV-P2-1: an instance's GPU calibration and KV capacity describe the same device). Applied at all three placement sites (startup, deferred `NodeReadyEvent`, autoscaler scale-up) via `cluster.applyPerInstanceKVCapacity`, immediately after the `HWConfigByGPU` execution-calibration override (issue #893) so the placed GPU is authoritative for capacity as well. **Precedence**: an explicit `--total-kv-blocks` disables per-instance recalc (every instance keeps that uniform global capacity); when both node pools and PD per-pool KV overrides are present, the placement-derived per-GPU capacity wins (mirrors how `HWConfigByGPU` overrides the resolved `HWConfig`). A per-GPU capacity smaller than the configured `--max-model-len` auto-caps that instance's `MaxModelLen` to the KV-feasible maximum. Missing pool memory or a capacity-calc error falls back to the inherited global capacity with a warning (never a panic). `blis replay`/`observe` reject node pools, so this is `blis run` only (INV-13 parity N/A). Distinct from #1315 (role-specific capacity correct, latency coefficients wrong) and #633 (per-role overrides that can't express mixed hardware within one role).

**Multi-node TP placement (#1529)**: A model instance can occupy WHOLE NODES across a single node pool, enabling multi-node tensor parallelism (e.g. GLM-5.2 at TP=16 on 2×8 H100). `PlacementManager.PlaceInstance` runs two global passes: **Pass 1** is the original single-node first-fit across all pools (kept verbatim — single-node placement is byte-identical, INV-6); **Pass 2** (whole-node cross-node) is reached only when no single Ready node in any matching pool fits `tpDegree`. Pass 2 is **whole-node occupancy**: eligible only when `tpDegree > gpus_per_node` AND `tpDegree % gpus_per_node == 0`, taking `tpDegree/gpus_per_node` fully-free Ready nodes (uniform per-node rank count — the shape vLLM's `mp` executor enforces; vLLM's Ray backend tolerates asymmetric spreads but discourages them, so this is deliberately stricter. vLLM's docs recommend TP-within-node + PP-across-nodes over multi-node TP entirely — see #1530). The fragmentation case (`tpDegree ≤ gpus_per_node`, no single node free) is NOT spanned — an asymmetric 2+1-style group is discouraged and not modeled; the instance stays pending. Cost = distinct-nodes-spanned × pool `cost_per_hour` (`InstanceCostPerHour`, all three placement sites; single-node is 1×, unchanged). Startup deferrals are summarized in one `logrus.Warn` (NodeReadyEvent, the retry path, has no `blis run` production caller, so a deferral is effectively a drop); a structurally-unsatisfiable `tpDegree` (never divisible by any pool) gets a distinct error. **Cross-node latency is now priced by #1530** (see *Inter-node network cost* above): the trained-physics comm bases charge a spanning collective at a blended intra/inter-node bandwidth derived from this placement. The one-time stderr `logrus.Warn` on first span remains, restated as a factual notice — whether the cost actually applies depends on the latency backend and the placed GPU's interconnect calibration, which the placement manager does not know, so `applyPlacementTopology` raises the specific diagnostic when it will not. `blis run` only (`replay`/`observe` reject node pools, INV-13 N/A). **Config constraints (panic at startup, tracked by #1543):** with node pools, a per-role TP override (`--prefill-tp`/`--decode-tp`) differing from the global `--tp` is rejected (placement/span/cost use the global TP), and every pool must have a distinct `gpu_type` (cost/capacity are resolved by first-match on `gpu_type`). Out of scope: pipeline parallelism (#1535, design-doc first) and cross-*pool* spanning. Known limit tracked in #1536: the autoscaler's scale-up headroom (`GreedyEngine.Optimize`) is span-unaware — it counts free GPUs, not free whole nodes, so it can propose a spanning scale-up that `PlaceInstance` then harmlessly rejects. (Per-variant `CostPerReplica` IS span-aware: the span-multiplied `inst.CostPerHour` flows through the collector to the analyzer; taking the first replica's cost is correct because the global-TP guard + unique-`gpu_type` requirement make all replicas of a variant span the same node count.)

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
