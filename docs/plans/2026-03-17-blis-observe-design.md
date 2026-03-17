# Design: `blis observe` CLI Command

**Status:** Draft
**Issue:** #659
**Date:** 2026-03-17
**Species:** Specification

---

## 1. Problem Statement

### Analysis Questions

`blis observe` exists to answer:

1. **What are the real per-request latencies (TTFT, E2E) of a production inference server under a given workload?** This provides ground-truth data for the observe/replay/calibrate pipeline.
2. **Does the simulator's queueing model track real queueing behavior?** By replaying the same workload through BLIS and comparing via `blis calibrate`, we measure fidelity gaps.
3. **How do prediction errors vary across load regimes?** Sub-saturation vs saturation calibration reveals where the simulator's abstractions break down.

### What It Does

`blis observe` accepts a WorkloadSpec, generates requests from distributions, dispatches them to a real inference backend at precise arrival times, and records per-request timing into TraceV2 files. It is the data collection leg of the observe/replay/calibrate pipeline.

### Critical Architectural Requirement

`blis observe` must consume the same workload generation pipeline as `blis run`. Any future WorkloadSpec extension (new arrival process, new session strategy, new client type) must automatically be available to both commands. The observe command must never have its own parallel request generation or session logic.

### Modeling Decisions

| Aspect | Treatment | Justification / What Is Lost |
|--------|-----------|------------------------------|
| Arrival timing | **Modeled** | Wall-clock dispatch matching WorkloadSpec arrival times |
| TTFT measurement | **Simplified** | Measured via first SSE chunk timestamp, not server-side first-token GPU event. Lost: server-internal prefill-to-first-token gap (~0.1-1ms) |
| E2E measurement | **Modeled** | Last chunk timestamp minus send timestamp |
| Token counting | **Simplified** | Prompt generated via word repetition; actual count from server `prompt_tokens`. Lost: tokenizer-dependent content effects on processing time (typically <1% for decoder-only models) |
| Server-side batching dynamics | **Omitted** | Black-box observation only. Lost: cannot attribute latency to batching decisions. Acceptable: calibrate compares E2E outcomes, not internal scheduling |
| Prompt content effects | **Omitted** | Synthetic prompts, no semantic variation. Lost: content-dependent attention patterns. Acceptable: latency is dominated by token count, not content |
| Network jitter | **Simplified** | RTT measured once in header; per-request variance not modeled. Lost: tail latency from network variability |
| Per-chunk ITL | **Omitted** | Only first/last chunk timestamps captured. Deferred to #655 |
| Session follow-ups | **Modeled** | Closed-loop sessions via SessionManager reuse, completion-driven follow-ups |
| Coordinated omission | **Simplified** | Dispatcher sleeps to target time; if dispatch falls behind, requests are sent late (not dropped). Traces record actual send time, so downstream analysis can detect backpressure |

### Scoping Evaluation (Banks et al.)

1. **Does the component affect the analysis questions?** Yes — observe collects the ground-truth data that calibrate compares against.
2. **Can it be replaced by a constant?** No — real server latency is the variable under study.
3. **Is the interaction with other components well-defined?** Yes — observe produces TraceV2 files consumed by replay and calibrate. No coupling beyond file format.
4. **Is there data to parameterize it?** Yes — WorkloadSpec provides all parameters. Server provides timing data.
5. **Can it be validated independently?** Yes — unit tests with mock HTTP server; integration test against real vLLM.
6. **What is the simplest version that answers the same questions?** Open-loop-only dispatch with a single request stream. Session support adds complexity but is necessary for multi-turn workload fidelity — without it, observe cannot generate the same request sequence as `blis run` for closed-loop workloads, breaking the calibration pipeline.

## 2. Design Decisions

Each decision lists alternatives considered, rationale, what breaks if wrong, and status.

### D1: Shared workload generation pipeline — **Implemented** (in `blis run`)

Both `blis run` and `blis observe` use the workload generation pipeline, which returns pre-generated requests from all clients plus session blueprints for closed-loop clients. These are not mutually exclusive — a mixed WorkloadSpec produces both.

**Alternatives:** (a) Observe-specific request generator. **Rejected:** violates the architectural requirement. Every WorkloadSpec extension would need dual implementation.
**What breaks if wrong:** Future extensions silently diverge between run and observe. Calibration becomes meaningless for workloads that exercise new features.

### D2: Both workload input paths — **Proposed**

Supports `--workload-spec` YAML and `--rate` + distribution flags (synthesis).

**Alternatives:** (a) `--workload-spec` only. **Rejected:** Distribution flags enable quick experimentation without writing YAML.
**What breaks if wrong:** Users write unnecessary YAML for simple measurements. Low risk.

### D3: Single-goroutine dispatcher with async completions — **Proposed**

A main dispatcher goroutine rate-paces dispatch. Completion goroutines feed session follow-ups back via channel. Single point of scheduling control.

**Alternatives:** (a) Timer-based dispatch via `time.AfterFunc` per request — no blocking sleep, but harder to bound concurrency and drain cleanly. (b) Worker pool with N goroutines pulling from a shared queue — natural concurrency bound but workers sleeping wastes goroutines; ordering not guaranteed.
**What breaks if wrong:** If the single-goroutine model can't keep up with target request rates, dispatch latency accumulates. In practice, the per-iteration cost (sleep + semaphore acquire + goroutine launch) is ~1-5 microseconds, supporting well above 100K rps. The real bottleneck is semaphore contention when the server is slow, not dispatcher throughput. Mitigated by recording actual send time (coordinated omission detection).

### D4: Observe-specific flags only — **Proposed**

No sim config flags (`--total-kv-blocks`, `--routing-policy`, etc.). Observe talks to a real server, not the DES. Adding `observe` does not change behavior of any existing command.

**Alternatives:** (a) Reuse `registerSimConfigFlags` for forward compatibility. **Rejected:** Exposes ~30 irrelevant flags, confusing UX.
**What breaks if wrong:** If future observe features need sim config (e.g., co-simulation), flags must be added then. Low risk.

### D5: Default streaming=true — **Proposed**

Streaming gives true TTFT via first SSE chunk. Non-streaming only gets approximate first-byte timing (server buffers entire response).

**Alternatives:** (a) Default non-streaming — simpler HTTP handling but loses TTFT accuracy. (b) Per-client from WorkloadSpec — most flexible but adds dispatch path complexity.
**What breaks if wrong:** Some server configurations may not support streaming SSE. The `--no-streaming` flag provides fallback.

### D6: Simple prompt content — **Proposed**

Word repetition approximates token count. Server-reported `prompt_tokens` provides ground truth.

**Alternatives:** (a) Load real prompts from a dataset file. **Rejected:** Adds dataset loading complexity, new schema fields, tokenizer dependency. Deferred to #660.
**What breaks if wrong:** For models with significant content-dependent latency variation (e.g., vision models with image tokens), synthetic prompts may underestimate processing time. The modeling decisions table acknowledges this as an omission.

### D7: SessionManager concurrency strategy — **Proposed**

`SessionManager.OnComplete` is documented as single-threaded (DES event loop assumption). In observe mode, completions arrive concurrently from HTTP goroutines. All `OnComplete` calls are serialized through a single goroutine that reads from a completion channel, preserving the single-threaded contract. Follow-ups are written to `followUpCh` by this serializer goroutine, not by HTTP completion goroutines directly.

**Alternatives:** (a) Add mutex to SessionManager. **Rejected:** Modifying a sim/workload type for a cmd/ use case violates separation of concerns. (b) Create observe-specific session tracker. **Rejected:** Duplicates logic, breaks the shared-pipeline requirement.
**What breaks if wrong:** If serialization bottlenecks throughput (unlikely — OnComplete is CPU-only, ~microseconds), completion processing becomes the bottleneck instead of HTTP I/O.

## 3. Behavioral Contracts

### ObserveOrchestrator Contract

**Observes:** WorkloadSpec (via GenerateWorkload), HTTP responses (via RealClient), wall-clock time.

**Controls:** Request dispatch timing, concurrency bound, session follow-up scheduling.

**Owns:** In-flight request tracking, active session state (via SessionManager), trace recording buffer.

**Invariants:**
- **OBS-INV-1 Request conservation:** `dispatched_count == recorded_ok + recorded_error + recorded_timeout` at drain completion. Every dispatched request produces exactly one trace record.
- **OBS-INV-2 Concurrency bound:** At any instant, `in_flight_count <= max_concurrency`.
- **OBS-INV-3 Session serialization:** All `SessionManager.OnComplete` calls are made from a single goroutine, preserving the single-threaded contract.
- **OBS-INV-4 Warmup exclusion:** The first `warmup_count` requests (by dispatch index, regardless of outcome) are dispatched but excluded from the exported trace. `len(trace_records) == total_dispatched - min(warmup_count, total_dispatched)`. Edge case: if `warmup_count >= total_dispatched`, the trace is empty (valid — all requests were warmup).
- **OBS-INV-5 Timestamp ordering:** For every trace record: `arrival_time_us <= send_time_us <= first_chunk_time_us <= last_chunk_time_us` (when status is "ok").

**Events consumed:** HTTP responses (external), timer expiry (wall-clock dispatch), session follow-ups (internal channel).

**Events produced:** Trace records (written to TraceV2 files).

**Extension friction:** Adding a new dispatch strategy (e.g., closed-loop rate adaptation) requires modifying the dispatcher goroutine (~1 file). Adding a new recording field requires modifying the adapter and recorder (~2 files).

### Session Adapter Contract

The adapter bridges HTTP responses to the SessionManager's expected interface.

**GIVEN** a completed HTTP request with server-reported token counts
**WHEN** the adapter constructs the request representation for OnComplete
**THEN** the session state field reflects HTTP outcome (success → completed, error/timeout → timed out), the progress indicator equals input tokens plus server-reported output tokens, and the output tokens array has length equal to server-reported completion_tokens.

**GIVEN** a session request that receives an HTTP error or timeout
**WHEN** the adapter passes it to OnComplete
**THEN** OnComplete cancels the session (no follow-up generated).

**GIVEN** a session request that completes successfully and is not the final round
**WHEN** OnComplete returns a follow-up request
**THEN** the follow-up's arrival time is the completion wall-clock plus the blueprint's think-time, and it is dispatched at that wall-clock time.

### Drain Contract

**GIVEN** all pre-generated requests have been dispatched and all HTTP responses received
**WHEN** no active sessions remain (all completed, cancelled, or errored) AND no requests are in-flight
**THEN** the follow-up channel is closed, the dispatcher exits, and trace export begins.

**Failure mode — backpressure:** If the dispatcher falls behind the target rate (e.g., semaphore full), requests are dispatched late. The trace records actual `send_time_us` (not target arrival time), so downstream analysis can detect backpressure by comparing `arrival_time_us` vs `send_time_us`.

**Failure mode — error storm:** If all HTTP requests fail (server down), all sessions cancel immediately. Drain completes after the last in-flight request times out. The trace records all errors.

**Failure mode — semaphore deadlock:** Cannot occur — semaphore is acquired before goroutine launch, released after HTTP response. No nested acquisition.

## 4. Time-Base Convention

- **`arrival_time_us`**: Relative microseconds from time origin (matching WorkloadSpec semantics, starting at 0). Rebased to absolute wall-clock by adding `start_wall` for dispatch scheduling, but stored as relative in the trace for replay compatibility.
- **`send_time_us`, `first_chunk_time_us`, `last_chunk_time_us`**: Absolute wall-clock microseconds (`time.Now().UnixMicro()`).
- **Session think-time**: Wall-clock duration. Follow-up arrival time = current wall-clock + think-time (in microseconds), stored as relative by subtracting `start_wall`.

This mixed convention matches the existing trace recording implementation and is consumed correctly by `blis calibrate` (which computes TTFT as `first_chunk - send`, both absolute).

## 5. CLI Flags

### Required
| Flag | Type | Description |
|------|------|-------------|
| `--server-url` | string | Inference server endpoint |
| `--model` | string | Model name for API requests |
| `--trace-header` | string | Output path for TraceV2 header YAML |
| `--trace-data` | string | Output path for TraceV2 data CSV |

### Workload input (one required)
| Flag | Type | Description |
|------|------|-------------|
| `--workload-spec` | string | Full WorkloadSpec YAML path |
| `--rate` | float | Requests per second (distribution synthesis path) |

### Optional
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--api-key` | string | "" | Bearer token for auth |
| `--server-type` | string | "vllm" | Server type for endpoint construction |
| `--max-concurrency` | int | 256 | Max simultaneous in-flight requests |
| `--warmup-requests` | int | 0 | Requests to exclude from trace |
| `--seed` | int64 | 42 | RNG seed (overrides WorkloadSpec) |
| `--horizon` | int64 | from spec | Observation horizon in microseconds |
| `--num-requests` | int | from spec | Max requests to generate |
| `--no-streaming` | bool | false | Disable streaming (use non-streaming HTTP) |
| Distribution flags | various | — | Same as `blis run` distribution mode |

## 6. TraceV2 Output

### Header
```yaml
trace_version: 2
time_unit: us
created_at: "2026-03-17T10:00:00Z"
mode: real
warm_up_requests: 10
workload_spec: "spec.yaml"
server:
  type: vllm
  model: meta-llama/Llama-3.1-8B-Instruct
network:
  measured_rtt_ms: 0.5
```

### Data
Standard TraceV2 CSV with all columns defined in `sim/workload/tracev2.go` (canonical source for column schema). Session follow-ups have populated `session_id` and `round_index > 0`. See Section 4 for time-base convention.

## 7. Testing Strategy

### Verification (correctness invariants)

| Test | Invariant | What It Verifies |
|------|-----------|------------------|
| Request conservation | OBS-INV-1 | `dispatched == recorded_ok + recorded_error + recorded_timeout` |
| Concurrency bound | OBS-INV-2 | Peak concurrent requests ≤ `--max-concurrency` |
| Session serialization | OBS-INV-3 | Data race detector (`-race`) passes with concurrent sessions |
| Warmup exclusion | OBS-INV-4 | Trace record count = total - warmup |
| Timestamp ordering | OBS-INV-5 | `arrival ≤ send ≤ first_chunk ≤ last_chunk` for ok records |
| Session follow-ups | Adapter contract | Round-2 arrives after think-time; context accumulates |
| Session cancellation | Adapter contract | HTTP 500 → no follow-up; session terminated |
| Pipeline parity | D1 | Same WorkloadSpec + seed → same round-0 request sequence as `blis run` (equality scoped to calibration-relevant fields: arrival time, input token count, output token count, session ID, round index — not token ID content, which differs between sim and HTTP paths) |
| TraceV2 round-trip | Format | Export → load → field equality |
| Error storm drain | Drain contract | All-error workload completes, all records captured |

### Validation (fidelity against real systems)

- **Timing tolerance:** Open-loop dispatch within ±10ms of target arrival time (mock server).
- **Success criteria for calibrate pipeline:** Observe → replay → calibrate produces a CalibrationReport. No specific MAPE threshold at this stage (that's calibrate's concern).
- **Falsification:** If `send_time_us - (start_wall + arrival_time_us)` consistently exceeds 100ms, the dispatcher architecture is inadequate for the workload rate.

### What would falsify the design?

- SessionManager reuse produces incorrect follow-up sequences (field mapping adapter is wrong).
- Dispatcher can't sustain >1000 rps (single-goroutine bottleneck).
- Coordinated omission makes traces unusable for calibration (send times don't match arrival times).

## 8. What This Design Does NOT Cover

- Per-chunk ITL timestamps (#655 Bug 5 — deferred)
- Prompt dataset loading (#660 — deferred)
- Prefix group token sharing (#660 Priority 2 — deferred)
- Multi-server dispatch (single `--server-url`)
- Live progress reporting / metrics dashboard
- Automatic retry on transient errors (errors recorded, not retried)
- `stream_options` for streaming usage extraction (#660 Priority 2.5)
- Chat completions endpoint support (#660 Priority 2.5)
- `finish_reason` extraction (#660 Priority 2.5)
