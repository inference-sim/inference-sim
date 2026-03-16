# Client Behavior Model: Request Timeouts + Completion-Driven Sessions

**Status:** Draft
**Species:** Specification
**Closes:** #627
**Parent issues:** #529 (reasoning workload hang), #578 (blackbox MaxModelLen gap)

---

## 1. Problem

BLIS models the server (scheduler, KV cache, preemption) faithfully but models the client as a fire-and-forget injector with infinite patience. Three real-world client behaviors are missing:

1. **Request timeouts.** Real clients have timeouts. When a request exceeds its deadline, the client cancels it. This is the escape valve that bounds preemption spirals in production — without it, BLIS hangs indefinitely on reasoning workloads (#529).

2. **Closed-loop sessions.** Multi-turn/reasoning workloads send round N+1 only after round N's response arrives. BLIS pre-generates all rounds with fixed inter-arrival gaps, ignoring completion timing. Under load, this produces unrealistic arrival patterns.

3. **Session cancellation.** When a round times out, real clients abandon the session — subsequent rounds are never sent. BLIS has no concept of session-level state.

### Analysis questions this design answers

- AQ-1: At what timeout threshold does a reasoning workload stabilize instead of hanging? (Quantitative: timeout duration → completion rate curve)
- AQ-2: How does closed-loop session modeling change throughput and TTFT predictions vs open-loop? (Quantitative: >10% change in capacity planning recommendation = significant)
- AQ-3: What fraction of sessions are cancelled under various overload conditions? (Quantitative: cancellation rate by utilization level)
- AQ-4: How does per-SLO-class timeout differentiation affect request conservation and fairness?

### Simplest version analysis

Timeouts alone (without sessions) solve the immediate #529 hang: a request stuck in a preemption spiral is cancelled after 300s. However, timeouts alone still pre-generate all rounds — 4800 requests injected simultaneously instead of 960 round-0s. The 5x injection amplification distorts queueing behavior and makes AQ-1 answers dependent on an unrealistic arrival pattern. Closed-loop sessions remove this distortion and enable AQ-2 (arrival pattern fidelity). Sessions without timeouts don't bound the spiral. Both are needed for correctness (AQ-1) and fidelity (AQ-2).

### vLLM reference behavior

In vLLM v1, client disconnection triggers the scheduler's `finish_requests()` method which:
1. Removes the request from the running list or waiting queue (two-pass: collect then remove)
2. Sets status to `FINISHED_ABORTED` (distinct from `FINISHED_ERROR` for unexpected exceptions)
3. Frees KV blocks via the KV cache manager (reverse order, tail blocks first)
4. Emits a journey event with finished status

vLLM has no server-side per-request timeout — the timeout is always client-side (HTTP disconnect triggers abort). This is exactly what our design models.

### Real-system correspondence

| BLIS component | Real-system correspondent | Notes |
|---|---|---|
| Timeout event | Client HTTP timeout → TCP disconnect → abort in vLLM | Modeled as instantaneous; real propagation latency omitted |
| Timed-out state | vLLM `FINISHED_ABORTED` / `FinishReason.ABORT` | |
| KV release on timeout | vLLM KV cache manager `free()` | |
| Session manager | **No real-system correspondent** | vLLM/OpenAI/llm-d have no server-side session concept. BLIS-specific modeling addition for capacity planning |
| Completion callback | Client receiving response and deciding next action | |

## 2. Modeling Decisions

| Behavior | Status | Simplification / justification | What breaks if wrong |
|---|---|---|---|
| Client request timeout | **Modeled** | Instantaneous cancellation at exact deadline tick. Real propagation latency omitted (sub-ms, negligible vs 300s window) | If propagation latency matters: add a configurable cancellation delay (1 field) |
| KV block release on abort | **Modeled** | Immediate release. No abort cleanup cost (vLLM's two-pass removal is sub-ms) | If cleanup cost matters: add to `SchedulingProcessingTime` (#482) |
| Closed-loop session timing | **Simplified** | Think time is a fixed value per session (from ClientSpec), not sampled from a distribution. Real think time varies by orders of magnitude (sub-second for agents, 10-60s for humans). Lost: arrival time jitter, bursty re-engagement. Fixed value is sufficient for AQ-1 (stabilization threshold depends on timeout, not think-time variance) and AQ-2 (load correlation is binary: present or absent) | If think-time variance matters for capacity planning: promote to `DistSpec` (reuse existing distribution framework, 1 field change). The `DistSpec` type already supports gaussian, exponential, pareto, constant |
| Session cancellation on timeout | **Modeled** | All remaining rounds cancelled when any round times out. Real clients may retry or resume | If retry matters: cancellation rate would be overestimated. See retry omission below |
| Partial output on timeout | **Omitted** | Tokens generated before timeout are discarded. BLIS doesn't simulate streaming delivery. No current AQ depends on partial output | If streaming fidelity needed: requires output-streaming model (new subsystem) |
| Server-side abort cost | **Omitted** | vLLM's abort path overhead is <1ms, negligible vs step times | If abort cost matters: include in `SchedulingProcessingTime` (#482) |
| Client retry on timeout | **Omitted** | Not modeled. Impact: BLIS underestimates server load under timeout storms (retries increase effective arrival rate by 2-5x in practice). This makes the timeout model **optimistic** — the predicted stabilization threshold (AQ-1) is a lower bound. In real systems, retry amplification can shift the saturation point by 2-5x. For AQ-3, omitting retries means cancellation rates are underestimated (retried requests may succeed, reducing apparent cancellation) | If retry storms matter: add retry policy (new subsystem, ~3 files). The 2-5x amplification factor is from production experience with exponential backoff clients |
| Tiered KV cleanup on timeout | **Modeled** | Existing `ReleaseKVBlocks` handles both GPU and CPU tiers | |
| Open-loop → closed-loop transition | **Simplified** | Multi-turn workloads switch from pre-generated to completion-driven. **This is a breaking behavioral change for existing multi-turn workload specs** — fewer concurrent requests, different arrival patterns, different metrics. Non-session workloads remain open-loop (unchanged). **Opt-out shipped in this PR:** `closed_loop` field on ClientSpec (default: true for reasoning/multi-turn clients). Users who need open-loop behavior set `closed_loop: false` to preserve pre-generation of all rounds | If the default should be false (opt-in): change default, add deprecation warning for open-loop multi-turn |

## 3. Design Decisions

| Decision | Choice | Alternative considered | What breaks if wrong | Status |
|---|---|---|---|---|
| PR scope | Both timeouts + sessions in one PR | Timeouts only first | Sessions depend on the completion callback; separate PRs wire it twice | Proposed |
| Timeout mechanism | Dedicated timeout event in event queue | Check deadlines during step processing | Step-based check misses queued requests stuck behind full batch | Proposed |
| Timeout scheduling location | In enqueue (after all guards pass) | In arrival injection (before guards) | Scheduling before guards causes INV-1 double-counting for dropped requests | Proposed |
| Past-due timeout guard | If deadline ≤ current tick at scheduling time, immediately timeout | Schedule past-due event anyway | Past-due event fires on next pop, but request is in ambiguous state between arrival and enqueue | Proposed |
| Event priority: completion before timeout | Step events fire before timeout events at equal timestamps | No priority (heap-order-dependent) | Non-deterministic outcome (INV-6 violation) at same-tick completion/timeout | Proposed |
| Session callback | Completion callback function on Simulator | Interface, or Simulator imports SessionManager | Function is simplest single-method abstraction. If wrong: refactor to interface (2 files) | Proposed |
| SessionManager location | Workload package | sim/ kernel | Would add workload logic to kernel. If wrong: move (3-4 files) | Proposed |
| Timeout configuration | Per-client field on ClientSpec. Default: 300s. Configurable in workload spec YAML. Expected range: 30-600s for interactive, 600-3600s for batch | Global CLI flag | Can't differentiate by SLO class. If wrong: add CLI flag (1 file) | Proposed |
| Cluster routing for follow-ups | Follow-up rounds enter routing pipeline | Hardcode to same instance | Hardcoding produces unrealistically good cache hits. Note: without prefix-affinity scorer, follow-ups scatter and get zero cache hits — users should enable prefix-affinity for multi-turn workloads | Proposed |
| Generator signature | Unchanged. New function returns sessions alongside requests | Change existing return type | Many existing call sites break. If wrong: mechanical migration | Proposed |
| Distinct timeout state | Yes (maps to vLLM FINISHED_ABORTED) | Reuse completed with flag | Lose timeout/completion distinction in metrics. If wrong: add flag (2 files) | Proposed |

## 4. Components

### 4.1 Request Model Changes

A new terminal state (`timed_out`) is added alongside `completed`. The updated state machine:

```
queued ──→ running ──→ completed
  │            │
  └────────────┴──→ timed_out
```

Both `completed` and `timed_out` are terminal — no further transitions. This updates INV-2 (request lifecycle).

Each request carries a deadline (absolute tick) computed as `arrival_time + timeout`. Zero means no timeout. The deadline is set during workload generation, not by the simulator kernel.

**INV-1 update:** `injected = completed + still_queued + still_running + dropped_unservable + timed_out`

A new metric counter tracks timed-out requests.

### 4.2 Timeout Event

**Extension type:** Kernel modification (new event type in the simulation kernel). Not one of the four standard extension types (policy template, subsystem module, backend swap, tier composition). This is architecturally significant — it adds a client-behavior concept to the server-simulation kernel. The mitigation is that the event is self-contained (no inference-specific logic in execution) and the callback mechanism delegates all client logic to the workload package.

**Classification:** Mixed exogenous/endogenous. Round-0 timeout events are exogenous — scheduling time is determined by workload input (client patience), independent of simulation state. Follow-up round timeout events are endogenous — their deadline depends on when the previous round completed (which depends on server load). This affects common random numbers experiments: round-0 timeouts are CRN-safe, follow-up timeouts are not.

**Module contract:**
- **Observes:** Request state (completed? timed-out?), request's container (wait queue or running batch)
- **Controls:** Request state transition to timed-out, KV block release, container removal, callback invocation
- **Owns:** Nothing — the event is ephemeral. All mutated state belongs to the request, KV cache, queue, or batch
- **Invariants:** (a) No double-processing: if request is already terminal, timeout is a no-op (BC-3). (b) No double-counting: timeout events are only scheduled for requests that pass all enqueue guards (BC-5). (c) Computed-token tracking cleaned up on timeout (prevents memory leak)
- **Events produced:** None directly. The completion callback may produce follow-up arrival events
- **Events consumed:** Timeout event itself
- **Extension friction:** One-time infrastructure cost for this PR: ~5 files (Event interface gains Priority(), EventQueue.Less updated, event.go for TimeoutEvent, simulator.go for handler/scheduling, request.go for new state). Marginal cost for adding the *next* event type after this PR: 2 files (event definition + priority constant)

**Priority ordering:** At equal timestamps, events fire in this order (lowest priority number first):

| Event type | Priority | Rationale |
|---|---|---|
| ArrivalEvent | 0 | External input enters system first |
| QueuedEvent | 1 | Request enters queue after arrival processing |
| StepEvent | 2 | Batch processing (may complete requests) |
| ScheduledEvent | 3 | Observational (no state mutation) |
| RequestLeftEvent | 4 | Observational (no state mutation) |
| TimeoutEvent | 5 | Client-side cancellation fires last — completion wins over timeout at equal ticks (BC-12) |

This extends the per-instance event queue from timestamp-only to `(timestamp, priority, seqID)` ordering, matching the cluster event queue's scheme. The seqID (monotonic counter incremented on each push) ensures full determinism when two events of the same type share a timestamp.

**Backward compatibility of priority ordering:** The current per-instance heap uses timestamp-only ordering. Same-timestamp events are ordered by heap internals (not deterministic across Go versions — this is a pre-existing INV-6 weakness). Adding priority+seqID makes same-timestamp ordering fully deterministic, which is an INV-6 improvement. However, this may change the ordering of existing same-timestamp events compared to current behavior. If golden dataset entries have same-timestamp event pairs, their output may change. This must be verified during implementation: if golden dataset output changes, regenerate golden values (R12) and document the priority-ordering improvement as the cause.

**Scheduling:** When a request successfully passes all enqueue guards, a timeout event is scheduled at the request's deadline. The scheduling and enqueue happen in this order: (1) count input tokens (`TotalInputTokens += len(InputTokens)`), (2) check past-due guard, (3) if not past-due: enqueue and schedule timeout event, (4) if past-due: immediately timeout.

**Past-due guard:** If `deadline ≤ current_tick` at scheduling time, the request is immediately timed out **after counting input tokens but before entering the wait queue**. Input tokens are counted because the request was received by the server (arrival event fired, queueing delay processed, all guards passed) — the timeout is a client-side cancellation of a valid request, not a server-side rejection. The request is counted as `timed_out` (not `dropped_unservable`) in INV-1. The timeout event is not scheduled.

**Callback tick specification:** The completion callback receives a `tick` parameter representing when the request left the system:
- Normal completion in `processCompletions`: `tick = now + currStepAdvance` (the step's completion time)
- Length-capped completion: `tick = now + currStepAdvance` (same as normal)
- Timeout event: `tick = e.time` (the deadline tick)
- Dropped-unservable in `EnqueueRequest`: `tick = sim.Clock` (equals the QueuedEvent timestamp — `ProcessNextEvent` sets `Clock = ev.Timestamp()` before calling `Execute`, so `sim.Clock` inside `EnqueueRequest` is the QueuedEvent's scheduled time)
- Past-due timeout: `tick = sim.Clock` (same reasoning — the past-due check runs inside `EnqueueRequest` which is called from `QueuedEvent.Execute`)

This distinction matters for INV-10 (session causality): the follow-up round's arrival time = `tick + ThinkTimeUs`. Using the wrong tick would place follow-ups at incorrect times.

**Execution behavior:**
- If the request is already in a terminal state (completed, timed-out): no-op
- Otherwise: transition to timed-out, release KV blocks, remove from wait queue or running batch (using new-slice construction, NOT in-place modification — matches the `remaining` slice pattern in `processCompletions`, avoids R21 violation), clean up per-request execution tracking state, check work-conserving property (BC-18), invoke completion callback with tick = event timestamp

**Running batch removal pattern:** The timeout handler must build a new `[]*Request` slice excluding the timed-out request and assign it to `RunningBatch.Requests`. It must NOT use index-based deletion or `slices.Delete` which modifies the slice in place. This is the same pattern used by `processCompletions` which builds a `remaining` slice. Rationale: other code may hold references to the original slice header (R21 — Go `range` captures slice header at entry).

### 4.3 Completion Callback

**Extension type:** Kernel modification (new field on Simulator).

**Module contract:**
- **Observes:** Completed/timed-out/length-capped request, current tick
- **Controls:** Whether follow-up requests are generated (via return value or side-effect)
- **Owns:** Nothing — stateless bridge between kernel and workload logic
- **Invariants:** (a) Must not directly mutate wait queue, running batch, or KV cache. Returned requests are injected through the event queue. (b) In cluster mode, follow-up requests must enter the routing pipeline (BC-9). In single-instance mode, follow-up requests are injected directly
- **Events produced:** Arrival events for follow-up requests (indirectly, via injection)
- **Events consumed:** Completion/timeout/length-cap notifications
- **Extension friction:** To add a second callback (e.g., session-level completion): 1 field + 1 invocation site = 2 files

Invocation points:
1. After recording metrics for a normally completed request
2. After recording metrics for a length-capped request
3. After timeout cleanup
4. After any request is dropped as unservable in `EnqueueRequest` (invoked for ALL dropped requests, not just session requests — the callback returns nil for non-session requests, keeping the kernel session-agnostic while enabling INV-11 session completeness for heterogeneous clusters)

**Verification mechanism for cluster mode:** The session completeness invariant (INV-11, Section 7) detects silently dropped follow-ups — if a session ends without completing all rounds or being explicitly cancelled/horizon-interrupted, the invariant fails.

### 4.4 Session Manager

**Extension type:** Subsystem module (new module with its own interface, state, and behavioral contract).

**Scoping evaluation (Banks et al.):**
1. *Does the component help answer an analysis question?* Yes — AQ-2 (closed-loop arrival patterns) and AQ-3 (session cancellation rates)
2. *What accuracy is needed?* Round-level timing correlation with server load. Fixed think time is sufficient (see modeling decisions)
3. *Can data requirements be satisfied?* Think time comes from the `ThinkTimeUs` field on the reasoning spec (already exists in workload YAML). Default: value from `MultiTurnSpec.ThinkTimeUs` (currently 0 in most specs — users should set it for realistic behavior). Expected range: 0 (automated agents) to 60,000,000 µs (60s, human think time). Configurable per-client via workload spec
4. *What is the cost of inclusion?* 1 new file, ~200 lines. Session blueprints add ~200 bytes per session in memory
5. *What breaks if we omit it?* AQ-2 is unanswerable. Open-loop multi-turn generates 5x more concurrent requests than closed-loop, distorting queueing predictions
6. *What is the simplest version?* The current design IS the simplest version — one strategy (generate-on-completion), no factory, no configuration knob for strategy selection

**Module contract:**
- **Observes:** Completion callbacks (request reference + tick), request state (completed vs timed out vs length-capped)
- **Controls:** Whether to generate a follow-up round, with what parameters (arrival time, token content, deadline)
- **Owns:** Per-session lifecycle state: current round, accumulated context tokens, session status (active/completed/cancelled/horizon-interrupted). All state is exclusively owned — no other module reads or writes it
- **Invariants:**
  - **INV-10 (Session causality):** `round[N+1].ArrivalTime >= round[N].CompletionTime + ThinkTimeUs`. Boundary: ThinkTimeUs = 0 produces equality
  - **INV-11 (Session completeness):** Every session reaches exactly one terminal state: completed (all rounds done), cancelled (a round timed out), or horizon-interrupted (simulation ended mid-session). No session is silently abandoned
- **Events produced:** Follow-up requests (via return value to caller, who injects them)
- **Events consumed:** Completion callbacks
- **Extension friction:** To add a new session strategy (e.g., retry-on-timeout, speculative prefetch): 1 file (new strategy). Current design has no factory or strategy selection — a single hardcoded strategy. **R13 deviation:** single implementation without an interface is acceptable here because (a) the session manager is an internal module, not an extension point exposed to users, (b) adding a factory before a second implementation exists would be premature abstraction, and (c) the callback-based contract is strategy-agnostic by construction. When a second strategy is needed, add a factory + CLI/YAML field for selection (~3 files total)

**Behavior:**

A session blueprint captures the generation parameters for a multi-turn conversation: identity, turn limits, context growth strategy, think time, timeout budget, token length samplers, a per-session deterministic RNG (seeded from the client RNG for INV-6), prefix tokens, and workload metadata. Blueprints are immutable after creation.

Per active session, the manager tracks lifecycle state sufficient to enforce INV-10/INV-11 and generate follow-up rounds.

On completion callback:
1. Look up session by request's session ID. If not found (non-session request), return nil
2. If request timed out: mark session cancelled, return nil
3. If request was dropped as unservable (e.g., follow-up round exceeds KV capacity on a heterogeneous-capacity instance): mark session cancelled, return nil. **Note:** this requires a fourth callback invocation point — the enqueue guard drop path must invoke the callback for requests that have a session ID. Without this, INV-11 is violated for heterogeneous-instance clusters
4. If request was length-capped (force-completed at MaxModelLen boundary): generate round N+1 with truncated context. Length-capping means the output was truncated, not that the request failed — real clients receiving a length-capped response typically continue the session. The accumulated context will include the truncated output tokens
5. If current round is the final round: mark session completed, return nil
6. Compute round N+1 arrival time = completion tick + think time. If arrival time > horizon: mark session horizon-interrupted, return nil (BC-19 — prevents phantom requests that break INV-1)
7. Generate round N+1: sample input/output lengths, prepend accumulated context if enabled, set deadline, prepend prefix tokens. Return the new request

**Failure modes:**
- Empty session (MaxRounds = 0): No round-0 generated. Manager never receives a callback
- ThinkTimeUs = 0: Valid — produces arrival = completion tick (boundary of INV-10)
- Timeout = 0 on ClientSpec: Use default (300s). Validated at ClientSpec validation time: must be > 0 or omitted
- Session ID collision: Prevented by sequential counter-based ID generation (not UUID), which eliminates collision by construction
- Horizon interruption: When simulation ends mid-session, remaining rounds are never generated. Session status transitions to horizon-interrupted. The manager's state is cleaned up during finalization
- Same-session duplicate completion: If a callback fires for a session that has already advanced past the reported round (should not happen since sessions generate one round at a time), the callback is a no-op — the session's round counter has already advanced, so the stale completion is ignored

**Timeout-preemption interaction:** A request can be preempted (moved from running to queued, KV blocks released by preemption) and then time out while re-queued. The timeout handler finds the request in the wait queue (not running batch) and releases KV blocks. Since preemption already released blocks, `ReleaseKVBlocks` on a request with zero allocated blocks is a safe no-op (verified: existing implementation handles empty `RequestMap[req.ID]` gracefully — `delete` on missing key is a no-op in Go, and the release loop iterates an empty slice).

### 4.5 Workload Generation Changes

A new function produces both the initial request set and session blueprints. The existing generator function is preserved for backward compatibility. It internally delegates to the new function and returns only the requests.

For session clients (reasoning with multi-turn): only round-0 requests are generated. A session blueprint is created per session. Per-session RNG is derived from the existing workload generation RNG partition (`SubsystemWorkloadGen`). No new RNG partition needed.

**Common random numbers (CRN) consideration:** Closed-loop sessions break CRN for multi-turn workloads — two configurations with the same seed produce different arrival streams after the first round diverges (because follow-up timing depends on server load). This is inherent to closed-loop modeling and acceptable: the alternative (open-loop) doesn't model load-dependent arrivals at all.

For non-session clients: identical to current behavior.

### 4.6 Cluster-Mode Integration

In cluster mode, the completion callback delegates to the session manager. Follow-up requests enter the routing pipeline (admission → routing → instance injection), allowing the prefix-affinity scorer to make optimal decisions.

The cluster aggregates timed-out request counts in the same pattern as dropped-unservable counts.

**Design constraint:** The session manager assumes single-threaded invocation. The cluster event loop is currently single-threaded, satisfying this constraint. If the cluster simulator later adds concurrent event processing, the session manager would need synchronization.

**Stale snapshot consideration:** When a follow-up round enters routing, the snapshot may not yet reflect KV blocks freed by the completing round (INV-7: KVUtilization is periodic). The prefix-affinity scorer still benefits because the prior round's prefix hash is in the instance's cache index regardless of snapshot freshness. Under high load, the composite scorer may route follow-ups away from the cache-warm instance due to load balancing — this is intentional and matches the design decision that follow-ups should be routed, not hardcoded.

**Timeout-horizon interaction:** Requests arriving near the horizon with deadlines beyond it (e.g., arrival at `horizon - 100s`, deadline at `horizon + 200s`) will NOT time out — the event loop stops at horizon, and the timeout event is never popped. These requests are counted as `still_queued` or `still_running` in INV-1, not as `timed_out`. The `timed_out` counter reflects only requests that actually timed out during the simulation. Users should be aware that extending the horizon may increase the timed-out count.

**No-op default:** When no session blueprints exist (non-session workloads), the session manager is not instantiated. The completion callback is nil or returns nil for every request. No code path through the session manager is exercised. All existing workloads produce byte-identical output (verified via golden dataset regression in BC-14).

### 4.7 Workload Path Coverage

| Path | Timeout | Closed-loop | Notes |
|------|---------|-------------|-------|
| Distribution/preset | Default 300s deadline | N/A | No sessions |
| Reasoning/multi-turn | Per-client or default 300s | Yes — round-0 only pre-generated | Core use case |
| inference-perf with multi-turn | Same as reasoning | Same as reasoning | Via inference-perf expansion |
| ServeGen | Default 300s deadline | N/A | Same as distribution |
| Trace v2 replay | Default 300s deadline | N/A | Traces have real timing |
| Observe mode (real backend) | Own HTTP timeout (5 min) | N/A | No changes |

## 5. DES Design Review Checklist

| # | Question | Answer |
|---|----------|--------|
| 1 | Analysis questions? | AQ-1 through AQ-4 (Section 1) |
| 2 | Modeled / simplified / omitted? | Section 2 table (9 entries with justification and reversal cost) |
| 3 | New events classified? | TimeoutEvent: mixed exogenous/endogenous — round-0 exogenous, follow-up endogenous (Section 4.2) |
| 4 | Event priority? | Full priority table for all 6 event types (Section 4.2) |
| 5 | State vs statistics? | TimeoutEvent mixes state mutation with metric increment, consistent with existing `processCompletions`. Known debt — tracked as part of a broader separation effort (not filed as standalone issue because the pattern is pervasive in the existing codebase) |
| 6 | New randomness? | Per-session RNG from existing `SubsystemWorkloadGen`. No new partition. CRN broken for closed-loop (inherent, documented in Section 4.5) |
| 7 | New state? | Request: deadline field, timed-out terminal state. SessionManager: per-session lifecycle state. Metrics: timed-out counter. New statistics: timed-out count, session completion/cancellation counts |
| 8 | Verification? | BC-1 through BC-19 (Section 6). INV-1 5-term conservation. INV-4 post-timeout. INV-8 edge case c (BC-18). INV-10 session causality. INV-11 session completeness. #529 before/after regression. Horizon follow-up guard (BC-19) |
| 9 | Validation? | Section 8: H-Sessions experiment, #529 regression, deferred real-vLLM comparison |
| 10 | Simplest version? | Section 1: Timeouts alone solve hang, sessions needed for arrival fidelity. Both shipped together |

## 6. Behavioral Contracts

**BC-1: Timeout cancels queued request.** GIVEN a request with a deadline, still in the wait queue when the deadline tick arrives → THEN the request transitions to timed-out, KV blocks released, removed from wait queue, timed-out counter increments.

**BC-2: Timeout cancels running request.** GIVEN a request in the running batch at its deadline tick → THEN removed from running batch, KV blocks released, per-request execution tracking state (computed-token map entry) cleaned up, transitions to timed-out. If running batch becomes empty and WaitQ has items, a step event is scheduled (INV-8).

**BC-3: Timeout no-op for completed request.** GIVEN a request that completed normally before its deadline → WHEN timeout event fires → THEN no state change, no metric increment.

**BC-4: INV-1 conservation with timeouts.** GIVEN any workload → WHEN simulation ends → THEN `injected == completed + still_queued + still_running + dropped_unservable + timed_out`.

**BC-5: Dropped-unservable requests get no timeout event.** GIVEN a request rejected by enqueue guards → THEN no timeout event scheduled. Counted only as `dropped_unservable`.

**BC-6: Closed-loop round generation.** GIVEN a 3-round session, round 0 completes at tick T with think time D → THEN round 1 generated with arrival time = T + D.

**BC-7: Session cancellation on timeout.** GIVEN a 5-round session, round 2 times out → THEN rounds 3-4 never generated. Session marked cancelled.

**BC-8: Context accumulation in closed-loop.** GIVEN context growth = "accumulate", round 0 (input I₀, output O₀) completes → THEN round 1's input begins with I₀ + O₀ accumulated tokens.

**BC-9: Cluster routing for follow-up rounds.** GIVEN cluster mode, round 0 completes on instance_0 → THEN round 1 enters cluster routing pipeline.

**BC-10: Determinism.** Same seed + workload spec → byte-identical output across runs, including timeout and session outcomes.

**BC-11: Backward compatibility.** Non-session workloads with default timeout and all requests completing within 300s produce identical output to current code.

**BC-12: Completion wins at equal timestamps.** GIVEN step completion and timeout at the same tick → THEN step event fires first (priority ordering), request completes normally, timeout is no-op.

**BC-13: INV-4 conservation after timeout.** GIVEN a running request with N KV blocks, when it times out → THEN `allocated_blocks + free_blocks = total_blocks` holds after release.

**BC-14: Horizon-interrupted session.** GIVEN a 5-round session, rounds 0-2 complete, simulation horizon reached before round 3 completes → THEN session marked horizon-interrupted. Round 3 counted as `still_queued` or `still_running` in INV-1. Rounds 4 never generated.

**BC-15: Preempt-then-timeout is safe.** GIVEN a running request that is preempted (KV released, moved to wait queue) and then times out while queued → THEN KV release is a no-op (zero blocks allocated), request removed from wait queue, timed-out counter increments. No double-free.

**BC-16: Length-capped session continues.** GIVEN a session where round N is length-capped (force-completed at MaxModelLen) → THEN session generates round N+1 with accumulated context including the truncated output tokens. The session is NOT cancelled — length-capping is a completion, not a failure.

**BC-17: Dropped-unservable follow-up cancels session.** GIVEN a session where round N+1 is dropped by enqueue guards (e.g., exceeds KV capacity on a smaller instance) → THEN session marked cancelled via the callback. INV-11 session completeness holds.

**BC-18: Queued-timeout with empty batch triggers work-conserving check (defense-in-depth).** GIVEN a queued request that times out when the running batch is empty (no step event scheduled) and WaitQ still has other items after removal → THEN a step event is scheduled at the current tick to process the remaining queued requests (INV-8 work-conserving property). **Note:** This state (`stepEvent == nil && WaitQ.Len() > 0`) may not be reachable with current FormBatch semantics — `scheduleNextStep` already schedules a StepEvent when WaitQ has items after an empty batch. BC-18 is a defense-in-depth guard: it costs one `if` check and prevents a livelock if future FormBatch changes create this state.

**BC-19: Follow-up rounds beyond horizon are not generated.** GIVEN a session where round N completes and round N+1's computed arrival time (completion_tick + think_time) exceeds the simulation horizon → THEN round N+1 is NOT generated, no arrival event is injected, and the session is marked horizon-interrupted. This prevents phantom requests that would be registered in Metrics.Requests but never processed, breaking INV-1.

## 7. Invariants

**INV-1 (Request conservation):** Updated: `injected = completed + still_queued + still_running + dropped_unservable + timed_out`. In cluster mode, the full pipeline formula remains `num_requests = injected + rejected` where `injected` is the 5-term sum above. The `InjectedRequests` computation in `SaveResults` must be updated to include `timed_out`. High-risk change — INV-1 has broken twice before (#386, #387). Requires explicit 5-term conservation test in both single-instance and cluster mode.

**INV-2 (Request lifecycle):** Updated state machine adds `timed_out` terminal state.

**INV-3 (Clock monotonicity):** Unaffected.

**INV-4 (KV cache conservation):** Explicitly tested after timeout-triggered KV release (BC-13). Both single-tier and tiered cache.

**INV-5 (Causality):** Scoped to completed requests only. The chain `arrival_time <= enqueue_time <= schedule_time <= completion_time` does not apply to timed-out requests (which may time out before scheduling) or dropped-unservable requests (which never enter the queue). For timed-out requests, a weaker causality holds: `arrival_time <= enqueue_time <= timeout_time` where `timeout_time = deadline`.

**INV-6 (Determinism):** Requires event-type priority ordering (Section 4.2 priority table). Per-session RNG seeded from client RNG.

**INV-8 (Work-conserving):** After timeout frees KV/queue slots, work-conserving property is maintained. Three cases: (a) Timeout removes a running request — a StepEvent already exists (the running batch was non-empty), so the next step's batch formation will pick up waiting requests. (b) Timeout removes a queued request while a running batch exists — the running batch's StepEvent handles scheduling. Queued requests have zero allocated KV blocks (preemption releases blocks before re-queuing), so no KV is freed. (c) **Edge case: timeout fires with empty running batch and non-empty WaitQ** — this can occur when KV pressure caused FormBatch to preempt all requests (empty batch, all re-queued, stepEvent = nil). If a queued request times out in this state and WaitQ still has items, INV-8 requires a StepEvent. **The timeout handler must check `stepEvent == nil && WaitQ.Len() > 0` after removing the request and schedule a StepEvent if needed** (matching the work-conserving check in `QueuedEvent.Execute` and `scheduleNextStep`).

**INV-10 (Session causality):** NEW. `round[N+1].ArrivalTime >= round[N].CompletionTime + ThinkTimeUs`. Boundary: ThinkTimeUs = 0 produces equality.

**INV-11 (Session completeness):** NEW. Every session reaches exactly one terminal state: completed, cancelled, or horizon-interrupted.

## 8. Validation Strategy

**Verification (correctness):**
- One test per behavioral contract (BC-1 through BC-19)
- INV-1 conservation: explicit 5-term test. Both single-instance and cluster mode
- INV-4 post-timeout: `allocated + free = total` after timeout of running request (both tiers)
- INV-10 session causality: including ThinkTimeUs=0 boundary
- INV-11 session completeness: first-round timeout, last-round timeout, mid-session timeout, horizon interruption
- Golden dataset regression: verify byte-identical output for existing test entries (BC-11/BC-14)
- All tests are law-tests (verify invariants), not golden-value-tests
- JSON output schema: `timed_out_requests` field uses `omitempty` (consistent with `kv_allocation_failures`)

**Validation (fidelity):**
- **H-Sessions hypothesis experiment:** Under reasoning workloads at 80% utilization with 4 instances, closed-loop sessions produce inter-round gaps that correlate with instance queue depth (Pearson r > 0.5 across 5 seeds). Control: open-loop configuration with same workload and seeds (expected r ≈ 0). Workload: 200 sessions × 5 rounds, mean input 256, mean output 128, poisson arrival, 5 seeds. Null hypothesis: closed-loop inter-round gap is independent of server load (r ≈ 0)
- **#529 before/after regression test:** (a) Control: run the #529 workload (5-round reasoning, output_len=1448, 7463 KV blocks) WITHOUT timeouts — confirm the hang still exists (zero completions after 10x expected simulation time, or still_running > 0 with preemption count growing). This proves the mechanism is necessary. (b) Treatment: same workload WITH timeouts (300s default) — confirm simulation completes within 1x horizon, with timed-out requests properly accounted in INV-1. This proves the mechanism is sufficient
- **Real-vLLM comparison:** Deferred. File tracking issue: compare BLIS timeout rates against vLLM abort rates from sim-to-real validation traces

**Falsification criteria:**
- If default 300s timeout fires for >1% of requests in workloads that complete fully on real vLLM → default too aggressive
- If closed-loop sessions produce identical throughput to open-loop at sub-saturation → feature adds complexity with no fidelity benefit (refuted by simplest-version analysis, but empirical confirmation needed)
- If session cancellation rate exceeds 20% under moderate load (70% utilization) → timeout too tight for normal reasoning workloads

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Orphaned timeout events after completion | No-op guard: check terminal state first. O(1) per orphan |
| INV-1 conservation regression | Timeout event scheduled AFTER enqueue guards (BC-5). Explicit 5-term test. High-risk: INV-1 broke twice before |
| INV-4 post-timeout KV accounting | Explicit conservation test (BC-13). Both tiers |
| Past-due timeout (deadline ≤ current tick at scheduling) | Immediate in-place timeout in enqueue, no event scheduled |
| Non-deterministic same-tick ordering | Full event priority table (Section 4.2). BC-12 |
| Computed-token tracking leak | Cleanup in timeout handler, matching preemption cleanup pattern |
| Timeout-preemption double-free | Safe: preemption releases blocks; timeout on re-queued request finds empty block list (Go no-op) |
| Session manager non-determinism | Per-session RNG from client RNG. Counter-based session IDs |
| Cluster follow-up uses stale snapshot | Consistent with INV-7. Prefix cache index is fresh |
| Dropped follow-up silently ends session | Session manager handles dropped-unservable follow-ups as cancellation (step 3 in callback). INV-11 catches abandonment |
| Timeout events beyond horizon never fire | Requests counted as still_queued/still_running in INV-1. Documented in Section 4.6 |
| Default 300s changes golden dataset | Golden tests complete in <1s. Timeout events are no-ops (BC-3). Priority ordering preserves existing event sequence for non-timeout events |
| Callback modifies batch during iteration | Contract prohibits direct mutation. Injection through event queue only |
| State/statistics mixing in timeout handler (R14) | Known debt: timeout handler spans request state, KV cache, queue/batch, and metrics — same multi-concern pattern as existing `processCompletions`. Decomposition deferred to a future PR that addresses both together (the Step() decomposition in #393 provides the pattern). Not filed as standalone issue because the debt is pre-existing and pervasive |
| Timeout YAML field zero-value ambiguity (R9) | The `timeout` field on ClientSpec should use `*int64` (pointer type) so that zero is distinguishable from "not set." Nil = use default (300s). Zero = no timeout (infinite). Positive = explicit timeout. This follows the R9 pattern established for KVOffloadThreshold |
| Priority ordering changes golden dataset | Adding `(timestamp, priority, seqID)` to per-instance EventQueue makes same-timestamp ordering deterministic (INV-6 improvement). If existing golden dataset entries have same-timestamp events, output may change. Verify during implementation; regenerate golden values if needed (R12) |
| In-place RunningBatch modification (R21) | Timeout handler must use new-slice construction (exclude timed-out request, assign new slice), NOT in-place `slices.Delete`. Matches `processCompletions` pattern. R21: `range` captures slice header at entry |
| TotalInputTokens for past-due timeouts | Past-due timeouts count input tokens (request was received and validated, just timed out). Input token counting happens BEFORE the past-due check in `EnqueueRequest`. Consistent with "timed out" INV-1 accounting |
| Wrong callback tick breaks INV-10 | Each invocation point passes the correct tick: `now + currStepAdvance` for step completions, `e.time` for timeouts, `sim.Clock` for enqueue drops. Follow-up arrival = tick + ThinkTimeUs — wrong tick produces wrong arrival time |
