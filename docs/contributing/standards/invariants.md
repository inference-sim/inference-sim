# BLIS System Invariants

Invariants are properties that must hold at all times during and after simulation. They are verified by invariant tests (see R7) and checked during self-audit (Step 4.75).

**Hypothesis family mapping:** INV-1 through INV-3, INV-5, and INV-6 belong to the **Scheduler invariants (safety/liveness)** family. INV-4 (KV cache conservation), INV-7 (signal freshness), INV-8 (work-conserving property), and INV-9 (oracle knowledge boundary) belong to the **Structural model** family. See `docs/contributing/standards/experiments.md` for hypothesis family definitions.

## INV-1: Request Conservation

**Statement:** `injected_requests == completed_requests + still_queued + still_running + dropped_unservable + timed_out` at simulation end (all levels).

**Full pipeline:** `num_requests == injected_requests + rejected_requests` (from anomaly counters).

**Verification:** `sim/cluster/cluster_test.go` — conservation tests. Conservation fields (`still_queued`, `still_running`, `injected_requests`) are included in CLI JSON output.

**Evidence:** Issue #183 — a silently-dropped request violated conservation for months.

**Experimental validation:** H12 confirmed conservation across 10 policy configurations (67 invariant checks) — including round-robin, least-loaded, weighted (multiple scorer configs), SJF, priority-FCFS, token-bucket admission, and always-busiest. H8 confirmed conservation under extreme KV pressure (15 configurations). Full preemption-path validation is blocked by the panic bug (#293).

**Additional evidence (hardening wave):** Issue #498, fix #504 — `InjectArrival` silently accepted requests with `ArrivalTime > Horizon`, registering them in `Metrics.Requests` but never firing the arrival event. This broke conservation accounting (LHS included the request, RHS never completed it). Fix: log warning on beyond-horizon injection.

---

## INV-2: Request Lifecycle

**Statement:** Requests transition `queued -> running -> completed`. No invalid transitions. Requests not completed before horizon remain in current state.

**Verification:** State machine assertions in request processing code.

---

## INV-3: Clock Monotonicity

**Statement:** Simulation clock never decreases. Every event's timestamp >= the previous event's timestamp.

**Verification:** Clock is advanced in the event loop only via min-heap extraction, which guarantees non-decreasing order.

---

## INV-4: KV Cache Conservation

**Statement:** `allocated_blocks + free_blocks = total_blocks` at all times.

**Verification:** Checked after every allocation/deallocation. Transactional allocation with rollback on mid-loop failure (R5).

**Operational note (H8):** KV cache pressure exhibits a sharp cliff, not gradual degradation. In H8's workload, performance was identical above ~2200 blocks and collapsed below it (4.7x TTFT P99 increase with just 4.5% fewer blocks). Below ~1000 blocks, the preempt-requeue cycle can livelock (see R19). Capacity planning formula: `threshold ≈ rate / num_instances × (input_tokens + output_tokens) / block_size`.

**Additional evidence (hardening wave):** Two KV conservation bugs discovered in March 2026: (1) Issue #492, fix #502 — prefill capacity pre-check over-estimated by up to 1 block (partial last-block fill not accounted for), causing false allocation failures that triggered unnecessary preemptions. (2) Issue #501, fix #506 — TieredKVCache CPU→GPU reload could produce an inverted range (`newStart >= endIndex`), causing a slice-bounds panic in block allocation. Both bugs directly affected the allocation/deallocation balance that INV-4 protects. (See also #519 in INV-8 — the range-loop livelock primarily violated the work-conserving property, not block-level conservation.)

---

## INV-5: Causality

**Statement:** `arrival_time <= enqueue_time <= schedule_time <= completion_time` for every request.

**Verification:** Per-request metric timestamps recorded at each lifecycle stage. Invariant tests verify ordering for all completed requests.

---

## INV-6: Determinism

**Statement:** Same seed must produce byte-identical stdout across runs.

**Verification:** Run same configuration twice with same seed; diff stdout. Wall-clock timing goes to stderr (not stdout).

**Common violation sources:**
- Go map iteration feeding output ordering (R2)
- Floating-point accumulation order dependencies
- Wall-clock-dependent randomness (must use PartitionedRNG)
- Stateful scorers with non-deterministic internal state

---

## INV-7: Signal Freshness Hierarchy

**Statement:** Routing snapshot signals have tiered freshness due to DES event ordering and configurable staleness.

| Signal | Owner | Freshness (interval=0) | Freshness (interval>0) | Updated By |
|--------|-------|------------------------|------------------------|------------|
| InFlightRequests | Cluster | Synchronous | Synchronous | `RoutingDecisionEvent.Execute()` (increment), completion detection (decrement) |
| QueueDepth | Instance | Immediate | Periodic | `QueuedEvent.Execute()` |
| BatchSize | Instance | Immediate | Periodic | `StepEvent.Execute()` |
| KVUtilization | Instance | Immediate | Periodic | `FormBatch()` → `AllocateKVBlocks()` |
| CacheHitRate | Instance | Immediate | Periodic | `FormBatch()` |

**Design implication:** When `--snapshot-refresh-interval > 0`, all Prometheus-sourced signals (QueueDepth, BatchSize, KVUtilization) share the same scrape interval — matching real vLLM deployments where all three are exposed via the same `/metrics` endpoint. `InFlightRequests` remains synchronous (gateway-local counter, not Prometheus-sourced).

`EffectiveLoad()` = `QueueDepth + BatchSize + InFlightRequests`. The synchronous `InFlightRequests` term compensates for Periodic staleness in the other two terms.

**Verification:** H3 hypothesis experiment, H29 snapshot-staleness experiment (see [`hypothesis-archive` branch](https://github.com/inference-sim/inference-sim/tree/hypothesis-archive/hypotheses)).

**Evidence:** Issues #282, #283. At rate=5000, kv-utilization-only routing produces 200x worse distribution uniformity than queue-depth. Issue #463: unified Prometheus staleness model.

---

## INV-8: Work-Conserving Property

**Statement:** After every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. The simulator must not idle while there is work waiting.

**Verification:** `sim/simulator_test.go` — `TestWorkConserving_StepRestartsWhenWaitQNonEmpty`. Deterministic test with `MaxRunningReqs=1`, two requests arriving simultaneously. Without the property, the second request is stranded forever (no arrival to trigger a new StepEvent). With the property, both complete.

**Evidence:** H-MMK experiment (PR #325) — without the work-conserving fix, W_q error was 151,000% at ρ=0.3. After fix, error dropped to 47% (remaining gap is discrete step processing, not a bug).

**Additional evidence (hardening wave):** Issue #349, fix #519 — Go `range` over mutable `RunningBatch.Requests` during `FormBatch` Phase 1 visited evicted requests, triggering 102K+ cascading preemptions with zero completions. The simulator never made forward progress (zero completed requests = INV-8 violation). See R21.

**Code location:** Search for `// Work-conserving:` comment in `sim/simulator.go` — the `else` branch of `len(remaining) > 0` checks `WaitQ.Len() > 0` and schedules a new `StepEvent`.

**Hypothesis family:** Structural model (same as INV-4, INV-7).

---

## INV-9: Oracle Knowledge Boundary

**Statement:** Servability decisions — enqueue guard (`EnqueueRequest`), admission control (`AdmissionPolicy`), routing (`RoutingPolicy`), and priority scoring (`PriorityPolicy`) — must not read `Request.OutputTokens` or `len(Request.OutputTokens)`. The control plane uses `Request.MaxOutputLen` (client-declared output budget) for sequence-length checks against `MaxModelLen`. When `MaxOutputLen == 0` (no budget), only input length is checked; the proactive MaxModelLen cap in `FormBatch` (clamping to `maxModelLen-1-ProgressIndex`) and the completion boundary in `processCompletions` (`PI >= maxModelLen-1`) enforce output growth limits. Only the execution engine (`executeBatchStep`, `processCompletions`, `recordRequestCompletion`, `FormBatch` step planning) may access `OutputTokens` for token generation, completion detection, and per-step resource allocation.

**Rationale:** In real inference serving (vLLM), the engine does not know actual output length at admission time — only the client's declared `max_tokens` budget. BLIS's `Request.OutputTokens` is oracle knowledge (pre-determined for simulation). Using it for servability decisions would make the simulator's control plane behave differently from a real system, invalidating capacity planning results. See issue #567 ("Architectural Principle: Oracle Knowledge Boundary").

**Scope:** The boundary applies to *servability* decisions (admit/reject/route), not to all scheduler operations. `FormBatch` legitimately reads `OutputTokens` for decode-phase step planning (whether to allocate a decode token), which mirrors vLLM's scheduler reading sequence state for per-step execution. The distinction: "should this request enter the system?" (servability — no oracle) vs. "what should this request do in the current step?" (execution — oracle allowed).

**Verification:** `sim/simulator_test.go` — `TestEnqueueRequest_MaxOutputLen_OracleKnowledgeBoundary`: a request with `OutputTokens=1000` but `MaxOutputLen=0` and `MaxModelLen=512` is NOT rejected (input=200 < 512 passes input-only check), proving the enqueue guard does not peek at `OutputTokens`. Grep-based verification: `admission.go`, `routing.go`, `routing_scorers.go`, `routing_prefix_scorer.go`, `scheduler.go`, `priority.go` contain zero references to `OutputTokens`.

**Evidence:** Issue #567 — the original implementation's BC-4 fallback (`effectiveMaxOutput = len(r.OutputTokens)`) violated this boundary. Fixed in the same PR after convergence review caught it.

**Hypothesis family:** Structural model (same as INV-4, INV-7, INV-8).

---

## INV-10: Session Causality

**Statement:** For all rounds N in a closed-loop session: `round[N+1].ArrivalTime >= round[N].CompletionTime + ThinkTimeUs`. Boundary: ThinkTimeUs = 0 produces equality.

**Verification:** `sim/workload/session_test.go` — `TestSession_RoundGeneration_CorrectArrivalTime` verifies the arrival time formula. The ThinkTimeUs=0 boundary is inherent in the formula.

**Evidence:** Design doc `docs/plans/2026-03-13-client-behavior-model-design.md` — INV-10 definition. Guaranteed by construction in `SessionManager.OnComplete`.

**Hypothesis family:** Scheduler invariants (safety/liveness) — causality chain for session rounds.

---

## INV-11: Session Completeness

**Statement:** Every session reaches exactly one terminal state: completed (all rounds done), cancelled (a round timed out or was dropped), horizon-interrupted (simulation ended mid-session), or budget-exhausted (concurrency mode: global follow-up request cap reached). No session is silently abandoned.

**Verification:** `sim/workload/session_test.go` — tests cover all terminal paths: `TestSession_TimeoutCancels_NoMoreRounds` (cancelled), `TestSession_FinalRound_Completes` (completed), `TestSession_BeyondHorizon_NotGenerated` (horizon-interrupted), `TestSession_DroppedFollowUp_CancelsSession` (cancelled via drop). Budget-exhausted path verified via `TestConcurrencyMode_EndToEnd_SessionFollowUps` (budget exhaustion stops follow-up generation).

**Evidence:** Design doc INV-11 definition. The `SessionManager.OnComplete` method transitions sessions to exactly one terminal state before returning nil. The `budget_exhausted` state is reached when the shared follow-up budget (set via `SetFollowUpBudget` for `--concurrency` mode) is depleted — the session's unlimited-rounds flag would otherwise continue generating follow-ups, but the global cap takes precedence.

**Hypothesis family:** Structural model — session lifecycle completeness.

---

## PD Disaggregation Invariants

### INV-PD-1: KV Completeness

**Statement:** For every disaggregated request, `decode_enqueue_time >= kv_transfer_completion_time`. A decode sub-request must not be enqueued before its KV transfer completes.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_RequestCompletesFullPath` checks DecodeEnqueueTime >= TransferCompleteTime for every parent request. Runtime defensive check in `DecodeRoutingEvent.Execute()`.

**Evidence:** By event priority ordering: KVTransferCompletedEvent (priority 6) schedules DecodeRoutingEvent (priority 7) at the same timestamp. DecodeEnqueueTime is set in DecodeRoutingEvent which fires after transfer completion.

### INV-PD-2: Pool Exclusivity

**Statement:** Prefill sub-requests route only to prefill pool instances; decode sub-requests route only to decode pool instances.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_PrefillRoutedToPrefillPool` and `TestDisaggregation_DecodeRoutedToDecodePool` verify pool role for every parent request's prefill and decode instance assignments.

**Evidence:** `buildPoolFilteredSnapshots(role)` filters routing snapshots to only include instances of the specified pool role before passing to the routing policy.

### INV-PD-3: Transfer Conservation

**Statement:** `initiated_transfers == completed_transfers` at simulation end, provided all transfers complete within the simulation horizon. At bounded horizons, the difference (`initiated - completed`) represents in-flight transfers accounted for in the `pdInTransfer` conservation correction (see INV-1 PD correction in `cluster.go`).

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_TransferConservation` asserts equality and expected count (uses unbounded horizon).

**Evidence:** `transfersInitiated` incremented in `KVTransferStartedEvent.Execute()`, `transfersCompleted` incremented in `KVTransferCompletedEvent.Execute()`. Every started event schedules exactly one completed event.

### INV-PD-4: Phase Causality

**Statement:** For every disaggregated request: `arrival <= prefill_enqueue <= prefill_complete <= transfer_start <= transfer_complete <= decode_enqueue <= completion`.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_PhaseCausality` checks the full causal chain for every parent request.

**Evidence:** Each phase transition is enforced by DES event ordering: earlier phases schedule later-phase events at `time >= current_time`.

### INV-PD-5: Pool Stability

**Statement:** Pool membership is fixed at construction time and never changes during simulation.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_PoolStability` compares `PoolMembership()` before and after `Run()`.

**Evidence:** `BuildPoolMembershipFromIndices` is called once in `NewClusterSimulator` and stored in `cs.poolMembership`. No code path in `Run()` modifies this map.

### INV-PD-6: Metric Map Parent Granularity

**Statement:** After `Run()` completes on a disaggregated cluster, every per-request metric map (`RequestE2Es`, `RequestTTFTs`, `RequestITLs`, `RequestSchedulingDelays`, `RequestCompletionTimes`, `Requests`) contains only parent-level request IDs. No key may have a `_prefill` or `_decode` suffix. Completed parents contribute exactly one entry per map (keyed by parent ID); dropped or incomplete parents contribute no entry.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_MetricProjection_NoSubRequestKeys` checks all six maps for suffix-free keys; `TestDisaggregation_MetricProjection_DroppedParent_NoSubRequestKeys` verifies the invariant holds when decode KV allocation fails. `TestDisaggregation_MetricProjection_NoOp` verifies the projection is a no-op for non-disaggregated clusters.

**Evidence:** `projectPDMetrics()` in `sim/cluster/cluster.go` is called after `aggregateMetrics()` and the conservation correction. It unconditionally deletes the `pfx` and `dec` keys for every parent request, and conditionally inserts a parent-keyed entry only for completed parents (`CompletionTime > 0 && DecodeInstanceID != ""`).

### INV-PD-6b: CompletionTime Includes PostDecodeFixedOverhead

**Statement:** For all successfully decoded parent requests (`DecodeInstanceID != ""`), `parent.CompletionTime` equals the cluster clock at decode completion plus the decode instance's `PostDecodeFixedOverhead()`. For backends where overhead is 0 (blackbox, roofline, cross-model), `CompletionTime` equals the raw cluster clock tick. For `trained-roofline` (overhead ≈ 1850 µs), `CompletionTime` exceeds the raw clock by the overhead amount. This ensures that `projectPDMetrics()` computes `RequestE2Es[parentID] = CompletionTime - ArrivalTime` consistently with how `recordRequestCompletion` computes non-PD E2E (which also adds `PostDecodeFixedOverhead`). Note: the non-PD path applies the overhead conditionally when `len(req.OutputTokens) > 0`; the PD path applies it unconditionally, which is safe because decode sub-requests always inherit `OutputTokens` from the original request via `KVTransferCompletedEvent.Execute`.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_CompletionTime_IncludesNonZeroOverhead` verifies that `E2E_with_overhead − E2E_without_overhead == overhead` exactly for trained-roofline clusters (directly exercises the bug-fix site). `TestDisaggregation_CompletionTime_GeqAllPriorPhaseTimestamps` verifies `CompletionTime >= DecodeEnqueueTime` and `CompletionTime >= TransferCompleteTime` (phase causality preserved). `TestDisaggregation_E2E_IncludesOverhead_ZeroOverheadRegression` verifies `RequestE2Es[parentID] == CompletionTime − ArrivalTime` and `E2E >= TTFT` for blackbox (overhead=0) clusters.

**Evidence:** `detectDecodeCompletions()` in `sim/cluster/cluster.go` stamps `parent.CompletionTime = c.clock + inst.PostDecodeFixedOverhead()`. Fixed in issue #846.

### INV-P2-1: Pool-Config Consistency

**Statement:** Per-pool hardware overrides produce a valid `SimConfig` for each pool role: zero-valued `PoolOverrides` is a no-op (backward-compatible), non-nil fields override only the specified fields, and the global `SimConfig` is never mutated.

**Verification:** `sim/cluster/resolve_test.go` — `TestINV_P2_1_PoolConfigConsistency` verifies observable KV capacity differences between pools pre-simulation via `FreeKVBlocks()`; `TestINV_P2_1_RequestConservation` verifies INV-1 holds under heterogeneous pool configuration.

**Evidence:** `ResolvePoolConfig` performs a struct copy and applies only non-nil/non-zero overrides. `resolveConfigForRole` is called in the instance construction loop in `NewClusterSimulator`, before any simulation state is created.

---

### INV-P2-2: Fair-Share KV Transfer Bandwidth

**Statement:** When `--pd-transfer-contention` is enabled, the effective bandwidth available to each concurrent KV transfer is `total_bandwidth / active_transfers`, where `active_transfers` is the count of transfers in flight at the moment the new transfer starts (inclusive of the new transfer). With a single transfer in flight, the full bandwidth is used (`active_transfers == 1`, divisor == 1). This invariant gates the transfer duration formula in `KVTransferStartedEvent.Execute()`.

**Verification:** `sim/cluster/transfer_contention_test.go`:
- `TestTransferContention_INVP22_EffectiveBandwidthFormula` — golden test for the N=1 duration (9 µs with 10 blocks at 10 GB/s)
- `TestTransferContention_INVP22_N2FormulaExact` — golden test for the N=2 duration (17 µs with same payload at 5 GB/s effective)
- `TestTransferContention_INVP22_DivisorLaw` — invariant test: `duration(N) / duration(1) ≈ N` for N ∈ {1,2,3,4,5,8,10} with monotonicity
- `TestTransferContention_INVP22_FairShareBandwidth` — end-to-end: concurrent transfers record peak >= 1 when multiple requests arrive simultaneously

**Evidence:** PR9 (`sim/cluster/pd_events.go`, `KVTransferStartedEvent.Execute()`). Gated behind `PDTransferContention` flag (off by default for backward compatibility). The `activeTransfers` counter is incremented before the divisor is applied, ensuring the new transfer receives a fair share of the bandwidth with every other transfer currently in flight.
