# BLIS System Invariants

Invariants are properties that must hold at all times during and after simulation. They are verified by invariant tests (see R7) and checked during self-audit (Step 4.75).

## How to use this registry

**The resolution rule (#1721):** *if an invariant ID appears in code, it must be resolvable from this registry.* An ID a reader cannot look up is worse than no ID — `INV-A` was cited 11 times and printed `"INV-A violation on node %s"` at runtime while being documented nowhere. The registry therefore lists **every** `INV-*` ID cited in `sim/` or `cmd/`.

Nothing enforces that automatically yet, so re-check it when adding an ID:

```bash
git grep -ho 'INV-[A-Za-z0-9-]*' -- 'sim/**' 'cmd/**' | sort -u
```

Every result must appear in the index below. (`INV-6-safe` in `sim/workload/fixed_accumulate_test.go` is a known false positive — it is the adjective "INV-6-safe", not an ID.) A test that fails when a cited ID is undocumented is tracked in #1735.

Resolvable does not mean restated in full. Where a design plan is already authoritative for a feature-local invariant (the LoRA control plane), this registry carries a **pointer line** and the plan keeps the prose. Full entries live here for everything else.

**Grouping.** Entries are grouped so a reader can tell a whole-system guarantee from one module's internal rule at a glance:

| Group | What it means | If it is violated |
|---|---|---|
| **Run-level** | A property of the whole run's accounting, output, or forward progress. Applies to any run of a configuration it is defined for, rather than to one subsystem. | The simulator is wrong, wherever the bug lives. |
| **Subsystem** | Scoped to one subsystem or one optional feature. | That subsystem is wrong; elsewhere the invariant is not applicable. |
| **Code-boundary** *(cross-cutting)* | A rule about which code may read or do what — not a statement about simulation state. | The code is structurally wrong even if today's output looks right. |

The first two grade by **scope**; code-boundary grades by *kind*, so it sits alongside them rather than between them, and code-boundary rules also appear inside subsystem entries (`INV-L6`, and `INV-P2-1`'s no-global-mutation clause).

**Section order within a group is by ID, with one deliberate exception:** an invariant that *mirrors* another is sited beside the one it mirrors, because the contrast is the point (`INV-14` sits after `INV-2`, and `INV-15` after `INV-3`). The Index below is the authoritative by-ID lookup; do not read section order as a numbering claim.

Grouping is **orthogonal to the hypothesis-family mapping** below: the family says which experimental campaign validates an invariant, the group says how far its blast radius reaches.

**Hypothesis family mapping:** INV-1 through INV-3, INV-5, INV-6, INV-10 (session causality), and INV-13 (run/replay parity) belong to the **Scheduler invariants (safety/liveness)** family. INV-4 (KV cache conservation), INV-7 (signal freshness), INV-8 (work-conserving property), INV-9 (oracle knowledge boundary), INV-11 (session completeness), and INV-12 (Phase 1 completeness under priority preemption) belong to the **Structural model** family. The PD disaggregation invariants (INV-PD-*) and pool/transfer invariants (INV-P2-*) below are feature-scoped structural invariants for disaggregated serving. The six **enforcement-anchored** invariants promoted by #1772 (INV-14 … INV-19) belong to no family: each is anchored by a runtime assertion in production code rather than by an experimental campaign, which is what made them discoverable at all (see the note below).

**Enforcement-anchored entries (#1772).** INV-14 … INV-19 were promoted by reading every runtime assertion whose message says an internal state is impossible — a `panic`, or in INV-19's case a `logrus.Warnf` — and declaring the property it protects. Each was *already* written down in executable form before it had an ID; none is inferred from a structural pattern. Two consequences for a reader. First, each entry names an **Enforced** site as well as a **Verification** test, and the enforced site is the primary anchor: a reviewer editing that function is the audience. Second, the method has a floor but no ceiling — a property upheld without being checked is invisible to it, so the absence of an entry for a subsystem is not evidence that the subsystem has no invariants. Config-struct `Validate()` methods are deliberately excluded: they guard *user input*, not system state, and R3 already covers them.

## Index

Ordered by ID, since the common use is resolving an ID seen in code.

| ID | Group | Subsystem |
|---|---|---|
| [INV-1](#inv-1-request-conservation) Request conservation | Run-level | — |
| [INV-2](#inv-2-request-lifecycle) Request lifecycle | Run-level | — |
| [INV-3](#inv-3-clock-monotonicity) Clock monotonicity | Run-level | — |
| [INV-4](#inv-4-kv-cache-conservation) KV cache conservation | Subsystem | KV cache |
| [INV-5](#inv-5-causality) Causality | Run-level | — |
| [INV-6](#inv-6-determinism) Determinism | Run-level | — |
| [INV-7](#inv-7-signal-freshness-hierarchy) Signal freshness hierarchy | Subsystem | routing / observability |
| [INV-8](#inv-8-work-conserving-property) Work-conserving property | Run-level | — |
| [INV-9](#inv-9-oracle-knowledge-boundary) Oracle knowledge boundary | Code-boundary | control plane |
| [INV-10](#inv-10-session-causality) Session causality | Subsystem | sessions |
| [INV-11](#inv-11-session-completeness) Session completeness | Subsystem | sessions |
| [INV-12](#inv-12-phase-1-completeness-under-priority-preemption) Phase 1 completeness | Subsystem | batch formation |
| [INV-13](#inv-13-runreplay-parity) Run/replay parity | Run-level | — |
| [INV-14](#inv-14-instance-lifecycle-transitions) Instance lifecycle transitions | Run-level | — |
| [INV-15](#inv-15-arrival-ordering-into-the-cluster) Arrival ordering into the cluster | Run-level | — |
| [INV-16](#inv-16-gateway-queue-counter-consistency) Gateway queue counter consistency | Subsystem | gateway queue |
| [INV-17](#inv-17-shed-victim-is-the-flow-tail) Shed victim is the flow tail | Subsystem | gateway queue |
| [INV-18](#inv-18-prefix-cache-lru-structural-consistency) Prefix-cache LRU structural consistency | Subsystem | routing / observability |
| [INV-19](#inv-19-scale-decisions-carry-a-non-zero-delta) Scale decisions carry a non-zero delta | Subsystem | autoscaling |
| [INV-A](#inv-a-gpu-conservation) GPU conservation | Subsystem | cluster placement |
| [INV-A2](#inv-a2-placement-failure-visibility) Placement failure visibility | Code-boundary | autoscaler / actuator |
| [INV-BC-DP1](#inv-bc-dp1-dense-dp1-step-time-byte-identity) Dense DP=1 step-time byte-identity | Subsystem | latency model |
| [INV-L1 … INV-L7](#lora-control-plane) LoRA control plane — *stated in the LoRA design plan; pointer lines here* | Subsystem | LoRA |
| [INV-P2-1, INV-P2-2](#pools-and-kv-transfer) Pools and KV transfer | Subsystem | pools / KV transfer |
| [INV-PD-1 … INV-PD-6b](#pd-disaggregation) PD disaggregation | Subsystem | PD |
| [INV-W3](#inv-w3-cohort-expansion-purity) Cohort expansion purity | Subsystem | workload generation |
| [NS-6](#ns-6-catalog-is-authoritative-and-read-only) Catalog is authoritative and read-only | Code-boundary | CLI / model catalog |

---

## Run-level invariants

These hold for every configuration. A violation is a simulator bug regardless of which subsystem introduced it.

### INV-1: Request Conservation

**Statement:** At simulation end every injected request is accounted for in exactly one bucket. The canonical form is the **twelve-term cluster** equation:

`injected_requests == completed_requests + still_queued + still_running + dropped_unservable + timed_out + routing_rejections + gateway_queue_depth + gateway_queue_shed + gateway_queue_rejected + gateway_evicted + gateway_expired + encode_routing_rejections`

**Single-instance specialisation:** a single-instance simulation has no cluster router, no gateway queue and no encode pool, so the last seven terms are identically zero and the equation reduces to five:

`injected_requests == completed_requests + still_queued + still_running + dropped_unservable + timed_out`

**Which form to assert:** the five-term form is valid **only** for a single-instance simulation. A cluster-level assertion must use all twelve terms. Before #1721 this statement claimed the five-term form held "at simulation end (all levels)" and presented the twelve-term equation as a secondary "extension" — which is self-contradicting, and is why #1720 found 18 cluster-level tests asserting the narrower equation. Any shared conservation helper should be generated from the twelve-term list above.

**Cluster-only term definitions (issues #882, #1190, #1228, #1193, #1264):** the seven cluster-only buckets cover routing rejections, the gateway queue, in-flight gateway eviction, TTL expiry, and encode-pool routing. `gateway_queue_depth` counts requests still in the gateway queue at horizon. `gateway_queue_shed` counts requests shed (evicted victims) from the gateway queue due to capacity limits. `gateway_queue_rejected` counts requests rejected from the gateway queue when no displaceable victim exists (either no sheddable entries, or all sheddable entries have equal/higher priority than the incoming request). `gateway_evicted` counts requests evicted in-flight from instances when the system is saturated and higher-priority requests are waiting (Phase 4, #1228). `gateway_expired` counts requests removed from the gateway queue when their TTL expires (Phase 6, #1193); default off (TTL=0). `encode_routing_rejections` (GAP-4, #1264) counts requests rejected at the encode routing stage when the encode pool has zero routable instances; the term is always zero when `--encode-instances 0` (default). Each terminal bucket is mutually exclusive — a request lands in exactly one. Single-instance simulations have no cluster router, gateway queue or encode pool; all **seven** cluster-only terms are always zero there.

**Full pipeline:** `num_requests == injected_requests + rejected_requests` (from anomaly counters).

**Verification:** every cluster-level conservation assertion in the tree goes through one shared helper (#1720). `sim/cluster/gateway_queue_ttl_test.go` — `TestINV1_Conservation_GatewayTTLExpiry` is the only conservation test that drives a real TTL expiry, so it is the one that exercises the `gateway_expired` bucket; the bucket is zero in every other fixture, which is why three assertions could omit it and still pass. Conservation fields (`still_queued`, `still_running`, `injected_requests`) are included in CLI JSON output; the seven cluster-only counters are **not** (see below).

**How to assert it (#1720).** Do not hand-roll a sum. There is one helper per observation layer, and each takes `total` and `rejected` separately rather than a pre-subtracted `injected`, so a fixture that must not reject anything can pass a `noRejections` constant and have a spurious rejection break the equation. Note what that does *not* do: folding the pipeline clause into the same equality means a failure cannot be attributed to one clause or the other, and `total == injected + rejected` is an identity once `injected` is derived that way. Two call sites check the pipeline clause separately, against the length of the `Metrics.Requests` map — the only observation of injected that is independent of the counters. The map is not a general-purpose baseline. The four drop guards, the drain-redirect path and the PD parent collapse delete from it, so a fixture exercising any of those undercounts. Timeout paths are the opposite trap: they do **not** delete, so a timed-out request stays in the map and a fixture with timeouts overcounts. Both call sites guard the precondition they rely on.

- `assertClusterINV1Conservation` (`sim/cluster/inv1_conservation_test.go`) — the twelve-term equation; use it for anything cluster-level. `total` must come from a source independent of the metrics under test, normally `len(requests)`. The helper's doc comment states the three preconditions under which that identity holds: an eager slice request source, all arrivals inside the horizon, and no session follow-ups.
- `assertInstanceINV1Conservation` (same file) and `assertINV1Conservation` (`sim/inv1_conservation_test.go`) — the five-term specialisation, for per-instance metrics only.
- `clusterConservationHolds` (`cmd/dp_placement_test.go`) — checks what stdout can prove: that `injected_requests` is positive and equals the count the fixture generated. It does *not* compare `injected_requests` against the five-term sum of the same object, because `MetricsOutput` defines it as that sum, so the equality is a tautology. It cannot check the twelve-term form at all: the seven cluster-only counters live on `cluster.RawMetrics` and are not serialised into the JSON `MetricsOutput`, so no CLI-level test can assert the twelve-term form (#1746).

Two guard tests keep that arrangement honest. `TestINV1_HelperMatchesRegistry` diffs the helpers' term lists against the two equations stated above, so editing this section without editing the tests fails rather than drifting. `TestINV1_NoInlineClusterConservationSums` and its `sim/` counterpart walk the AST of every test file in their package and reject a newly hand-rolled sum. Both call `internal/invariantscan`, which resolves locals and `+=` accumulation, walks subtraction and comparison as well as addition, and distinguishes a conservation ledger from the other multi-bucket expressions in the tree (INV-5's non-vacuity floors, the shed-accounting identity) by requiring an instance-side bucket. Laundering through locals was the shape of one of the three sites that omitted a bucket, and it is the reason the guard is AST-based rather than a grep. `cmd/` has no such guard: its one conservation check parses stdout field names, which no AST walk over Go expressions would see.

**Known-unasserted clause.** The exclusivity promise above — *each terminal bucket is mutually exclusive, a request lands in exactly one* — is checked by nothing, and no aggregate sum can check it: a request double-counted in one bucket cancels a request lost from another. Asserting it requires per-request terminal-bucket tagging, which is a production change; tracked in #1745. Recorded here so the equality is not mistaken for a completeness proof.

**Evidence:** Issue #183 — a silently-dropped request violated conservation for months.

**Experimental validation:** H12 confirmed conservation across 10 policy configurations (67 invariant checks) — including round-robin, least-loaded, weighted (multiple scorer configs), SJF, priority-FCFS, token-bucket admission, and always-busiest. H8 confirmed conservation under extreme KV pressure (15 configurations). Full preemption-path validation is blocked by the panic bug (#293).

**Additional evidence (hardening wave):** Issue #498, fix #504 — `InjectArrival` silently accepted requests with `ArrivalTime > Horizon`, registering them in `Metrics.Requests` but never firing the arrival event. This broke conservation accounting (LHS included the request, RHS never completed it). Fix: log warning on beyond-horizon injection.

### INV-2: Request Lifecycle

**Statement:** Requests transition `queued -> running -> completed`. No invalid transitions. Requests not completed before horizon remain in current state.

**Verification:** State machine assertions in request processing code.

**Known weakness (#1721):** this is the thinnest entry in the registry — it names no test, and the three states above are fewer than the request states the code actually distinguishes. Correcting the statement was raised separately and is deliberately out of scope here; it is recorded so a reader does not mistake brevity for a tight specification.

**Contrast with INV-14 (#1772):** the *instance* lifecycle is the same class of invariant with the opposite protection profile — it has a legal-transition table, a single mutation choke point that panics, and an edge-table test, while INV-2 has a registry entry and no enforcement (request state is written directly at many sites). The two entries are sited adjacently for exactly that reason.

### INV-14: Instance Lifecycle Transitions

**Statement:** an instance's `State` changes only along an edge of `validInstanceTransitions`. Any other transition is a programming error and panics. Two clauses:

1. **Legal edges.** `Scheduling → {Loading, Terminated}`, `Loading → {WarmingUp, Active, Terminated}`, `WarmingUp → {Active, Draining, Terminated}`, `Active → {Draining, Terminated}`, `Draining → {Terminated}`, `Terminated → {}`. The edge set is **monotone**: every legal edge advances the instance along `Scheduling < Loading < WarmingUp < Active < Draining < Terminated` and none regresses, so a terminated instance never becomes active again and a draining one never reloads.
2. **The seeding carve-out.** When `State == ""` — lifecycle tracking is not enabled for the run — the *first* `TransitionTo` call is accepted **unvalidated** and seeds `State`; every subsequent call on that instance is validated normally. This is a documented backward-compatibility path, not an oversight, and it is stated here rather than left for a reader to discover: it means clause 1 constrains an instance's second and later transitions, and a test that asserts an illegal edge panics must set `State` first or it will observe the seeding path instead.

**Why run-level:** an instance that regresses to `Active` after `Terminated` corrupts routing, placement accounting and the run's forward progress, not one subsystem's internals — the blast radius is the whole run's output.

**Enforced:** `sim/cluster/instance.go` — the table at `:416`, and `TransitionTo` at `:434` with two panics: an unrecognised source state (`:436`) and an illegal edge (`:439`). `TransitionTo` is the **only** writer of `State` outside construction, which is what makes the table load-bearing rather than advisory.

**Verification:** `sim/cluster/instance_lifecycle_test.go` — `TestInstanceStateMachine_ValidTransitions` (legal/illegal edge table asserting a panic on exactly the illegal edges) and `TestInstanceStateMachine_NoBackwardTransitions` (the monotonicity law, checked over the whole table rather than over sampled edges, so a newly added backward edge fails without the table's test cases being extended). `TestINV14_SeedingCarveOutAcceptsAnyFirstTransition` covers clause 2, which the edge table cannot reach — it assigns `State` before every call, so it never exercises the empty-state path.

### INV-3: Clock Monotonicity

**Statement:** Simulation clock never decreases. Every **processed** event's timestamp >= the previous processed event's timestamp — except when restoring an optimistic advance after a lazily-cancelled event, where no event was processed.

**Verification:** Clock is advanced in the event loop only via min-heap extraction, which guarantees non-decreasing order. `sim/simulator_test.go` — `TestINV3_ClockNeverDecreases` drives the loop one event at a time and asserts the processed-timestamp sequence is non-decreasing (with a non-vacuity gate, and at least one orphaned timeout present so the single-instance lazy-cancellation skip is exercised). The *cluster* `prevClusterClock` restore described below is **not** covered by any test.

**The carve-out, and why it is not a violation:** `sim/cluster/cluster.go` advances the cluster clock to an instance's next event time *before* the event is known to be real, then restores the previous value when `ProcessNextEvent()` turns out to have skipped a lazily-cancelled `TimeoutEvent` (`prevClusterClock`). A no-op orphaned timeout must not advance the cluster clock, so the advance is undone. Nothing was processed across that pair of assignments, which is why the statement is scoped to *processed* events. One further case: the horizon guard between the advance and the restore reads the optimistically advanced clock and can `break` out of the loop, leaving the advance in place unrestored — benign, because the cluster's reported `SimEndedTime` comes from per-instance `Finalize` as `min(Clock, Horizon)`, progress snapshots clamp, and `ClusterSimulator.Clock()` has no production caller. This also means a test asserting on the raw `c.clock` field would fail here and look like a real bug — assert on processed event timestamps instead.

**Not to be confused with INV-15 (#1772):** INV-3 constrains the clock the simulator *advances*; INV-15 constrains the arrival stream the simulator is *given*. They are adjacent because the arrival-hook guard used to cite INV-3, and the two have different repair procedures — see below.

### INV-15: Arrival Ordering into the Cluster

**Statement:** the fresh-arrival stream delivered to `ClusterSimulator`'s arrival hook is non-decreasing in effective arrival time. For consecutive hook firings, `timeUs(n+1) >= timeUs(n)`. Clearing the hook (`SetArrivalHook(nil)`) resets the floor to zero, so a subsequently installed hook starts from a clean baseline rather than inheriting the previous stream's high-water mark.

**Why it is its own entry, and not INV-3 or INV-6.** Before #1772 the guard attributed itself to "INV-3/INV-6 violation". It is neither. INV-3 is a property the event loop *establishes* — the min-heap makes processed timestamps non-decreasing whether or not anything asserts it. INV-6 is a property of stdout. This is a **precondition on an input**: whatever feeds the hook (the workload generator, closed-loop session follow-up scheduling, the `blis run` trace writer) must supply ordered arrivals, and nothing inside the cluster can make that true. The distinction is operational, not taxonomic — it is the difference between the two repairs a violation admits:

| If it were INV-3 | It is INV-15 |
|---|---|
| Fix the clock or the event ordering | Fix the arrival source |

**What it buys.** The hook exists so `blis run --trace-output` can capture TraceV2 records at the arrival boundary (#1440) instead of sorting a post-run request list. Ordered-by-construction is the whole reason that replacement is sound: records emerge already in arrival order, so the exported trace needs no downstream sort, and INV-13 (replay reads the same trace the run produced) rests on it. A violation would silently produce an out-of-order trace, which is why it is a panic rather than a warning.

**Scope:** fresh arrivals only. `REDIRECT` re-injections (`req.Redirected`) do not fire the hook at all and are outside the statement — the drain policy reroutes them internally and a trace record for them would be spurious or duplicated. Beyond-horizon follow-ups likewise never reach the hook, because their `ClusterArrivalEvent` never executes.

**Enforced:** `sim/cluster/cluster.go:740` — `fireArrivalHook` panics when `timeUs < lastArrivalHookTime`, naming the request and both timestamps.

**Verification:** `sim/cluster/arrival_hook_test.go` — `TestArrivalHook_PanicsOnNonMonotonicArrival` (the guard fires), `TestSetArrivalHook_NilClearResetsMonotonicityFloor` (the floor-reset clause), and `TestArrivalHook_FiresOncePerInitialRequest` / `TestArrivalHook_FiresForSessionFollowUps` (the ordinary streams satisfy it, so the guard is not vacuously true).

### INV-5: Causality

**Statement:** `arrival_time <= enqueue_time <= schedule_time <= completion_time` for every request. Four timestamps, so three links — each of which must be asserted on its own.

**Verification:** `sim/cluster/causality_test.go` — `TestINV5_Causality_FullChain` asserts each link separately against a fixture with the gateway queue active, splitting the enqueue link at the *measured* gateway dispatch instant (`Request.GatewayDispatchTime`) rather than inferring it from the instance-side delay. `TestINV5_Causality_SingleInstancePath` covers the no-gateway path, where the chain reduces to `arrival <= schedule <= completion`.

Two things to preserve if you touch it. Both metric-map reads must be two-value: the scheduling-delay and completion-time keys are deleted and only conditionally re-added for PD parent requests, so a bare index yields zero and silently inverts the comparison. And each link carries its own non-vacuity floor, because a request still in the gateway queue at the horizon legitimately has no dispatch timestamp — one global counter would either fail for the wrong reason or hide a fixture that stopped exercising the gateway.

**Which timestamps.** All in ticks (µs) on the shared cluster clock. `arrival` is `Request.ArrivalTime`, never re-stamped as a request moves between instances. `enqueue` is `Request.GatewayEnqueueTime`, assigned from the router-state clock in `FlowControlAdmission.Admit` and **reset to 0** for rejected and shed-victim requests — so 0 means "no gateway timestamp", and a fixture whose arrivals start at tick 0 cannot distinguish the two cases. `schedule` is `ArrivalTime + Metrics.RequestSchedulingDelays[id]`, since the stored delay is arrival-relative. `completion` is `Metrics.RequestCompletionTimes[id]`, absolute and `float64`; compare in `float64` rather than truncating.

**The weaker property, and why it is not this invariant (#1720, Finding 2).** Many tests assert `TTFT >= 0` and `E2E >= TTFT`. That is a valid property and worth keeping, but both quantities are measured *from arrival*, so together they only say the last token follows the first: the enqueue and schedule links are untouched by it. Before #1720 every test citing INV-5 asserted only that pair.

### INV-6: Determinism

**Statement:** Same seed must produce byte-identical stdout across runs.

**Verification:** Run same configuration twice with same seed; diff stdout. Wall-clock timing goes to stderr (not stdout).

**Common violation sources:**
- Go map iteration feeding output ordering (R2)
- Floating-point accumulation order dependencies
- Wall-clock-dependent randomness (must use PartitionedRNG)
- Stateful scorers with non-deterministic internal state

**Transient sub-invariant — eager ≡ lazy (holds until A8, #1442):** While both the eager generator (`GenerateWorkload`/`GenerateRequests`) and the lazy streaming source (`GenerateWorkloadLazy`, behind `--lazy-generation`, #1441) exist, they MUST produce byte-identical request streams and session blueprints for any seed and any lazy-supported spec — and consequently byte-identical stdout. Determinism is load-bearing for lazy mode: the two paths consume RNG draws in the same order, so a re-ordering breaks both this equality and INV-6 itself. Enforced at the generator layer by the property test `sim/workload/parity_property_test.go` (`TestProperty_EagerEqualsLazy_RequestStreams`), which samples ≥100 random `(seed, spec)` draws (env-gated extended mode via `BLIS_PROPERTY_DRAWS`) and asserts stream + blueprint equality; and at the CLI layer by `cmd/parity_run_replay_test.go` (`TestParity_RunStdout_DeterministicAndEagerLazyIdentical` for stdout, `TestParity_RunReplay_TraceByteIdentity_Matrix` for the exported trace). This invariant is transient by design: it is retired when eager generation is removed (A8, #1451), after which lazy is the sole path.

### INV-8: Work-Conserving Property

**Statement:** After every step completion, if `WaitQ.Len() > 0`, a `StepEvent` must exist in the event queue. The simulator must not idle while there is work waiting.

**Verification:** `sim/simulator_test.go` — `TestWorkConserving_StepRestartsWhenWaitQNonEmpty`. Deterministic test with `MaxNumSeqs=1`, two requests arriving simultaneously. Without the property, the second request is stranded forever (no arrival to trigger a new StepEvent). With the property, both complete.

**Evidence:** H-MMK experiment (PR #325) — without the work-conserving fix, W_q error was 151,000% at ρ=0.3. After fix, error dropped to 47% (remaining gap is discrete step processing, not a bug).

**Additional evidence (hardening wave):** Issue #349, fix #519 — Go `range` over mutable `RunningBatch.Requests` during `FormBatch` Phase 1 visited evicted requests, triggering 102K+ cascading preemptions with zero completions. The simulator never made forward progress (zero completed requests = INV-8 violation). See R21.

**Code location:** Search for `// Work-conserving:` comment in `sim/simulator.go` — the `else` branch of `len(remaining) > 0` checks `WaitQ.Len() > 0` and schedules a new `StepEvent`.

**Hypothesis family:** Structural model (same as INV-4, INV-7).

### INV-13: Run/Replay Parity

**Statement:** For any configuration supported by both `blis run` and `blis replay`, a trace exported via `blis run --trace-output` and replayed via `blis replay --session-mode fixed` with identical flags MUST produce identical per-request TTFT, E2E, and aggregate metrics. At the CLI boundary the enforceable form of that is stronger and is what new parity tests should assert: **byte-identical stdout**, which subsumes the per-request distributions, the cache/latency aggregates, and the conservation counters in one assertion (`cmd/dp_replay_parity_test.go`'s `TestINV13_RunReplayParity_MoEDPPlacement` is the model; pair it with a non-vacuity gate and a conservation companion, since byte-identity alone is satisfied by two identically-wrong runs). "Identical flags" includes `--horizon`: the two commands have different *defaults* for it (`blis run` defaults to unlimited; `blis replay` auto-computes 2x the max arrival time), so a parity comparison must pass `--horizon` explicitly on both legs or the replay leg may truncate a still-draining run. This is a difference in defaults, not in behavior — the invariant is over the resolved configuration. Features not yet supported by replay (autoscaler, node pools) MUST cause a `logrus.Fatalf` at startup when those flags are explicitly set or when a policy bundle containing those features is passed — never silent degradation or a mere warning. MoE data parallelism as placement (`--dp > 1` on an MoE model) was run-only from #1531 and is **supported on both paths since #1556**: `blis run` and `blis replay` resolve the placement through the one shared `cmd.resolveDPPlacement`, so identical flags over the same trace produce identical metrics. Expert parallelism alongside it (`--dp>1` + `--enable-expert-parallel`) is likewise supported on both paths since #1548: the logical EP-group width is re-supplied from the CLI on both legs (a model-level input, not a trace field — the same treatment `--kv-cache-dtype` gets), so identical flags over the same trace produce identical metrics. PD disaggregation + `--dp>1` is **supported on both paths since #1553** (each pool spawns N per-rank replicas; `blis replay` reproduces it byte-identically). The guard that remains still `logrus.Fatalf`s rather than degrades: the model autoscaler + `--dp>1` is **rejected** (#1553 decision — dp-group co-scaling is undefined). On replay the autoscaler and node pools are additionally rejected unconditionally (before DP is considered) as unreplayable features.

**Verification:** `cmd/replay_test.go` — `TestINV13_RunReplayParity_PD` verifies that a PD-disaggregated cluster produces matching `RequestTTFTs` and `RequestE2Es` maps when the same requests are run directly vs. through the trace-export-then-replay path. `TestReplayCmd_AutoscalerBundleFatal` and `TestReplayCmd_NodePoolsBundleFatal` verify fatal exit for unsupported features. For MoE `--dp>1` DP-as-placement (#1531, #1556), `cmd/dp_replay_parity_test.go` — `TestINV13_RunReplayParity_MoEDPPlacement` asserts that a trace exported by `blis run --dp N` replays with the same flags to byte-identical stdout (per-replica and cluster-aggregate metrics), `TestReplayCmd_MoEDPPlacement_SpawnsReplicas` asserts the `--num-instances × N` expansion plus INV-1 conservation on the replay path, and `TestReplayCmd_MoEDPPlacement_PD_Parity` asserts PD + `--dp N` now replays byte-identically since #1553 (`TestINV13_RunReplayParity_MoEDPPlacement`'s `tp2-ep` parity case covers the #1548 EP-on topology positively); `cmd/dp_placement_test.go` — `TestResolveDPPlacement_MutatesDeploymentVars` and `TestResolveDPPlacement_DenseModelIsInert` pin the shared resolver both commands call (what it writes to the deployment vars, and that a guard error or an inactive plan writes nothing).

**Under lazy generation (#1441, #1442, #1458, #1459, #1460):** run/replay parity must hold whether the run used the eager generator or the lazy streaming source (`--lazy-generation`). Before lazy mode, parity was structural by construction (both code paths consumed the same eagerly-built request list); with lazy mode in play it is enforced only by tests. The authoritative enforcement point is `cmd/parity_run_replay_test.go`: `TestParity_RunReplay_TraceByteIdentity_Matrix` asserts eager and lazy runs export byte-identical traces across a coverage matrix (single-turn chatbot, multi-turn accumulate cohort, single-session reasoning, multi-session reasoning, concurrency clients, and time-varying / per-window workloads), and `TestParity_RunReplay_INV13_BothModes` asserts an eager-sourced and a lazy-sourced trace replay to identical per-request TTFT/E2E. As of #1460 the lazy source streams every workload class — there is no eager-fallback sentinel remaining.

**Evidence:** `cmd/replay.go::replayCmd.Run` wires the same `cluster.DeploymentConfig` field set as `runCmd` (`cmd/root.go`). This parity surface is broader than PD disaggregation alone: it spans PD fields (`PrefillInstances`, `DecodeInstances`, `SharedInstances`, `PDDecider`, `PDPrefixThreshold`, `PDTransferBandwidthGBps`, `PDTransferBaseLatencyMs`, `PDTransferContention`, `PrefillScorerConfigs`, `DecodeScorerConfigs`, `PrefillOverrides`, `DecodeOverrides`), encode-stage fields (`EncodeInstances`, `EncodeDecider`), the flow-control fields, tier-shedding (`TierShedThreshold`, `TierShedMinPriority`), GAIE thresholds (`GAIEQDThreshold`, `GAIEKVThreshold`), `TenantBudgets`, and `InstanceLifecycle`. Both call sites carry an `INV-13 SYNC POINT` comment marking the literal that must stay in sync; treat that comment — not a fixed field count — as the authoritative enumeration, since it grows as features are added. Autoscaler and node-pool checks use `logrus.Fatalf` to prevent silent divergence.

**Hypothesis family:** Scheduler invariants (safety/liveness) — same as INV-1, INV-5, INV-6.

---

## Code-boundary invariants

These constrain **which code may read or do what**. They are not statements about simulation state, so a violation can leave today's output correct while making the next change unsafe. This is a *kind*, not a scope: each entry below names the scope it applies to, and code-boundary rules also appear inside subsystem entries (`INV-L6`, and `INV-P2-1`'s no-global-mutation clause).

### INV-9: Oracle Knowledge Boundary

**Statement:** Servability decisions — enqueue guard (`EnqueueRequest`), admission control (`AdmissionPolicy`), routing (`RoutingPolicy`) — must not read `Request.OutputTokens` or `len(Request.OutputTokens)`. The control plane uses `Request.MaxOutputLen` (client-declared output budget) for sequence-length checks against `MaxModelLen`. When `MaxOutputLen == 0` (no budget), only input length is checked; the proactive MaxModelLen cap in `FormBatch` (clamping to `maxModelLen-1-ProgressIndex`) and the completion boundary in `processCompletions` (`PI >= maxModelLen-1`) enforce output growth limits. Only the execution engine (`executeBatchStep`, `processCompletions`, `recordRequestCompletion`, `FormBatch` step planning) may access `OutputTokens` for token generation, completion detection, and per-step resource allocation.

**Rationale:** In real inference serving (vLLM), the engine does not know actual output length at admission time — only the client's declared `max_tokens` budget. BLIS's `Request.OutputTokens` is oracle knowledge (pre-determined for simulation). Using it for servability decisions would make the simulator's control plane behave differently from a real system, invalidating capacity planning results. See issue #567 ("Architectural Principle: Oracle Knowledge Boundary").

**Scope:** The boundary applies to *servability* decisions (admit/reject/route), not to all scheduler operations. `FormBatch` legitimately reads `OutputTokens` for decode-phase step planning (whether to allocate a decode token), which mirrors vLLM's scheduler reading sequence state for per-step execution. The distinction: "should this request enter the system?" (servability — no oracle) vs. "what should this request do in the current step?" (execution — oracle allowed).

**Bounded carve-out — the Phase-2 decode-sub-request grant cap (#1657).** `FormBatch`'s Phase-2 admission loop caps a PD decode sub-request's speculative-decode grant `g` by its distance to the completion boundary (`Request.completionProgressIndex()`, oracle-derived from `len(OutputTokens)`). This sits in the *admission* loop, not the running-request loop, so it is explicitly carved out rather than covered by the sentence above. It is safe because the cap is **monotone-shrinking and floored at 1**: it can only lower the granted token count, never below 1, so it can never reject a request, never reorder admissions, and never change *whether* a request is admitted — only how many tokens that one step grants it. The one visible effect is that a capped grant reserves *less* KV and debits *less* of the batched-token budget, so admission under KV pressure becomes marginally **more** likely, never less. Any future oracle read in this loop that could gate admission (a `break`/`continue`/reject on an `OutputTokens`-derived value) is a violation.

**Verification:** `sim/simulator_test.go` — `TestEnqueueRequest_MaxOutputLen_OracleKnowledgeBoundary`: a request with `OutputTokens=1000` but `MaxOutputLen=0` and `MaxModelLen=512` is NOT rejected (input=200 < 512 passes input-only check), proving the enqueue guard does not peek at `OutputTokens`. Source-scan verification: `TestINV9_OracleKnowledgeBoundary_NoOutputTokensInControlPlane` checks that the sim/ control-plane files contain zero references to `OutputTokens` — **and none to `completionProgressIndex()`** (#1657), the package-private accessor that derives a request's terminal `ProgressIndex` from `len(OutputTokens)`: it is oracle-derived, so calling it from a servability path would smuggle the oracle past a plain `OutputTokens` grep. Its legitimate callers are execution-side only (`processCompletions`, `FormBatch` decode-step sizing). Until #1720 the test checked only the first pattern, so the accessor clause this entry claims was unasserted.

**The file set is a glob, not a list (#1720, Finding 3).** The scan resolves `routing*.go`, `admission*.go`, `scheduler*.go`, `slo*.go`, `saturation*.go` and `router_state.go` under `sim/`, minus an explicit `simControlPlaneExemptions` map (empty today). A new routing scorer is therefore covered by default, and exempting one is a named, reviewable act; a stale exemption naming a deleted file fails the test. The previous hardcoded list omitted `router_state.go`, `routing_nohit_lru_scorer.go` and `routing_precise_prefix_scorer.go` — all three clean, but two of them scorers, so the next scorer would have escaped too. The four `sim/cluster/` files stay an explicit list: they legitimately carry the `TotalOutputTokens` metric aggregate and need line-level scanning, and no glob separates them from the rest of the package — so a new *cluster* control-plane file must still be registered by hand, which is a known gap.

**Evidence:** Issue #567 — the original implementation's BC-4 fallback (`effectiveMaxOutput = len(r.OutputTokens)`) violated this boundary. Fixed in the same PR after convergence review caught it.

**Hypothesis family:** Structural model (same as INV-4, INV-7, INV-8).

### INV-A2: Placement Failure Visibility

**Statement:** A failed `PlaceInstance` must be logged, never silently dropped — on both the autoscaler decision path and the actuator path.

**Why this tier:** this is an error-handling and observability rule (a specialization of R1, "never a silent `continue`"), not a property of simulation state. A silently dropped scale-up leaves the cluster under-provisioned with no signal, and the autoscaler's stabilization windows have already been consumed, so the next opportunity is delayed too.

**Verification:** enforced by construction at each site rather than by a dedicated test — `sim/cluster/autoscaler.go:174` (the `Actuator` interface contract, where the requirement is stated), `sim/cluster/autoscaler.go:376` (`logrus.Errorf` when decisions are dropped because no actuator is wired, naming the decision count and the consumed stabilization windows), `sim/cluster/direct_actuator.go:33` (documented contract: individual failures are always logged), `:56` (a missing `PlacementManager` returns an error rather than skipping), and `:71` (`logrus.Errorf` per failed `PlaceInstance`, preserving model id and variant).

**Cited in code:** 5 times, all in the two files above.

**Registry note (#1721):** cited by ID since the autoscaler landed, documented nowhere until #1721.

### NS-6: Catalog is Authoritative and Read-Only

**Statement:** A model runs **if and only if it is in the catalog**, and no run may change the catalog. Two clauses:

1. **Read-only, refuse-on-miss.** `cmd.resolveModelConfig` reads the model's `config.json` from the catalog located by `--catalog` / `BLIS_CATALOG` (`<catalog>/<short-name>/config.json`; #1731 — there is no default, no search path, and a run naming no catalog is refused naming both forms). A model with no entry — or an entry that is not a HuggingFace config — is **refused, naming the path the entry belongs at**. No `blis run` / `blis replay` / `blis observe` invocation creates or modifies a catalog file, and none makes a HuggingFace request.
2. **The deployment is chosen, never inferred.** `--hardware` and `--tp` are required on both `blis run` and `blis replay` (`cmd.requireDeploymentFlags`); omitting either is refused **naming the missing flag**.

**Why this tier:** like INV-9 and INV-A2 this constrains *what the code may do* rather than describing simulation state. A run that writes a catalog entry can leave that run's output perfectly correct while making the catalog — the thing every later run and every calibration reads — drift by accident.

**Rationale:** before #1733, an absent `config.json` triggered a HuggingFace download that was **written into `model_configs/`**, so running an unknown model *added a catalog entry as a side effect* — the exact mechanism by which a catalog silently accumulates unreviewed, unprovenanced models. Independently, `--hardware`/`--tp` were looked up per-model in `defaults.yaml` behind a `logrus.Warnf`, so a run could complete and emit metrics for a deployment nobody chose. Both are now refusals (R1: no warn-and-continue).

**Scope and non-scope:** NS-6 binds the **CLI boundary only**. Library callers that construct a `sim.ModelConfig` or call `sim.NewModelHardwareConfig` directly are unaffected and need no flags — the many test files that build configs through the Go API are deliberately outside it. `blis observe` satisfies clause 1 vacuously (it resolves no model config) and takes neither deployment flag (it places no instances).

**Byte-identity (INV-6):** NS-6 changes which inputs are *required*, not any number. Supplying `--hardware`/`--tp` equal to what `defaults.yaml` would have supplied reproduces the pre-#1733 stdout exactly. #1768 has since **deleted** that `defaults:` block (and `DefaultConfig`/`GetHFRepo`) — the policy had no consumer left, so nothing is lost, and the byte-identity evidence is unaffected: its golden was captured from a run that supplied *neither* flag. One consequence worth knowing: a hand-maintained `defaults.yaml` still carrying a `defaults:` block is now refused at load (strict parsing, R10) rather than parsed and ignored.

**Verification:** `cmd/hfconfig_test.go` — `TestResolveModelConfig_AbsentFromCatalog_RefusedNamingPath` (refusal names the path), `..._CreatesNothing` (nothing written, catalog directory not created), `TestResolveModelConfig_MalformedCatalogEntry_RefusedAndPreserved` (refused, entry byte-preserved), `TestResolveModelConfig_ResolutionInvariant` (one-step catalog resolution with no fallback, #1731). `cmd/ns6_catalog_test.go` — `TestNS6_NoRuntimeFetch_StaticGuard` (removed fetch symbols and any `huggingface.co` string literal are absent from `cmd/`'s production sources; the resolver imports no network package and makes no file-creating call), `TestNS6_MissingDeploymentFlagIsRefusedByName` (subprocess: each omission exits 1 naming only the omitted flag), `TestNS6_DeploymentFlagsRequiredOnRunAndReplay` (INV-13 parity), `TestNS6_ObserveTakesNoDeploymentFlags`. The INV-6 claim rests directly on `TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline`'s pre-#1733 golden; the anchor test that cross-checked it against `defaults.yaml`'s own GPU/TP keys was removed with those keys by #1768, whose own contracts (the trimmed file parses, a stale `defaults:` block is refused, the deleted surface cannot return) live in `cmd/defaults_block_removal_test.go`. `cmd/docs_examples_test.go` — `TestDocExamplesPassDeploymentFlags` (every documented example passes both flags).

**ID note:** `NS-6` is numbered by the operator-basis design note reconciling Discussion #1700, not by this registry's `INV-*` sequence; it is listed here because it is cited in `cmd/` and the resolution rule above is about *resolvability*, not about the prefix. Delivered by #1733 (R1 task S6, tracker #1727).

---

## Subsystem invariants

Scoped to one subsystem or one optional feature. Outside that subsystem they are not applicable.

### KV cache

#### INV-4: KV Cache Conservation

**Statement:** `allocated_blocks + free_blocks = total_blocks` at all times.

**Verification:** Checked after every allocation/deallocation. Check-then-act pre-check gate before any state mutation (vLLM parity); post-pre-check `popFreeBlock() == nil` panics (structurally unreachable in single-threaded DES). `FreeBlockCnt` maintained in lockstep by `appendToFreeList`/`removeFromFreeList`. `verifyBlockConservation()` provides independent free-list walk for debug-mode assertions.

**Operational note (H8):** KV cache pressure exhibits a sharp cliff, not gradual degradation. In H8's workload, performance was identical above ~2200 blocks and collapsed below it (4.7x TTFT P99 increase with just 4.5% fewer blocks). Below ~1000 blocks, the preempt-requeue cycle can livelock (see R19). Capacity planning formula: `threshold ≈ rate / num_instances × (input_tokens + output_tokens) / block_size`.

**Additional evidence (hardening wave):** Two KV conservation bugs discovered in March 2026: (1) Issue #492, fix #502 — prefill capacity pre-check over-estimated by up to 1 block (partial last-block fill not accounted for), causing false allocation failures that triggered unnecessary preemptions. (2) Issue #501, fix #506 — TieredKVCache CPU→GPU reload could produce an inverted range (`newStart >= endIndex`), causing a slice-bounds panic in block allocation. Both bugs directly affected the allocation/deallocation balance that INV-4 protects. (See also #519 in INV-8 — the range-loop livelock primarily violated the work-conserving property, not block-level conservation.)

### Batch formation

#### INV-12: Phase 1 Completeness Under Priority Preemption

**Statement:** After Phase 1 of `FormBatch` completes, every non-preempted running request in decode phase has `NumNewTokens > 0`, provided the token budget was not exhausted and `MaxModelLen` did not cap the request. No running request is silently skipped due to index drift from non-tail eviction.

**Context:** With `--preemption-policy priority`, the preemption victim may be at any index in the running batch (not just the tail). Removing an element at index `i < reqIndex` shifts subsequent elements left by one. Without the `reqIndex -= adjustment` correction (analog of vLLM `scheduler.py:853` `req_index -= 1`), the Phase 1 loop skips the shifted element.

**Verification:** `sim/batch_formation_test.go` — `TestPreemption_Priority_Phase1Completeness`: verifies that after non-tail eviction where `victimIdx < reqIndex`, ALL remaining running requests receive decode tokens (NumNewTokens > 0). The index adjustment is tested with [bg, crit, std] batch where bg is evicted at index 0 while processing crit at index 1.

**Trivially satisfied for FCFS:** With `--preemption-policy fcfs` (default), victims are always at the batch tail (`victimIdx == len-1 >= reqIndex`), so `adjustment == 0` and no element skipping is possible.

**Hypothesis family:** Structural model (same as INV-4, INV-7, INV-8, INV-9).

### Gateway queue

#### INV-16: Gateway Queue Counter Consistency

**Statement:** `GatewayQueue.totalLen` always equals the number of entries actually held across all priority bands and flows, and each `priorityBand.totalLen` equals the number held across that band's flows. The consequence the code asserts on: **a dequeue that finds the counter positive must return an entry.** A positive counter with nothing to hand back is a desync, not an empty queue.

**Relationship to INV-4:** the same lockstep-counter shape one subsystem over. INV-4's entry already documents this pattern for KV blocks (*"`FreeBlockCnt` maintained in lockstep by `appendToFreeList`/`removeFromFreeList`"*); here the paired mutators are `Enqueue` / `removeEntryByIndex` / the `dequeueFrom*` family, and the equality is over a two-level structure (queue → band → flow) rather than one free list, so there are two counters to keep in step rather than one.

**Why it is worth declaring.** The failure mode is not a wrong number, it is a **stuck queue**: a `totalLen` that over-counts makes `Dequeue` believe there is work forever, and a `totalLen` that under-counts makes it return `nil` while requests sit in a flow. Neither is visible in any metric — the first surfaces as a panic in whatever test happens to drain a queue, the second as requests silently stranded until the horizon (where they land in `gateway_queue_depth` and keep INV-1 balanced, so conservation does *not* catch it).

**Enforced:** three panics in `sim/cluster/gateway_queue.go`, all of the form "counter positive, dequeue returned nothing" — `Dequeue` at `:401`, `DequeueGated`'s no-bands case at `:419`, and `DequeueGated`'s per-band case at `:453`. Three sites for one property is deliberate: the two dequeue entry points and the per-band inner loop each have their own route to a desync, and the band-level panic names the band so the report localises to one counter.

**Verification:** `sim/cluster/gateway_queue_invariant_test.go` — `TestINV16_CounterConsistency_AcrossMutations` walks the bands and flows after every mutation in a mixed enqueue / dequeue / shed / TTL-removal sequence and compares the walked count against both `totalLen` levels and `requestIndex`, so a counter that drifts fails on the equality rather than waiting for a dequeue to trip the panic. `TestINV16_PositiveCounterAlwaysDequeues` drains a multi-band, multi-flow queue to empty through both `Dequeue` and `DequeueGated`, asserting every call with a positive counter yields a request.

#### INV-17: Shed Victim Is the Flow Tail

**Statement:** within a flow, the entry chosen for shedding is always the **tail** — the highest `seqID`. Flows are append-only in `seqID` order, so removing the shed victim is a truncation: it never disturbs index 0, which is the head `Dequeue` and the fairness policies read.

**Why it is worth declaring.** This is the guard against the index-drift bug class INV-12 covers on the batch-formation side, and the two entries are worth reading together: there, removing an element at `i < reqIndex` shifted the survivors left and silently skipped one; here, removing a non-tail element would shift the survivors left and silently reorder the flow's FIFO — changing which request dispatches next without any counter or metric moving. `removeEntryByIndex` is written as a truncation (`flow.requests[:last]`) precisely because it may assume tail-ness, so the assumption has to be asserted rather than commented.

**Where the property comes from:** `findGlobalShedVictim` tie-breaks toward the **highest** `seqID` within the lowest-priority flow, and `seqID` increases monotonically with append order. So tail-ness is a consequence of the victim-selection rule, not an independent constraint — which is exactly why it can be broken from a distance: a change to victim selection (a new fairness or criticality rule that prefers an older entry) breaks a truncation two functions away.

**Enforced:** `sim/cluster/gateway_queue.go:370` — `removeEntryByIndex` panics when `idx != len(flow.requests)-1`, naming both indices.

**Verification:** `sim/cluster/gateway_queue_invariant_test.go` — `TestINV17_ShedVictimIsFlowTail` fills one flow, sheds under capacity pressure, and asserts the surviving entries are the original prefix with the head unchanged (a truncation, not a reorder); `TestINV17_NonTailRemovalPanics` asserts the guard fires on a non-tail index, so the panic is not dead code.

### Routing and observability

#### INV-7: Signal Freshness Hierarchy

**Statement:** Routing snapshot signals have tiered freshness due to DES event ordering and configurable staleness.

| Signal | Owner | Freshness (interval=0) | Freshness (interval>0) | Updated By |
|--------|-------|------------------------|------------------------|------------|
| InFlightRequests | Cluster | Synchronous | Synchronous | `RoutingDecisionEvent.Execute()` (increment), completion detection (decrement) |
| PreemptionCount | Instance | Immediate | Periodic | `CachedSnapshotProvider.Snapshot()` — routed through `ObservabilityConfig.PreemptionCount` like other instance signals |
| QueueDepth | Instance | Immediate | Periodic | `QueuedEvent.Execute()` |
| BatchSize | Instance | Immediate | Periodic | `StepEvent.Execute()` |
| KVUtilization | Instance | Immediate | Periodic | `FormBatch()` → `AllocateKVBlocks()` |
| CacheHitRate | Instance | Immediate | Periodic | `FormBatch()` |
| cacheQueryFn (precise-prefix-cache, no-hit-lru) ¹ | Instance (via CachedSnapshotProvider) | Ground truth (synchronous) | Periodic (CacheBlocks interval, default 50ms) | `CachedSnapshotProvider.RefreshCacheIfNeeded()` in `buildRouterState()` |
| LoadingSnapshots (TotalKvCapacityTokens) ² | Instance (direct accessor) | Synchronous | Synchronous | `buildRouterState()` reads `inst.TotalKvCapacityTokens()` directly — bypasses `snapshotProvider` entirely; unaffected by `--snapshot-refresh-interval` or `--cache-signal-delay` |

¹ `cacheQueryFn` freshness is governed by `--cache-signal-delay` (default 50ms), which maps to `ObservabilityConfig.CacheBlocks`. The "interval=0" / "interval>0" columns for this row refer to `--cache-signal-delay`. Cache block staleness is now managed by `CachedSnapshotProvider` alongside other signals (#1060).

² `LoadingSnapshots` carries `Model`, `GPUType`, `TPDegree`, `CostPerHour`, and `TotalKvCapacityTokens`; this row covers `TotalKvCapacityTokens` freshness because it is the only autoscaler-relevant field that is derived at runtime (from KVCache), though stable after construction (the others are literal config constants set at `NewInstanceSimulator`). All fields are always synchronous regardless of observability configuration. Used by `DefaultCollector` to populate `ModelSignals.PendingTotalKvCapacityTokens` for pending supply accounting in `V2SaturationAnalyzer` (#1109).

**Design implication:** When `--snapshot-refresh-interval > 0` (default: 50000µs = 50ms, llm-d parity), all Prometheus-sourced signals (QueueDepth, BatchSize, KVUtilization) share the same scrape interval — matching real vLLM deployments where all three are exposed via the same `/metrics` endpoint. Set `--snapshot-refresh-interval 0` for oracle/immediate mode. `InFlightRequests` remains synchronous (gateway-local counter, not Prometheus-sourced). When `--cache-signal-delay > 0` (default: 50ms), prefix cache query closures use periodic snapshots of each instance's `HashToBlock` map, managed by `CachedSnapshotProvider` alongside other signal snapshots. The 50ms default models aggregate signal staleness from production llm-d. Set `--cache-signal-delay 0` for oracle mode (live cache state).

`EffectiveLoad()` = `QueueDepth + BatchSize + InFlightRequests`. The synchronous `InFlightRequests` term compensates for Periodic staleness in the other two terms. The `queue-depth` scorer reads `QueueDepth` only (GIE parity); `EffectiveLoad()` is used by `load-balance`, `least-loaded`, `always-busiest`, and admission policies. The `active-requests` scorer reads `InFlightRequests` only (synchronous). The `running-requests` scorer reads `BatchSize` (Periodic/Immediate). The `load-aware` scorer reads `QueueDepth` only (Periodic/Immediate), with a linear threshold at 128.

**Verification:** H3 hypothesis experiment, H29 snapshot-staleness experiment (see [`hypothesis-archive` branch](https://github.com/inference-sim/inference-sim/tree/hypothesis-archive/hypotheses)). **No Go test anchors this invariant, deliberately:** the freshness hierarchy is a property of the observability configuration across the whole snapshot pipeline, and asserting it in-process would pin the current snapshot plumbing rather than the tiering law. It is named in 8 production files; adding a Go test for it is issue #1718's out-of-scope item 3, awaiting a decision on what the assertion should be.

**Evidence:** Issues #282, #283. At rate=5000, kv-utilization-only routing produces 200x worse distribution uniformity than queue-depth. Issue #463: unified Prometheus staleness model.

#### INV-18: Prefix-Cache LRU Structural Consistency

**Statement:** in every `lruBlockCache` (one per instance, inside `PrefixCacheIndex`), the intrusive doubly-linked list and the `lookup` map describe the same set of blocks. The load-bearing corollary the code asserts on: **`tail == nil` if and only if `len(lookup) == 0`.** So a cache the map reports as non-empty always has a tail to evict, and eviction at capacity can never fail.

**Why it is worth declaring.** The two structures are maintained by separate statements — `touch` writes `lookup[h]` and calls `pushHead`, `evictOldest` calls `delete(lookup, ...)` and `removeNode` — so a new mutator that updates one and not the other compiles, passes, and leaves the cache subtly wrong. The failure is **silent until it isn't**: a list that has lost nodes still answers `MatchLength` queries, just with fewer hits, so the prefix-affinity and `precise-prefix-cache` scorers quietly route on a degraded index and the only symptom is worse cache locality — a performance result a reader would attribute to the workload. It becomes loud only when `len(lookup) >= capacity` finally drives an eviction into a nil tail.

**Scope:** the *router-side approximate* index, which is deliberately an approximation of real per-instance KV state (the router does not query instances). INV-18 is about the index's internal self-consistency, not about its agreement with the instance's actual cache — that divergence is by design and is what `--cache-signal-delay` (INV-7) governs.

**Enforced:** `sim/prefix_cache_index.go:131` — `evictOldest` panics on a nil tail, and its message already said *"(invariant violation)"* before the invariant had an ID. The two nil-node guards in `pushHead` (`:139`) and `removeNode` (`:155`) are the same family.

**Verification:** `sim/prefix_cache_index_invariant_test.go` — `TestINV18_ListAndLookupStayConsistent` walks the list forward and backward after each of a long sequence of `RecordBlocks` calls that crosses capacity repeatedly, asserting the walked node set equals the `lookup` key set, the head/tail terminals are correct in both directions, and the `tail == nil ⟺ len(lookup) == 0` biconditional holds at every step (including the empty cache, the one-entry cache, and the at-capacity cache). The biconditional is checked in **both** directions: a nil tail with a non-empty map is the panic case, while a non-nil tail with an empty map is the mirror leak — an orphaned node the map can no longer reach, which degrades routing without ever reaching the panic. `TestINV18_EvictOldestPanicsOnInconsistentState` asserts the guard itself fires on a hand-built inconsistent cache, so the entry's "eviction at capacity can never fail" clause does not rest on dead code.

### Autoscaling

#### INV-19: Scale Decisions Carry a Non-Zero Delta

**Statement:** every `ScaleDecision` an `Engine.Optimize` implementation emits has `Delta != 0`. A zero delta is an engine bug, and the pipeline's response is **warn and skip the decision** — the tick continues and the run does not fail.

**Why the warn-only choice is the invariant's point.** The property itself is small; what is worth writing down is the policy. BLIS applies both policies to corrupted internal state and only one of them was documented: `activeTransfers` going negative in `pd_events.go` deliberately **fails the run** (INV-P2-2), while a zero-delta decision is logged and dropped. Both are "an internal producer emitted something impossible". The reason to differ is that a zero delta is *inert* — skipping it actuates nothing and leaves cluster state exactly as a correct engine would have — whereas a negative transfer count means the accounting has already diverged and every later number is suspect. Recording that reasoning is what stops the next reviewer from "fixing" the asymmetry in either direction.

**Consequences of the skip, which are what a reader needs.** The decision is dropped *before* the stabilization-window gate, so it neither starts nor advances a window. It is also excluded from the `modelsWithScaleUp` / `modelsWithScaleDown` sets built just above, so a model whose only decision this tick was a zero delta counts as having **lost** its signal and has its timers reset. A buggy engine emitting zero deltas therefore does not merely fail to scale — it can hold a model permanently outside its stabilization window. That is still the right behaviour (a zero delta carries no direction, so it cannot legitimately sustain either timer), but it is not obvious from the `continue`.

**Enforced:** `sim/cluster/autoscaler.go:287` — `logrus.Warnf` naming the model, then `continue`. Warn-only by design (R1 is satisfied: the failure is observable, not silent).

**Verification:** `sim/cluster/autoscaler_invariant_test.go` — `TestINV19_ZeroDeltaDecisionSkippedNotFatal` drives the pipeline with an engine that emits a zero-delta decision and asserts the actuator is never called and the tick completes; `TestINV19_ZeroDeltaDoesNotSustainStabilizationTimer` asserts a zero delta interleaved into a scale-up stream resets the window rather than advancing it, pinning the timer consequence above.

### Cluster infrastructure and placement

#### INV-A: GPU Conservation

**Statement:** For every node in the placement inventory, `allocated_gpus + free_gpus == total_gpus`.

**Verification:** `PlacementManager.VerifyConservation()` (`sim/cluster/infra_placement.go:185`) walks every node in sorted ID order (R2, so the first-violation report is deterministic) and returns a descriptive error naming the node and all three counts. `sim/cluster/infra_placement_test.go` asserts it after a successful placement, after a *failed* placement, after a partial placement, after a full placement, and after `ReleaseInstance`; `sim/cluster/infra_node_test.go` carries the dedicated companion invariant test (T012).

**Relationship to INV-4:** the same conservation shape one level up the hierarchy. INV-4 conserves *blocks* inside one instance's KV cache; INV-A conserves *GPUs* inside a node's inventory. Both have a runtime verifier and both are checked around every allocate/release — there is no principled reason to document one and not the other.

**Cited in code:** 11 times — the verifier and its error string in `sim/cluster/infra_placement.go`, plus assertions in `infra_placement_test.go` and `infra_node_test.go`. Re-derive with `git grep -o 'INV-A[^2]' -- 'sim/**' 'cmd/**' | wc -l`; a plain `INV-A` pattern also matches the five `INV-A2` sites and returns 16.

**Registry note (#1721):** this invariant had a runtime checker and 11 citations while being absent from this registry. It is the finding that motivated the resolution rule at the top of this document.

---

### Workload generation

#### INV-W3: Cohort Expansion Purity

**Statement:** Cohort expansion is a pure function — the same `(cohorts, seed)` always produces identical output.

**Relationship to INV-6:** a component-level restatement of INV-6 (determinism), scoped to one function. It is listed here because it is cited by ID in code; the run-level guarantee it serves, and the one that is actually tested end-to-end, is INV-6.

**Verification:** documented at the expansion entry point (`sim/workload/cohort.go:10`). There is no INV-W3-specific test — determinism is covered at the run level by INV-6's byte-identity tests, which a purity regression here would break.

**Cited in code:** 1 time.

**Registry note (#1721):** cited by ID, documented nowhere until #1721.

---

### Sessions and closed-loop replay

#### INV-10: Session Causality

**Statement:** For all rounds N in a **closed-loop** session: `round[N+1].ArrivalTime >= round[N].CompletionTime + ThinkTimeUs`. Boundary: ThinkTimeUs = 0 produces equality.

**Scope (#1692):** INV-10 is scoped to closed-loop arrival *regeneration* — it constrains arrivals the `SessionManager` derives from sim completion. The `fixed-accumulate` replay mode (`blis replay --session-mode fixed-accumulate`) is **exempt by design**: it injects every round at its *recorded* arrival time (open-loop, like `--session-mode fixed`) regardless of sim completion, precisely so N large prefills pile into the scheduler at the real clock and produce genuine queueing delay. Chaining arrivals to sim completion (INV-10) is the self-throttling feedback loop that mode exists to break, so INV-10 does not apply to it. `fixed-accumulate` still reconstructs the growing accumulate-delta input (the closed-loop reconstruction) — only the arrival source differs.

**Verification:** `sim/workload/session_test.go` — `TestSession_RoundGeneration_CorrectArrivalTime` verifies the arrival time formula. The ThinkTimeUs=0 boundary is inherent in the formula.

**Evidence:** Design doc `docs/plans/2026-03-13-client-behavior-model-design.md` — INV-10 definition. Guaranteed by construction in `SessionManager.OnComplete`.

**Hypothesis family:** Scheduler invariants (safety/liveness) — causality chain for session rounds.

#### INV-11: Session Completeness

**Statement:** Every session reaches exactly one terminal state: completed (all rounds done), cancelled (a round timed out or was dropped), horizon-interrupted (simulation ended mid-session), or budget-exhausted (concurrency mode: global follow-up request cap reached). No session is silently abandoned.

**Verification:** `sim/workload/session_test.go` — tests cover all terminal paths: `TestSession_TimeoutCancels_NoMoreRounds` (cancelled), `TestSession_FinalRound_Completes` (completed), `TestSession_BeyondHorizon_NotGenerated` (horizon-interrupted), `TestSession_DroppedFollowUp_CancelsSession` (cancelled via drop). Budget-exhausted path verified via `TestConcurrencyMode_EndToEnd_SessionFollowUps` (budget exhaustion stops follow-up generation).

**Evidence:** Design doc INV-11 definition. The `SessionManager.OnComplete` method transitions sessions to exactly one terminal state before returning nil. Issue #1657 — a speculative-decoding step advanced `ProgressIndex` PAST the completion boundary, so `OnComplete` read `actualOutputLen > len(OutputTokens)` and cancelled an entire valid closed-loop session as accounting corruption. The fix caps the granted token count at the completion boundary in `FormBatch` (generation time), keeping the over-cap branch a genuine corruption detector. The `budget_exhausted` state is reached when the shared follow-up budget (set via `SetFollowUpBudget` for `--concurrency` mode) is depleted — the session's unlimited-rounds flag would otherwise continue generating follow-ups, but the global cap takes precedence.

**Hypothesis family:** Structural model — session lifecycle completeness.

### Latency model

#### INV-BC-DP1: Dense DP=1 Step-Time Byte-Identity

**Statement:** For a **dense** model (`NumLocalExperts < MoEMinExperts`) at `DP=1` with expert parallelism off, the `trained-physics` `StepTime` MUST be byte-identical to the pre-#1419 value across the full TP matrix. The DP/EP refactor (#1419) splits the monolithic TP all-reduce term into per-class terms (`tTpAttention + tTpDenseFFN [+ tMoEReduce]`) and routes MoE expert cost through `sim.ExpertPlacement`; for dense `DP=1` this is value-preserving: `tTpAttention + tTpDenseFFN = V(numLayers, tp) + V(numDenseLayers, tp) = V(2·numLayers, tp)` (since `numDenseLayers == numLayers`, `numMoELayers == 0`), exactly the old `allReduceUnits = 2·numDenseLayers + numMoELayers` term, and every `/(tp·dp)` divisor reduces to `/tp` at `dp=1`.

MoE-model step time **intentionally changes** at `DP=1`/EP-off (B1 routed-expert weight scoping + the newly-charged `tMoEReduce`); that is a deliberate fidelity gain, not a parity regression. Since #1548 an MoE model with `--enable-expert-parallel` also intentionally differs from its EP-off self (the MoE-FFN collective becomes dispatch/combine, and routed-expert weights shard over the EP group) — the invariant covers dense models and the EP-off MoE baseline, never the EP-on toggle.

**Verification:** `sim/latency/trained_physics_dpep_test.go` — `TestINVBCDP1_DenseStepTimeByteIdentical` (golden across TP∈{1,2,4,8}) with companion `TestINVBCDP1_DenseDP1Determinism` (dense step time is invariant to the EP flag and the MoE comm backend, and deterministic across calls).

**Evidence:** Dense experiments in the trained-physics golden dataset (`testdata/trained_physics_iter29.json`) are unchanged across the #1419 refactor; only the four Llama-4 Scout (MoE) experiments shift.

**Hypothesis family:** Latency-model invariants (correctness/conservation).

### LoRA control plane

The LoRA control-plane invariants are stated in full in `docs/plans/2026-07-15-lora-control-plane-design.md` §9, which stays **authoritative**. They are listed here as pointer lines so an `INV-L*` citation in code can be resolved (the resolution rule at the top of this document).

| ID | Statement (abbreviated — the plan is authoritative) | Relationship | Code citations |
|---|---|---|---|
| **INV-L1** | No-op inertness: with an empty registry and unset adapter capacity, output is byte-identical to the pre-feature build. | specializes INV-6 | 2 |
| **INV-L2** | Capacity bound: for every instance at every point in the run, `\|resident adapters\| <= configured capacity`. | — | 1 |
| **INV-L3** | Cold-load charge: load latency is `>= 0`, charged exactly once per cold `(adapter, instance)` transition, never when the adapter is warm. | — | 6 |
| **INV-L4** | Memory conservation: per instance, `allocated_blocks + free_blocks + adapter_reserved == total`, with `adapter_reserved` fixed at startup. | extends INV-4 | 11 |
| **INV-L5** | No eviction of in-use: an adapter is *in use* from the moment a request referencing it enters the running batch until that request completes; an in-use adapter is never evicted, and this persists through preemption (a preempted request has not completed). Also covers the victim reserved at load-start, which is never in use. | — | 7 |
| **INV-L6** | Oracle boundary: adapter-aware routing/servability reads adapter id, rank and resident sets only — never `Request.OutputTokens`. | specializes INV-9 | 0 |
| **INV-L7** | Backend parity: roofline and trained-physics apply an identical adapter-overhead factor for the same batch (R23). | — | 0 |

Counts are a snapshot over `sim/` and `cmd/`. Re-derive any row with:

```bash
git grep -o 'INV-L4' -- 'sim/**' 'cmd/**' | wc -l
```

**INV-L2's code anchor (#1721).** INV-L2 was documented in the plan and cited nowhere in code — a declared capacity bound with nothing pointing at it. #1721 anchors it at the enforcement point the plan's §7 identifies: `Simulator.maybeStartAdapterLoad` (`sim/simulator.go`) commits the LRU eviction victim and reserves the slot at load **start**, so the reserved slot counts toward capacity for the whole load window. That timing is what makes the bound hold — committing the victim at load *completion* would let a request arriving mid-load resurrect the soon-to-be-evicted adapter and exceed capacity. The structural guarantee is in `sim/lora/resident_set.go`: `Store` admits a new id only when a slot is free **or** an eviction succeeds, and returns `false` when the set is full and every entry is pinned. That conditional — not the commit timing — is what makes the bound hold. What load-start commitment buys is that the `Store` at load completion can never fail, plus exact eviction accounting and the plan's §12 no-deadlock argument. Note the ID is written at the load-start site, so `git grep INV-L2` lands there rather than here.

**INV-L6 and INV-L7 have no code anchor.** Both are declared in the plan and both have tests — `sim/latency/adapter_overhead_test.go`'s `TestStepTime_BackendParity_IdenticalFactorApplication` is INV-L7's, and the servability paths carry no `OutputTokens` reference for INV-L6 — but neither test nor production site names the ID. That is the *inverse* of the gap the resolution rule addresses (a declared promise with no citation, rather than a citation with no statement), so this registry records it rather than fixing it. Unlike INV-L2, neither has a runtime message that would leave a reader stranded.

---

### PD disaggregation

#### INV-PD-1: KV Completeness

**Statement:** For every disaggregated request, `decode_enqueue_time >= kv_transfer_completion_time`. A decode sub-request must not be enqueued before its KV transfer completes.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_RequestCompletesFullPath` checks DecodeEnqueueTime >= TransferCompleteTime for every parent request. Runtime defensive check in `KVTransferCompletedEvent.Execute()`.

**Evidence:** Both `TransferCompleteTime` and `DecodeEnqueueTime` are set in `KVTransferCompletedEvent.Execute()` at the same simulation tick (`e.time`), so the invariant holds by construction.

#### INV-PD-2: Pool Exclusivity

**Statement:** Prefill sub-requests route only to prefill pool instances; decode sub-requests route only to decode pool instances.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_PrefillRoutedToPrefillPool` and `TestDisaggregation_DecodeRoutedToDecodePool` verify pool role for every parent request's prefill and decode instance assignments.

**Evidence:** `buildPoolFilteredSnapshots(role)` filters routing snapshots to only include instances of the specified pool role before passing to the routing policy.

#### INV-PD-3: Transfer Conservation

**Statement:** `initiated_transfers == completed_transfers` at simulation end, provided all transfers complete within the simulation horizon. At bounded horizons, the difference (`initiated - completed`) represents in-flight transfers accounted for in the `pdInTransfer` conservation correction (see INV-1 PD correction in `cluster.go`).

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_TransferConservation` asserts equality and expected count (uses unbounded horizon).

**Evidence:** `transfersInitiated` incremented on every entry to `KVTransferStartedEvent.Execute()` — both the happy path (via `scheduleTransferCompletion`) and the drop-at-start path (via `dropAtStart`, issue #1343). `transfersCompleted` incremented on every entry to `KVTransferCompletedEvent.Execute()`. Every started event schedules exactly one completed event — successful reservations schedule a real completion at `start + duration`; drop-at-start schedules a zero-duration degenerate completion at the same tick so the counters stay paired even when decode-side reservation fails.

#### INV-PD-4: Phase Causality

**Statement:** For every disaggregated request: `arrival <= prefill_enqueue <= prefill_complete <= transfer_start <= transfer_complete <= decode_enqueue <= completion`.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_PhaseCausality` checks the full causal chain for every parent request.

**Evidence:** Each phase transition is enforced by DES event ordering: earlier phases schedule later-phase events at `time >= current_time`.

#### INV-PD-5: Pool Stability

**Statement:** Pool membership is fixed at construction time and never changes during simulation.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_PoolStability` compares `PoolMembership()` before and after `Run()`.

**Evidence:** `BuildPoolMembershipFromIndices` is called once in `NewClusterSimulator` and stored in `cs.poolMembership`. No code path in `Run()` modifies this map.

#### INV-PD-6: Metric Map Parent Granularity

**Statement:** After `Run()` completes on a disaggregated cluster, every per-request metric map (`RequestE2Es`, `RequestTTFTs`, `RequestITLs`, `RequestSchedulingDelays`, `RequestCompletionTimes`, `Requests`) contains only parent-level request IDs. No key may have a `_prefill` or `_decode` suffix. A **served** parent — one that reached a terminal `CompletionTime` with a decode instance assigned AND whose decode sub-request has ≥1 un-discarded `ITL` entry at finalization (a surviving first-token measurement) — contributes exactly one entry per map (keyed by parent ID). Incomplete parents, and **terminal-but-no-token** parents, contribute **no** entry — they produced no surviving output and are accounted for by the drop/timeout counters (`DroppedUnservable` / `TimedOutRequests`) only, never by the latency distributions (issue #1511). The no-token terminal shapes are: decode-side drops (nil `DecodeSubReq`); decode-timeout-while-queued (`DecodeSubReq` non-nil, empty `ITL`, never ran a step); and preempted-then-timed-out (`DecodeSubReq` non-nil but `ITL` discarded by a preemption — `sim/batch_formation.go` clears `ITL` on preempt — with a timeout before re-emitting).

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_MetricProjection_NoSubRequestKeys` checks all six maps for suffix-free keys; `TestDisaggregation_MetricProjection_DroppedParent_NoSubRequestKeys` verifies the suffix clause holds when decode KV allocation fails; `TestDisaggregation_DroppedParent_NotInLatencyMaps` verifies a no-token terminal parent appears in none of the latency maps while remaining counted in `DroppedUnservable` (issue #1511). `TestDisaggregation_MetricProjection_NoOp` verifies the projection is a no-op for non-disaggregated clusters. `sim/cluster/pd_e2e_causality_test.go` — `TestPDParentMetrics_NoTokenExcludedFromLatency` (table over the served/no-token taxonomy) and `TestPDParentMetrics_TTFTSumConsistentAcrossDrop` (the aggregate `TTFTSum == Σ RequestTTFTs` law with a drop present).

**Evidence:** `projectPDMetrics()` in `sim/cluster/cluster.go` is called after `aggregateMetrics()` and the conservation correction. It unconditionally deletes the `pfx` and `dec` keys for every parent request, and conditionally inserts a parent-keyed entry only for **served** parents (`served := CompletionTime > 0 && DecodeInstanceID != "" && DecodeSubReq != nil && len(DecodeSubReq.ITL) > 0`). A terminal-but-no-token parent additionally has its prefill TTFT rolled out of the aggregate `TTFTSum`, so `TTFTSum == Σ RequestTTFTs` holds across the no-token terminal outcomes this fix restores (BC-3, issue #1511). This is **not** a general PD law: a preempted-and-re-prefilled decode sub-request re-stamps its own `FirstTokenTime` into `TTFTSum`, and `projectPDMetrics` deletes `RequestTTFTs[dec]` without rolling that decode term back — a pre-existing residual tracked in #1628, out of scope for #1511.

#### INV-PD-6b: CompletionTime Includes PostDecodeFixedOverhead

**Statement:** For all successfully decoded parent requests (`DecodeInstanceID != ""`), `parent.CompletionTime` equals the cluster clock at decode completion plus the decode instance's `PostDecodeFixedOverhead()`. For roofline, overhead is 0, so `CompletionTime` equals the raw cluster clock tick. For trained-physics, overhead is α₁ (≈ 777 µs), so `CompletionTime` exceeds the raw clock by that amount. `parent.CompletionTime` is a lifecycle field (it drives session follow-up scheduling and phase-causality checks); the decode instance's `PostDecodeFixedOverhead` is also carried by the decode sub-request's own recorded E2E (`recordRequestCompletion` adds it, matching the non-PD path).

Note (issue #1513): the client-visible parent E2E metric is **not** `parent.CompletionTime − ArrivalTime`. `parent.CompletionTime` is stamped on the cluster clock at the completion-*detection* tick and omits the decode step's own advance, so that formula under-counts E2E — below a single decode step (`ITL[0]`) and below TTFT for short outputs, violating INV-5. `projectPDMetrics()` instead reconstructs the arrival→last-token span from the decode sub-request's own per-instance metrics, in two clock frames:

- **Primary (non-preempted):** the decode sub-request never ran prefill (`FirstTokenTime == 0`), so its own E2E is schedule-relative → `E2E = decodeSchedulingDelay + decodeOwnE2E`. This guarantees INV-5 **by construction on the primary path** (`decodeOwnE2E ≥ ITL[0]`, so `E2E ≥ TTFT`).
- **Preempted-and-re-prefilled:** preemption resets `ProgressIndex` and re-prefill stamps an arrival-relative `FirstTokenTime`, so `decodeOwnE2E` is already the full arrival→completion span → `E2E = decodeOwnE2E` (discriminated on `FirstTokenTime != 0`; adding `decodeSchedulingDelay` would double-count the pre-decode wait — a regression the guard prevents).
- **Fallback:** a parent that **served a token** but whose decode sub-request never recorded its own E2E — e.g. it emitted the first token and then timed out mid-generation — falls back to the `parent.CompletionTime − ArrivalTime` value (no regression). The three no-token terminal outcomes (decode-side drop at transfer start / transfer complete, decode-timeout-while-queued) no longer reach this branch: they are excluded from the projection entirely and contribute no E2E entry (issue #1511, INV-PD-6). INV-5 on this fallback branch is **not** guaranteed by construction — a served-token request whose deadline lands inside the post-first-token OTPT window can leave TTFT marginally above E2E; this residual micro-edge is unchanged from main and orthogonal to #1511.

`projectPDMetrics` also keeps `RequestCompletionTimes[parentID] = ArrivalTime + E2E` consistent with the projected E2E.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_CompletionTime_LifecycleField_IncludesOverhead` verifies the lifecycle field directly: `CompletionTime_with_overhead − CompletionTime_without_overhead == overhead` (this pins `parent.CompletionTime = c.clock + PostDecodeFixedOverhead()`; the E2E-metric differential below no longer does, since E2E stopped deriving from `parent.CompletionTime`). `TestDisaggregation_CompletionTime_IncludesNonZeroOverhead` verifies the overhead also flows into the E2E metric (`E2E_with − E2E_without == overhead`, because `decodeOwnE2E` carries the overhead while `decodeSchedulingDelay` is independent of it). `TestDisaggregation_CompletionTime_GeqAllPriorPhaseTimestamps` verifies `CompletionTime >= DecodeEnqueueTime` and `CompletionTime >= TransferCompleteTime` (phase causality preserved). `sim/cluster/pd_e2e_causality_test.go` (issue #1513) verifies the reconstructed E2E satisfies INV-5 (`E2E ≥ TTFT`) for short outputs, is `≥ ITL[0]` and `≥ decodeOwnE2E`, reconstructs the arrival→completion span, keeps `RequestCompletionTimes` consistent, is `≥` the non-PD baseline for 1-token requests, and (via `TestPDParentE2E_ProjectionBranches`) covers the primary/preempted/fallback/negative-guard branches at unit granularity — including the preemption double-count guard.

**Evidence:** `detectDecodeCompletions()` in `sim/cluster/cluster.go` stamps `parent.CompletionTime = c.clock + inst.PostDecodeFixedOverhead()` (overhead fixed in issue #846). `projectPDMetrics()` computes the parent E2E metric as `decodeSchedulingDelay + decodeOwnE2E` (E2E under-count fixed in issue #1513).

### Pools and KV transfer

#### INV-P2-1: Pool-Config Consistency

**Statement:** Per-pool hardware overrides produce a valid `SimConfig` for each pool role: zero-valued `PoolOverrides` is a no-op (backward-compatible), non-nil fields override only the specified fields, and the global `SimConfig` is never mutated.

**Verification:** `sim/cluster/resolve_test.go` — `TestINV_P2_1_PoolConfigConsistency` verifies observable KV capacity differences between pools pre-simulation via `FreeKVBlocks()`; `TestINV_P2_1_RequestConservation` verifies INV-1 holds under heterogeneous pool configuration. `sim/cluster/kv_autocalc_test.go` — `TestStartupPlacement_PerGPUKVCapacity`, `TestDeferredPlacement_PerGPUKVCapacity`, and `TestAutoscalerScaleUp_PerGPUKVCapacity` verify that node-pool-placed instances size KV capacity from their actual placed GPU (#1522).

**Evidence:** `ResolvePoolConfig` performs a struct copy and applies only non-nil/non-zero overrides. `resolveConfigForRole` is called in the instance construction loop in `NewClusterSimulator`, before any simulation state is created. For node-pool placement (#1522), `applyPerInstanceKVCapacity` recomputes each placed instance's `TotalKVBlocks` from its pool's `gpu_memory_gib` at all three placement sites (startup, deferred `NodeReadyEvent`, autoscaler scale-up), immediately after the `HWConfigByGPU` execution-calibration override (#893) — so an instance's GPU calibration and KV capacity describe the same device. Gated by `KVAutoCalcConfig.Enabled` (off ⇒ no-op, INV-6); an explicit `--total-kv-blocks` disables it (uniform global capacity).

#### INV-P2-2: Fair-Share KV Transfer Bandwidth

**Statement:** When `--pd-transfer-contention` is enabled, the effective bandwidth available to each concurrent KV transfer is `total_bandwidth / active_transfers`, where `active_transfers` is the count of transfers in flight at the moment the new transfer starts (inclusive of the new transfer). With a single transfer in flight, the full bandwidth is used (`active_transfers == 1`, divisor == 1). This invariant gates the transfer duration formula in `KVTransferStartedEvent.Execute()`.

**Verification:** `sim/cluster/transfer_contention_test.go`:
- `TestTransferContention_INVP22_EffectiveBandwidthFormula` — golden test for the N=1 duration (9 µs with 10 blocks at 10 GB/s)
- `TestTransferContention_INVP22_N2FormulaExact` — golden test for the N=2 duration (17 µs with same payload at 5 GB/s effective)
- `TestTransferContention_INVP22_DivisorLaw` — invariant test: `duration(N) / duration(1) ≈ N` for N ∈ {1,2,3,4,5,8,10} with monotonicity
- `TestTransferContention_INVP22_FairShareBandwidth` — end-to-end: concurrent transfers record peak >= 1 when multiple requests arrive simultaneously

**Evidence:** PR9 (`sim/cluster/pd_events.go`, `KVTransferStartedEvent.Execute()`). Gated behind `PDTransferContention` flag (off by default for backward compatibility). The `activeTransfers` counter is incremented before the divisor is applied, ensuring the new transfer receives a fair share of the bandwidth with every other transfer currently in flight.
