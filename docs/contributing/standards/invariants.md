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
| [INV-AX1 … INV-AX4](#agentic-replay) Agentic replay: conservation, dependency causality, lane bound, sending window | Subsystem | agentic replay |
| [INV-BC-DP1](#retired-latency-model-invariants) Dense DP=1 step-time byte-identity — *retired* | Subsystem | latency model |
| [INV-K1](#inv-k1-no-blis-side-draft-cost) No BLIS-side draft cost | Subsystem | latency model |
| [INV-K2](#inv-k2-rejected-drafts-do-not-advance-computed) Rejected drafts do not advance Computed | Subsystem | latency model / batch formation |
| [INV-M1](#inv-m1-measured-window-accounting-closes) Measured-window accounting closes | Subsystem | metrics output |
| [INV-M2](#inv-m2-derived-aggregates-are-exactly-their-definitions) Derived aggregates are exactly their definitions | Subsystem | metrics output |
| [INV-M3](#inv-m3-cluster-cache-hit-rate-is-prompt-lookup-weighted) Cluster cache hit rate is prompt-lookup-weighted | Subsystem | metrics output / KV cache |
| [INV-M4](#inv-m4-only-the-phase-decides-what-is-measured) Only the phase decides what is measured | Subsystem | metrics output |
| [INV-K3](#inv-k3-a-sequences-bounded-state-is-held-on-demand-and-gated-at-admission) A sequence's bounded state is held on demand and gated at admission | Subsystem | KV cache |
| [INV-K4](#inv-k4-a-speculating-request-holds-its-verify-width-and-lookahead) A speculating request holds its verify width and lookahead | Subsystem | KV cache / batch formation |
| [INV-K5](#inv-k5-the-recorded-acceptance-is-the-simulated-one) The recorded acceptance is the simulated one | Code-boundary | CLI / latency model |
| [INV-K6](#inv-k6-the-offload-tiers-hold-whole-chunks) The offload tiers hold whole chunks | Subsystem | KV cache / offload |
| [INV-L1 … INV-L7](#lora-control-plane) LoRA control plane — *stated in the LoRA design plan; pointer lines here* | Subsystem | LoRA |
| [INV-P2-1, INV-P2-2](#pools-and-kv-transfer) Pools and KV transfer | Subsystem | pools / KV transfer |
| [INV-R1](#inv-r1-admission-overhead-runs-on-the-endpoints-render-threads-in-arrival-order) Admission overhead runs on the endpoint's render threads, in arrival order | Subsystem | latency model / front end |
| [INV-PD-1 … INV-PD-8](#pd-disaggregation) PD disaggregation | Subsystem | PD |
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

**Data-parallel lockstep (#1909):** for the ranks of one DP group the property holds per group: while any rank has waiting or running work, the group's single `StepEvent` is armed in rank 0's queue and every rank's step guard points at it (`sim/dp_lockstep.go`). Verified after every event of random multi-rank workloads by the rapid property `TestLockstepGroup_WorkConservingAndWaveEnd` (`sim/dp_lockstep_test.go`), which also checks that a wave ends on a sync step fewer than 16 dummy steps after its last request.

**Code location:** Search for `// Work-conserving:` comment in `sim/simulator.go` — the `else` branch of `len(remaining) > 0` checks `WaitQ.Len() > 0` and schedules a new `StepEvent`.

**One forward pass at a time.** Work-conserving never means overlapping: a step triggered by new work on an idle instance (an arrival, a timeout, an injected decode sub-request, `ScheduleStepIfIdle`) starts no earlier than the end of the step still executing (`Simulator.idleStepTime`). A step whose requests all finish empties the batch at the step's start, so before this clause an arrival during it began the next step at once; a P/D prefill pool, whose every sub-request finishes in its one prefill step, ran its steps in parallel. Verified by `sim/step_busy_test.go` — `TestStep_NoStepStartsBeforeThePreviousEnds` (rapid) and `TestStep_EverySiteWaitsForTheExecutingStep` (each of the four sites) (#1944). vLLM's async scheduling, which overlaps the next step's scheduling with the current one's execution, is not modelled.

**Hypothesis family:** Structural model (same as INV-4, INV-7).

### INV-13: Run/Replay Parity

**Statement:** For any configuration supported by both `blis run` and `blis replay`, a trace exported via `blis run --trace-output` and replayed via `blis replay --session-mode fixed` with identical flags MUST produce identical per-request TTFT, E2E, and aggregate metrics. At the CLI boundary the enforceable form of that is stronger and is what new parity tests should assert: **byte-identical stdout**, which subsumes the per-request distributions, the cache/latency aggregates, and the conservation counters in one assertion (`cmd/dp_replay_parity_test.go`'s `TestINV13_RunReplayParity_MoEDPPlacement` is the model; pair it with a non-vacuity gate and a conservation companion, since byte-identity alone is satisfied by two identically-wrong runs). "Identical flags" includes `--horizon`: the two commands have different *defaults* for it (`blis run` defaults to unlimited; `blis replay` auto-computes 2x the max arrival time), so a parity comparison must pass `--horizon` explicitly on both legs or the replay leg may truncate a still-draining run. This is a difference in defaults, not in behavior — the invariant is over the resolved configuration. Features not yet supported by replay (autoscaler, node pools) MUST cause a `logrus.Fatalf` at startup when those flags are explicitly set or when a policy bundle containing those features is passed — never silent degradation or a mere warning. Both commands build the deployment through the same kernel path from the same scenario (`--scenario`/`--scenarios`/`--registry`), so MoE data parallelism (a scenario pool with `dp > 1`), expert parallelism and P/D disaggregation replay byte-identically. The model autoscaler with `dp > 1` is **rejected** (dp-group co-scaling is undefined), and on replay the autoscaler and node pools are rejected unconditionally as unreplayable features.

**Verification:** `cmd/replay_test.go` — `TestINV13_RunReplayParity_PD` verifies that a PD-disaggregated cluster produces matching `RequestTTFTs` and `RequestE2Es` maps when the same requests are run directly vs. through the trace-export-then-replay path. `TestReplayCmd_AutoscalerBundleFatal` and `TestReplayCmd_NodePoolsBundleFatal` verify fatal exit for unsupported features. For MoE DP-as-placement (#1531, #1556), `cmd/dp_replay_parity_test.go` — `TestINV13_RunReplayParity_MoEDPPlacement` asserts that a trace exported by a `dp > 1` run replays with the same flags to byte-identical stdout (per-replica and cluster-aggregate metrics), `TestReplayCmd_MoEDPPlacement_SpawnsReplicas` asserts the `--num-instances × N` expansion plus INV-1 conservation on the replay path, and `TestReplayCmd_MoEDPPlacement_PD_Parity` asserts PD + `dp > 1` replays byte-identically since #1553 (`TestINV13_RunReplayParity_MoEDPPlacement`'s `tp2-ep` parity case covers the #1548 EP-on topology positively); `cmd/dp_placement_test.go` — `TestResolveDPPlacement_MutatesDeploymentVars` and `TestResolveDPPlacement_DenseModelIsInert` pin the shared resolver both commands call (what it writes to the deployment vars, and that a guard error or an inactive plan writes nothing).

**Under lazy generation (#1441, #1442, #1458, #1459, #1460):** run/replay parity must hold whether the run used the eager generator or the lazy streaming source (`--lazy-generation`). Before lazy mode, parity was structural by construction (both code paths consumed the same eagerly-built request list); with lazy mode in play it is enforced only by tests. The authoritative enforcement point is `cmd/parity_run_replay_test.go`: `TestParity_RunReplay_TraceByteIdentity_Matrix` asserts eager and lazy runs export byte-identical traces across a coverage matrix (single-turn chatbot, multi-turn accumulate cohort, single-session reasoning, multi-session reasoning, concurrency clients, and time-varying / per-window workloads), and `TestParity_RunReplay_INV13_BothModes` asserts an eager-sourced and a lazy-sourced trace replay to identical per-request TTFT/E2E. As of #1460 the lazy source streams every workload class — there is no eager-fallback sentinel remaining.

**Evidence:** `cmd/replay.go::replayCmd.Run` wires the same `cluster.DeploymentConfig` field set as `runCmd` (`cmd/root.go`). This parity surface is broader than PD disaggregation alone: it spans PD fields (`PrefillInstances`, `DecodeInstances`, `SharedInstances`, `PDDecider`, `PDPrefixThreshold`, `PrefillScorerConfigs`, `DecodeScorerConfigs`, `PrefillOverrides`, `DecodeOverrides`), encode-stage fields (`EncodeInstances`, `EncodeDecider`), the flow-control fields, tier-shedding (`TierShedThreshold`, `TierShedMinPriority`), GAIE thresholds (`GAIEQDThreshold`, `GAIEKVThreshold`), `TenantBudgets`, and `InstanceLifecycle`. Both call sites carry an `INV-13 SYNC POINT` comment marking the literal that must stay in sync; treat that comment — not a fixed field count — as the authoritative enumeration, since it grows as features are added. Autoscaler and node-pool checks use `logrus.Fatalf` to prevent silent divergence.

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

1. **Read-only, refuse-on-miss.** `cmd.resolveModelConfig` reads the model's `config.json` from the catalog located by `--catalog` / `BLIS_CATALOG` (`<catalog>/models/<short-name>/config.json`, the sole clone-root layout since #1771; #1731 — there is no default, no search path, and a run naming no catalog is refused naming both forms). A model with no entry — or an entry that is not a HuggingFace config — is **refused, naming the path the entry belongs at**. No `blis run` / `blis replay` / `blis observe` invocation creates or modifies a catalog file, and none makes a HuggingFace request.
2. **The deployment is chosen, never inferred.** The deployment comes from a required scenario (`--scenario` / `--scenarios`, a blis-schemas Scenario + Deployment file) plus `--registry` on both `blis run` and `blis replay`; omitting one is refused. The `--hardware`/`--tp` flags that carried this clause before the kernel-exclusive change no longer exist.

**Why this tier:** like INV-9 and INV-A2 this constrains *what the code may do* rather than describing simulation state. A run that writes a catalog entry can leave that run's output perfectly correct while making the catalog — the thing every later run and every calibration reads — drift by accident.

**Rationale:** before #1733, an absent `config.json` triggered a HuggingFace download that was **written into `model_configs/`**, so running an unknown model *added a catalog entry as a side effect* — the exact mechanism by which a catalog silently accumulates unreviewed, unprovenanced models. Independently, the deployment was once looked up per-model in `defaults.yaml` behind a `logrus.Warnf`, so a run could complete and emit metrics for a deployment nobody chose. Both are now refusals (R1: no warn-and-continue).

**Scope and non-scope:** NS-6 binds the **CLI boundary only**. Library callers that construct a `sim.ModelConfig` or call `sim.NewModelHardwareConfig` directly are unaffected and need no flags — the many test files that build configs through the Go API are deliberately outside it. `blis observe` satisfies clause 1 vacuously (it resolves no model config) and takes neither deployment flag (it places no instances).

**Byte-identity (INV-6):** NS-6 changes which inputs are *required*, not any number. A hand-maintained `defaults.yaml` still carrying a removed block is refused at load (strict parsing, R10) rather than parsed and ignored.

**Verification:** `cmd/hfconfig_test.go` — `TestResolveModelConfig_AbsentFromCatalog_RefusedNamingPath` (refusal names the path), `..._CreatesNothing` (nothing written, catalog directory not created), `TestResolveModelConfig_MalformedCatalogEntry_RefusedAndPreserved` (refused, entry byte-preserved), `TestResolveModelConfig_ResolutionInvariant` (one-step catalog resolution with no fallback, #1731). `cmd/ns6_catalog_test.go` — `TestNS6_NoRuntimeFetch_StaticGuard` (removed fetch symbols and any `huggingface.co` string literal are absent from `cmd/`'s production sources; the resolver imports no network package and makes no file-creating call), `TestNS6_ObserveTakesNoDeploymentFlags`. The scenario/registry requirement is enforced by the kernel deployment path shared by `run` and `replay` (INV-13). The INV-6 claim rests directly on `TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline`'s pre-#1733 golden; the anchor test that cross-checked it against `defaults.yaml`'s own GPU/TP keys was removed with those keys by #1768, whose own contracts (the trimmed file parses, a stale `defaults:` block is refused, the deleted surface cannot return) live in `cmd/defaults_block_removal_test.go`.

**ID note:** `NS-6` is numbered by the operator-basis design note reconciling Discussion #1700, not by this registry's `INV-*` sequence; it is listed here because it is cited in `cmd/` and the resolution rule above is about *resolvability*, not about the prefix. Delivered by #1733 (R1 task S6, tracker #1727).

---

## Subsystem invariants

Scoped to one subsystem or one optional feature. Outside that subsystem they are not applicable.

### KV cache

#### INV-4: KV Cache Conservation

**Statement:** `allocated_blocks + free_blocks = total_blocks` at all times.

**Verification:** Checked after every allocation/deallocation. Check-then-act pre-check gate before any state mutation (vLLM parity); post-pre-check `popFreeBlock() == nil` panics (structurally unreachable in single-threaded DES). `FreeBlockCnt` maintained in lockstep by `appendToFreeList`/`removeFromFreeList`. `verifyBlockConservation()` provides independent free-list walk for debug-mode assertions. Every running request holds KV for every token it has computed: a prefix-cache hit is capped at `InputLen - 1` tokens in whole blocks (an offload reload at `InputLen - 1` tokens), so a fully cached prompt -- resumed after preemption or admitted over another request's prefix -- recomputes its last token and claims its blocks (#1943, vLLM v0.31.0 `get_computed_blocks` and `_update_waiting_for_remote_kv`); `TestEveryRunningRequestHoldsKVForItsComputedTokens` (`sim/kv`, rapid) checks it at every allocation, `TestAFullyCachedPromptRecomputesItsLastBlockAndHoldsItsKV` (`sim`) pins both cases.

**Operational note (H8):** KV cache pressure exhibits a sharp cliff, not gradual degradation. In H8's workload, performance was identical above ~2200 blocks and collapsed below it (4.7x TTFT P99 increase with just 4.5% fewer blocks). Below ~1000 blocks, the preempt-requeue cycle can livelock (see R19). Capacity planning formula: `threshold ≈ rate / num_instances × (input_tokens + output_tokens) / block_size`.

**Additional evidence (hardening wave):** Two KV conservation bugs discovered in March 2026: (1) Issue #492, fix #502 — prefill capacity pre-check over-estimated by up to 1 block (partial last-block fill not accounted for), causing false allocation failures that triggered unnecessary preemptions. (2) Issue #501, fix #506 — TieredKVCache CPU→GPU reload could produce an inverted range (`newStart >= endIndex`), causing a slice-bounds panic in block allocation. Both bugs directly affected the allocation/deallocation balance that INV-4 protects. (See also #519 in INV-8 — the range-loop livelock primarily violated the work-conserving property, not block-level conservation.)

#### INV-K3: A Sequence's Bounded State Is Held on Demand and Gated at Admission

**Statement:** After every allocation a request holds exactly `SequenceState.Hold(computed, inFlight)` blocks for its bounded per-sequence state -- window rings, compressor and indexer state, bounded recurrent state -- where `computed` is its tokens computed before the step and `inFlight` those scheduled in it: the kernel's `SequenceBoundedBytesAt` in whole pool blocks (#1941, blis-latency-kernel#46/#56). That is vLLM's on-demand hold: `allocate_slots` allocates a window group's blocks for the scheduled tokens after `remove_skipped_blocks` frees those the window has passed (`single_type_kv_cache_manager.py:178-238, 687-725, 1153-1175` at v0.31.0), so the hold grows through a prefill chunk and settles at the steady `SequenceFixedBytes` while decoding; a shrink frees its blocks before any are taken. The admission cap (`SequenceAdmissionBytes`, `max_admission_blocks_per_request`) is only a gate: a request's first allocation is refused unless its whole prompt and its state at `SequenceState.Gate(prompt)` = each window group at `min(cdiv(prompt), cap)` are free (`full_sequence_must_fit`, `kv_cache_manager.py:517-534`, `single_type_kv_cache_manager.py:216-229`). State blocks hold no tokens or hash, are freed with the request, and are counted in INV-4. A model whose kernel reports no bounded per-sequence state has no `SequenceState`, no gate, and byte-identical output.

**Enforced:** `sim/kv` `KVCacheState.AllocateKVBlocks` (`admissionGateRefuses`, `stateDelta`, `resizeState`), `ReleaseKVBlocks`; the offload tiers check `sequenceFits` before reloading and charge through `chargeSequence`. The functions are `sim/kernelmodel` `Settings.SequenceState`.

**Verification:** `TestSequenceStateHoldsWhatVLLMAllocatesOverALifecycle` (`sim/kernelmodel`: the real DSV4.1 kernel through the store over a prefill -> decode lifecycle; exact hold every step, recycling chunks, steady decode, never the cap); `TestSequenceState_LowerTheAdmissionCeilingByTheFixedTerm` (rapid), `TestSequenceState_ChargedOnceAndFreedWithTheSequence`, `TestSequenceState_RefusalIsSideEffectFree`, `TestSequenceState_DoNotTakeTheCachedPrefixBeingClaimed`, `TestARefusedReloadAdmissionHasNoSideEffects`, `TestAFullyReloadedAllocationHoldsItsSlotsAndStateBlocks` (`sim/kv`); `TestTheBudgetChargesTheKernelsAdmissionReservationInWholeBlocks`, `TestRaisingMemoryUtilizationNeverLowersTheSequenceCeiling`, and the startup-figure golden `TestKVCapacityMatchesTheEnginesLoggedFigure` (vLLM's logged "GPU KV cache size", a request at its admission cap; a startup report, not a runtime hold) (`sim/kernelmodel`).

#### INV-K4: A Speculating Request Holds Its Verify Width and Lookahead

**Statement:** On a speculating instance (`K > 0`), after any allocation a request holds `max(blocks held before, token blocks, ceil(min(end + L, max_model_len) / block))` token and spare blocks besides its INV-K3 state blocks, where `end` is its computed tokens plus everything scheduled this step and `L` is vLLM's `num_lookahead_tokens` for the method (vLLM v0.31.0 `allocate_slots`, `num_tokens_need_slot`, `kv_cache_manager.py:536-538`; a running request's slots are never handed back, `single_type_kv_cache_manager.py:223-228`); a speculating decode schedules and debits `1 + K` of `max_num_batched_tokens` (fewer only when the budget or `max_model_len` leaves less), never less than it advances; every scheduled request also debits the drafter's `max_num_new_slots_for_drafting` from the input budget (`stepBudget`, scheduler.py:682, 825-826, 1073, 1313-1314); a preemption returns what the victim debited, draft slots included. Progress is still the accepted count (INV-K2). A run without speculation reserves nothing extra and is byte-identical.

**Enforced:** `sim/kv` `KVCacheState.AllocateKVBlocks` (`slotTarget`, `reserveSlots`, spare blocks consumed first by `takeBlock`); `sim/batch_formation.go` (`SpecVerifyWidth`, `scheduledTokens`); `cluster.NewInstanceSimulator` derives the reservation from the `SpeculativeConfig` (`SpeculativeConfig.KV`).

**Verification:** `TestSpeculativeSlots_HoldTheVerifyWidthAndLookahead` (rapid), `TestSpeculativeSlots_TokensFillTheReservedBlocksFirst`, `TestSpeculativeSlotsStopAtMaxModelLen`, and `TestEveryRunningRequestHoldsKVForItsComputedTokens` (rapid: the exact hold after every allocation under every store with state, speculation, max_model_len and preemption at once) (`sim/kv`); `TestSpecDecode_ADecodeDebitsItsVerifyWidthFromTheTokenBudget`, `TestSpecDecode_APreemptedDecodeReturnsItsVerifyWidth`, `TestSpecDecode_TheSimulatorSchedulesTheVerifyWidth`, `TestSpecDecode_LookaheadReservationPreemptsEarlierUnderKVPressure`, `TestSpecDecode_ReservationPropertiesAroundSaturation` (rapid; tokens conserved across K), `TestSpecDecode_AcceptanceLengthOneChangesPriceNotProgress`, `TestLookaheadTokensFollowVLLMsNumLookaheadTokens`, `TestSpecDecode_ParallelDraftSlotsDebitTheInputBudget`, `TestDraftSlotsFollowVLLMsMaxNumNewSlotsForDrafting`, `TestSpecDecode_TheSimulatorChargesTheDraftersSlots` (`sim`); `TestInstanceKVReservesTheSpeculativeConfigsSlots` (`sim/cluster`).

#### INV-K6: The Offload Tiers Hold Whole Chunks

**Statement:** With `blocks_per_chunk = b`, the offload tiers behave as follows.
- *Keys.* Each tier entry is one chunk of `b` GPU blocks, keyed by the hash of its last block. The chunk keyspace is therefore a subset of the GPU block hashes.
- *Capacity.* The CPU tier holds at most `cpu_bytes_to_use // (per_block_bytes × b)` chunks and evicts whole chunks in LRU order.
- *Transfers.* Every tier transfer (cascade write, promotion read) moves `b × per_block_bytes`.
- *Loads.* A load (CPU→GPU reload, or the reloadable boundary batch formation credits) starts at the chunk containing the GPU-cached boundary. It ends on a chunk boundary, spans at least one chunk past the boundary or is empty, and the pure query and the commit reach the same boundary.
- *Stores.* A store offers only whole chunks of the offloadable (prompt-clamped) range.
- *b = 1* is block-granular offload, byte-identical to the pre-chunk mechanism.

vLLM v0.31.0 sources:
- `offloading/scheduler.py:418-436` (`update_offload_keys`);
- `:448-479` (`storable_chunks`);
- `:818-821`, `:841`, `:888-891` (lookup start and "less than a chunk" skip);
- `kv_offload/cpu/spec.py:108-121` (chunk sizing).

**Enforced:** `sim/kv/offload_chunks.go` (`tailOf`, `cpuRun`, `loadableBlocks`) through `consultAndReload`, `ReloadablePrefixEnd`, `classifyFetch`, `MirrorToCPU` and the CPU-tier prefix counts; `KVOffloadConfig.CPUChunkCapacity` / `CheckFitsOneChunk` (refused at the CLI, run, replay and per P/D pool).

**Verification:** `TestOffloadChunks_LoadsAreWholeChunksAndTheQueryIsTheCommit` (rapid), `TestOffloadChunks_AlignedPromptReloadsTheSameAtEveryWidth` (metamorphic), `TestOffloadChunks_StoreWholeChunksKeyedByTheirLastBlock`, `TestOffloadChunks_ReloadWholeChunksAndNeverLessThanOne`, `TestOffloadChunks_LRUEvictsWholeChunks`, `TestOffloadChunks_TransfersMoveWholeChunks`, `TestNewOffloadCache_Validation` (`sim/kv`); `TestKVOffloadConfig_CPUTierHoldsWholeChunks` (`sim`); `TestRunCmd_ScenarioOffloadBlock` (`cmd`: `blocks_per_chunk 4` beside a scenario block, run and replay). All existing offload tests run at `b = 1`.

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

### Metrics output

These bind the measured-window aggregates of the `--metrics-path` file (`sim/metrics_measured.go`,
#1938) and the cluster cache hit rate (#1904). The definitions mirror aiperf 0.12.0; the
code comments cite the aiperf lines.

#### INV-M1: Measured-Window Accounting Closes

**Statement:** every request whose `Phase` is measured (not `warmup`) and that was sent before
the horizon ends in exactly one of `measured_completed_requests`; `cancelled_requests` (in
flight when the window closed -- queued, running, or in the P/D pipeline including KV transfer
-- or completed after an explicit `MeasuredWindow.EndUs`); `context_overflow_requests` (prompt,
or prompt plus output budget, past `MaxModelLen`); and `failed_requests` (a client timeout,
which aiperf records as an error; any other drop; admission or routing rejection; gateway shed,
eviction or expiry; decode-KV drop). It holds for colocated and disaggregated runs. The one
open term is a request still in the gateway queue at the horizon. `completed_requests` stays
the run-wide INV-1 term.

**Enforced:** `Metrics.RecordUnserved` beside every instance drop counter, `Metrics.RecordTimeout`
beside every timeout counter (it also marks the row so it is not cancelled too),
`ClusterSimulator.recordUnserved` beside every cluster rejection, shed and decode-KV drop, and
`projectPDMetrics`, which counts a measured parent still in the pipeline at the horizon
(`MeasuredPipelineCancellations`) when it removes its rows.

**Verification:** `TestMeasuredAccountingClosesOverClusterRuns` and
`TestMeasuredAccountingClosesOverDisaggregatedRuns` (rapid, over random phases, deadlines,
oversized prompts, token-bucket admission, tight decode KV and horizons; each run asserts it
exercised the paths it claims); `TestBuildMeasured_InvariantsHoldOverRandomRequestSets`;
`TestEnqueueRequest_UnservedDropsAreCountedByPhase`; `TestBuildMeasured_ATimeoutIsAFailureNotACancellation`.

#### INV-M2: Derived Aggregates Are Exactly Their Definitions

**Statement:** a request's aiperf latency is `FirstTokenTime + ΣITL`, the time of its last
streamed token. vLLM v0.31.0 queues the final token's output (`v1/engine/output_processor.py:745`)
before it frees the request (`:759`) and records finish stats (`:766`), and the chat stream
yields that chunk (`entrypoints/openai/chat_completion/serving.py:828`) before usage and
`[DONE]`, so the latency model's completion overhead, which E2E carries, is not in it.
`request_itl = (latency − ttft)/(osl − 1)` over `osl ≥ 2`; `interactivity_pNN == 1000 /
request_itl_pNN_ms` exactly; `output_tok_per_user_mean` averages `1000/request_itl` over records
with a non-zero ITL only; `output_throughput_tok_s × measured_window_s` equals the measured output
tokens; every `*_pNN` is numpy-linear over one value per measured request. Under time scaling by
`s` the latency statistics and the window scale by `s` and the rates by `1/s`. Duplicating every
request keeps every mean and the extreme percentiles and doubles counts and throughput; interior
linear percentiles are not invariant under duplication (`[0,1]` p90 0.9, `[0,0,1,1]` p90 1.0).

**Verification:** `TestBuildMeasured_PinsTheAiperfDefinitions`, `TestCalculatePercentile_MatchesNumpyLinear`,
`TestMeasured_LatencyEndsAtTheLastTokenNotAtTheCompletionOverhead` (non-zero completion overhead),
`TestBuildMeasured_PerUserMeanSkipsZeroITLRecords`, `TestBuildMeasured_TheWindowEndIsInclusive`,
`TestBuildMeasured_InvariantsHoldOverRandomRequestSets`, `TestBuildMeasured_ScalingTimeScalesLatenciesAndInvertsRates`,
`TestBuildMeasured_DuplicatingEveryRequestKeepsMeansAndExtremes`, and
`TestMetricsPath_DSparkRequestITLIsTheStepIntervalOverTheAcceptedLength`.

#### INV-M3: Cluster Cache Hit Rate Is Prompt-Lookup-Weighted

**Statement:** `cache_hit_rate` is summed prompt-token prefix hits over summed prompt-token
queries, recorded per admission as vLLM v0.31.0 counts them (`v1/core/sched/scheduler.py:1252-1256`,
`v1/core/kv_cache_manager.py:253-262`): a new request's prompt is queried once, a re-admission
after preemption is not counted (`v1/metrics/stats.py:132-143`), and decode growth and decode-pool
admissions query nothing. It is the GPU prefix cache's local rate; `cpu_tier_hit_rate` is the
offload tier's share of the same queried tokens. So the cluster rate lies within the rates of the
instances that queried anything, and adding idle or decode-only instances leaves it unchanged.
`KVThrashingRate` aggregates its own counts the same way.

**Enforced:** `FormBatch` records `ScheduledRequest.Lookup` on a new prefill admission;
`Metrics.recordPrefixLookup`; `aggregateMetrics` divides the sums.

**Verification:** `TestAggregateCacheHitRate_IdleInstancesDoNotDiluteIt`,
`TestAggregateCacheHitRate_AddingIdleInstancesChangesNothing` (rapid),
`TestAggregateCacheHitRateLiesWithinTheInstanceRates` (rapid),
`TestAggregateCacheHitRate_IsTokenWeighted`, `TestAggregateCacheHitRate_TheDecodePoolDoesNotDiluteIt`,
`TestPrefixLookups_APreemptedReadmissionIsNotANewQuery`.

**Evidence:** #1904: four instances with three idle reported 0.2181 where the busy instance's
rate was 0.8723.

#### INV-M4: Only the Phase Decides What Is Measured

**Statement:** adding warm-up requests (with their decode steps and prefix-cache lookups) to a
run's records changes no measured aggregate, `cache_hit_rate` and `cpu_tier_hit_rate` included.
With no phase set and no explicit window the measured set is every completed request:
`step_interval_*` equals the legacy `itl_*`, and `e2e_*` equals stdout's when the completion
overhead is zero. The file's `ttft_*` population is completed measured requests; stdout's (kept
under `whole_run`) also counts a request that emitted a first token and did not complete, so the
two agree only when there is none.

**Verification:** `TestBuildMeasured_AddingWarmupRequestsChangesNothingMeasured` (rapid, with
lookups), `TestBuildMeasured_WithNoPhaseEveryCompletedRequestIsMeasured` (step series, E2E and the
TTFT population), `TestAggregatedMetrics_CarriesTheMeasuredWindowInputs`.

### Autoscaling

#### INV-19: Scale Decisions Carry a Non-Zero Delta

**Statement:** every `ScaleDecision` an `Engine.Optimize` implementation emits has `Delta != 0`. A zero delta is an engine bug, and the pipeline's response is **warn and skip the decision** — the tick continues and the run does not fail.

**Why the warn-only choice is the invariant's point.** The property itself is small; what is worth writing down is the policy. BLIS applies both policies to corrupted internal state: corrupted accounting that every later number depends on fails the run, while a zero-delta decision is logged and dropped. Both are "an internal producer emitted something impossible". The reason to differ is that a zero delta is *inert* — skipping it actuates nothing and leaves cluster state exactly as a correct engine would have — whereas a negative transfer count means the accounting has already diverged and every later number is suspect. Recording that reasoning is what stops the next reviewer from "fixing" the asymmetry in either direction.

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

### Agentic replay

These bind `blis replay --agentic-scenario inferencex-agentx-mvp` (#1937; driver
`sim/workload/agentx_replay.go`), which reproduces aiperf 0.12.0's session-tree replay. They
hold for any corpus, concurrency, window, grace period and service times; the behaviour they
constrain is aiperf's (the code cites its source), and `TestAgentXRuntimeMatchesAiperf`
checks it send for send against aiperf's own runtime.

#### INV-AX1: Agentic Request Conservation

**Statement:** every request the replay sends is injected into the cluster exactly once and
ends exactly once, with one outcome:

| outcome | how it ends | the stream then |
|---|---|---|
| `completed` | a response (non-P/D, or a P/D parent's completion, reported by ID) | continues |
| `error` | a client timeout (aiperf's 6 h default); a cluster refusal -- admission or routing rejection, flow-control rejection or shed, gateway TTL expiry, in-flight eviction, P/D decode-KV drop (`SetUnservedHook`); an engine refusal other than context length | continues (aiperf records an error and dispatches the next turn) |
| `context_overflow` | the engine refuses the prompt, or prompt plus output budget, past max_model_len | stops (non-final turn: trajectory death; a root's tree drains and recycles) |
| `cancelled` | still in flight -- queued, running, in transfer or in the gateway queue -- when the run stops at the end of the grace period | -- (aiperf keeps no record) |

A request the cluster re-injects while in flight (a REDIRECT drain) is the same request, not
a new one. A superseded send (a timer the idle cap moved, a send held behind a barrier, a
timer cancelled at the window end) is refused by the arrival gate before admission and is
never a request, so INV-1 holds over the sent requests unchanged. At the end, the replay's
in-flight count equals the cancelled count, which equals what the cluster still holds. No
profiling request ends after the grace end, and the profiling phase completes no earlier
than the window end (and, as aiperf's does, 10 s after the grace end when requests were
cancelled while branch work was pending).

**Enforced:** `AgentXReplay.Gate` (one confirmed send per dispatch; redirects pass),
`OnComplete` / `OnUnserved` (one outcome, matched by request ID), `Finish` (the cancelled
outcome); `ClusterSimulator.gateArrival` before `injectedByClass`, `notifyUnserved` in
`recordUnserved`.

**Verification:** `TestAgentXInvariantsHoldOverRandomCorpora` (rapid, harness, with random
timeouts, refusals and overflows, and a coverage check that every outcome occurs); through
the real cluster, `TestAgentXThroughPDDisaggregation`, `TestAgentXThroughFlowControlTTL`,
`TestAgentXThroughAdmissionRejection`, `TestAgentXThroughRedirectDrain`,
`TestAgentXContextOverflowEndsTheStream` (sim/cluster) and
`TestReplayAgentX_DeterministicConservingAndTagged` (cmd).

#### INV-AX2: Dependency Causality

**Statement:** a profiling send of turn k of a stream instance happens only after (a) the
same instance's turn k-1 completed (sends of one instance are consecutive turns, one in
flight at a time); (b) every recorded replay-barrier predecessor of the turn in its tree
completed, or was completed history at the lane's t*; and (c) for a root turn gated by
SPAWN_JOIN, every child instance of the gating branches sent its final turn and it
completed. A stream instance sends each turn at most once.

**Enforced:** `submit` / `barrierReady` (barrier), `maybeSuspend` / `childDone` (joins),
`dispatchNext` (closed loop).

**Verification:** `TestAgentXInvariantsHoldOverRandomCorpora`,
`TestAgentXServiceTimesDoNotReorderAStream` (rapid); hand-computed timelines in
`agentx_replay_test.go`.

#### INV-AX3: Lane Bound

**Statement:** each lane holds at most one session tree at a time: a lane's next tree sends
its first request no earlier than the last completion of the tree before it (the tree's root
and every descendant drained), so at most C trees are live, while requests in flight may
exceed C (subagents take no lane).

**Enforced:** `maybeDrain` (aiperf's SessionTreeRegistry) is the only path that recycles a
lane.

**Verification:** `TestAgentXInvariantsHoldOverRandomCorpora`.

#### INV-AX4: Sending Window

**Statement:** warm-up requests are sent with max_tokens 1 and only before the profiling
phase starts (the warm-up's last return); profiling requests are sent only in
[profile start, profile start + duration); every request carries `Phase` `warmup` or
`profile` (`grace` once it completes after the window end).

**Enforced:** `canSend`, `closeWindow`.

**Verification:** `TestAgentXInvariantsHoldOverRandomCorpora`;
`TestAgentXClosedLoopWarmupAndRecycle`.

### Latency model

#### INV-K1: No BLIS-side Draft Cost

**Statement:** For a speculating pool (`engine.speculative.num_spec_tokens = K > 0`), the adapter's step time for a batch equals `max(1, NoOverlap)` of blis-latency-kernel's `StepTime` for the verify-width batch: each decode scheduling `1 + K`, each prefill chunk its own tokens, under a decode threshold of at least `1 + K`. BLIS adds no draft term: the kernel (v0.2.0+) prices the drafter's passes, context stack and head inside `StepTime`, and its weights and caches in `FixedBytes`, so any BLIS-side draft cost would count them twice.

**Enforced:** `sim/kernelmodel` `batchOf` (the only place a BLIS batch becomes a kernel batch) and `Model.StepTime`.

**Verification:** `TestNoDraftCostIsAddedOnTheBLISSide` (property test on the DeepSeek-V4.1-Flash DSpark scenario: BLIS's price equals the kernel's price of an independently built verify-width batch, and that price exceeds the same batch on the same deployment without a drafter); `TestTheDecodeThresholdCoversTheVerifyWidth`.

#### INV-K2: Rejected Drafts Do Not Advance Computed

**Statement:** After a speculative decode step, a request's `ProgressIndex` (the kernel's `ReqShape.Computed` on the next step) advances by the tokens granted as accepted (`1 <= g <= 1 + K`, the mean `1 + α·K` through the fractional carry), never by the verify width. The kernel reads `Computed` for KV reads, ring sizes, sparse selections and drafting eligibility, so a verify-width advance would price every later step at a context the engine rolled back.

**Enforced:** `sim/simulator.go` decode-phase `ProgressIndex += NumNewTokens`, with `NumNewTokens` the granted advance from `FormBatch`; `batchOf` substitutes the verify width only in the kernel's `Scheduled`.

**Verification:** `TestRejectedDraftsDoNotAdvanceComputed` (`sim/kernelmodel`; a real DSpark simulation at α = 0.502); `TestSpecDecode_OutputTokenConservation` (`sim`).

#### INV-K5: The Recorded Acceptance Is the Simulated One

**Statement:** A speculating run's acceptance is exactly one `acceptance.Resolve` resolution (blis-schemas#65) from the registry profile, the workload, the scenario pin and the CLI, highest layer winning, with AL in `[1, K+1]` for the deployment's K and no silent fallback; the results file's `acceptance.length` and `rate` are the `SpeculativeConfig` the simulator advanced requests by (`length = 1 + rate·K`), and `source_layer`/`entry` name where it came from. A run that does not speculate records none. Run and replay under the same flags resolve the same acceptance (INV-13).

**Enforced:** `cmd/speculative_acceptance.go` (`resolveAcceptance`, `acceptanceRecordOf`), `cmd/root.go` `resolveSpeculativeConfig`.

**Verification:** `TestAcceptancePrecedence`, `TestAcceptanceKSweepPicksTheProfileAtEachK`, `TestAcceptanceRefusals`, `TestAcceptanceIsAbsentWithoutSpeculation`, `TestAcceptanceRunReplayParity` (`cmd`).

#### INV-R1: Admission Overhead Runs on the Endpoint's Render Threads, in Arrival Order

**Statement:** With the front end on (`FrontendConfig.RendererNumWorkers = W > 0`; the CLI default is vLLM's `renderer_num_workers = 1`), a request's admission overhead (`LatencyModel.QueueingTime`, the kernel's `AdmissionOverhead`: chat template + tokenization) runs on one of its endpoint's render threads. Each endpoint has `api_server_count` API servers (vLLM's per-mode default: dp behind an internal- or hybrid-balancer replica, else 1) of `W` threads each. Arrivals go to the servers round-robin. Within a server a request starts at `max(arrival, earliest thread free)` in arrival order and holds that thread for its overhead, so a server never runs more than `W` renders at once and a request waits only while all `W` are busy (FIFO M/G/W). The request enters the wait queue when its render ends. Every arrival is rendered exactly once. A render that costs nothing, or a stage with a thread per request, gives `arrival + overhead`, which is the inert front end's uncontended delay; with zero overhead the run is byte-identical to the inert one. An internal/hybrid replica's ranks share one stage. An external-balancer rank and a P/D pool rank each have their own. A P/D decode sub-request is not rendered again.

vLLM v0.31.0: `BaseRenderer._executor = ThreadPoolExecutor(max_workers=renderer_num_workers)` carries the tokenize step (`vllm/renderers/base.py:117-133`) and the V4 renderer's template step (`vllm/renderers/deepseek_v4.py:32-37`), one pool per API-server process. The API-server count defaults are in `vllm/entrypoints/cli/serve.py:105-127`.

**Limitation:** BLIS picks a replica's rank when the request reaches the endpoint, whereas vLLM's balancer picks it after rendering. The cost itself stays in the kernel and registry.

**Enforced:** `sim.Renderer.Admit` (called from `ArrivalEvent.Execute`); `sim.ShareRenderer` from `cluster.buildDPReplicas`.

**Verification:** `TestRenderer_EachServerIsAFIFOQueueOfItsThreads` (rapid), `TestRenderer_FreeOrUncontendedRenderIsTheBareOverhead` (rapid, metamorphic), `TestRenderQueue_ABurstWaitsForTheRenderThread`, `TestRenderQueue_FreeRenderIsByteIdenticalToNoRenderStage`, `TestRenderQueue_EveryArrivalIsRenderedOnce` (`sim`); `TestRenderStage_BelongsToTheEndpoint`, `TestRenderStage_AReplicaRendersOnItsDPServers` (`sim/cluster`); `TestFrontendFlagsDefaultToVLLMs` (`cmd`).

#### Retired latency-model invariants

BLIS prices every step with `blis-latency-kernel`; the in-repo roofline and trained-physics backends were removed (kernel-exclusive change). The invariants below constrained those backends and are **retired**. Latency-pricing invariants now belong to `blis-latency-kernel`.

| ID | Statement (as it was) | Status |
|---|---|---|
| **INV-BC-DP1** | Dense DP=1 trained-physics step time byte-identical across the #1419 DP/EP refactor. | Retired: roofline/trained-physics removed. |
| **INV-L7** | Backend parity: roofline and trained-physics apply an identical LoRA adapter-overhead factor. | Retired: roofline/trained-physics removed; one backend remains. |

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
| **INV-L7** | *Retired* — see [Retired latency-model invariants](#retired-latency-model-invariants). | — | 0 |

Counts are a snapshot over `sim/` and `cmd/`. Re-derive any row with:

```bash
git grep -o 'INV-L4' -- 'sim/**' 'cmd/**' | wc -l
```

**INV-L2's code anchor (#1721).** INV-L2 was documented in the plan and cited nowhere in code — a declared capacity bound with nothing pointing at it. #1721 anchors it at the enforcement point the plan's §7 identifies: `Simulator.maybeStartAdapterLoad` (`sim/simulator.go`) commits the LRU eviction victim and reserves the slot at load **start**, so the reserved slot counts toward capacity for the whole load window. That timing is what makes the bound hold — committing the victim at load *completion* would let a request arriving mid-load resurrect the soon-to-be-evicted adapter and exceed capacity. The structural guarantee is in `sim/lora/resident_set.go`: `Store` admits a new id only when a slot is free **or** an eviction succeeds, and returns `false` when the set is full and every entry is pinned. That conditional — not the commit timing — is what makes the bound hold. What load-start commitment buys is that the `Store` at load completion can never fail, plus exact eviction accounting and the plan's §12 no-deadlock argument. Note the ID is written at the load-start site, so `git grep INV-L2` lands there rather than here.

**INV-L6 has no code anchor.** It is declared in the plan, and the servability paths carry no `OutputTokens` reference, but no test or production site names the ID. That is the *inverse* of the gap the resolution rule addresses (a declared promise with no citation, rather than a citation with no statement), so this registry records it rather than fixing it. Unlike INV-L2, it has no runtime message that would leave a reader stranded.

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

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_MetricProjection_NoSubRequestKeys` checks all six maps for suffix-free keys; `TestDisaggregation_MetricProjection_DroppedParent_NoSubRequestKeys` verifies the suffix clause holds when decode KV allocation fails; `TestDisaggregation_DroppedParent_NotInLatencyMaps` verifies a no-token terminal parent appears in none of the latency maps while remaining counted in `DroppedUnservable` (issue #1511). `TestDisaggregation_MetricProjection_NoOp` verifies the projection is a no-op for non-disaggregated clusters. `sim/cluster/pd_e2e_causality_test.go` — `TestPDParentMetrics_NoTokenExcludedFromLatency` (table over the served/no-token taxonomy) and `TestPDParentMetrics_TTFTSumConsistentAcrossDrop` (the aggregate `TTFTSum == Σ RequestTTFTs` law with a drop present). `sim/cluster/completion_index_test.go` — `TestCompletionIndex_ATimeoutMidGenerationIsNotACompletion` checks, on both the colocated and the PD path, that a metrics-file row has an E2E exactly when it has a `completion_index`, and that the indices are 1..completed.

**Evidence:** `projectPDMetrics()` in `sim/cluster/cluster.go` is called after `aggregateMetrics()` and the conservation correction. It unconditionally deletes the `pfx` and `dec` keys for every parent request, and conditionally inserts a parent-keyed entry only for **served** parents (`served := CompletionTime > 0 && DecodeInstanceID != "" && DecodeSubReq != nil && len(DecodeSubReq.ITL) > 0`). A terminal-but-no-token parent additionally has its prefill TTFT rolled out of the aggregate `TTFTSum`, so `TTFTSum == Σ RequestTTFTs` holds across the no-token terminal outcomes this fix restores (BC-3, issue #1511). This is **not** a general PD law: a preempted-and-re-prefilled decode sub-request re-stamps its own `FirstTokenTime` into `TTFTSum`, and `projectPDMetrics` deletes `RequestTTFTs[dec]` without rolling that decode term back — a pre-existing residual tracked in #1628, out of scope for #1511.

#### INV-PD-6b: CompletionTime Includes PostDecodeFixedOverhead

**Statement:** For all successfully decoded parent requests (`DecodeInstanceID != ""`), `parent.CompletionTime` equals the cluster clock at decode completion plus the decode instance's `PostDecodeFixedOverhead()`. The overhead is the decode pool kernel's `CompletionOverhead()` (a host overhead), so `CompletionTime` exceeds the raw clock by exactly that amount. `parent.CompletionTime` is a lifecycle field (it drives session follow-up scheduling and phase-causality checks); the decode instance's `PostDecodeFixedOverhead` is also carried by the decode sub-request's own recorded E2E (`recordRequestCompletion` adds it, matching the non-PD path).

Note (issue #1513): the client-visible parent E2E metric is **not** `parent.CompletionTime − ArrivalTime`. `parent.CompletionTime` is stamped on the cluster clock at the completion-*detection* tick and omits the decode step's own advance, so that formula under-counts E2E — below a single decode step (`ITL[0]`) and below TTFT for short outputs, violating INV-5. `projectPDMetrics()` instead reconstructs the arrival→last-token span from the decode sub-request's own per-instance metrics, in two clock frames:

- **Primary (non-preempted):** the decode sub-request never ran prefill (`FirstTokenTime == 0`), so its own E2E is schedule-relative → `E2E = decodeSchedulingDelay + decodeOwnE2E`. This guarantees INV-5 **by construction on the primary path** (`decodeOwnE2E ≥ ITL[0]`, so `E2E ≥ TTFT`).
- **Preempted-and-re-prefilled:** preemption resets `ProgressIndex` and re-prefill stamps an arrival-relative `FirstTokenTime`, so `decodeOwnE2E` is already the full arrival→completion span → `E2E = decodeOwnE2E` (discriminated on `FirstTokenTime != 0`; adding `decodeSchedulingDelay` would double-count the pre-decode wait — a regression the guard prevents).
- **Timed out mid-generation:** a parent that **served a token** but whose decode sub-request never recorded its own E2E — it emitted the first token and then timed out — did not complete, so it gets **no** E2E, completion time or `completion_index`, exactly as a non-PD request timed out mid-generation (`simulator.go` records E2E only on completion). It keeps its TTFT, scheduling delay and request row, since the user did receive that token, and is counted in `TimedOutRequests`. (Before #1851 it took `parent.CompletionTime − ArrivalTime`, which put its deadline in the E2E distribution and gave the metrics file a row with an E2E but no completion index.) The three no-token terminal outcomes (decode-side drop at transfer start / transfer complete, decode-timeout-while-queued) are excluded from the projection entirely (issue #1511, INV-PD-6). With no fallback, INV-5 holds by construction for every parent that has an E2E.

`projectPDMetrics` also keeps `RequestCompletionTimes[parentID] = ArrivalTime + E2E` consistent with the projected E2E.

**Verification:** `sim/cluster/disaggregation_test.go` — `TestDisaggregation_CompletionTime_LifecycleField_IncludesOverhead` verifies the lifecycle field directly: `CompletionTime_with_overhead − CompletionTime_without_overhead == overhead` (this pins `parent.CompletionTime = c.clock + PostDecodeFixedOverhead()`; the E2E-metric differential below no longer does, since E2E stopped deriving from `parent.CompletionTime`). `TestDisaggregation_CompletionTime_IncludesNonZeroOverhead` verifies the overhead also flows into the E2E metric (`E2E_with − E2E_without == overhead`, because `decodeOwnE2E` carries the overhead while `decodeSchedulingDelay` is independent of it). `TestDisaggregation_CompletionTime_GeqAllPriorPhaseTimestamps` verifies `CompletionTime >= DecodeEnqueueTime` and `CompletionTime >= TransferCompleteTime` (phase causality preserved). `sim/cluster/pd_e2e_causality_test.go` (issue #1513) verifies the reconstructed E2E satisfies INV-5 (`E2E ≥ TTFT`) for short outputs, is `≥ ITL[0]` and `≥ decodeOwnE2E`, reconstructs the arrival→completion span, keeps `RequestCompletionTimes` consistent, is `≥` the non-PD baseline for 1-token requests, and (via `TestPDParentE2E_ProjectionBranches`) covers the primary/preempted/timed-out/negative-guard branches at unit granularity — including the preemption double-count guard.

**Evidence:** `detectDecodeCompletions()` in `sim/cluster/cluster.go` stamps `parent.CompletionTime = c.clock + inst.PostDecodeFixedOverhead()` (overhead fixed in issue #846). `projectPDMetrics()` computes the parent E2E metric as `decodeSchedulingDelay + decodeOwnE2E` (E2E under-count fixed in issue #1513).

#### INV-PD-7: Prefill KV Hold

**Statement:** With `DeploymentConfig.PDPrefillKVHold` (every kernel P/D run: a NIXL handoff), a finished prefill sub-request's KV blocks stay allocated on its prefill instance from prefill completion until one of these:

- **(a)** The decode side's pull completes (`KVTransferCompletedEvent`).
- **(b)** The decode engine aborts a request it had received: its prompt can never fit the decode KV, so the decode side rejects it. The blocks are freed at once.
- **(c)** The decode engine never received the request (decode instance missing or unroutable at transfer start). No notice comes and no heartbeat renews the lease, so the blocks are freed at `prefill_complete + lease`.
- **(d)** The decode engine had the request and vanished: unroutable mid-pull, or while the request waited for decode KV. The blocks are freed at `max(prefill_complete + lease, vanish + 2/3·lease)`.

In case (d), heartbeats go out every `lease/6` and each extends the lease to `2/3·lease` from its send. So the true expiry lies in `[vanish + lease/2, vanish + 2/3·lease]`, which is 15–20 s after the vanish at vLLM's 30 s default; BLIS takes the upper end.

A prompt that fits the decode KV but finds it full waits in the decode engine's FCFS queue (head-of-line, as vLLM's waiting queue), not dropped, and its prefill KV stays held meanwhile. Decode-side timeouts of a waiting request are not modelled.

The held KV is a subset of what the prefill instance has allocated, equal to it when nothing else is queued or running there. It never exceeds the instance's capacity, and it is fully released by the end of an unbounded run.

**Source:** vLLM v0.31.0:

- `nixl/pull_scheduler.py:247-266`: `request_finished` returns `delay_free_blocks` with a lease deadline.
- `:213-232`: an abort on the decode side before the pull, with `do_remote_prefill` still set, notifies the prefill side to free; a request the decode engine never saw sends nothing.
- `nixl/base_worker.py:2902-3031`: `finished_sending` frees on pull completion.
- `:3056-3096`, `base_scheduler.py:81-87` and `base_worker.py:563-567`: the 30 s lease, heartbeats every `lease//6`, each extending to `2/3·lease`.
- `sched/scheduler.py`: the waiting queue stops at the first request it cannot allocate.

**Verification:**

- `sim/cluster/pd_kv_hold_test.go`:
  - `TestPrefillKVHold_HeldFromPrefillEndUntilThePullCompletes`;
  - `TestPrefillKVHold_HeldBlocksGateTheNextPrefill` (the hold costs capacity);
  - `TestPrefillKVHold_DropAtTransferStartFreesAtOnce` (case b);
  - `TestPrefillKVHold_VanishedDecodeFreesAtLeaseExpiry` (case d, both terms);
  - `TestPrefillKVHold_AmpleCapacityChangesNoLatency` (metamorphic);
  - `TestPrefillKVHold_NeverLeaksAndNeverExceedsCapacity` (rapid: subset, equality, capacity, no leak, INV-1).
- `sim/cluster/pd_decode_kv_wait_test.go`:
  - `TestPDDecodeKVWait_FullDecodeKVWaitsRatherThanDrops`;
  - `TestPDDecodeKVWait_HeadOfLineBlocks`;
  - `TestPDDecodeKVWait_DecodeThatNeverHadTheRequestLeavesTheLease` (case c);
  - `TestPDDecodeKVWait_DrainedDecodeDropsItsWaiters` (case d, from the queue).

**Evidence:** `sim.Request.RetainKVOnFinish` makes `processCompletions` skip the release. `ClusterSimulator.startPrefillHold` and `releasePrefillHold` (`sim/cluster/pd_kv_hold.go`) count each held block once across the holds that share it. `serveDecodeKVWaiters` (`sim/cluster/pd_events.go`) runs the decode KV queue.

#### INV-PD-8: Decode-Side Load Path

**Statement:** A decode sub-request's reservation takes the decode instance's own GPU prefix-cache hit first -- never on blocks another pull is still filling, which join the prefix cache only when that pull lands (`delay_cache_blocks`, `sched/scheduler.py:1216, 3052-3085`) -- (whole blocks, never the prompt's last token, none with prefix caching off) and the handoff moves and is priced for only the remaining blocks (`ParentRequest.PulledKVBlocks`). The decode instance's CPU offload tier is never consulted for that reservation, so under `--pd-decider always` a decode-pool CPU tier changes no latency.

**Source:** vLLM v0.31.0 `v1/core/sched/scheduler.py:525-537, 949` (local hit, then `get_num_new_matched_tokens` for the rest), `v1/core/kv_cache_manager.py:290-299` (hit capped at `num_tokens - 1`), `nixl/pull_scheduler.py:149-177` (pull only the unhashed blocks), `multi_connector.py:134-141, 429-448` (the first connector reporting tokens loads them; NIXL, listed first, claims a remotely prefilled prompt).

**Verification:** `sim/cluster/pd_decode_load_test.go` — `TestPDDecodeLoad_PullsOnlyWhatTheDecodeCacheLacks`, `TestPDDecodeLoad_LocalHitStopsBeforeTheLastToken`, `TestPDDecodeLoad_DecodeCPUTierNeverServesAHandoff` (metamorphic), `TestPDDecodeLoad_InFlightBlocksAreNotAPrefixHit`; `sim/kv/remote_kv_test.go`; `sim/cluster/pd_offload_test.go` — `TestPDOffload_PrefillTierServesALaterTurnsPrefix` (the prefill pool's tier does serve its own prompts).

**Evidence:** `sim.Simulator.ReserveRemoteKV` allocates through `sim.GPUTierAllocator` when the store has one.

### Pools and KV transfer

#### INV-P2-1: Pool-Config Consistency

**Statement:** Per-pool hardware overrides produce a valid `SimConfig` for each pool role: zero-valued `PoolOverrides` is a no-op (backward-compatible), non-nil fields override only the specified fields, and the global `SimConfig` is never mutated.

**Verification:** `sim/cluster/resolve_test.go` — `TestINV_P2_1_PoolConfigConsistency` verifies observable KV capacity differences between pools pre-simulation via `FreeKVBlocks()`; `TestINV_P2_1_RequestConservation` verifies INV-1 holds under heterogeneous pool configuration.

**Evidence:** `ResolvePoolConfig` performs a struct copy and applies only non-nil/non-zero overrides. `resolveConfigForRole` is called in the instance construction loop in `NewClusterSimulator`, before any simulation state is created. The simulator no longer derives KV capacity itself: on the CLI each pool's `TotalKVBlocks` (and `MaxModelLen`, `MaxNumSeqs`, `MaxNumBatchedTokens`) is the kernel's per-data-parallel-rank answer for that pool's engine, carried in as `PoolOverrides`, and a node-pool-placed instance keeps the block count it was configured with. The per-GPU KV re-sizing at placement (#1522) was removed with the in-repo KV auto-calculation (kernel-exclusive change).

#### INV-P2-2: KV Transfer Contention

**Statement:** With `--pd-transfer-contention` (`DeploymentConfig.PDTransferContention`), concurrent handoffs share the links they cross. Each handoff's demand -- payload, floor, and links with their full rates -- is the kernel's `PDTransferDemand` (blis-schemas `kernel.TransferDemand`); `kvtransfer.LinkArbiter` divides every link max-min fairly (progressive filling) among the undrained handoffs on it, re-dividing whenever one starts, drains or finishes. The floor is per handoff and never shared: a handoff completes at the later of start + floor and the tick its last byte drains. Laws: (1) a handoff alone completes at exactly `PDTransferTime` in ticks (the schemas law `PDTransferTime == PDTransferDemand.AloneTime()`, checked for every rank pair at startup and every handoff at run time); (2) no link carries more than its rate, and every undrained handoff has a saturated bottleneck link on which none is faster (max-min fairness); (3) handoffs on links nobody else uses finish exactly as alone; (4) doubling every link's rate never delays a handoff; (5) on one shared link set (every 1P1D handoff), no handoff is faster than alone and one more handoff never makes another finish earlier; (6) INV-PD-3 and INV-1 hold, at any horizon; (7) deterministic (INV-6), and run → replay byte-identical (INV-13). Max-min fairness is not monotone across different link sets in general (more traffic on one link can speed a flow on another); `TestLinkArbiter_MaxMinIsNotMonotoneAcrossLinks` pins the counterexample so no law claims it. Off (the default), every handoff is priced as if alone, byte-identical to before.

**Verification:** `sim/kvtransfer/links_test.go` (laws 1-5, 7; rapid; `TestLinkArbiter_UnchangedRateKeepsItsAccounting` pins law 3's exactness by construction), `sim/cluster/pd_transfer_contention_test.go` (one-at-a-time identical, shared decode NIC doubles each, disjoint links identical, never faster than alone + INV-PD-3 + INV-1 at a horizon, refusals, wakes bounded by starts + finish ticks), `cmd/kernel_pd_fidelity_test.go` (CLI, run → replay).

**Evidence:** `scheduleTransferCompletion` starts the handoff in the arbiter; `PDLinkWakeEvent` completes finished handoffs through the same `KVTransferCompletedEvent` an uncontended one takes. The handoff's queueing delay is its duration minus `TransferAloneTicks`, reported per run (`pd.transfer_queue_delay_*`).

**Coverage limit:** the kernel's demand is one rank's pull at one rank's link rate, exact when the pools' TP widths match (blis-latency-kernel `pdTransferBytes`). NIXL's transfers do not share a LinkID with offload-tier I/O (the kernel prices those over the host link with `TierTime`), so handoffs and CPU-tier traffic do not contend.
