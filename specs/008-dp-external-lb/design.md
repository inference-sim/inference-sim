# DP external load balancing — per-rank router-visible endpoints

Design source: inference-sim#2699 (agreed RFC, rev 3). Tracking issue carries the live copy; this is the committed snapshot for the feature branch.

---

Addresses #2233. Follows from reviewer re-verification on #2688 of the module-boundary
evidence below, which confirmed an RFC is needed before implementing #2233 (CLAUDE.md:
"An RFC... precedes implementation for changes that introduce/modify module boundaries").

> **Revision note.** Rev 1 (addressing [the first review](https://github.com/inference-sim/inference-sim/issues/2699#issuecomment-7168721250))
> corrected the modeling justification (the ranks are already separate engines, so the views
> lose no skew), added holes for per-rank KV events and for the leader/member driver, covered
> P/D (both refusals), fixed the `EndpointRanks` wiring and the `sim/cluster` import rule, and
> named the proof run. Rev 2 (addressing [the re-review](https://github.com/inference-sim/inference-sim/issues/2699#issuecomment-7169414354)
> and the [M3.2.1 scope change](https://github.com/inference-sim/inference-sim/issues/2699#issuecomment-7169512014))
> withdrew the false claim that `kickGroup` was already rewritten for in-flight batches and
> made that rewrite an explicit H2 contract (BC-H2-5), set `StepDue` true when *any* rank has
> work (BC-H1-3), gave the proof run a numeric hit-rate gate and definition, and moved the
> feature **into** M3.2.1 as a stacked PR on #2688 merged before the tag. Rev 3 (addressing
> [the rev-2 re-review](https://github.com/inference-sim/inference-sim/issues/2699#issuecomment-7169618799))
> restated BC-H1-1 as per-rank KV *ownership* (a lockstep group cannot drive a single-rank
> step) and recorded in H2 that per-rank KV events need no router-fork change (in-process
> subscription per `inst.ID()`) and saturation needs only each instance's own metrics. See the
> reply comments for the point-by-point diffs.

## Feature Description

**What:** Let the host run a data-parallel (DP) deployment under *external* load balancing
— where the router, not vLLM's internal balancer, picks which DP rank serves each request
— by giving the router one routable endpoint per rank, instead of collapsing the whole DP
group into a single host-side instance.

**Why:** Today `cmd/engine_driver_build.go` refuses `parallel.data_parallel_load_balancing:
external` for any engine that runs its DP group through the engine contract
(`logrus.Fatalf`, "does not drive parallel.data_parallel_load_balancing external yet"), and
`engines/vllm/core/build/policy.go`'s `requireOneEndpoint` (#2512) refuses the same mode on a
P/D pool — a scenario modeling this deployment shape cannot run at all. Where it is
approximated by routing all requests to the single collapsed-group instance, the router sees
one endpoint for the whole group and cannot express per-rank prefix-cache locality: measured
evidence from a production-like P/D deployment running `--data-parallel-multi-port-external-lb`
(router sees one endpoint per rank) shows prefix-cache hit rate at 0.98-0.998, against 0.74
from BLIS's current collapsed-group modeling of the same workload. Closing that gap needs the
router to see, and route across, individual ranks — and needs each rank to publish its own KV
events, which today only rank 0 does.

**How it behaves:** A scenario with `parallel.data_parallel_load_balancing: external` stands
up N router-visible endpoints for an N-rank DP group instead of refusing to build. Each
endpoint admits, schedules, reports metrics and publishes KV events for its own rank; the
router picks the rank. The group still steps in lockstep as a unit (one scheduled step per
wave, dummy batch, all-reduce) — external balancing changes *who the router sees*, not how
vLLM synchronises the ranks. No new CLI flags — this fills in behavior an existing scenario
field already declares but the host currently rejects. (Hybrid is out of scope — see below;
this RFC does not build hybrid-shaped endpoints. A `dp_local=1` hybrid deployment resolves to
external at `ResolveDPLoadBalancing` and is covered by the external path; every other hybrid
shape is deferred.)

**Scope in:**
- External load balancing for a DP group, colocated **and** P/D: N router-visible endpoints,
  each one rank's view of the shared group, for the colocated pool and for each of a
  disaggregated deployment's prefill and decode pools.
- Removing **both** refusals for the external case: `cmd/engine_driver_build.go` (colocated)
  and `requireOneEndpoint` in `engines/vllm/core/build/policy.go` (P/D).
- Per-rank KV events: under external balancing every rank publishes, and rank k's events
  reach only endpoint k, so the router's prefix index sees all ranks (not just rank 0).
- Per-rank metrics and admission visibility sufficient for prefix-aware routing to measurably
  raise cache-hit rate over the current single-collapsed-instance modeling.

**Scope out (deferred — separate RFC/issue if a later coverage census needs it):**
- Hybrid load balancing, where an endpoint is a node of several ranks (generalizing the
  instance-per-rank assumption to instance-per-endpoint-of-ranks). #2233 estimates this at
  ~6h on top of the external case, but it is a separate module-boundary question (what
  `PickRank` means when an "endpoint" already spans ranks) and is not needed to close the
  cache-hit gap this RFC targets, which is specifically the external, multi-port case. The
  only hybrid shape in scope is the degenerate `dp_local=1` one that already resolves to
  external.

**Modeling decisions:**
- A DP rank's engine is **already a separate engine** in BLIS today: `DPGroup` builds one
  engine per rank, each with its own scheduler, KV cache, batch formation and metrics
  (`engines/vllm/core/vllm/dpgroup.go:128-140`), matching vLLM's `DPEngineCoreProc`
  (one scheduler and KV cache per rank, ranks syncing the step and the wave via dummy batch
  and all-reduce). Exposing each rank as its own router-visible endpoint therefore loses **no**
  inter-rank skew — the per-rank engines already exist; the external-balancing path exposes
  them individually instead of flattening them behind rank 0. This is why N per-rank views
  keep the lockstep, padding, decoder replay (#2681) and EP pricing exact.
- The group is stepped **once per lockstep step** by a single leader (rank 0's driver), and
  that step's batch and result are split across the per-rank drivers so each keeps its own
  report. This is a modeling choice, not a simplification of fidelity: it mirrors vLLM
  synchronising the ranks at the step and wave boundary, and keeps all pricing inside
  `substrate/kernel/`'s existing group step-time accounting rather than forking a per-rank
  cost path (see Trade-offs). What a true independent-per-rank simulation (separate step
  clocks diverging freely) would add — free inter-rank clock drift within a wave — vLLM does
  not exhibit under DP, because the ranks synchronise every step; so nothing real is omitted.

## Baseline Analysis (Step 0)

Traced by hand rather than `archon-go` — the question is about specific call sites, which a
targeted grep answers more precisely than the health/impact view:

- `EngineDriver.JoinGroup` (`sim/engine_driver_group.go:18`) and the leader/member
  group-step path it drives (`kickGroup`, `splitGroupBatch`, `splitGroupResult`,
  `ShareAdmissionStage`, `PickRank`) are **present in `sim`** but have zero non-test callers:
  nothing in `sim`/`cmd` calls `JoinGroup`. The path is dormant for **two** reasons, both of
  which this RFC must close.
  - First, no engine implements the leader's `groupEngine` interface
    (`StepDue`/`LockstepSteps`/`LastActive`, `sim/engine_driver_group.go:156-160`), so a leader
    reaches `kickGroup`'s panic. That interface was written against the retired compatibility
    engine's `RankView` — H1 supplies an implementation for the live engine.
  - Second, **`kickGroup` as written models the compatibility engine's step semantics, not the
    current engine's**: it calls `Schedule` once and `Complete` on that batch immediately,
    applying the result *at the step's start* (its own comment says so, `:164`) — the
    `ApplyAtStepStart` branch, which no non-test code sets (`git grep ApplyAtStepStart`). The
    normal driver's `kick` (`sim/engine_driver.go:717-751`) instead schedules batches while the
    engine admits them, keeps them **in flight**, and completes each at its step end
    (`driverStepEnd`); the current `DPGroup` engine uses that in-flight model. So activating the
    leader path requires **rewriting `kickGroup` to the in-flight model** (scheduled below as
    BC-H2-5), not merely supplying a `groupEngine`. The path is the right mechanism, but it is
    not yet rewritten for the live engine — the earlier draft's claim that it already was is
    withdrawn.
- `DPGroup`'s endpoint surface (`AddToEndpoint`/`Endpoints()`/`EndpointRanks(e)`, in
  `engines/vllm`, exercised by `engines/vllm`'s own tests) has zero callers in `sim`/`cmd` —
  built engine-side, never wired host-side.
- `cmd/engine_driver_build.go:38-42` refuses external colocated DP with `logrus.Fatalf`
  exactly where it would otherwise build a per-rank-endpoint view.
- `requireOneEndpoint` (`engines/vllm/core/build/policy.go:99-112`, #2512) refuses external DP
  on a P/D pool, judged on the mode `ResolveDPLoadBalancing` resolves.
- `DPGroup.EnableKVEvents` (`engines/vllm/core/vllm/dpgroup.go:699-707`, #2677) enables rank
  0's KV events **only** — correct for the internal balancer (one endpoint, rank 0's pod) but
  wrong for external, where ranks 1..dp-1 would publish nothing and the router's prefix index
  could not see them.
- `sim/cluster/dp_replica.go`'s `buildDPReplicas` (`DPLockstepRanks`) registers N separately-
  built `GatewayHost` instances for MoE lockstep replicas and coordinates them at the cluster
  level (`eventMembers` advances the whole replica on one rank's event). It *groups* instances;
  it does **not** step a single shared group engine — that is the driver-level job above.
- `sim/cluster/dp_replica.go:116-121` (`endpointReplica`) already returns "this instance alone"
  (nil replica) when `sim.DPEachRankIsAnEndpoint` holds, i.e. under external balancing every
  rank is its own endpoint — the read side this RFC's `EndpointRanks` wiring relies on.

## Holes

### H1: `engines/vllm` — per-rank endpoint view engine + the leader's group-step surface

- **Responsibility:** Expose one `DPGroup`'s N already-separate rank engines as N independent
  `Engine`-contract views (one per endpoint), and give rank 0's view the surface the `sim`
  leader uses to step the shared group once per lockstep step.
- **Surface:**
  - A rank-scoped `Engine` per endpoint: `Add` on endpoint k's view routes to that rank's
    engine only (via the existing `AddToEndpoint`); each view reports only its own rank's
    metrics, snapshot, KV events and batch plan.
  - Rank 0's view additionally satisfies the leader's group-step surface (`StepDue`,
    `LockstepSteps`, `LastActive`) so the existing `sim` leader/member path drives it; and
    `PickRank` reports the rank the balancer would pick given per-rank in-flight counts (used
    only when the host, not the router, balances — a no-op for external).
  - `EnableKVEvents` becomes mode-aware: under external balancing it enables **every** rank,
    and each rank's emitted events carry its rank index so they reach only endpoint k
    (llm-d subscribes each rank endpoint at the socket port + rank-index offset,
    `extractor.go:71`; vLLM offsets each rank's KV-events port by its rank). Under internal/
    unset it stays rank-0-only, exactly as #2677.
  - The build-time `requireOneEndpoint` refusal is lifted for the external case (colocated and
    P/D), since the engine now supports one endpoint per rank.
- **Allowed imports:** the existing `DPGroup`/group-state and `engine` contract types it
  already wraps; no new dependency on `sim`/`cmd`.
- **Contracts:**
  - BC-H1-1: a request's KV lands only on the rank that holds it — adding a request on
    endpoint k's view grows rank k's KV occupancy and no other rank's. (Restated from the
    earlier "only one rank's step": the group steps every rank together per `LastActive`, so a
    single-rank step cannot be driven; the testable property is per-rank KV *ownership*, not
    per-rank stepping.) [evidenced: property_test — add a request on rank k, step the group,
    and assert no rank j≠k gains KV from it]
  - BC-H1-2: with requests assigned to ranks exactly as the internal balancer assigns them
    (the no-op / single-endpoint case), stepping the group through rank 0's view reproduces
    the current collapsed-instance step timeline (step times, finish times, per-request token
    counts); the per-endpoint reports are that timeline partitioned across the N views and
    sum/merge to the collapsed path's aggregate metrics. (Under external routing the router's
    assignment differs, so per-rank batches and the timeline legitimately differ — that is the
    feature, not a regression.) [evidenced: differential_test against the current
    collapsed-group engine path under a fixed internal-equivalent assignment, on aggregate
    metrics and per-request token timings]
  - BC-H1-3: rank 0's view satisfies `groupEngine` such that the `sim` leader steps the group
    exactly once per lockstep step (`LockstepSteps` advances by one per `kickGroup`), with
    `LastActive` naming the ranks that ran. **`StepDue(now)` is true whenever *any* rank has
    work, not only rank 0** — a member's arrival while rank 0 is idle calls `kickMember`, which
    steps the leader only when `StepDue` holds (`sim/engine_driver_group.go:270-277`); if
    `StepDue` tracked rank 0 alone, a group with work only on ranks 1..dp-1 would stall.
    [evidenced: differential_test against the collapsed path's step count for an identical
    workload, plus a property_test that a step fires when only a non-zero rank has work]
  - BC-H1-4: under external balancing every rank publishes KV events, and an event emitted by
    rank k is tagged with index k so it is deliverable only to endpoint k; under internal/
    unset only rank 0 publishes. [evidenced: property_test — for each rank k, the events it
    emits carry index k and no other, across a multi-rank step]
  - BC-H1-5: a rank's view never observes a step before the group's shared clock has advanced
    to it (no time travel across ranks). [evidenced: property_test]
- **Invariants:** INV-3 (clock monotonicity) extended across views of one group; INV-6
  (determinism) — same resolved input at the same revisions ⇒ byte-identical per-endpoint
  reports *for the same topology* (N views), independent of iteration order.
- **Extension type:** backend swap (alternative `Engine`-contract presentation over existing
  `DPGroup` state) plus the `groupEngine` implementation the dormant leader path needs.
- **No-op default:** `parallel.data_parallel_load_balancing` unset or `internal` — the group
  is built and exposed as one instance exactly as today; H1's per-endpoint views, mode-aware
  KV events and the lifted refusal are reached only when `external` is requested (or a
  `dp_local=1` hybrid that resolves to external). Byte-identical output otherwise.

### H2: `sim` + `sim/cluster` — rewrite the leader step for in-flight batches, register N endpoints

- **Responsibility:** Rewrite the dormant `sim` leader step (`kickGroup`) to the current
  engine's in-flight-batch model, and register one `InstanceSimulator` per endpoint view (built
  at cmd, H3) as N router-visible endpoints wired through that leader/member path — one leader
  steps the shared group while each endpoint applies its own requests and results — for the
  colocated pool and for each P/D pool.
- **Surface:**
  - In `sim`: rewrite `kickGroup` so the leader schedules the group's step and keeps its batch
    **in flight**, completing it at the step's end (`driverStepEnd` semantics) and scheduling
    the next step while the current is in flight — mirroring the normal driver's `kick`
    (`sim/engine_driver.go:717-751`) and the live `DPGroup` engine, rather than the
    compatibility engine's schedule-then-complete-at-once (`ApplyAtStepStart`) path it holds
    today. The per-driver batch/result split (`splitGroupBatch`/`splitGroupResult`) is reused.
  - In `sim/cluster`: given N already-built per-rank `InstanceSimulator`s for an
    externally-balanced group (constructed at cmd's single vLLM site, H3), each reachable at its
    own `InstanceID`: register them as N endpoints; make rank 0's driver the leader and
    `JoinGroup` the rest as members; `ShareAdmissionStage` the ranks of one endpoint-set so a
    request is admitted once before the rank is chosen (INV-R1); and have `EndpointRanks(id)`
    return **the single instance** for that endpoint id (each rank is its own endpoint under
    external balancing), which is already the non-grouped behavior — so `GatewayHost`/`sim/llmd`
    need no change. For a P/D deployment, do this for the prefill and the decode pools
    independently.
  - Per-rank KV events need **no router-fork change**: the gateway subscribes each
    `InstanceSimulator` individually (`subscribeKVEvents`, keyed by `inst.ID()`,
    `sim/llmd/kvevents.go:29-55`) and calls the router's ingest **in-process**
    (`g.host.Schedule`), not over ZMQ topics. So once H1 enables every rank's events, the N
    rank instances simply produce N event streams the gateway already knows how to subscribe
    and deliver. Saturation needs nothing beyond each instance's own metrics either: the
    router's saturation detectors are per-endpoint plugins (`utilization-detector`,
    `concurrency-detector`, `max-saturation-detector`, `sim/llmd/support.go:74-77`) that read
    each endpoint's scraped metrics, which the gateway renders per `InstanceID`
    (`sim/llmd/gateway.go:221`); so N rank instances give N metric sets with no extra work.
- **Allowed imports:** `sim`, `sim/cluster` (the existing `GatewayHost`, `InstanceSimulator`,
  the `dp_replica.go` registration pattern, and the `sim` leader/member surface — `JoinGroup`,
  `ShareAdmissionStage`, `kickGroup`, the in-flight machinery `kick` already uses). It works
  entirely through the `sim.EngineDriver`/`engine.Engine` contract and does **not** import
  `engines/vllm` (per CLAUDE.md the vLLM engine is imported by non-test `sim`/`cmd` code only at
  the single cmd build site).
- **Contracts:**
  - BC-H2-1: the router sees exactly N endpoints for an N-rank externally-balanced group, each
    individually admitting, scheduling and reporting KV events, and `EndpointRanks(id)` returns
    the one instance for each. [evidenced: differential_test against the non-grouped
    single-instance registration behavior in `dp_replica.go`]
  - BC-H2-2: one leader steps the shared group and the step's batch/result reach each endpoint's
    own driver (each endpoint's report counts only its own requests). [evidenced:
    property_test — the leader's `kickGroup` is the only driver that calls the group's
    `Schedule`/`Complete`; members' requests appear only in their own reports]
  - BC-H2-3: removing the two refusals does not change behavior for any scenario that does not
    request external DP load balancing. [evidenced: metamorphic_test — rerun the full existing
    scenario suite, byte-identical `Result`]
  - BC-H2-4: a request is admitted once per endpoint before the rank is fixed (shared admission
    stage), so N endpoints neither create nor drop requests relative to one collapsed instance.
    [evidenced: property_test tied to INV-1 over a multi-rank run]
  - BC-H2-5: the rewritten `kickGroup` keeps the group's step batch in flight and completes it
    at the step's end (and schedules the next step while one is in flight), so for the
    internal-equivalent assignment the leader path's per-step schedule/complete timeline equals
    the normal driver's `kick` over the same engine — it does **not** apply results at the
    step's start. This is the contract BC-H1-2 depends on; without it the leader path models the
    compatibility engine, not the live one. [evidenced: differential_test — leader-path step
    start/end times and in-flight depth equal the collapsed single-instance `kick` path for an
    identical internal-equivalent workload]
- **Invariants:** INV-1 (request conservation — N endpoints must not create or drop requests
  relative to one collapsed instance); INV-6 (determinism, within a fixed topology); INV-13
  (run/replay parity — a trace captured against N rank-endpoints replays to the same per-rank
  metrics).
- **Extension type:** subsystem module (extends the existing cluster/router integration and
  activates the dormant `sim` leader/member path; no new top-level package).
- **No-op default:** identical to H1 — internal/unset load balancing builds the single
  collapsed instance exactly as today, and the leader/member path is wired only for an
  externally-balanced group.

### H3: `cmd` — build the per-rank views at the single vLLM site and lift the colocated refusal

- **Responsibility:** At the one site CLAUDE.md permits a non-test `cmd` import of
  `engines/vllm` (`cmd/engine_build.go`), construct the N per-rank endpoint views (H1) and the
  N `InstanceSimulator`s, hand them to H2's registration, and lift the external refusal in
  `cmd/engine_driver_build.go`.
- **Surface:** when a deployment's engine config resolves to external DP load balancing, build
  one endpoint view per rank over the shared group (instead of the single collapsed instance),
  set the instance count to replicas × dp and `DPLockstepRanks`, and pass the instances to H2;
  do not `logrus.Fatalf` the external case. Keep the single collapsed-instance build for
  internal/unset. This is the only hole that imports `engines/vllm`.
- **Allowed imports:** `sim`, `sim/cluster`, and `engines/vllm` (the single permitted site).
- **Contracts:**
  - BC-H3-1: for internal/unset load balancing the build is byte-identical to today (one
    collapsed instance, one endpoint). [evidenced: metamorphic_test — existing scenario suite
    byte-identical]
  - BC-H3-2: for external load balancing the build produces N endpoint views sharing one group,
    registered through H2, for the colocated pool and for each P/D pool. [evidenced:
    differential_test — endpoint count equals replicas × dp for colocated, prefill and decode]
- **Invariants:** INV-6 (determinism); the no-op default (BC-H3-1) is the byte-identity
  guarantee.
- **Extension type:** integration wiring (no new package; the single engine-construction site).
- **No-op default:** internal/unset — the collapsed single-instance build, unchanged.

## Trade-offs

| Decision | Alternatives | Why this approach | What breaks if wrong |
|---|---|---|---|
| **Reuse the `sim` leader/member driver path** (`JoinGroup`/leader step), implementing `groupEngine` on H1's rank-0 view (H1) **and rewriting the leader step for in-flight batches** (H2/BC-H2-5), so one leader steps the shared group and each endpoint applies its own share | (a) Have each endpoint view step the group independently — but that steps the group N times per lockstep step, double-counting the wave; (b) coordinate only at the cluster level like `buildDPReplicas` — but that *groups* N separate engines and never steps a single shared group; (c) write a fresh group-step driver from scratch — but the member split (`splitGroupBatch`/`splitGroupResult`), admission sharing and wake logic already exist and are correct | The ranks are one synchronised group with one lockstep step per wave; exactly one driver must trigger that step and split its batch/result per rank. The member-split/admission/wake machinery is already correct; only `kickGroup`'s step model is wrong (it applies at step start, the compatibility-engine path). So the work is bounded: implement `groupEngine` (H1) and rewrite the one step method to the in-flight model `kick` already uses (H2/BC-H2-5), not revive a whole subsystem and not avoid the path (`buildDPReplicas` cannot step a shared group) | If the in-flight rewrite (BC-H2-5) is skipped or wrong, the leader models the compatibility engine, not the live one, and BC-H1-2 (match the collapsed `kick` path) fails. This is why BC-H2-5 is an explicit contract with its own differential test, not a residual risk |
| **Model each rank as a per-rank view over the one shared group**, not an independently-simulated engine with its own copy | Simulate N fully independent engines with free inter-rank clock drift | The ranks are already separate engines inside `DPGroup` (own scheduler/KV/metrics) but they synchronise every step and wave; a view exposes each rank's real engine without flattening it, so no skew is lost, while the group still steps as a synchronised unit (as vLLM does) | If real DP ranks were *not* step-synchronised (they are), the view would under-model free drift — but vLLM's dummy-batch/all-reduce wave makes them synchronised, so this is faithful |
| **Scope out hybrid** (endpoint = node of several ranks), keeping only the `dp_local=1` hybrid that resolves to external | Do all three of #2233's parts (view, external wiring, hybrid) in one RFC | The measured 0.74-vs-0.98 gap is specifically the external, multi-port case; hybrid redefines what `PickRank`/an "endpoint" means at the routing boundary and deserves its own trade-off discussion once external lands and is validated | If a near-term workload needs true hybrid, it is a follow-up RFC building on H1/H2's endpoint-view engine rather than from scratch |
| **Price each endpoint view's KV/compute as the group's existing cost-model accounting** split per rank, not a new per-rank cost model | Build a separate cost-model entry point for rank-views | `substrate/kernel/` is the only latency backend; the group step time already prices the lockstep/padding/dummy-batch/EP cost (#2681), and the views must stay inside it, not fork a parallel pricing path | A separate pricing path would silently diverge from the single-instance case and break BC-H1-2's differential test |

## Validation / proof run (BC-H2-3 → the named run)

- **Scenario:** a DP-8 GLM-style **P/D** deployment (1 prefill pool + 1 decode pool, the group
  dp=8) run twice on the identical workload: once with
  `parallel.data_parallel_load_balancing: internal` (today's single-collapsed-instance path,
  the baseline) and once with `external` (this RFC's N-endpoint path), prefix-aware routing on.
- **Metric definition (declared before the run):** prefix-cache hit rate =
  (Σ cached prefix tokens over all completed requests) ÷ (Σ total prompt tokens over all
  completed requests), token-weighted, aggregated across the group's N endpoints over the run
  — the quantity BLIS already prices as the cached share of a request's prompt (`CachedTokens`
  reflected in `Computed`), reported in the committed `Result` field the run names.
- **Numeric gate (pre-registered):** the external run's hit rate ≥ **0.90** *and* ≥ internal
  baseline + **0.15** absolute (baseline is ≈0.74 for the collapsed path on this workload).
  That is a directional target toward the 0.98-0.998 production range, not a match (0.98-0.998
  reflects a real deployment). The run commits both numbers as a before/after pair; if it does
  not clear the gate the actual figures are reported as-is rather than tuned to pass (a
  negative result is a valid outcome, not a reason to refit).
- **Oracle:** #2652 (`oracle: run external data-parallel balancing through vLLM's
  `DPAsyncMPClient`, mocks.json entry at the next golden refresh`) is the real-engine oracle
  for the per-rank behavior; this RFC's directional gate stands until that golden refresh lands
  and lets the per-rank timeline be checked against the oracle.
- **Estimate:** ~11h for the inference-sim external piece (revised up from #2233's ~8h to add
  the `kickGroup` in-flight rewrite the earlier draft had mis-stated as already done), split
  H1 ≈ 4h (endpoint view + `groupEngine` + mode-aware KV events + P/D refusal lift) + H2 ≈ 6h
  (the `kickGroup` in-flight rewrite with its differential test, BC-H2-5, ≈ 3h; registration +
  leader/member wiring + `EndpointRanks` ≈ 3h) + H3 ≈ 1h (cmd construction + colocated refusal
  lift), plus the proof run.

## Delivery Order

```
H1 (engines/vllm endpoint-view engine + groupEngine + per-rank KV events + P/D refusal lift)
   — no dependencies on sim/cmd, can start first
H2 (sim kickGroup in-flight rewrite + sim/cluster N-endpoint registration + leader/member wiring)
   — the kickGroup rewrite and registration are testable against a stub groupEngine in parallel
     with H1; needs H1's engine for the end-to-end proof run
H3 (cmd: build the views at the single vLLM site + lift the colocated refusal)
   — depends on H1 (the view engine) and H2 (the registration + leader step)
```

Three holes. `engines/vllm`, `sim` and `sim/cluster` are directories of **this** module, so
this is one repository and one review set delivered in dependency order (engine hole → sim/
cluster hole → cmd wiring) — not a cross-repo release.

**Release train (per the scope change in the thread):** this feature ships **inside M3.2.1**,
as a **stacked PR on #2688's branch (`m321/dp-prefill-kv-fidelity`), merged before the M3.2.1
tag** — #2688 stays as-is and is approved on its own. The stack is based off #2688's branch
rather than `main` to stay compatible with #2688's `EnableKVEvents` change that H1 extends.
**Done means:** the RFC points above are resolved; the stacked PR carries the H1-H3 contracts
as tests; the DP8 P/D external-balancing proof run clears the pre-registered numeric hit-rate
gate above; and CI is green. The next step is encoding these holes into a `.archon` plan with
sub-issue 0 (feature branch + plan, no implementation code) per
`.github/agent-prompts/templates/rfc-to-plan.md`.
