# LoRA Control-Plane Subsystem

**Status:** Draft (pending convergence review)
**Date:** 2026-07-15
**Feature area:** `sim/` + `sim/cluster/` + `sim/latency/` — new `sim/lora/` module
**Species:** System Overview (multi-PR feature spanning identity, memory, latency, routing, metrics)
**Spec-kit artifacts:** `specs/007-lora-control-plane/` (spec, plan, research, data-model, contracts, tasks)
**Tracking:** epic `inference-sim/inference-sim#1464`; PRs #1471, #1465–#1470
**Source rationale:** `~/Projects/blis/lora-control/docs/blis-lora-extension.md`

---

## 1. Motivation

BLIS today has no notion of a LoRA adapter: every request runs against a single base model, and instances have no finite adapter capacity, no adapter-load cost, and no adapter-aware routing. That makes it impossible to study the questions LoRA-serving research actually asks — where to place adapters, when to migrate them, and how to route requests so hot adapters stay resident. This design gives BLIS the minimum adapter awareness needed to develop and evaluate those placement/routing policies in a fast, deterministic, GPU-free harness, so a policy tuned against BLIS transfers to llm-d.

The design principle (from the source rationale) is deliberately narrow: **keep BLIS's calibrated engine; port only the Digital Twin's adapter *physics* (three cost terms) plus an adapter identity that rides BLIS's already-tested `Model`-tag plumbing.**

---

## 2. Scope

**Extension-Framework classification.** This feature is a **Subsystem Module** (a new module with its own state, events, and behavioral contract — not a Policy Template, Backend Swap, or Tier Composition), which is why a design doc and a no-op default are mandatory. It carries *one nested* **Policy Template** — the LoRA-aware routing scorer (§11) — layered on the module's resident-set signal. Everything below follows the Subsystem Module recipe (design-guidelines §5.3).

**In scope:**
- A per-request **adapter identity** (opaque string id) riding the existing model-tag path, backed by a pre-declared **adapter registry** (`id → rank`).
- A per-instance **resident-adapter set** with a configurable capacity and LRU eviction.
- The three DT **cost terms** as deltas onto the calibrated base: cold-load latency (by rank), per-step compute overhead (by unique adapters × max rank), per-adapter HBM footprint.
- A **LoRA-aware routing scorer** — the control surface where placement/routing policies attach.
- Adapter-aware **metrics** (load/eviction counts, TTFT-by-adapter, per-adapter throughput).

**Explicitly out:**
- Adapter *training*, rank/accuracy trade-offs, adapter internal architecture.
- Cross-GPU bin-packing **placement as a solver** (this design provides the surface such solvers plug into; the solvers are the research payload, not this feature).
- Multi-base-model-per-instance beyond the existing model-tag mechanism.

**Deferred (follow-on):**
- **Dynamic** runtime KV↔adapter memory negotiation (this design uses a **static** footprint subtraction; see Decision D2).
- An in-repo adapter-profiling / coefficient-fitting harness (calibration is limited to importing pre-fitted DT tables).
- Adapter *migration* mechanics (the scorer enables migration research; explicit migration events are future work).

---

## 3. Modeling Decisions

| Aspect | Modeled | Simplified | Omitted |
|---|---|---|---|
| Adapter identity | Opaque global string id per request; registry maps id→rank | Single base model per adapter (rides model-tag) | Adapter versioning, adapter content/weights |
| Adapter residency | Finite per-instance slot set with LRU eviction, pinning of in-use adapters | Uniform per-instance capacity | Per-adapter placement quotas; tiered adapter storage |
| Cold-load cost | Pre-admission blocking latency `base + ceil(bytes/bw)`, serialized per instance | Latency shape reused from tiered-KV transfer; bytes from rank | Concurrent/bandwidth-shared loads; unload cost (~0, per DT) |
| Compute overhead | Multiplicative step-time factor `1 + (K6(r_max)/K7(r_max))·A_B` — the DT term normalized to the no-adapter baseline; unique-adapter count `A_B` × max-rank tier `r_max` (D4) | Rank enters only as the coefficient-tier key (max-rank-in-batch), not a continuous multiplicand; relative multiplier onto calibrated base (=1.0 at `A_B=0` by construction) | Per-adapter GEMM microarchitecture; kernel fusion effects |
| HBM footprint | Static per-adapter bytes (from rank) subtracted from KV budget | Fixed block count set once; representative footprint × capacity | Dynamic runtime KV↔adapter renegotiation |
| Routing | Adapter-affinity scorer reading resident sets (Periodic freshness) | Warm-preference via min-max normalized affinity | Global optimal placement; predictive prefetch |
| Metrics | Per-adapter load/eviction counts, TTFT, throughput | Attribution by adapter id | Per-tenant fairness accounting |

**What each simplification/omission loses (Banks criteria 1, 2, 5).** Each row answers a specific analysis question and states the lost real-system behavior:
- *Single base model per adapter* (identity) — loses shared-adapter-name-across-base-models; unused in the DT's single-model regime, so negligible for the placement questions. Its *omitted* items (adapter versioning, adapter content/weights) are irrelevant to placement/routing timing questions. Schema-migration path in D3.
- *Uniform per-instance capacity* (residency) — loses heterogeneous-GPU (different HBM) slot counts; acceptable while validation targets homogeneous clusters, revisit for heterogeneous fleets. Its *omitted* items (per-adapter placement quotas; tiered adapter storage) are deferred: quotas are a placement-policy concern (research payload on the scorer surface, not the harness), and tiered adapter storage is out of scope alongside the static-memory decision (D2).
- *Serialized, non-overlapped cold loads* (load cost) — loses bandwidth-shared/pipelined loads and prefetch-hiding, and (per the head-of-line decision, §7) concurrent warm-batch execution during a background load; also treats unload cost as ~0 (per the DT). This is the DT's own assumption, so it does not widen the BLIS-vs-DT gap but caps real-system fidelity (see D1, §7, §15).
- *Static HBM subtraction* (memory) — loses runtime KV↔adapter renegotiation; mis-answers "can KV be freed to fit an adapter?" — explicitly out of scope (D2).
- *Per-adapter GEMM microarchitecture / kernel fusion* (overhead) — folded into the fitted coefficients; below the accuracy target the analysis needs (criterion 2).
- *Global-optimal placement / predictive prefetch* (routing) — deliberately omitted; these are the *research payload* the scorer surface enables, not the harness.
- *Per-adapter attribution only* (metrics) — loses cross-adapter/per-tenant fairness accounting; acceptable because fairness policy is not among the stated analysis questions (§1/§16).

**Simplest version that answers the questions (Banks criterion 6):** the minimal load-bearing set is (1) the three DT cost terms — cold-load latency (by rank), the per-step compute-overhead factor (D4), and static per-adapter HBM footprint; (2) a finite per-instance capacity with LRU eviction; and (3) a single affinity scorer, off by default. (The cost terms and eviction are each defended below as necessary; the scorer is off by default so the no-op case is truly inert.) Coarser variants were rejected by sensitivity reasoning, one per load-bearing element: a *uniform* (rank-blind) load latency erases the by-rank cold-start signal placement research optimizes (cost term 1); *omitting the compute-overhead factor entirely* erases the batch-composition-driven throughput/ITL degradation that per-adapter throughput validation targets (cost term 2, D4); *omitting HBM footprint accounting entirely* would let adapters and KV double-book the same memory, violating INV-L4 and making the memory-pressure analysis question unanswerable (cost term 3); *random* eviction erases the locality the scorer exploits (capacity/LRU); and *no explicit load event* (TTFT delta only) cannot model queueing behind a serialized load. Each retained element is the coarsest form that still moves TTFT/throughput/churn — refine only with evidence against the DT.

---

## 4. Concept Model & Module Interactions

The LoRA subsystem is a new domain module that threads through the existing request lifecycle without owning the event loop:

- **Workload → Request**: the generator stamps an adapter id on requests (from client/cohort specs). The registry (`id → rank`) is declared once.
- **Routing (cluster)**: the adapter-affinity scorer reads each instance's resident-adapter set (exposed via the routing snapshot) and biases placement toward warm instances.
- **Batch formation (instance)**: a cold request (adapter not resident on its instance) is **gated pre-admission** — held out of the runnable batch until an adapter-load completes.
- **Step execution (instance)**: the latency model multiplies base step time by the adapter-overhead factor derived from the unique adapters in the batch.
- **Memory accounting**: resident adapters reserve HBM, reducing KV capacity.
- **Metrics**: per-adapter aggregates are collected as derived statistics.

The registry and cost model live in a new `sim/lora/` subpackage that **registers into `sim/` via `init()`** (no reverse import), preserving the unidirectional dependency rule. Cross-cutting fields (the request's adapter id, the snapshot's resident-set view, the subsystem config) live in `sim/` where the existing bridge types are.

---

## 5. Module Contract (per §4.3)

| Aspect | Contract |
|---|---|
| **Observes** | Request adapter id; adapter registry (`id → rank`); per-instance resident-adapter set; batch composition (for the overhead factor); instance memory budget |
| **Controls** | Whether a request is gated pending a cold load; which adapter is evicted under capacity pressure; the cold-load latency, per-step overhead factor, and HBM reservation; the adapter-affinity routing score |
| **Owns** | The adapter registry (immutable after config); each instance's resident-adapter set (residency order, pinned set, in-flight load). Per-adapter **statistics** are *not* owned here — they are owned by the metrics module, derived from `sim/lora`'s load/evict events (state/statistics separation, §8) |
| **Invariants** | INV-L1…INV-L7 (§9) |
| **Events** | Produces and consumes its own **adapter-load-completion** event (endogenous, §7). The gating decision itself is a *synchronous check* during batch-formation, not consumption of a separate event type |
| **Extension friction** | The designated extension surface is the routing scorer — a **Policy Template**, ~3 files (scorer, registration, validation message). Two other future extension points sit at *different* taxonomy tiers, with different friction: an alternate **cost table** is a data-only drop-in (~0 code, §11); an alternate **eviction policy** (e.g., LFU) is a **Backend Swap** — the resident set has no eviction-policy interface at v1, so it requires Phase-A interface extraction before a Phase-B alternative (§5.4), a heavier recipe than the scorer/cost-table cases. The subsystem's own first-build footprint is inherently cross-cutting; see §18 |

---

## 6. Interface Design

Two design rules dominate here, both from the constitution (Principle III, R13): **prefer extending existing behavioral contracts over minting single-implementation interfaces.**

- **Adapter cost accessor (behavioral contract).** The latency model needs adapter rank to compute the overhead factor, but requests carry only the id (identity decision, D3). The durable requirement: the model resolves the overhead through a **single pure, batch-scoped query it holds** (given a batch, return its overhead multiplier; per-adapter rank lookup is internal to that query — not a second method) — *not* by reading request fields and *not* by reading `OutputTokens` (INV-L6), and *without* changing the batch-timing method's existing signature. The same query is available to both latency backends, so the contract serves ≥2 implementations (R13). The query is **batch-scoped** for the factor (given a batch, return its overhead multiplier). Because registry completeness is validated at construction (every adapter id used by the workload exists in the registry — §12/§14), the query never encounters an unknown id at runtime. When the query is absent (no adapters), the factor is exactly 1.0 (no-op). Whether the query is injected at construction or wired another way is a micro-plan choice.
- **Resident-set view on the routing snapshot (behavioral contract).** The scorer reads resident adapters through a new field on the existing routing snapshot (an established bridge type), not a new interface. The scorer itself is a standard scorer registered in the existing scorer set. When no adapters are configured the field is empty and the scorer is neutral (no-op).
- **Batch-formation gating check (behavioral contract).** Batch Formation must be able to ask, for a request it is about to consider, "is this request's adapter resident on this instance?" — a pure boolean query the instance's resident set answers. `sim/lora` owns the authoritative answer (the resident set); Batch Formation only reads it. Naming it lets Batch Formation and `sim/lora` be developed and tested independently (§4.6). When no adapters are configured the query is trivially "resident" (no gating).
- **Adapter-reserved memory query (behavioral contract).** The KV-capacity/memory-budget accounting (owned outside `sim/lora`) obtains the instance's adapter-reserved bytes through a **pure query** `sim/lora` answers from its footprint model — not by reaching into `sim/lora`'s internal state (R14). Under the static model (D2 / INV-L4) the answer is the **fixed capacity-based reservation** (`capacity × per-slot footprint`), computed once at startup — a constant — so the memory module subtracts it once when it sets the KV block count. When no adapters are configured the query returns 0 and KV capacity is unaffected (INV-L1). This is the fourth boundary the subsystem crosses (a config-time computation, not a runtime-varying signal).
- **Registry & resident set** are concrete types in `sim/lora/`, queried through pure accessors (no destructive reads; Principle III `Get`/`Consume` separation observed).

The subsystem must be testable in isolation with mocked adjacent modules: the resident set is a pure data structure; the cost model is a pure function of rank + config; the scorer's LoRA-affinity term is a pure function of `(request adapter, resident sets)` that is then **composed into the weighted routing profile** alongside the existing signals (queue depth, KV utilization) — it is one weighted term, not a standalone router.

---

## 7. Event Integration

**New event: adapter-load completion.**
- **Classification:** **endogenous** — triggered by internal state (a cold request reaching batch formation on an instance whose resident set lacks its adapter), not by workload arrival.
- **Semantics:** when a request's adapter is cold, the instance begins a load. It **commits the eviction victim and reserves the slot at load-*start*** — selecting the LRU non-pinned adapter *now* (never a pinned/in-use one, INV-L5); the reserved slot counts toward capacity for the whole load duration and cannot be re-pinned mid-load. (Committing the victim at load-start, not load-completion, is what makes INV-L2 and §12's no-deadlock argument sound: a request arriving mid-load cannot resurrect the soon-to-be-evicted adapter.) An adapter-load-completion event is scheduled at `now + LoadLatency(rank)`; at completion the adapter is resident and the gated request becomes batch-eligible. **Intra-pass ordering:** within a single batch-formation pass at time *t*, requests made batch-eligible by a load completing at *t* are admitted (pinning their adapters) **before** any new cold-gate/eviction decision in that same pass — so a just-loaded, not-yet-pinned adapter cannot be selected as another request's eviction victim in the same tick (protects INV-L5 against a same-timestamp race).
- **Same-adapter coalescing (INV-L3).** Additional cold requests for an adapter whose load is *already in flight* on the instance **attach to that same load's completion** — they do **not** start a second load and incur **no separate load-latency charge**, satisfying INV-L3 ("charged exactly once per (adapter, instance) cold transition"). "At most one load per instance at a time" is across *all* adapters, and there is at most one in-flight load per adapter.
- **Request state during gating (INV-2, INV-12).** A gated request stays in the `queued` state (a *gated-pending-load* sub-state); it is **not** promoted into the running batch — batch formation may *inspect* it at the queue head (that is the head-of-line gating mechanism) but never *admits* it into the running batch (the inspection mechanism is a micro-plan choice), so INV-12 (Phase-1 completeness of running requests) is trivially preserved — gating never produces a running request without its adapter. On load completion the request enters the runnable set; the load-completion handler MUST **guarantee a step-formation opportunity exists at *t*** (enqueueing a StepEvent through the same mechanism any wait-queue arrival uses if one is not already scheduled), so INV-8 holds across the transition *out* of gating — not merely during the wait. (Because adapter-memory feasibility is a startup guarantee under the static reservation, §12, every gated request's adapter is guaranteed a slot — there is no runtime adapter-unservable termination.)
- **Serialization:** at most one adapter loads per instance at a time; further cold loads on the same instance queue behind the in-flight load. This models a shared host→GPU copy path and matches the DT's blocking behavior.
- **Head-of-line behavior during a load (modeling decision).** While an instance has an in-flight adapter load, batch formation on that instance is **gated for the load duration** — the instance's work that step is the load itself (INV-8), so a gated request at the wait-queue head blocks the requests behind it (including warm ones) until the load completes. This is the **DT-faithful serialized-load model** (the instance stalls behind its load) and is the natural fit for the existing head-only Phase-2 admission loop (which inspects the queue head and stops on an admission barrier). It is a *conscious modeling choice with a throughput cost*: it forgoes running a warm batch concurrently with a background load, and it means a high-priority cold request (under a `priority-fcfs` scheduler) can hold the head across steps. The **non-blocking alternative** — skip past a gated head to admit warm requests behind it — is a documented future refinement; it is a Phase-2 *control-flow* rewrite (Phase 2 only inspects the queue head today), **not** blocked by missing queue primitives (the wait queue already supports full iteration and by-identity removal, as `TimeoutEvent` uses), and it diverges from the DT's blocking reference. Because this choice directly shapes the per-adapter throughput/TTFT the validation strategy measures (§15), it is called out here rather than left to the micro-plan.
- **Tie-breaking (relative ordering invariant).** The event participates in the existing `(timestamp, priority, seqID)` ordering. **Ordering contract:** an adapter-load completion at time *t* MUST be ordered ahead of any step-formation event at the same timestamp *t* on that instance, so the newly-resident adapter's request is batch-eligible for the step formed at *t*. This is a *relative* priority requirement (load-completion precedes co-timed step-formation); the numeric constant realizing it is assigned in the micro-plan against the concrete event-loop priorities — deferring the *number* (not the ordering) keeps the doc free of premature implementation detail (§3.4).
- **Boundedness, ordering & accounting.** The per-instance pending-load queue is bounded by the instance's in-flight requests (each gated request waits on exactly one pending load, and multiple same-adapter requests share a single load per the coalescing rule above; loads drain one at a time). Cold requests are gated in the order batch-formation encounters them, which is **whatever the active per-instance `InstanceScheduler` orders the wait-queue as that step** (FCFS by default; `priority-fcfs`/`sjf`/`reverse-priority` schedulers reorder it — the WaitQ is re-ordered each step before batch formation). This is independent of the *cluster-level* `--dispatch-order` (gateway→instance) flag, a separate mechanism. The subsystem introduces no separate sequencing state — it inherits the scheduler's order. Adapter-memory feasibility is guaranteed **at startup** by the static reservation (§9 INV-L4, §12), so there is **no per-request runtime "adapter cannot fit" drop** — the pending-load queue only ever holds requests whose adapters are guaranteed a pre-reserved slot. (Non-adapter oversizing still routes to the existing `dropped_unservable` path, unchanged.)
- **Interaction with INV-8 (work-conserving):** the load event is itself scheduled work, so a gated request does not make the simulator "idle while work waits" — the pending load is the work.

No existing event types change semantics. When no adapters are configured, no adapter-load events are ever created (no-op).

---

## 8. State Ownership

- **Adapter registry** — created once from config at **simulation construction (cluster *or* single-instance mode)**; **immutable** thereafter; read by the workload generator (validation), the cost model (rank), and the scorer. No mutation after init. (Single-instance mode constructs its own registry from the same config; it does not depend on the cluster layer.)
- **Resident-adapter set** — one per instance; created by the instance; **mutated only by that instance**, on three triggers: (1) *touch* on a warm reference; (2) at cold-load **start**, *commit the eviction victim and reserve the slot* (§7); (3) at load **completion**, mark the adapter resident. Never read or mutated across instance boundaries. For routing, each instance's resident id set is exposed to the routing snapshot as a **point-in-time view — frozen as of snapshot build, not continuously updated** (whether realized as a value copy or an immutable reference is a micro-plan choice) — so the scorer reads a *stale-by-design* view whose freshness is **Periodic** (D5), consistent with "no shared mutable state across module boundaries."
- **Multi-instance coordination.** The router sees all instances' resident sets through one aggregated snapshot (each instance's local id set collected into the shared routing snapshot the router reads), refreshed on the snapshot cadence. Under rapid churn the view can lag actual residency — the scorer therefore biases, never guarantees, warm placement (D5); a mis-predicted warm target simply incurs a cold load on arrival.
- **Autoscaler / horizontal-scaling composability.** The per-instance invariants (INV-L2, INV-L4) are phrased per instance and compose with autoscaler/node-pool instance churn without modification: a newly-added instance starts with an empty resident set; a removed instance's invariants become moot. Cluster-level adapter placement across a *changing* instance set (adapter re-placement on scale events) is **not** evaluated here — it is research payload on top of the scorer surface. This is a **documented scope boundary, not a runtime guard** — distinct from INV-13, whose autoscaler/node-pool clause is specifically a `blis replay` fail-fast (`logrus.Fatalf`) for run/replay parity, not a `blis run`-time mechanism; this design adds no such gate for LoRA + autoscaler, it simply does not implement cross-scale re-placement. (Note: node pools are BLIS's heterogeneous-GPU-fleet construct, so this also inherits the uniform-capacity simplification flagged in §3.)
- **Single-instance mode.** Because gating, memory reservation, and the step-overhead factor are all instance-local, the subsystem works unchanged in single-instance (non-cluster) mode; only the routing scorer is inapplicable there (routing runs in cluster mode only).
- **Per-adapter statistics** — derived accumulators, written by the metrics path, never read back into any control decision (state/statistics separation).

---

## 9. Invariants

New invariants introduced by this design (prefixed INV-L to namespace them; they specialize or extend the constitution's INV-1…INV-11):

- **INV-L1 (no-op inertness).** With an empty registry and unset adapter capacity, simulation output is byte-identical to the pre-feature build. *(Specializes INV-6; the load-bearing constraint.)*
- **INV-L2 (capacity bound).** For every instance at every point in the run, `|resident adapters| ≤ configured capacity`.
- **INV-L3 (cold-load charge).** Cold-load latency is `≥ 0`, charged exactly once per cold (adapter, instance) transition, and never charged when the adapter is already resident (warm).
- **INV-L4 (memory conservation).** Per instance, `allocated_blocks + free_blocks + adapter_reserved = total` at all times, where `adapter_reserved` is the **fixed, capacity-based reservation** (`capacity × per-slot footprint`, sized from the max declared rank) computed **once at startup and held constant** — the static model of D2. Adapters churn *within* the pre-reserved slots; the reservation does **not** vary with which specific adapters are currently resident (a runtime KV↔adapter renegotiation is the deferred dynamic tier). Reservation is scoped to the instance's own budget. *(Extends INV-4.)*
- **INV-L5 (no eviction of in-use).** An adapter is **in use** from the moment a request referencing it enters the running batch until that request completes; an in-use (pinned) adapter is never evicted — and pinning **persists through preemption** (a preempted request has not completed, so its adapter stays pinned). This also covers the eviction victim reserved at load-start (§7), which is never a pinned adapter. *(Behavioral definition — independent of how pinning is represented.)*
- **INV-L6 (oracle boundary).** Adapter-aware routing/servability reads adapter id, rank, and resident sets only — never `Request.OutputTokens`. *(Specializes INV-9.)*
- **INV-L7 (backend parity).** The roofline and trained-physics backends apply an identical adapter-overhead factor for the same batch. *(R23.)*

Existing invariants that must continue to hold: **INV-1** (request conservation — adapters partition accounting; an adapter that cannot fit on any eligible instance is counted under the **existing `dropped_unservable`** terminal state — the same path as an oversized request, §12 — so the INV-1 identity is unchanged and nothing is silently dropped; no new identity term is introduced), **INV-2** (request lifecycle — the *gated-pending-load* state is a transient sub-state within `queued`, not a new top-level lifecycle state; no invalid transitions), **INV-3** (`StepTime ≥ 1`; load-completion timestamp ≥ now), **INV-5** (`enqueue ≤ schedule` — the gate delays schedule, not enqueue), **INV-6** (determinism), **INV-7** (signal freshness — the resident-set snapshot field is Periodic, per D5), **INV-8** (work-conserving), **INV-12** (Phase-1 completeness — trivially preserved: a gated request never enters the running batch, so it is structurally outside INV-12's domain, §7), **INV-13** (run/replay parity — the adapter id round-trips through TraceV2 so replay reproduces per-adapter metrics, §14; unsupported replay ⇒ `logrus.Fatalf`, never silent degradation).

---

## 10. Decisions with Trade-offs

### D1 — Cold-load gating: pre-admission block (not per-step stall)
**Decision.** A cold request is held out of the batch until its adapter loads; the load latency is charged before scheduling and shows up in the request's TTFT; loads serialize per instance.
**Alternatives.** (a) *Per-step stall* — drain a pending-latency accumulator inside the step, reusing the tiered-KV drain pattern; rejected because it charges the delay after the request is already batched, muddying the cold-vs-warm TTFT signal and entangling with preemption / Phase-1 batch completeness. (b) *Non-blocking/prefetch* — no dedicated blocking term; rejected because it discards the cold-start TTFT tail that placement research optimizes.
**What breaks if wrong.** Real engines (vLLM/SGLang) may overlap or pipeline loads with compute; if so, our TTFT tails are overstated. Our only calibrated reference (the DT) shares the blocking assumption, so BLIS-vs-DT agreement is a **fidelity ceiling, not ground truth** against a real server (see §15). Overlapped loading is a future refinement, not a first-cut requirement.

### D2 — Adapter memory: static subtraction (not dynamic tier)
**Decision.** Reserve per-adapter HBM statically from the KV budget; the block count stays fixed once.
**Alternatives.** (a) A full dynamic runtime KV↔adapter tradeoff (a second memory tier analogous to the CPU KV tier); rejected for the first cut because it needs cross-consumer budget-negotiation plumbing that does not exist today. Deferred, flagged as follow-on. (b) A *middle ground* — reserve footprint only for pinned/in-flight adapters and lazily reclaim idle non-pinned slots under KV pressure; also deferred (it still needs a reclamation trigger tied to the KV budget), but it is the natural first step toward the dynamic tier if static rigidity proves too coarse.
**What breaks if wrong.** Overstates memory rigidity in a specific direction — it *understates* achievable adapter density/concurrency whenever idle KV headroom could otherwise have been reclaimed to admit one more adapter, biasing placement toward the conservative. Acceptable for capacity-planning questions and keeps INV-L4 a clean equality; the middle-ground (b) above is the escape hatch if this bias proves material.

### D3 — Adapter identity: global string id + pre-declared registry (not per-request rank, not compound key)
**Decision.** Requests carry an opaque adapter id; a registry maps `id → rank`; single base model per adapter.
**Alternatives.** (a) Stamp rank on every request — rejected: duplicates the registry and can disagree across requests for the same adapter. (b) `(base_model, name)` compound key — rejected as unnecessary plumbing while ids are globally unique. **Verified against the DT** (`experiments/dt_driver.py`): flat string ids, a parallel `served_adapters_sizes` registry, one base model per run.
**What breaks if wrong.** Multi-base-model clusters sharing an adapter name would need the compound key; deferred and low-risk. **Migration path:** moving to a `(base_model, id)` key is *additive* — it widens the registry key only; requests still carry the id, routing still filters by model tag, and the scorer contract (id ∈ resident set) is unchanged. No consumer breaks, so the deferral is not a footgun.

### D4 — Compute overhead: relative multiplier (not absolute constants, not additive)
**Decision.** Multiply base step time by the DT's adapter-overhead term **normalized to the no-adapter baseline**: `factor = (K7(r_max) + K6(r_max)·A_B) / K7(r_max) = 1 + (K6(r_max)/K7(r_max))·A_B`, where `A_B` is the count of unique adapters in the batch and `r_max` is the **maximum rank among them** — rank enters by selecting the fitted-coefficient *tier* keyed on max-rank-in-batch (this is what "unique adapters × max rank" in §2/§3 means; both the count and the max rank enter). Normalizing by `K7` makes the factor **exactly 1.0 at `A_B=0` by construction, independent of the fitted `K7` value** — this is load-bearing for INV-L1 (byte-identical no-op) and is precisely the "relative multiplier onto BLIS's calibrated base" framing (§6); it avoids a raw `(K6·A_B + K7)` form, which would evaluate to `K7 ≠ 1.0` at zero adapters and break the no-op. A mixed-rank batch uses the max-rank tier (conservative — the largest adapter dominates the GEMM cost). **Short-circuit:** `r_max` is defined only for `A_B>0`; when `A_B=0` (a batch with no adapter requests, even in a configured system) the factor is exactly 1.0 **without** evaluating any tier — this is the per-batch no-op, distinct from the global no-adapters-configured case (§6). Each tier's `K7(r) > 0` is enforced by config validation (§14) so the `/K7` normalization never divides by zero (R11).
**Alternatives.** (a) Drop the DT's absolute step-time constants — rejected: they are H100/vLLM-specific and cannot sit on BLIS's separately calibrated base. (b) Additive overhead — rejected: the DT's fitted form is multiplicative on the backbone term and additive misfits at large batch.
**Validity range.** The normalized `1 + (K6(r_max)/K7(r_max))·A_B` form and its per-rank fitted coefficients are calibrated over the DT's profiled batch-size and rank ranges; batch sizes or ranks far outside that envelope are extrapolation and carry no fidelity guarantee. The design charges the factor as a bounded multiplier (**exactly 1.0 with no adapters; strictly > 1.0 with ≥1 unique adapter; never < 1.0**) and **clamps out-of-envelope inputs (rank/batch beyond the calibrated range) to the nearest calibrated boundary rather than extrapolating**, so it degrades gracefully instead of inverting or blowing up.
**What breaks if wrong.** Factor-form misfit at extreme batch sizes. Note the factor multiplies *every* step (prefill and decode, since `StepTime` sees the whole batch), so it contributes to **both** absolute TTFT and steady-state throughput/ITL; the cold-vs-warm **differential** TTFT test isolates D1 (the factor cancels in the difference), while **steady-state throughput** isolates D4. Bounded by the ≤20% per-adapter throughput error claim (§15) over the DT's calibrated envelope, and by the extrapolation caveat above.

### D5 — Scorer signal freshness: Periodic (not synchronous)
**Decision.** The resident-set view on the routing snapshot has **Periodic** freshness (like queue-depth/KV-utilization), Immediate when the refresh interval is 0.
**Alternatives.** A synchronous per-request resident-set query (like in-flight requests); rejected because it couples routing to live instance state and breaks the snapshot abstraction — and Periodic is the natural fit for how a GAIE/llm-d-style endpoint picker scrapes metrics periodically (assumed parity based on that scraping design; not verified against llm-d source — treat as an assumption, not an established fact). The scorer must document this tier (R17).
**What breaks if wrong.** Slightly stale routing under rapid churn; acceptable and parity-faithful. A secondary failure mode is a **correlated stampede** — several cold requests arriving between refreshes all biased toward the same believed-warm instance, converging simultaneously; this is inherent to periodic-snapshot pickers and is mitigated by the scorer being one *weighted* term (queue-depth/KV-utilization terms still spread load), not decision-flipping for Periodic vs. Synchronous.

### D6 — Configuration: a dedicated `LoRAConfig` module sub-config (constitution amendment) — resolves C1

*(C1 is the `/speckit.analyze` finding that this feature introduces a 7th `SimConfig` sub-config, which the constitution's Principle VI currently caps at "exactly 6.")*
**Decision.** Add one module-scoped sub-config for the subsystem (capacity, cost coefficients, registry). This makes `SimConfig` hold **7** sub-configs, whereas Principle VI currently states "exactly 6."
**Analysis.** The rule's intent (R16) is "config grouped by module"; the Extension Framework explicitly sanctions new Subsystem Modules, each of which owns config. "Exactly 6" merely counts today's modules.
**Resolution.** Amend Principle VI from *"composed of exactly 6 embedded sub-configs"* to *"embeds at most one sub-config per domain module (6 at v1.0.0); a new Subsystem Module that introduces module-scoped config adds exactly one (never fragmented), and one with no config needs adds none."* This is a **MINOR** constitution bump (v1.1.0 — expanded guidance, not a backward-incompatible redefinition) and must run the amendment procedure (convergence review, atomic template update) as a companion change landing with PR1. **Exact before/after text, affected files, and gate are in Appendix A.**
**Alternatives.** Spread LoRA fields across KVCache/Latency/Workload sub-configs — rejected: fragments one module across three configs, breaks R16 cohesion and independent validation. Keep "exactly 6" and treat the 7th as a permanent justified complexity entry — rejected as it leaves a standing rule conflict every future Subsystem Module re-hits.
**What breaks if wrong.** The loosened rule could be *stretched* — a future contributor labels a policy template, backend swap, or tier composition a "Subsystem Module" to justify its own sub-config, eroding R16 cohesion (the exact abuse the amendment text names). Mitigation: the amendment binds "Subsystem Module" to the precise Extension-Framework definition (Appendix A), and the fallback if even the amendment is unsound is in Appendix A ("If the amendment fails review"). If mis-stretched anyway, the cost is config sprawl in `SimConfig`, caught in code review against the Extension-Framework classification.

---

## 11. Extension Points

- **Placement/routing policies** attach at the **adapter-affinity scorer** and, more generally, at any scorer reading the resident-set snapshot field. `Tantawi2025` server placement, Toppings rank-aware routing, and dLoRA-style migration heuristics are all expressible as scorers (or scorer compositions) over this signal. **Default behavior:** the scorer is *not* in the default routing profile; routing is unchanged until a user weights it in.
- **Adapter cost model** is a query contract; a re-profiled or alternative cost table (e.g., a different GPU) is a drop-in without touching call sites.
- **Eviction policy** is LRU today; an alternate policy (e.g., LFU) is a **Backend Swap** — Phase A must extract an eviction-policy interface from the resident set before Phase B adds the alternative (§5.4). No such interface exists at v1 (matches the §5 friction-cell classification).
- **Memory model** is static today; the dynamic (runtime KV↔adapter) tier is a future Tier-Composition extension over the KV budget.
- **Non-default example:** a warm-affinity scorer weighted `lora-affinity:2, queue-depth:1, kv-utilization:1` biases hot adapters to stay resident while still load-balancing.

---

## 12. Failure Modes

| Condition | Handling | Boundary |
|---|---|---|
| Invalid config (negative rank/capacity, zero bandwidth, adapters present with zero capacity, request id absent from registry, **any tier `K7(r) ≤ 0` or NaN/Inf** — guards the `/K7` normalization divisor, R11/R3 — validated for the per-rank tier table, not just CLI flags) | Reject at startup with a clear message | CLI (`cmd/`) → `logrus.Fatalf`; library constructor → `panic` |
| Adapter reservation (`capacity × per-slot footprint`) cannot fit alongside model weights + minimum KV | **Rejected at startup** (config validation) — under the *static* reservation (§9 INV-L4), memory feasibility is a one-time check, so there is **no per-request runtime "adapter cannot fit" path**; the existing `dropped_unservable` runtime path remains for non-adapter oversizing, unchanged | CLI `logrus.Fatalf` / library `panic` (R3); R22 sizing pre-check |
| Capacity pressure with all resident adapters pinned (in use) | The cold request waits, then loads once a slot frees. **No deadlock:** a pinned adapter belongs to a *running* request, which makes forward progress each step (INV-8) and completes in finite time, unpinning its slot → LRU eviction → the waiting load proceeds. The wait is bounded by the slowest in-flight request's remaining work, not unbounded. | Bounded wait — forward-progress guarantee (INV-8) + capacity bound (R19) |
| In-flight load whose sole waiting request departs while gated — via the **always-on per-request `TimeoutEvent`** (SLO deadline; the common path — it removes a still-`queued` gated request) **or** the opt-in `--in-flight-eviction` | The load completes anyway; the adapter simply becomes resident with no immediate consumer and is later reclaimed by LRU. No invariant is violated; the departed request is counted **once under its own terminal bucket** — `timed_out` for a `TimeoutEvent` departure, `gateway_evicted` for in-flight eviction — not double-counted by the load path. (Note: gateway-queue TTL and queue-shedding act only *pre-dispatch*, so they cannot reach an already-routed, gated request; the always-on `TimeoutEvent` and opt-in in-flight eviction are the mechanisms that can.) If either mechanism instead removes a *running* request whose adapter is pinned, that adapter unpins on the request's departure (pinning is defined by running-batch membership, INV-L5), then becomes LRU-eligible normally. | Existing `timed_out` / `gateway_evicted` accounting (unchanged); LRU reclamation |
| No adapters configured | Subsystem fully inert (INV-L1) | N/A |

---

## 13. Default Behavior (no-op — mandatory)

With no request carrying an adapter and no capacity configured: no adapter-load events, an overhead factor of exactly 1.0, no HBM reservation, a neutral scorer, and no new stdout fields. Output is byte-identical to today (INV-L1 / INV-6). This is verified by a golden baseline captured before the feature and re-checked after each PR.

---

## 14. Configuration Surface

- **Module sub-config** (`LoRAConfig`, see D6): per-instance adapter capacity; cold-load base latency and bandwidth; per-step overhead coefficients; per-rank footprint; and the adapter registry (list of `id → rank`).
- **Workload specs**: an optional adapter id per client/cohort (empty ⇒ base-model-only).
- **CLI flags**: `--lora-*` mirrors of the numeric config, plus `--lora-scorer-weight` to compose the scorer into the routing profile. Each flag guarded by "changed?" before applying defaults (R18); every numeric validated for zero/negative/NaN/Inf (R3); strict YAML parsing (R10); pointer types where zero is a meaningful value (R9).
- **Trace round-trip (INV-13).** The adapter id is part of the exported **TraceV2** per-request record and **round-trips through `--trace-output` → `blis replay`**, so a LoRA-tagged trace replays with identical per-adapter metrics (run/replay parity). Silently dropping the adapter id on export (replaying a LoRA trace adapter-blind) is a parity violation; if for any reason adapter replay is unsupported, `blis replay` MUST `logrus.Fatalf` at startup rather than degrade silently (matching INV-13's autoscaler/node-pool stance).
- **Validation** is independent and module-local (Principle VI).

---

## 15. Validation Strategy

**Verification (correctness — invariants):**
- INV-L1 no-op byte-identity: **automated CI gate** — exact (byte-for-byte) diff of the same-seed adapter-blind run against a committed golden baseline (canonical scenario set); a stdlib/Go-version drift that changes formatting is treated as a baseline-regeneration event (R12), not a silent pass. Additionally, a **non-unit-`K7` scenario** (populated coefficients, adapter-free batch `A_B=0`) must still yield the normalized factor `= 1.0` exactly — guarding the D4 no-op-normalization regression (§10), which a `K7=1.0`-default-only test would miss.
- Out-of-envelope clamping (D4): a test with rank/batch **beyond the DT's calibrated envelope** asserts the overhead factor is clamped to the nearest calibrated boundary — it stays `≥ 1.0` and neither inverts nor blows up (§10 Validity Range).
- **Rank/uniqueness sensitivity (D4)** — the *interior* behavior, and a **PR4 gate** (not just PR7): for fixed `A_B > 0`, the factor **strictly increases** with the max-rank tier; the factor is monotonic in unique-adapter count `A_B`; a duplicate adapter in a batch counts once toward `A_B`. Without this, a rank-ignoring bug (always resolving one tier) would pass INV-L7 (both backends share the bug), INV-L1/L6, and the clamping test — caught only by the *optional* PR7 fidelity comparison. This test makes the core compute-overhead physics a mandatory PR4 gate.
- INV-L2 capacity bound: integration invariant test asserting `|resident| ≤ capacity` at every step over a M>N-adapter run (concrete: capacity small, distinct adapters ≫ capacity), **counting the reserved-but-not-yet-resident slot of an in-flight load toward capacity during the load window**.
- **Load-start eviction commitment (resurrection race)**: a dedicated scenario — while adapter A's eviction is committed and its replacement is loading, a second cold request for A arrives mid-load; assert A cannot be "resurrected" (the commit stands, capacity is respected) — this guards the timing detail §7 calls load-bearing for INV-L2 and §12's no-deadlock argument, so a regression to load-*completion* commitment is caught.
- **Pending-load gating order**: assert cold requests are gated in the order the active instance scheduler presents them that step (FCFS by default; a `priority-fcfs`/`sjf` scheduler reorders them), and that this is independent of the cluster-level `--dispatch-order` flag.
- INV-L3 cold-load charge: **two** contract tests — (a) `TTFT(cold) − TTFT(warm) ≈ LoadLatency(rank)` for the same input; (b) repeated warm requests pay zero load cost — plus a **serialization** test (two co-arriving cold requests for *distinct* adapters on one instance load sequentially, not in parallel — D1) and a **coalescing** test (two co-arriving cold requests for the *same* adapter share one load and are charged the load latency exactly once — INV-L3).
- INV-L4 memory conservation: KV-capacity invariant test asserting the **fixed capacity-based `adapter_reserved`** reduces usable KV once at startup and the per-instance equality `allocated + free + adapter_reserved = total` holds throughout a run with adapter churn — the reservation is *constant* (adapters load/evict within the pre-reserved slots without changing the byte reservation), so the test samples the equality across churn but expects `adapter_reserved` unchanged.
- INV-L5 no eviction of in-use: three **integration invariant test** scenarios — (a) eviction-under-pressure at **capacity = 1 with the sole slot pinned** by a running request (the cold request waits, the pinned adapter is not evicted; and, asserting **liveness/INV-8**, once the pinned request completes the waiting load completes and its request runs — no deadlock); (b) **preemption-persistence** — a request whose adapter is pinned is *preempted* (removed from the running batch, per Phase-1/INV-12 mechanics) yet its adapter stays pinned and is re-admittable without a reload, since "in use" spans until completion not until preemption; (c) **intra-pass race** — a load completing at *t* is pinned before any co-timed eviction decision in the *same* batch-formation pass, so the freshly-loaded, not-yet-pinned adapter cannot be selected as another request's eviction victim that tick (the INV-L5 sibling of the INV-L2 resurrection-race scenario).
- INV-L6 oracle boundary: verified by **both legs of the INV-9 precedent** — (a) a **behavioral test** proving an observable admit/route/score result is invariant to `OutputTokens` (identical decision whether or not the oracle value is present), which catches indirect leaks a grep cannot; and (b) a **static/CI code-path check** that scorer/servability paths never reference `OutputTokens`. The paths read only adapter id, rank, resident set.
- INV-L7 backend parity: a **cross-backend parity test** asserting a **byte-identical** overhead factor for the same batch across roofline and trained-physics (exact equality, not a tolerance) (R23).
- INV-1 conservation: system-wide conservation test with adapters configured, focused on **in-flight eviction of a gated request** (evicted via `--in-flight-eviction` while waiting on a load), asserting it is counted **once** under the existing `gateway_evicted` term and not double-counted by the load path. Adapters add **no new runtime terminal state** — memory feasibility is a startup guarantee under the static reservation (§12), so there is no runtime adapter-unservable drop — hence INV-1 holds unchanged: `injected == completed + dropped_unservable + gateway_evicted + …` still balances (no new identity term).
- **In-flight eviction of a running pinned adapter** (§12): a running request whose adapter is pinned, removed via `--in-flight-eviction`, unpins its adapter (which becomes LRU-eligible) and is counted once under `gateway_evicted` — asserts no leaked pin and correct accounting (distinct from the *gated*-request in-flight-eviction case in the INV-1 test, and from scheduler preemption in INV-L5(b)).
- **Head-of-line gating behavior**: a scenario with a cold request at the wait-queue head and warm requests behind it asserts the modeled serialized-load behavior (§7) — the instance's batch formation is gated for the load duration — distinguishing it from a (future, non-modeled) skip-past behavior, so the throughput/TTFT-by-adapter measurements reflect the documented model.
- **LRU-victim correctness**: beyond the capacity bound (INV-L2) and no-eviction-of-pinned (INV-L5), a test asserts that under capacity pressure the evicted adapter is the *actual least-recently-used non-pinned* one (not merely *some* non-pinned adapter) — the locality property the affinity scorer and placement research depend on (§3).
- **Continuing invariants (INV-2/3/5/7/8/12)**: these are existing invariants with existing regression suites (INV-12 is trivially preserved — a gated request never enters the running batch — so it needs no dedicated test, see §7); the design does not re-specify them but relies on their suites plus gating-specific assertions where the gating path is new — e.g., INV-5 that gating moves `schedule_time` while leaving `enqueue_time` fixed; INV-8 that a load-completion re-enqueues a `StepEvent` when the wait-queue was otherwise idle; INV-7 that the resident-set snapshot field inherits the same Periodic-refresh plumbing already covered for `QueueDepth`/`KVUtilization`. Precise GIVEN/WHEN/THEN for these belong in the micro-plan (§3.5).
- INV-13 run/replay parity: export a LoRA-tagged trace via `--trace-output`, replay with identical flags, and assert **identical per-adapter metrics** — verifying the adapter id round-trips through TraceV2 (§14); an adapter-blind replay of a LoRA trace must not silently pass.

**Validation (fidelity — real data):**
- **Data source.** The reference is the Agullo et al. Digital Twin (arXiv:2508.08343); its runnable form is the `GPULLMAdapterOptimization` fork driven by `experiments/dt_driver.py` in the companion `lora-control` repo (CPU-only, numpy). PR7 imports the DT's fitted rank→latency tables and compares **both per-adapter TTFT and per-adapter throughput** — the DT's `simulate()` returns throughput, ITL, and TTFT (plus adapter load counts), so cold-load latency (INV-L3) is validated against the DT's TTFT and the compute-overhead factor (D4/INV-L7) against the DT's throughput/step-time output.
- **Metric & scope.** Target error **≤ 20% on both TTFT and per-adapter throughput** — cold-load latency (D1) manifests in TTFT, while the D4 overhead factor multiplies every step and so contributes to *both* TTFT and throughput; **steady-state throughput** is the metric that isolates D4, so validating only TTFT would leave D4 unvalidated. Metric note: report the DT's published accuracy in **its own metric (SMAPE, ~17–21% on TTFT vs real H100)**; MAPE and SMAPE are **not** on the same scale (MAPE is unbounded as the denominator → 0, SMAPE is bounded to [0, 200%]), so the BLIS-vs-DT target and the DT-vs-real figure are not directly comparable — the ceiling claim (below) is qualitative. Measured over the DT's calibrated rank set `{8,16,32}` and profiled batch ranges, for the two pre-fitted configs (Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct). Bounded to these two configs; other model/GPU configs need profiling not automated in-repo.
- **Transitive-trust caveat.** Per the DT's own validation vs a real H100, it is ~17–21% SMAPE on TTFT (its weakest axis) but only **~5% SMAPE on throughput** (its strongest) and ~10% on ITL. So the throughput leg rests on a firmer DT ceiling than the TTFT leg — but either way BLIS-matching-DT is a **fidelity ceiling, not ground truth**. This is the best calibrated reference available for the adapter axes; a real-server comparison is future work.
- **Falsification / abort path.** If the error metric (**MAPE**, the BLIS-vs-DT measure) exceeds 20% on a config: re-profile the adapter coefficients for that config, or mark the config **unsupported** and withhold the fidelity claim — the subsystem still ships (its cost terms remain qualitatively correct and no-op-safe), but the quantitative claim is scoped down rather than overstated. Failing the bound does **not** silently pass.
- **Claim gating on PR7.** All *quantitative* fidelity claims (the ≤20% target, SC-007, and the §1 "transfers to llm-d" claim insofar as it rests on measured fidelity) are **contingent on PR7 landing** — PR7 is the only PR that runs the DT comparison. Until PR7 merges, the subsystem ships with qualitatively-correct, no-op-safe physics but **asserts no quantitative fidelity claim**; the falsification path above can only fire once PR7 exists. (PR7 remains optional for shipping the *mechanism*; it is mandatory before the *fidelity claim* is asserted.)

**Determinism / common random numbers:** the subsystem introduces **no randomness** (LRU, cost terms, eviction are deterministic), so it neither needs a new `PartitionedRNG` subsystem nor perturbs existing streams — paired (common-random-number) comparisons of adapter-blind vs adapter-aware runs remain valid. Determinism is tested not only on the empty-registry path (INV-L1) but by a **same-seed byte-identity test on an *active* adapter run** (adapters actively loading/evicting), deliberately including simultaneous-touch and same-timestamp-completion tie conditions — **LRU ties are broken by insertion order** — to catch any map-iteration-order nondeterminism leaking into output (INV-6).

---

## 16. DES Design Review Checklist (§2.6)

| Question | Answer |
|---|---|
| What analysis questions does this help answer? | Adapter placement/migration/routing under finite GPU slots: TTFT (cold-start, by adapter), churn, per-adapter throughput, KV-vs-adapter memory pressure |
| What is modeled / simplified / omitted? | See §3 table |
| What events are introduced/modified? Exogenous/endogenous? | One new **endogenous** adapter-load-completion event; no existing event semantics change (§7) |
| How do new events interact with tie-breaking? | Ordered within `(timestamp, priority, seqID)` so a load completion precedes co-timed step formation (§7) |
| What new state is introduced? Who owns it? | Registry (immutable, built at simulation construction — cluster *or* single-instance, not cluster-dependent), per-instance resident set (instance-owned), per-adapter stats (metrics-owned) (§8) |
| What new metrics are derived? Incremental or on demand? | Per-adapter load/eviction counts, TTFT, throughput — accumulated incrementally, emitted only when adapters configured |
| How will correctness be verified? | INV-L1…L7 + INV-1 (§15) |
| How will fidelity be validated? | DT comparison (source: `dt_driver.py`) on two configs, ≤ 20% MAPE on **TTFT and per-adapter throughput** over the DT's calibrated envelope; bounded scope + transitive-trust caveat + falsification/abort path, **gated on PR7** (§15) |
| Does this introduce randomness? Which subsystem? | No new randomness; no new `PartitionedRNG` subsystem (§15) |
| Simplest version that answers the questions? | The three cost terms + per-instance capacity/LRU + affinity scorer (off by default); static memory, serialized blocking loads (§3) |

---

## 17. Phased Roadmap

Ordered by dependency; each PR is Small-tier and individually no-op-safe. Maps to the tracking epic #1464.

| PR | Issue | Scope | Gate invariant |
|---|---|---|---|
| **C-amend** (companion, lands with PR1) | constitution PR | Amend Principle VI to "one sub-config per module" (v1.1.0) — exact text in **Appendix A** | Amendment convergence review passes (zero CRIT+IMP) |
| PR1 | #1471 | Adapter identity plumbing + `LoRAConfig`/registry + per-adapter metrics | INV-L1, INV-1 |
| PR2 | #1465 | Resident-adapter set + capacity + LRU eviction | INV-L2, INV-L5 |
| PR3 | #1466 | Cold-load latency (pre-admission gate) + cost-model core | INV-L3, INV-5/8 |
| PR4 | #1467 | Per-step compute-overhead in `StepTime` (via cost accessor) | INV-L7, INV-6, **rank/uniqueness sensitivity** (§15) |
| PR5 | #1468 | Static HBM accounting | INV-L4 |
| PR6 | #1469 | LoRA-aware routing scorer | INV-L6, INV-7 |
| PR7 | #1470 | (opt) Calibration vs DT rank tables | INV-13 (run/replay), fidelity |

**Constitution companion:** the Principle VI amendment (D6 / C1) lands as its own governed change alongside PR1 (which introduces `LoRAConfig`).

**Parallel development.** Ordering above is by *dependency*, not a mandate to serialize. Once PR1 (registry + the request's adapter-id field) and PR2 (resident set) freeze the four boundary contracts of §6 (cost accessor, resident-set snapshot field, batch-formation gating check, adapter-reserved memory query), the three physics PRs and the scorer become **independent workstreams**: the physics PRs (PR3 cold-load gate, PR4 compute overhead, PR5 HBM accounting) and PR6 (scorer) are substantially parallelizable — each is developed against those frozen contracts with mocked neighbors. PR5's HBM accounting subtracts the **static, config-derived reservation** (`capacity × per-slot footprint`) via boundary #4, so it consumes a constant, not PR3's runtime state. Precise inter-PR dependency threading (e.g., PR5 reading the per-rank footprint function from the cost-model core; PR2's resident-set contract carrying the reserved-slot concept) is refined in the **macro plan** (§3.5); the design-level guarantee is that the four frozen boundary contracts decouple the workstreams. PR7 depends on PR3–PR5.

---

## 18. Touch-Point Acknowledgment (§4.5)

- **New routing scorer:** ~3 files (scorer, registration, validation message) — within the reference target for a policy template. Adding a *second* placement policy is likewise ~3 files, since it reuses the resident-set snapshot field.
- **Subsystem total (approximate).** The full feature touches on the order of **~14–15 files** across its 7 PRs: `sim/lora/` core (registry, resident set, cost model ≈ 3–4 new files), plus edits to request, workload spec, latency construction, KV-capacity, routing snapshot, cluster snapshot build, scorer set, metrics (2 files), and CLI (2 files). This **exceeds the ≤3-file policy-template target** — a conscious choice: adapter identity is genuinely cross-cutting. Confining it to ≤3 files would require a god-object violating layering (Principle I) and single-responsibility (R14). Footprint is minimized by **extending existing bridge types** (routing snapshot, latency-model construction) rather than minting interfaces, and by isolating registry/resident-set/cost-model logic in `sim/lora/` behind the constitution's mandated `init()` registration.
- **Known friction (dominant estimate uncertainty):** the general cost of a *new observable metric* is ~6 files (per §4.5, spanning production + test + doc + CLI). The per-adapter metrics (loads, evictions, TTFT, throughput) are variations threaded through **one** shared aggregation path, not N independent metrics, so they do not cost N×6 files; the "~2 files" in the count above is the core `RequestMetrics`/`MetricsOutput` edit, with the remaining per-metric friction (test/doc/CLI) folded into the other listed touch points. This metric friction is the largest source of the estimate's uncertainty (hence the "~" range); it is acknowledged, not fixed here.

---

## Appendix A — Constitution Amendment (Principle VI), companion to PR1

Resolves C1 (the 7th sub-config). Governed change; runs the amendment procedure (convergence review, zero CRITICAL + zero IMPORTANT; atomic template update) and lands with PR1.

**Before (v1.0.0, Principle VI):**
> - `SimConfig` is composed of exactly 6 embedded sub-configs; factory signatures MUST accept the narrowest sub-config (e.g., `NewKVStore(KVCacheConfig)`).

**After (v1.1.0, Principle VI):**
> - `SimConfig` embeds at most one sub-config per domain module (6 at v1.0.0: KVCache, Batch, Latency, ModelHardware, Policy, Workload). A new **Subsystem Module** — specifically the Extension-Framework type that adds a new module interface + events, *not* a policy template, backend swap, or tier composition (those add no sub-config) — that **introduces module-scoped configuration adds exactly one sub-config** (never fragmented across others); a Subsystem Module with no configuration needs adds none. Factory signatures MUST accept the narrowest sub-config (e.g., `NewKVStore(KVCacheConfig)`). Adding a sub-config WITHOUT a corresponding new Subsystem Module is the violation this rule guards against.

**Affected files (atomic):** `.specify/memory/constitution.md` (Principle VI text + version line → 1.1.0 + sync-impact note); any dependent template referencing "exactly 6"; `CLAUDE.md` if it restates the count. **Versioning:** MINOR (expanded guidance, not a backward-incompatible redefinition). **Gate:** amendment convergence review passes before PR1 merges. **If the amendment fails review:** LoRA does *not* ship with an unratified 7th sub-config. Three ordered options: (1) **interim** — ship with a **Complexity Tracking justification** (the plan-template's constitution-sanctioned mechanism for a justified violation), treating the single 7th sub-config as a documented, temporary deviation while a follow-up amendment attempt stays open — this is undesirable as a *permanent* state (D6's rejected "standing conflict") but acceptable as a bridge; (2) fall back to distributing config across existing sub-configs (rejected on cohesion grounds, D6); or (3) defer the feature until the config strategy is resolved.

## References

- Agullo et al., *Digital Twin for GPU LoRA serving*, arXiv:2508.08343 (the three cost terms; two calibrated configs).
- BLIS Constitution v1.0.0 (Principles I–VIII; INV-1…INV-11; R1–R23) and canonical `invariants.md` / `rules.md` (INV-12, INV-13).
- Source rationale: `~/Projects/blis/lora-control/docs/blis-lora-extension.md`; feasibility evidence: `experimental-testbed.md`.
- Spec-kit artifacts: `specs/007-lora-control-plane/`.
