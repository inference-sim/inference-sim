# Feature Specification: LoRA Control-Plane Subsystem

**Feature Branch**: `007-lora-control-plane`  
**Created**: 2026-07-15  
**Status**: Draft  
**Input**: User description: "~/Projects/blis/lora-control/docs/blis-lora-extension.md — Extend BLIS with LoRA control-plane capability: per-request adapter identity, per-instance loaded-adapter set with capacity and eviction, adapter cost physics (load latency, per-step compute overhead, HBM footprint), a LoRA-aware routing/placement scorer, and adapter-aware metrics — so adapter placement/migration/routing policies can be researched in a fast, deterministic, GPU-free harness."

## Clarifications

### Session 2026-07-15

- Q: When a request's adapter is cold (not resident on the target instance), where is the load latency applied? → A: **Pre-admission gate** — the request MUST NOT join a batch until its adapter is resident; the load latency is charged before scheduling (counts toward the request's queueing delay / TTFT), and adapter loads on an instance are processed one at a time (serialized). Matches the Digital Twin's blocking model.
- Q: How is each adapter's HBM footprint (the memory subtracted from the KV budget) determined? → A: **Derived from the adapter's rank** — rank is the single source of truth; there is no separate per-adapter byte field.
- Q: How is an adapter's identity keyed/scoped, and where does its rank live? → A: **Global string id + pre-declared registry.** An adapter is identified by a global string id, unique across the simulation, and belongs to a single base model. Adapters form a pre-declared registry mapping `id → rank`; requests reference an adapter by id only (rank is NOT stamped per request). Both load latency and HBM footprint are derived from the registry's rank. Mirrors the Digital Twin, whose calibrated reference runs use one base model per simulation.

## User Scenarios & Testing *(mandatory)*

The users of this feature are simulation researchers and capacity planners who
use BLIS to develop and evaluate LoRA adapter placement, migration, and
adapter-aware routing policies without access to real GPUs. Each story is a
slice of the "make BLIS LoRA-aware" journey that delivers standalone value.

### User Story 1 - Attribute requests and metrics to LoRA adapters (Priority: P1)

A researcher configures a workload in which requests carry a LoRA adapter
identity (with a rank), runs a simulation, and receives metrics broken down by
adapter — how many times each adapter was loaded, time-to-first-token (TTFT)
distribution per adapter, and per-adapter throughput.

**Why this priority**: This is the foundational plumbing. Without an adapter
identity riding through the request lifecycle and surfacing in metrics, none of
the downstream physics or routing can be observed or evaluated. It is also the
slice that most directly exercises the no-op-default constraint.

**Independent Test**: Configure a workload assigning adapters to requests, run,
and verify the results expose per-adapter load counts, TTFT, and throughput.
Separately, run any existing adapter-free scenario and verify output is
byte-identical to a run without the subsystem.

**Acceptance Scenarios**:

1. **Given** a workload where requests are tagged with two distinct adapters, **When** the simulation runs, **Then** the results report per-adapter TTFT and per-adapter throughput for both adapters.
2. **Given** a workload with no adapters configured and no adapter capacity set, **When** the simulation runs, **Then** the output is identical to the same run produced without the LoRA subsystem.
3. **Given** a mixed workload where some requests carry an adapter and some target the base model only, **When** the simulation runs, **Then** base-model requests are attributed to no adapter and adapter requests are attributed to their adapter.

---

### User Story 2 - Model finite per-instance adapter capacity with eviction (Priority: P1)

A researcher sets a maximum number of adapter slots per instance. The simulator
tracks which adapters are resident on each instance and, when serving a request
whose adapter is not resident would exceed the slot limit, evicts the
least-recently-used adapter to make room.

**Why this priority**: Finite GPU adapter slots are the core constraint that
placement research reasons about. Modeling capacity and eviction is what makes
adapter churn observable and is a prerequisite for any placement/routing policy
to matter.

**Independent Test**: Configure an instance with capacity N, drive traffic for
M > N distinct adapters, and verify the resident set never exceeds N at any
point and that evictions follow least-recently-used order.

**Acceptance Scenarios**:

1. **Given** an instance with adapter capacity N and traffic for M > N distinct adapters, **When** the simulation runs, **Then** the instance's resident adapter set never exceeds N adapters.
2. **Given** an instance at full capacity, **When** a request for a non-resident adapter is served, **Then** the least-recently-used resident adapter is evicted and the requested adapter becomes resident.
3. **Given** an adapter that is required by a currently in-flight request, **When** capacity pressure occurs, **Then** that adapter is not evicted while it is in use.

---

### User Story 3 - Model adapter cost physics (Priority: P1)

The simulator charges the physical costs of serving LoRA adapters: a one-time
load latency when a request's adapter is cold (not resident on the target
instance), a per-step compute overhead that grows with the number of unique
adapters in a batch and their maximum rank, and a per-adapter HBM footprint that
reduces the memory available for the KV cache.

**Why this priority**: Without cost physics, adapter attribution and capacity
are bookkeeping with no effect on latency, throughput, or memory — the outcomes
that placement policies exist to improve. This slice makes the simulation
physically meaningful.

**Independent Test**: With adapters configured, verify that (a) a request whose
adapter is cold shows measurably higher TTFT than an otherwise-identical warm
request, (b) batches containing more unique adapters (or higher rank) take
longer per step, and (c) usable KV-cache memory decreases as adapters load,
with total memory conserved.

**Acceptance Scenarios**:

1. **Given** two identical requests for the same adapter under identical load, the first cold and the second warm, **When** the simulation runs, **Then** the cold request's TTFT is higher by the modeled load latency and the warm request incurs no load latency.
2. **Given** two batches identical except that one contains more unique adapters (or a higher maximum rank), **When** each step executes, **Then** the batch with more unique adapters / higher rank has a longer step time.
3. **Given** an instance with adapters resident, **When** memory is accounted, **Then** the KV-cache capacity is reduced by the adapters' footprint and the sum of allocated, free, and adapter-reserved memory equals the total.

---

### User Story 4 - Route requests with a LoRA-aware placement scorer (Priority: P2)

A researcher enables a LoRA-aware routing/placement scorer. The scorer reads
each instance's resident adapter set and can prefer instances that already hold
a request's adapter (warm placement), reducing cold loads and eviction churn.
This scorer is the extension point where placement/routing research policies
attach.

**Why this priority**: This is the research payoff — the control surface for the
policies the harness exists to evaluate. It is P2 rather than P1 because it
delivers value only on top of adapter identity (US1) and the resident-adapter
set (US2).

**Independent Test**: Under a skewed adapter-popularity workload, run once with
an adapter-blind router and once with the LoRA-aware scorer enabled, and verify
the scorer routes requests toward instances already holding their adapter and
reduces total adapter loads/evictions.

**Acceptance Scenarios**:

1. **Given** an instance already holding a request's adapter and another that does not, **When** the LoRA-aware scorer ranks candidates, **Then** the instance holding the adapter is preferred (all else equal).
2. **Given** a skewed adapter-popularity workload, **When** run with the LoRA-aware scorer versus an adapter-blind router, **Then** total adapter loads and evictions are lower with the LoRA-aware scorer.
3. **Given** the LoRA-aware scorer is not selected, **When** the simulation runs, **Then** routing behavior is unchanged from today.

---

### User Story 5 - Validate adapter-cost fidelity against a calibrated reference (Priority: P3)

A researcher imports the pre-fitted adapter rank→latency tables for a supported
reference configuration and compares BLIS's adapter-aware TTFT predictions
against the reference to quantify fidelity.

**Why this priority**: Fidelity validation increases confidence in the ported
physics but is not required to begin policy research; it is optional and bounded
to the pre-fitted reference configurations.

**Independent Test**: For a supported reference configuration, import its
rank→latency table, run the adapter-aware simulation, and produce a comparison
report of predicted versus reference TTFT with an error metric.

**Acceptance Scenarios**:

1. **Given** a supported reference configuration with a pre-fitted rank→latency table, **When** the adapter-aware simulation is compared to the reference, **Then** a report is produced with a per-adapter error metric.
2. **Given** the comparison completes, **When** the error metric is evaluated, **Then** the adapter-aware TTFT prediction is within the agreed error bound for that configuration.

---

### Edge Cases

- **Adapters present but zero capacity**: A configuration that assigns adapters to requests while setting an instance adapter capacity of zero is rejected at startup with a clear error (rather than silently dropping adapters or looping on eviction).
- **Repeated adapter in one batch**: An adapter appearing on multiple requests in the same batch is counted once toward the unique-adapter compute overhead; its rank contributes once to the maximum.
- **Adapter too large to fit alongside KV**: If an adapter's footprint cannot fit in an instance's memory, the request is treated as unservable in the same way an oversized request is today — no negative or over-allocated memory.
- **Partial adapter workloads**: When only some requests carry adapters, base-model-only requests incur no adapter load latency, no per-step adapter overhead, and no adapter footprint.
- **Cross-instance churn**: An adapter resident on instance A does not make it warm on instance B; a request routed to B incurs a cold load on B.
- **Eviction of in-use adapter**: An adapter needed by a currently running request is not evicted until it is no longer in use.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST allow each request to carry an optional adapter identity — a global string id that references an entry in the pre-declared adapter registry — specified through workload configuration (per-client and per-cohort). The request references the adapter by id only; it does not carry the adapter's rank.
- **FR-001a (adapter registry)**: The set of adapters MUST be a pre-declared registry mapping each adapter's global string id (unique across the simulation, each belonging to a single base model) to its rank. Adapter rank, load latency, and HBM footprint are derived from this registry — not carried per request.
- **FR-002 (no-op default)**: When no request carries an adapter and no adapter capacity is configured, simulation output MUST be identical to output produced without the LoRA subsystem — no added latency, no memory charge, and no routing effect.
- **FR-003**: The system MUST report per-adapter metrics, including adapter load count, TTFT distribution by adapter, and per-adapter throughput.
- **FR-004**: The system MUST maintain, for each instance, the set of currently resident adapters, bounded by a configurable per-instance capacity.
- **FR-005**: When serving a request whose adapter is not resident would exceed an instance's capacity, the system MUST evict the least-recently-used resident adapter that is not currently in use.
- **FR-006**: An instance's resident adapter set MUST never exceed its configured capacity at any point in the simulation.
- **FR-007**: The system MUST charge a one-time load latency when a request's adapter is cold (not resident on the target instance) and MUST NOT charge load latency when the adapter is already resident (warm). The load latency is applied as a **pre-admission gate**: a cold request MUST NOT join a batch until its adapter is resident, the load latency is charged before scheduling (counted toward the request's queueing delay / TTFT), and concurrent adapter loads on the same instance are serialized.
- **FR-008**: Adapter load latency MUST be non-negative and MUST scale with the adapter's rank (obtained from the adapter registry) according to the configured cost model.
- **FR-009**: The system MUST add a per-step compute overhead that scales with the number of unique adapters present in the batch and the maximum rank among them.
- **FR-010**: The system MUST account for the HBM footprint of resident adapters — derived from each adapter's rank — by reducing the memory available to the KV cache while adapters are resident, such that total memory remains conserved (allocated + free + adapter-reserved = total).
- **FR-011**: The system MUST provide a LoRA-aware routing/placement scorer that can be selected among the available routing scorers and that reads each instance's resident-adapter set.
- **FR-012**: The LoRA-aware scorer MUST be able to prefer instances that already hold a request's adapter (warm placement).
- **FR-013**: Adapter-aware routing decisions MUST rely only on control-plane-visible signals (resident-adapter set, adapter identity, adapter rank) and MUST NOT read a request's realized output length, preserving the oracle knowledge boundary. The freshness of the resident-adapter signal used for routing MUST be documented.
- **FR-014**: Adapter cost effects MUST be applied consistently across the supported latency-model backends so that results agree between them where both are applicable.
- **FR-015 (determinism)**: For a given seed and adapter configuration, the system MUST produce byte-identical results across repeated runs.
- **FR-016**: Adapter configuration MUST be independently validatable, rejecting invalid values (e.g., negative capacity, negative rank, adapters present with zero capacity) at startup with a clear error message.
- **FR-017**: Request conservation MUST continue to hold with adapters configured — every injected request is accounted for in exactly one terminal or in-progress state at simulation end.
- **FR-018** *(optional, supports US5)*: The system MUST support importing pre-fitted adapter rank→latency tables for supported reference configurations and producing a comparison of adapter-aware predictions against the reference with an error metric.

### Key Entities *(include if feature involves data)*

- **Adapter**: A LoRA adapter identified by a global string id (unique across the simulation), belonging to a single base model, characterized by a rank. Requests reference an adapter by id.
- **Adapter registry**: The pre-declared mapping from each adapter's global string id to its rank. It is the single source of truth from which rank, load latency, and HBM footprint are derived. (The calibrated reference operates with one base model per run.)
- **Request (extended)**: An existing request that additionally carries an optional adapter id referencing the registry; when absent, the request targets the base model only. The request does not carry rank.
- **Resident-adapter set**: A per-instance collection of currently loaded adapters, bounded by a capacity and ordered by recency of use, supporting least-recently-used eviction and pinning of in-use adapters.
- **Adapter cost model**: The three cost terms that compose onto the calibrated base engine — load latency (by rank, additive), per-step compute overhead (by unique-adapters × maximum rank, relative), and per-adapter HBM footprint (memory reserved from the KV budget).
- **Adapter metrics**: Per-adapter aggregates surfaced in results — load counts, TTFT by adapter, and per-adapter throughput.
- **Rank→latency table**: A pre-fitted mapping (from the calibrated reference) from adapter rank to load latency, importable for fidelity comparison.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001 (no-op fidelity)**: Across every existing regression/golden scenario, running with the LoRA subsystem present but no adapters configured produces zero differences versus the pre-feature output.
- **SC-002 (config-only usability)**: A researcher can assign adapters to a workload and obtain per-adapter TTFT and throughput breakdowns through configuration alone, without modifying simulator code.
- **SC-003 (capacity invariant)**: In 100% of simulation steps across all adapter scenarios, no instance's resident adapter set exceeds its configured capacity.
- **SC-004 (cold-vs-warm effect)**: Under identical load, cold-adapter requests exhibit higher TTFT than warm-adapter requests for the same input, by the modeled load latency.
- **SC-005 (routing payoff)**: Under a skewed (e.g., Zipfian) adapter-popularity workload, enabling the LoRA-aware scorer reduces total adapter loads by at least 30% compared to an adapter-blind least-loaded router.
- **SC-006 (determinism)**: For a fixed seed and adapter configuration, repeated runs produce byte-identical results (zero-diff across runs).
- **SC-007 (bounded fidelity)**: For each of the two supported reference configurations, adapter-aware prediction error against the reference is within an agreed bound (target: MAPE ≤ 20%) on **both** per-adapter TTFT (where cold-load latency manifests) **and** per-adapter throughput (which isolates the per-step compute-overhead term).

## Assumptions

- **No-op-default subsystem**: The LoRA subsystem is classified as a Subsystem Module and is fully inert when unconfigured, consistent with the project's extension framework and its determinism invariant.
- **Deltas onto a calibrated base**: Adapter cost physics are ported as functional forms and fitted constants from an external Digital Twin reference and composed as deltas onto BLIS's separately calibrated base latency — load latency is additive, per-step overhead is a relative multiplier. Absolute step-time constants from the reference are not dropped in wholesale.
- **Static memory accounting first**: The initial adapter memory model is a static per-adapter subtraction from the KV budget. A dynamic runtime KV↔adapter memory tradeoff is deferred to future work.
- **Bounded calibration claim**: Fidelity claims are limited to the two pre-fitted reference configurations (Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct). Transferring adapter physics to other model/GPU configurations requires profiling that is not automated in-repo.
- **Cold-load gating resolved (pre-admission)**: A cold adapter load blocks the request pre-admission (it cannot join a batch until its adapter is resident); load latency is charged before scheduling and adapter loads serialize per instance (see Clarifications, FR-007). The design doc realizes this with a dedicated per-instance adapter-load-completion event that gates batch entry (design doc D1); it explicitly rejects a per-step pending-latency drain (which would charge the delay after batching, muddying the cold-vs-warm TTFT signal).
- **Adapter as registry entry**: Adapters are a pre-declared registry (`id → rank`), mirroring the Digital Twin's `served_adapters` / `served_adapters_sizes`; rank drives both load latency and HBM footprint. Requests reference adapters by id.
- **Single base model per instance**: Multi-base-model-per-instance beyond the existing model-tag mechanism is not addressed; adapter ids are globally unique and each maps to one base model.

## Out of Scope

- Adapter *training*, rank/accuracy trade-offs, and adapter internal architecture.
- Cross-GPU bin-packing placement as a *solver*. This feature provides the control surface (the scorer/placement extension point) on which such policies are researched; the policies themselves are separate research payload.
- Multi-base-model-per-instance beyond the existing model-tag mechanism.
- A dynamic, runtime-negotiated KV-cache ↔ adapter memory budget (only static subtraction is in scope).
- An in-repo adapter-profiling / coefficient-fitting harness (calibration is limited to importing pre-fitted reference tables).

## Dependencies

- The existing model-tag routing filter and per-model metrics plumbing (the path the adapter identity rides).
- The existing tiered-memory transfer / pending-latency machinery (the pattern reused for cold adapter loads).
- The existing latency model and routing snapshot (extended, not replaced), to keep new interfaces serving multiple implementations.
- An external calibrated reference (the Digital Twin and its two pre-fitted configurations) as the validation source for the adapter cost terms.
