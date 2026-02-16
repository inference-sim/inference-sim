# Agent Team Prompt: BLIS Architecture Simplification Assessment

> **Usage:** Paste the prompt below into Claude Code with Agent Teams enabled to launch
> the assessment team. The team will explore simplification opportunities in parallel
> and produce a revised macro plan.
>
> **Prerequisites:** `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings.json

---

## Prompt

Create an agent team with 4 teammates to assess architectural simplifications for the BLIS inference simulator. Use delegate mode — do NOT implement anything yourself; coordinate the teammates and synthesize their findings into a revised macro plan.

### Context

BLIS is a discrete-event LLM inference simulator in Go. We just merged PR4 of a 21-PR macro plan (`docs/plans/2026-02-11-macro-implementation-plan-v2.md`). The codebase is ~5,500 LOC across `sim/`, `sim/cluster/`, `sim/policy/`, and `cmd/`.

**Key change in requirements:** We are dropping the backward compatibility constraint that `sim.Simulator` and its constructors/public API remain unchanged. The ONLY hard requirement is that the golden dataset tests continue to pass (`testdata/goldendataset.json` — these verify exact metric output for specific workloads and seeds). Everything else — constructor signatures, public fields, package boundaries, CLI code paths — is fair game for simplification.

**Current architecture (post-PR4):**
- `sim.Simulator`: core single-instance engine with 25+ public fields, 17-parameter constructors, its own `EventQueue` (min-heap by timestamp only)
- `sim/cluster/InstanceSimulator`: thin wrapper around `*sim.Simulator` adding ID, run-once guard, observation methods (`QueueDepth()`, `BatchSize()`, `KVUtilization()`, `FreeKVBlocks()`)
- `sim/cluster/ClusterSimulator`: orchestrator with its OWN `ClusterEventQueue` (min-heap by timestamp+priority+seqID) containing `ClusterArrivalEvent`, `AdmissionDecisionEvent`, `RoutingDecisionEvent`. Has a shared-clock loop that interleaves cluster events and per-instance events.
- `sim/policy/admission.go`: duplicated `AdmissionPolicy` interface (also in `cluster.go`) due to import cycle
- `sim/cluster/workload.go`: duplicated workload generation logic (mirrors `sim/workload_config.go`)
- `cmd/root.go`: two completely separate code paths for `numInstances == 1` vs `> 1`

**The central question:** Can we simplify this architecture — particularly by using a SINGLE event queue instead of N+1 separate queues — while keeping the golden tests passing and making the remaining PRs (5-21) simpler to implement?

### Team Structure

Spawn 4 teammates. Each should read the codebase thoroughly before drawing conclusions. Every claim must cite file:line. No hallucinated behaviors.

**Teammate 1 — "unified-queue-analyst":**
Assess the feasibility of a single unified event queue that holds ALL events (both what are currently "cluster events" and "instance events"). Specifically:

1. Read `sim/simulator.go` (the event loop, `ProcessNextEvent`, `Step`, `makeRunningBatch`), `sim/event.go` (all 6 event types), `sim/cluster/cluster.go` (the shared-clock loop at lines 179-220), and `sim/cluster/cluster_event.go` (the 3 cluster event types).
2. Determine: can instance events (ArrivalEvent, QueuedEvent, StepEvent, etc.) be made to target a specific instance ID, so they can coexist in a single heap with cluster events? What would the unified `Event` interface look like?
3. Identify the key obstacle: instance events currently call `Execute(*Simulator)` which mutates a specific `sim.Simulator`. In a unified queue, how does an event know which instance to target? Options: (a) events carry an instance ID and the queue dispatches, (b) events carry a `*Simulator` pointer, (c) a wrapper event type.
4. Analyze the ordering implications. Currently instance events are timestamp-only; cluster events are (timestamp, priority, seqID). A unified queue needs a single consistent ordering. What should it be?
5. Assess impact on determinism. The current cluster loop processes cluster events before instance events at the same timestamp (`<=` on line 206 of cluster.go). How is this preserved in a unified queue?
6. Estimate LOC impact: how much code would change? What gets simpler? What gets more complex?
7. Prototype the unified `Event` interface and `EventQueue` ordering in pseudocode.

Deliver a feasibility report with a clear YES/NO recommendation and trade-offs.

**Teammate 2 — "simplification-scout":**
Survey ALL simplification opportunities enabled by dropping backward compatibility (except golden tests). Read the full codebase — every `.go` file in `sim/`, `sim/cluster/`, `sim/policy/`, and `cmd/`. For each simplification:

1. **Constructor collapse**: Can `NewSimulator`'s 17 params become an options struct? What about `NewSimulatorWithoutWorkload`? Can these be merged? Cite all call sites.
2. **Field privatization**: Which of `sim.Simulator`'s 25+ public fields can become private with accessor methods? Which fields does `InstanceSimulator` actually access? Which do tests access?
3. **Eliminate duplicated code**: (a) `AdmissionPolicy` interface defined in both `cluster.go:18` and `policy/admission.go:11` — can the import cycle be broken? (b) Workload generation in both `sim/workload_config.go` and `cluster/workload.go` — can one call the other? (c) Test helpers still duplicated anywhere after the `testutil` package?
4. **Unified CLI path**: `cmd/root.go` has two paths. Since `ClusterSimulator` with N=1 passes golden equivalence tests, can we eliminate the single-instance path entirely? What are the edge cases?
5. **Package restructuring**: Does `sim/policy/` need to be a separate package, or could admission/routing/scheduling interfaces live in `sim/cluster/`? What about moving core types to reduce import cycles?
6. **Dead code / over-engineering**: Any code that exists only for backward compat that we can now remove?

Rank simplifications by impact (LOC saved, complexity reduced, future PRs simplified) and risk.

**Teammate 3 — "plan-reviser":**
Given the simplifications identified by teammates 1 and 2, draft a REVISED macro plan for PRs 5-21. Use the format and quality bar from `docs/plans/macroplanprompt.md`. Specifically:

1. Read the current plan (`docs/plans/2026-02-11-macro-implementation-plan-v2.md`) thoroughly, especially the PR series starting at PR 5 (line ~1096).
2. Wait for findings from teammates 1 and 2 before drafting. Ask them questions if their reports are ambiguous.
3. If the unified queue is feasible, factor in a "PR 4.5" or amendment to PR 5 that introduces it. If not, explain what the revised architecture looks like.
4. Identify PRs that can be ELIMINATED or MERGED due to simplifications. The current plan has 21 PRs — can we get to 15 or fewer?
5. Re-evaluate the control plane / data plane separation. Is it still the right abstraction, or does the unified queue change this?
6. For each remaining PR, provide Tier 1 (human review summary, 15 lines) and note what changed from the v2.3 plan and why.
7. Ensure every PR is still independently reviewable and exercisable after merge.

Output format: A complete revised PR series (PR 5 onward) following the Phase 6 format from `docs/plans/macroplanprompt.md`.

**Teammate 4 — "devils-advocate":**
Challenge the proposals from teammates 1-3. Your job is to find hidden risks, coupling, and flawed assumptions. Specifically:

1. **Golden test fragility**: Read `sim/simulator_test.go`, `sim/cluster/instance_test.go`, and `sim/cluster/cluster_test.go`. What EXACTLY do the golden tests verify? Are there subtle ordering dependencies that a unified queue could break? The golden tests check bit-exact metrics — could event ordering changes affect TTFT/ITL/E2E even if final counts match?
2. **RNG stability**: Read `sim/rng.go`. The `PartitionedRNG` has a backward-compat special case for `SubsystemWorkload`. Would any proposed simplifications change the RNG draw sequence, breaking golden test determinism?
3. **Performance**: Would a single queue with N*events be slower than N small queues? Profile the heap operations: `O(log(total_events))` vs `O(log(events_per_instance))`. For N=16 instances with 10K events each, is this meaningful?
4. **Abstraction loss**: The current control plane / data plane separation models real systems (router vs instances). Does flattening into a single queue lose modeling fidelity that future PRs (P/D disaggregation, auto-scaling) need?
5. **Incremental migration risk**: Can we actually make these simplifications without a massive "rewrite PR" that's unreviewable? What's the safe migration path?
6. **What the plan gets wrong**: Read the v2.3 plan's risk register and design bug prevention sections. Do the proposed simplifications introduce NEW risks not covered there?

Deliver a risk report. For each risk, state severity (blocks adoption / complicates adoption / cosmetic) and propose a mitigation.

### Deliverables

After all 4 teammates report, synthesize their findings into:

1. **Decision matrix**: For each proposed simplification, a row with: description, LOC impact, risk level, golden-test-safe (Y/N), recommended (Y/N).
2. **Revised architecture diagram**: Show the simplified architecture (ASCII art, similar to the current plan's diagrams in section H).
3. **Revised PR series**: The plan-reviser's output, amended with the devil's advocate's mitigations.
4. **Migration strategy**: How to get from today's code to the simplified architecture safely, which PRs to do first.

Write the final synthesized assessment to `docs/plans/2026-02-13-simplification-assessment.md`.

### Ground Rules

- Every behavioral claim about existing code MUST cite file:line. No hallucinations.
- Read the actual code, not just CLAUDE.md or the plan documents. CLAUDE.md has had inaccuracies before.
- The golden dataset tests (`testdata/goldendataset.json`) are the single non-negotiable constraint. If a simplification risks breaking them, it must include a concrete mitigation.
- Prefer concrete pseudocode over hand-wavy descriptions.
- Challenge each other's findings. If you disagree with a teammate, say so explicitly and cite evidence.
