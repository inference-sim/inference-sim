# Prompt: Update Macro Plan to v3.0

Use an agent team with 2 teammates to update the macro implementation plan, incorporating the simplification assessment findings.

## Context

- **Current macro plan:** `docs/plans/2026-02-11-macro-implementation-plan-v2.md` (v2.3, 21 PRs)
- **Simplification assessment:** `docs/plans/2026-02-13-simplification-assessment.md` (reduces to 16 PRs total, 13 remaining)
- **PRs 1-4:** Already merged. Their entries in the plan stay as-is.
- **Goal:** Produce v3.0 of the macro plan as a single source of truth, replacing the assessment as the active planning document.

## Team Structure

### Teammate 1: "plan-updater"

Read both documents thoroughly. Then update the macro plan:

1. **Add v3.0 revision notes** to Section A (Revision Notes). Summarize:
   - Dropped backward compatibility constraint (golden tests remain)
   - Simplification assessment conducted (reference file path)
   - Constructor collapse, unified CLI path, field privatization, interface dedup
   - 5 PRs merged (Priority+Scheduler, AutoScaler+Actuation, TieredKV+Transfer, Traces+Counterfactual, Adapters combined, P/D+KVTransfer combined)
   - Unified event queue assessed but deferred
   - 21 PRs reduced to 16 total (13 remaining after PR 4)

2. **Replace Section I (PR Series)** from PR 5 onward with the assessment's revised PR series:
   - PR 5: Architectural Simplification (NEW)
   - PR 6: RoutingPolicy (was v2.3 PR 6)
   - PR 7: PriorityPolicy + InstanceScheduler (merged v2.3 PR 5 + PR 7)
   - PR 8: PolicyBundle (was v2.3 PR 8)
   - PR 9: RawMetrics + Pathological (was v2.3 PR 9)
   - PR 10: Workload Generator (was v2.3 PR 10)
   - PR 11: AutoScaler + Actuation (merged v2.3 PR 11 + PR 12)
   - PR 12: Tiered KV + Transfer (merged v2.3 PR 13 + PR 14)
   - PR 13: Decision Traces + Counterfactual (merged v2.3 PR 17 + PR 18)
   - PR 14: P/D Architecture + KV Transfer (merged v2.3 PR 15 + PR 16)
   - PR 15: Framework Adapters (merged v2.3 PR 19 + PR 20)
   - PR 16: Integration Tests (was v2.3 PR 21)

   Use the Tier 1 table format from the assessment. Include "v2.3 → v3.0 change" notes per PR.

3. **Replace Section J (Dependency DAG)** with the assessment's revised DAG. Update the parallel development matrix and timeline.

4. **Keep Sections B-H and K intact.** These are still accurate:
   - B: Repository Recon (update LOC counts if changed)
   - C: Objectives (add note about dropped backward compat)
   - D: Concept Model (unchanged)
   - E: Risk Register (add new risks from simplification)
   - F: Design Patterns (unchanged)
   - G: Interface Catalog (unchanged)
   - H: Architecture Diagram (update if needed to reflect unified CLI path)
   - K: Validation Strategy (unchanged)

5. **Add a v2.3 → v3.0 PR mapping table** (from the assessment's Appendix) so readers can trace the renumbering.

6. **Update the Interface Freeze checkpoint** from "after PR 8" (still correct number, just confirm the content).

7. **Update the Research-Ready checkpoint** (still after PR 9, confirm scope).

Write the updated plan to the SAME file: `docs/plans/2026-02-11-macro-implementation-plan-v2.md`
(The filename stays the same; the version number is in the document header.)

### Teammate 2: "consistency-checker"

After the plan-updater finishes, review the updated macro plan for:

1. **Cross-reference accuracy:**
   - Do all "Depends On" entries reference correct PR numbers?
   - Do all "Parallel With" entries reference correct PR numbers?
   - Are phase boundaries correct?
   - Do file path references match the actual codebase?

2. **Content consistency:**
   - Do the new PR entries in Section I match the concept model in Section D?
   - Does the risk register in Section E account for simplification risks?
   - Is the architecture diagram in Section H consistent with the unified CLI path?
   - Are LOC estimates reasonable given current codebase size?

3. **Completeness:**
   - Is every PR from the assessment represented?
   - Are the checkpoints (Mock Study, Interface Freeze, Research-Ready) correctly placed?
   - Does the timeline add up?

4. **No stale content:**
   - No references to old PR numbers (v2.3 numbering) in prose sections
   - No "PR 19", "PR 20", "PR 21" references (these are now PR 15, PR 16)
   - No references to dropped concepts (e.g., "separate single-instance path")

Flag any issues found. The plan-updater must fix them before the lead writes the final file.

## Ground Rules

- Read the actual assessment document, not summaries
- Preserve the quality and tone of the existing macro plan
- The v3.0 plan must be self-contained — a reader should not need the assessment document to understand the plan (though it can reference it for rationale)
- PRs 1-4 entries are historical record; do not modify them
- Every PR entry must include the "v2.3 → v3.0 change" annotation

## Deliverable

The updated macro plan, written to `docs/plans/2026-02-11-macro-implementation-plan-v2.md`.
