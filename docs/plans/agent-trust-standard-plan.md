# Agent Trust Boundary Standard Implementation Plan

**Goal:** Formalize three trust tiers for agent operations into a standards document so new sessions start with explicit trust boundaries instead of learning from MEMORY.md notes.
**Source:** [GitHub Issue #675](https://github.com/inference-sim/inference-sim/issues/675)
**Closes:** Fixes #675

**The problem today:** Hard-won lessons about agent reliability (from #381, #390, #430) exist only as informal notes in MEMORY.md. New sessions or contributors have no structured reference for which agent outputs to verify, leading to repeated trust failures.

**What this PR adds:**
1. A new standards document (`docs/contributing/standards/agent-trust.md`) defining three trust tiers: Trusted, Verify-after, and Never-trust
2. Documented failure modes with links to the PRs where they occurred
3. A CLAUDE.md pointer in the Standards section for discoverability
4. An updated source-of-truth map in `principles.md`

**Why this matters:** Agents are increasingly central to the BLIS development workflow (convergence reviews, plan execution, hypothesis experiments). Explicit trust boundaries prevent the same class of failures from recurring.

**Architecture:** Pure documentation change — four markdown files modified/created. No Go code changes.

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** New standards document in `docs/contributing/standards/`
2. **Adjacent blocks:** `principles.md` (source-of-truth map), `CLAUDE.md` (working copy pointer), `pr-workflow.md` (referenced for verification procedures)
3. **Invariants touched:** None (no code changes)
4. **Construction site audit:** N/A (no structs modified)

---

## Part 1: Design Validation

### A) Executive Summary

This PR creates `docs/contributing/standards/agent-trust.md` — a new standards document formalizing three trust tiers for agent operations. The tiers classify operations by the verification they require: Trusted (none), Verify-after (build/test/diff), and Never-trust (independent evaluation). The document includes known failure modes from real PRs (#381, #390, #430) as evidence. CLAUDE.md gets a 1-line pointer in the Standards section. The source-of-truth map in `principles.md` gets a new row.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Trust Tier Classification
- GIVEN a reader looking up any agent operation
- WHEN they consult `docs/contributing/standards/agent-trust.md`
- THEN they find it classified in exactly one of three tiers: Trusted, Verify-after, or Never-trust, with the required verification action for that tier

BC-2: Failure Mode Documentation
- GIVEN a reader wanting to understand why a tier exists
- WHEN they read the Failure Modes section
- THEN each failure mode links to the specific PR/issue where it occurred and describes the observable consequence

BC-3: Discoverability
- GIVEN a new session reading CLAUDE.md or a contributor browsing docs/contributing/
- WHEN they reach the Standards section
- THEN they find a pointer to `agent-trust.md` with a brief description of its purpose

BC-4: Source-of-Truth Map Consistency
- GIVEN the source-of-truth map in `principles.md`
- WHEN a reader looks up `agent-trust.md`
- THEN it appears as a canonical source with CLAUDE.md (pointer) and docs/contributing/index.md (landing page table) listed as working copies

**Negative contracts:**

BC-5: No Duplication of Existing Process
- GIVEN the existing verification gate in `pr-workflow.md` Step 4.5
- WHEN `agent-trust.md` describes the "Verify-after" tier
- THEN it cross-references `pr-workflow.md` rather than restating the exact verification commands

### C) Component Interaction

```
agent-trust.md (NEW - canonical source)
    ├── Referenced by: CLAUDE.md (pointer in Standards section)
    ├── Referenced by: principles.md (source-of-truth map row)
    └── Cross-references: pr-workflow.md (verification gate)
```

No data flow — pure documentation dependency.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue lists operations per tier as examples | Plan uses same operations but refines wording for clarity | CLARIFICATION |
| Issue says "1-2 line pointer" in CLAUDE.md without specifying section | Plan places pointer in Standards section alongside rules.md, invariants.md, principles.md | CLARIFICATION |
| Issue doesn't mention cross-referencing pr-workflow.md | Plan adds cross-references to avoid duplication (BC-5) | ADDITION |
| Issue doesn't mention docs/contributing/index.md | Plan adds row to Standards table for human contributor navigation | ADDITION |

### E) Review Guide

**Tricky part:** Getting the tier boundaries right — an operation classified as "Trusted" that should be "Verify-after" creates a false sense of safety. Scrutinize the operations listed in each tier.

**Safe to skim:** CLAUDE.md pointer (1 line) and source-of-truth map row (1 table row) — mechanical changes.

**Known debt:** None.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to create:
- `docs/contributing/standards/agent-trust.md` — the new standards document

Files to modify:
- `CLAUDE.md` — add 1-line pointer in Standards section
- `docs/contributing/standards/principles.md` — add row to source-of-truth map
- `docs/contributing/index.md` — add row to Standards table

No dead code. No Go changes.

### G) Task Breakdown

#### Task 1: Create agent-trust.md (BC-1, BC-2, BC-5)

**Files:** create `docs/contributing/standards/agent-trust.md`

**Test:** Verify document structure manually:
- Three tiers present with operations and verification requirements
- Failure modes section with PR/issue links
- Cross-reference to pr-workflow.md (not duplication)

**Impl:**

```markdown
# Agent Trust Boundaries

Agent operations have different reliability characteristics. This standard
defines three trust tiers so that sessions and contributors know which outputs
to verify.

## Trust Tiers

| Tier | Operations | Verification Required |
|------|-----------|----------------------|
| **Trusted** | File reads, searches, grep, lint output, build output | None — results are deterministic and verifiable by output |
| **Verify-after** | Code edits, construction site updates, file writes, refactoring | Run the [verification gate](../pr-workflow.md#after-convergence-verification-gate). |
| **Never-trust** | Convergence self-assessment, "all done" claims, severity classification, coverage claims, "0 issues found" reports | Human or orchestrator must independently evaluate the evidence |

### Trusted

Operations whose output is deterministic and machine-verifiable. The tool
either succeeds or fails visibly — there is no gray zone where the agent
could misinterpret the result.

Examples: `Read` (file contents), `Grep` (search results), `Glob` (file
matches), `go build` exit code, `golangci-lint` output.

### Verify-after

Operations that mutate state. The agent may believe it made the correct
change, but the only proof is running the build and test suite afterward.

Examples: code edits, struct field additions (construction site updates),
file creation, multi-file refactoring.

**Required verification:** Run the
[verification gate](../pr-workflow.md#after-convergence-verification-gate)
after any Verify-after operation.

### Never-trust

Subjective assessments where the agent's self-report has been empirically
unreliable. These require independent verification by a human or by an
orchestrator using different evidence than the agent's claim.

Examples: "all construction sites updated," "0 CRITICAL issues," "review
converged," "tests cover all contracts," "coverage is complete."

## Known Failure Modes

Each failure mode below was discovered in a real PR. The tier system exists
because these failures occurred.

### FM-1: Construction site misses (during #381 implementation)

**Tier violated:** Never-trust (the completeness *claim* was trusted without verification)

**What happened:** During SimConfig decomposition (#381 implementation), a
sub-agent reported "all construction sites updated" for a struct field addition.
Two construction sites were missed, causing silent field-zero bugs. The operation
itself (code edits) is Verify-after, but the agent's completeness claim ("all
sites updated") is Never-trust.

**Lesson:** Completeness claims about Verify-after operations are Never-trust.
Always `grep 'StructName{'` after the agent claims completion. See also R4.

### FM-2: Severity inflation/deflation (during #390 review)

**Tier violated:** Never-trust (treated as Trusted)

**What happened:** During a convergence review of #390 (hypothesis batch PR),
the reviewing agent reported "0 CRITICAL, 0 IMPORTANT" when the artifact
actually had 3 CRITICAL and 18 IMPORTANT issues. The team lead accepted the
self-report without independently reading the review output.

**Lesson:** Convergence self-assessment is a Never-trust operation. The
orchestrator must independently tally severity counts from the raw review
output, never from the agent's summary.

### FM-3: Premature convergence claim (#430)

**Tier violated:** Never-trust (treated as Trusted)

**What happened:** During a convergence review, the agent reported convergence
after a single round without re-running the review to verify that fixes
actually resolved the issues. The team lead accepted the claim.

**Lesson:** "Review converged" is a Never-trust claim. Convergence requires
evidence: a clean round with zero CRITICAL and zero IMPORTANT findings across
all perspectives. The orchestrator must verify the round ran and produced
clean results. See the [convergence protocol](../convergence.md).

## Relationship to Other Standards

- **Antipattern rules** ([rules.md](rules.md)): R4 (construction site audit)
  is the specific rule that FM-1 violates. The trust tiers provide the
  meta-framework for when to apply verification.
- **PR workflow** ([pr-workflow.md](../pr-workflow.md)): The verification gate
  in Step 4.5 is the procedural implementation of Verify-after tier
  requirements.
- **Convergence protocol** ([convergence.md](../convergence.md)): The
  convergence protocol's round-based evidence requirement is the procedural
  implementation of Never-trust tier requirements for review claims.
```

**Verify:** Read the created file, confirm all three tiers documented, all three failure modes linked to PRs, cross-references present.
**Commit:** `docs(standards): add agent trust boundary standard (BC-1, BC-2, BC-5)`

#### Task 2: Add CLAUDE.md pointer (BC-3)

**Files:** modify `CLAUDE.md`

**Test:** Verify the Standards section lists agent-trust.md.

**Impl:** In the `### Standards (what rules apply)` section, after the `experiments.md` line, add:

```markdown
- `docs/contributing/standards/agent-trust.md`: **Agent trust boundaries** — three trust tiers (Trusted, Verify-after, Never-trust) for agent operations, with known failure modes
```

**Verify:** Read CLAUDE.md Standards section, confirm pointer present.
**Commit:** `docs(claude-md): add agent trust boundary pointer (BC-3)`

#### Task 3: Update source-of-truth map (BC-4)

**Files:** modify `docs/contributing/standards/principles.md`

**Test:** Verify the source-of-truth map table contains the new row.

**Impl:** In the source-of-truth map table in `principles.md`, add after the "Engineering principles" row:

```markdown
| Agent trust boundaries | `docs/contributing/standards/agent-trust.md` | CLAUDE.md (pointer), `docs/contributing/index.md` (landing page table) |
```

**Verify:** Read principles.md source-of-truth map, confirm new row present.
**Commit:** `docs(principles): add agent-trust.md to source-of-truth map (BC-4)`

#### Task 4: Add contributing index entry (BC-3)

**Files:** modify `docs/contributing/index.md`

**Test:** Verify the Standards table lists agent-trust.md.

**Impl:** In the `## Standards` table in `docs/contributing/index.md`, after the "Experiment Standards" row, add:

```markdown
| [Agent Trust Boundaries](standards/agent-trust.md) | Three trust tiers for agent operations |
```

**Verify:** Read contributing index.md Standards table, confirm new row present.
**Commit:** `docs(contributing): add agent-trust.md to index (BC-3)`

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|--------------|
| BC-1 | Task 1 | Manual | Three tiers present with operations and verification |
| BC-2 | Task 1 | Manual | Three failure modes with PR links |
| BC-3 | Task 2 | Manual | CLAUDE.md Standards section includes pointer |
| BC-4 | Task 3 | Manual | principles.md map includes new row |
| BC-5 | Task 1 | Manual | Cross-references present, no command duplication |
| BC-3 | Task 4 | Manual | contributing index.md Standards table includes entry |

No Go tests needed — this is a documentation-only PR.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Tier boundaries too broad (e.g., "code edits" too vague) | Low | Medium | Specific examples listed per tier | Task 1 |
| Stale PR links | Low | Low | All referenced PRs (#381, #390, #430) are completed and merged | Task 1 |
| Source-of-truth map row format mismatch | Low | Low | Copy exact format from adjacent rows | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] N/A — no Go code (golangci-lint)
- [x] N/A — no shared test helpers
- [x] CLAUDE.md updated (Task 2)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: principles.md source-of-truth map updated (Task 3)
- [x] Deviation log reviewed — all deviations are CLARIFICATION or ADDITION
- [x] Each task produces complete content
- [x] Task dependencies correctly ordered (Task 1 before Tasks 2-3)
- [x] All contracts mapped to tasks
- [x] N/A — no golden dataset changes
- [x] N/A — no struct construction sites
- [x] N/A — not part of a macro plan

**Antipattern rules:** N/A for all 23 rules — this PR contains no Go code.

---

## Appendix: File-Level Implementation Details

### File: `docs/contributing/standards/agent-trust.md` (NEW)

- **Purpose:** Canonical standards document defining three trust tiers for agent operations
- **Sections:** Trust Tiers (table + detailed descriptions), Known Failure Modes (FM-1 through FM-3), Relationship to Other Standards
- **Cross-references:** rules.md (R4), pr-workflow.md (verification gate), convergence.md (round evidence)

### File: `CLAUDE.md` (MODIFY)

- **Purpose:** Add 1-line pointer in Standards section
- **Location:** After line `- docs/contributing/standards/experiments.md: ...`
- **Content:** Single bullet point with path and brief description

### File: `docs/contributing/standards/principles.md` (MODIFY)

- **Purpose:** Add row to source-of-truth map table
- **Location:** After the "Engineering principles" row (line ~94)
- **Content:** Single table row: `| Agent trust boundaries | docs/contributing/standards/agent-trust.md | CLAUDE.md (pointer), docs/contributing/index.md (landing page table) |`

### File: `docs/contributing/index.md` (MODIFY)

- **Purpose:** Add row to Standards table for human contributor navigation
- **Location:** After the "Experiment Standards" row (line ~45)
- **Content:** Single table row linking to agent-trust.md
