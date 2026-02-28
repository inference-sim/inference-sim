# Micro Plan Template (Single-PR Implementation Plan)

This template defines the output format for a single-PR implementation plan. Use this when planning any PR — from bug fixes to new features.

!!! note "For Claude Code users"
    The `writing-plans` skill generates plans from this template automatically.
    The agent prompt version is at [`micro-plan-prompt.md`](micro-plan-prompt.md).

The source of work may be a macro plan section, one or more GitHub issues, a design document, or a feature request.

The plan has **two audiences:**

1. A human reviewer who validates behavioral correctness (Part 1)
2. An implementer (human or agent) who executes the tasks (Part 2)

---

## Document Header

Every plan starts with this header:

- **Goal:** One sentence a non-contributor could understand — what capability does this PR add?
- **The problem today:** 2–3 sentences explaining what's missing or broken without this PR.
- **What this PR adds:** Numbered list of 2–4 concrete capabilities, each in plain language with a brief example.
- **Why this matters:** 1–2 sentences connecting this PR to the broader project vision.
- **Architecture:** 2–3 sentences about the technical approach (packages, key types, integration points).
- **Source:** Link to the source of work (macro plan section, issue numbers, design doc).
- **Closes:** GitHub issue numbers this PR will close on merge (e.g., `Fixes #183, fixes #189`).
- **Behavioral Contracts:** Reference to Part 1, Section B.

---

## Part 1: Design Validation (target <120 lines)

### A) Executive Summary

5–10 lines describing what this PR builds (plain language), where it fits in the system, adjacent components it interacts with, and any deviation flags.

### B) Behavioral Contracts

3–15 named contracts defining what this PR guarantees:

```
BC-N: <Name>
- GIVEN <precondition>
- WHEN <action>
- THEN <observable outcome>
- MECHANISM: <one sentence explaining how> (optional)
```

Group into: positive contracts (what MUST happen), negative contracts (what MUST NOT happen), error handling contracts.

**Quality gate:** Every THEN clause must describe **observable behavior**, not internal structure. If a THEN clause contains a concrete type name or internal field name, rewrite it. The THEN clause drives the test assertion — a structural THEN produces a structural test.

### C) Component Interaction

Text-based component diagram showing this PR's building block, adjacent components, data flow direction, and what crosses each boundary. Include API contracts and state ownership. Target: under 40 lines.

### D) Deviation Log

Table comparing the micro plan against the source document:

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| ... | ... | SIMPLIFICATION / CORRECTION / DEFERRAL / ADDITION / SCOPE_CHANGE |

### E) Review Guide

5–10 lines telling the reviewer: the tricky part, what to scrutinize, what's safe to skim, and known debt.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to create/modify (one-line each), key decisions, confirmation that no dead code exists.

### G) Task Breakdown (6–12 tasks)

Each task follows TDD format:

1. Write failing test
2. Run test to verify it fails
3. Implement minimal code to pass
4. Run test to verify it passes
5. Run lint check
6. Commit with contract reference

Each task must specify: contracts implemented (BC-X, BC-Y), files (create/modify/test), complete code in every step, exact commands with expected output.

**Task design rules:** Each task implements 1–3 related contracts. Complete code in every step (no "add validation" without showing exact code). Exact commands with expected output. Reference shared test infrastructure. Golden dataset updates if needed. Dependency ordering. No dead code. Behavioral assertions only (see [standards/principles.md](../standards/principles.md) for prohibited/required assertion patterns).

### H) Test Strategy

Map contracts to tasks and tests. Include invariant tests alongside golden tests — golden tests answer "did the output change?" while invariant tests answer "is the output correct?"

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestFoo_GivenX_ThenY |
| ... | ... | ... | ... |

Key invariants for this simulator (see [standards/invariants.md](../standards/invariants.md)):

- **Request conservation (INV-1):** completed + still_queued + still_running + dropped_unservable = injected
- **KV block conservation (INV-4):** allocated_blocks + free_blocks = total_blocks
- **Clock monotonicity (INV-3):** simulation clock never decreases
- **Causality (INV-5):** arrival_time ≤ enqueue_time ≤ schedule_time ≤ completion_time
- **Determinism (INV-6):** same seed produces byte-identical output across runs

### I) Risk Analysis

For each risk: description, likelihood (low/medium/high), impact (low/medium/high), mitigation (specific test or design choice), and which task mitigates it.

---

## Part 3: Quality Assurance

### J) Sanity Checklist

Before implementation, verify:

**Plan-specific checks:**
- [ ] No unnecessary abstractions.
- [ ] No feature creep beyond PR scope.
- [ ] No unexercised flags or interfaces.
- [ ] No partial implementations.
- [ ] No breaking changes without explicit contract updates.
- [ ] No hidden global state impact.
- [ ] All new code will pass golangci-lint.
- [ ] Shared test helpers used from existing shared test package (not duplicated locally).
- [ ] CLAUDE.md updated if: new files/packages added, file organization changed, plan milestone completed, new CLI flags added.
- [ ] No stale references left in CLAUDE.md.
- [ ] Documentation DRY: If this PR modifies a canonical source (docs/contributing/standards/rules.md, docs/contributing/standards/invariants.md, docs/contributing/standards/principles.md, docs/contributing/extension-recipes.md), all working copies in the source-of-truth map are updated. If a new file is added, it appears in the CLAUDE.md File Organization tree.
- [ ] Deviation log reviewed — no unresolved deviations.
- [ ] Each task produces working, testable code (no scaffolding).
- [ ] Task dependencies are correctly ordered.
- [ ] All contracts are mapped to specific tasks.
- [ ] Golden dataset regeneration documented (if needed).
- [ ] Construction site audit completed — all struct construction sites listed and covered by tasks.
- [ ] If this PR is part of a macro plan, the macro plan status is updated.

**Antipattern rules (full details in [standards/rules.md](../standards/rules.md)):**
- [ ] R1: No silent `continue`/`return` dropping data
- [ ] R2: Map keys sorted before float accumulation or ordered output
- [ ] R3: Every new CLI flag validated (zero, negative, NaN, Inf)
- [ ] R4: All struct construction sites audited for new fields
- [ ] R5: Resource allocation loops handle mid-loop failure with rollback
- [ ] R6: No `logrus.Fatalf` or `os.Exit` in `sim/` packages
- [ ] R7: Invariant tests alongside any golden tests
- [ ] R8: No exported mutable maps
- [ ] R9: `*float64` for YAML fields where zero is valid
- [ ] R10: YAML strict parsing (`KnownFields(true)`)
- [ ] R11: Division by runtime-derived denominators guarded
- [ ] R12: Golden dataset regenerated if output changed
- [ ] R13: New interfaces work for 2+ implementations
- [ ] R14: No method spans multiple module responsibilities
- [ ] R15: Stale PR references resolved
- [ ] R16: Config params grouped by module
- [ ] R17: Routing scorer signals documented for freshness tier
- [ ] R18: CLI flag values not silently overwritten by defaults.yaml
- [ ] R19: Unbounded retry/requeue loops have circuit breakers
- [ ] R20: Detectors and analyzers handle degenerate inputs (empty, skewed, zero)

---

## Appendix: File-Level Implementation Details

For each file to be created or modified, provide:

- **Purpose:** 1–2 sentences
- **Complete implementation:** All type definitions, function implementations, test code
- **Key notes:** Event ordering (Priority? Timestamp? Secondary tie-breaking?), RNG subsystem usage, metrics collection, state mutation, error handling approach

This section has no length limit. It should contain everything needed to implement the PR without further codebase exploration.
