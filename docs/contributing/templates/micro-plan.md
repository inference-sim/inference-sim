# Micro Plan Template (Single-PR Implementation Plan)

This template defines the output format for a single-PR implementation plan. Use when planning any PR — from bug fixes to new features.

The source of work is a GitHub issue, a sub-issue from an RFC, or a feature request.

---

## Compact Format (Small PRs)

For small PRs (≤3 files, mechanical changes, no new interfaces), use this streamlined format:

```
# [Title] Implementation Plan

**Goal:** One sentence a non-contributor could understand.
**Source:** Link to issue or sub-issue.
**Closes:** GitHub issue numbers (e.g., `Fixes #123`).

## Behavioral Contracts

BC-1: <Name>
- GIVEN <precondition>
- WHEN <action>
- THEN <observable outcome>

## Tasks

### Task 1: <Name> (BC-1)

**Files:** create/modify `path/to/file`, test `path/to/test`
**Test:** [complete test code]
**Impl:** [complete implementation code]
**Verify:** `go test ./path/... -run TestName`
**Lint:** `golangci-lint run ./path/...`
**Commit:** `type(scope): description (BC-1)`

## Sanity Checklist
- [ ] R1: no silent data loss (no bare `continue` in error paths)
- [ ] R2: sorted map iteration for deterministic output (INV-6)
- [ ] R4: all construction sites updated for new fields
- [ ] R6: no logrus.Fatalf in sim/ (library code)
- [ ] R8: no exported mutable maps
- [ ] INV-1: request conservation holds
- [ ] INV-6: determinism (same seed = byte-identical stdout)
```

---

## Full Format (Medium/Large PRs)

For PRs that change behavior, add interfaces, or touch 4+ files:

### Header

```
# [Title] Implementation Plan

**Goal:** One sentence a non-contributor could understand.
**The problem today:** 2-3 sentences — what's missing or broken.
**What this PR adds:** Numbered list of concrete capabilities.
**Source:** Link to issue, sub-issue, or RFC.
**Closes:** GitHub issue numbers.
```

### Part 1: Behavioral Contracts

Define what this PR guarantees. Every contract must have:
- A name (BC-1, BC-2, ...)
- GIVEN/WHEN/THEN with observable outcomes only
- No internal type names or field names in THEN clauses

```
BC-1: <Name>
- GIVEN <precondition>
- WHEN <action>
- THEN <observable outcome>
- Evidence: <property_test | differential_test | metamorphic_test>
```

### Part 2: Task Breakdown (6-12 tasks)

Each task follows TDD:
1. Write the failing test
2. Run test — verify it fails
3. Implement minimal code to pass
4. Run test — verify it passes
5. Lint: `golangci-lint run ./path/...`
6. Commit with contract reference

Tasks must be ordered so each can start given what comes before. Every task must have complete code — no "add validation" without showing exact code.

### Part 3: Sanity Checklist

Before marking plan complete, verify:

- [ ] All behavioral contracts have GIVEN/WHEN/THEN
- [ ] Every THEN clause describes observable behavior (no type names, no internal fields)
- [ ] Tasks are ordered by dependency (no task requires code from a later task)
- [ ] Every task has complete test + impl code
- [ ] R1: no silent data loss
- [ ] R2: sorted map iteration (INV-6)
- [ ] R4: all construction sites updated
- [ ] R6: no logrus.Fatalf in sim/
- [ ] R7: invariant tests alongside golden tests
- [ ] R8: no exported mutable maps
- [ ] R9: pointer types for YAML zero-value ambiguity
- [ ] R10: strict YAML parsing
- [ ] R19: division zero guards
- [ ] INV-1: request conservation holds
- [ ] INV-6: determinism preserved
- [ ] INV-13: run/replay parity (if applicable)
