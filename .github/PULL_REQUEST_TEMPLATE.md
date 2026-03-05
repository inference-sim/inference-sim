## Summary

<!-- 1-3 sentences: what does this PR do and why? -->

## Behavioral Contracts

<!-- List the GIVEN/WHEN/THEN contracts this PR implements. Reference the implementation plan if one exists. -->

- **BC-1:** ...
- **BC-2:** ...

## Antipattern Checklist

<!-- Check each item. If N/A, check it and note why. Full details: docs/contributing/standards/rules.md -->

**Critical (correctness):**
- [ ] No silent data loss — every error path returns error, panics, or increments counter (R1)
- [ ] Struct construction sites audited for new fields (R4)
- [ ] Resource allocation loops rollback on mid-loop failure (R5)
- [ ] No `logrus.Fatalf` or `os.Exit` in `sim/` packages (R6)
- [ ] Division by runtime-derived denominators guarded (R11)
- [ ] Unbounded retry/requeue loops have circuit breakers (R19)
- [ ] No `range` over slices that can shrink during iteration (R21)

**Important (quality):**
- [ ] Map keys sorted before float accumulation or ordered output (R2)
- [ ] Every new numeric parameter validated — CLI flags AND library constructors (R3)
- [ ] Invariant tests alongside golden tests (R7)
- [ ] No exported mutable maps — use `IsValid*()` accessors (R8)
- [ ] `*float64` for YAML fields where zero is valid (R9)
- [ ] YAML strict parsing with `KnownFields(true)` (R10)
- [ ] New interfaces work for 2+ implementations (R13)
- [ ] No method spans multiple module responsibilities (R14)
- [ ] Routing scorer signals documented for freshness tier (R17)
- [ ] CLI flag values not silently overwritten by defaults.yaml (R18)
- [ ] Detectors and analyzers handle degenerate inputs (R20)
- [ ] Pre-check estimates consistent with actual operation accounting (R22)
- [ ] Parallel code paths apply equivalent transformations (R23)

**Hygiene (maintenance):**
- [ ] Golden dataset regenerated if output changed (R12)
- [ ] Stale PR references resolved (R15)
- [ ] Config params grouped by module (R16)

## Invariants

<!-- Which invariants does this PR maintain or test? Full details: docs/contributing/standards/invariants.md -->

- [ ] Request conservation: injected == completed + queued + running (INV-1)
- [ ] Request lifecycle: queued -> running -> completed (INV-2)
- [ ] KV block conservation: allocated + free == total (INV-4)
- [ ] Causality: arrival <= schedule <= completion (INV-5)
- [ ] Clock monotonicity (INV-3)
- [ ] Determinism: same seed produces identical output (INV-6)
- [ ] N/A — this PR does not touch invariant-sensitive code

## Test Plan

<!-- How was this tested? List commands and key results. -->

```bash
go test ./...
golangci-lint run ./...
```

## Source

<!-- What motivated this PR? Link to: issue, design doc, macro plan section, or feature request. -->

Fixes #

## Documentation

- [ ] CLAUDE.md updated (if new files, packages, CLI flags, or invariants added)
- [ ] README updated (if user-facing behavior changed)
- [ ] CONTRIBUTING.md reviewed (if extension patterns changed)
