## Summary

<!-- 1-3 sentences: what does this PR do and why? -->

## Behavioral Contracts

<!-- List the GIVEN/WHEN/THEN contracts this PR implements. Reference the implementation plan if one exists. -->

- **BC-1:** ...
- **BC-2:** ...

## Antipattern Checklist

<!-- Check each item. If N/A, check it and note why. Full details: docs/contributing/standards/rules.md -->

- [ ] No silent data loss (R1)
- [ ] Map keys sorted before float accumulation (R2)
- [ ] CLI flags validated for zero, negative, NaN, Inf (R3)
- [ ] Struct construction sites audited for new fields (R4)
- [ ] Resource allocation loops rollback on mid-loop failure (R5)
- [ ] No `logrus.Fatalf` or `os.Exit` in `sim/` packages (R6)
- [ ] Invariant tests alongside golden tests (R7)
- [ ] Golden dataset regenerated if output changed (R12)

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
