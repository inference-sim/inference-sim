## Summary

<!-- 1-3 sentences: what does this PR do and why? -->

## Behavioral Contracts

<!-- List the GIVEN/WHEN/THEN contracts this PR implements. Reference the implementation plan if one exists. -->

- **BC-1:** ...
- **BC-2:** ...

## Antipattern Checklist

<!-- Check each item. If N/A, check it and note why. -->

- [ ] No silent `continue` or early `return` that drops data without error propagation
- [ ] No map iteration feeding float accumulation without sorted keys
- [ ] Every struct built in multiple places has a canonical constructor
- [ ] No `logrus.Fatalf` or `os.Exit` in library code (`sim/`, `sim/cluster/`, `sim/workload/`)
- [ ] Every new CLI flag validated for zero, negative, NaN, Inf
- [ ] Every loop that allocates resources handles mid-loop failure with rollback
- [ ] Invariant tests added or extended if request lifecycle, KV cache, or metrics are touched
- [ ] Golden dataset regenerated if output values changed

## Invariants

<!-- Which invariants does this PR maintain or test? -->

- [ ] Request conservation: injected == completed + queued + running
- [ ] KV block conservation: allocated + free == total
- [ ] Causality: arrival <= schedule <= completion
- [ ] Clock monotonicity
- [ ] Determinism: same seed produces identical output
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
- [ ] CONTRIBUTORS.md reviewed (if extension patterns changed)
