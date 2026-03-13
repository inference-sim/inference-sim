# Fix Stale References to Deleted training/ Files

- **Goal:** Remove dead `training/` path references from production code and config comments.
- **The problem today:** PR #560 removed the `training/` directory, but 6 comments in `sim/latency/crossmodel.go`, `sim/latency/trained_roofline.go`, and `defaults.yaml` still reference deleted files (`training/ledger.md`, `training/problem.md`, `training/DESIGN.md`, `training/output/fit/coefficients.json`). Contributors following these references hit dead ends.
- **What this PR adds:**
  1. Replaces stale `training/` paths with inline coefficient provenance summaries
  2. Directs readers to `defaults.yaml` as the canonical coefficient source and `git log` for historical derivation
- **Why this matters:** Eliminates contributor confusion from dead documentation links (R15: stale references).
- **Architecture:** Comment-only changes in 3 files. No behavioral changes, no new types, no tests needed.
- **Source:** GitHub issue #576
- **Closes:** Fixes #576
- **Tier:** Small (comment updates only, ≤3 files, no behavioral logic changes)

---

## Part 1: Design Validation

### A) Executive Summary

This PR updates 6 stale comments across 3 files that reference deleted `training/` directory files. Each reference is replaced with a self-contained provenance note: what the coefficients represent, where they live now (`defaults.yaml`), and how to find the original derivation (`git log -- training/`). No code logic changes.

### B) Behavioral Contracts

**BC-1: No stale training/ references in production code**
- GIVEN the `training/` directory was removed in PR #560
- WHEN a contributor reads `sim/latency/crossmodel.go`, `sim/latency/trained_roofline.go`, or `defaults.yaml`
- THEN no comment references a `training/` path

**BC-2: Coefficient provenance is self-contained**
- GIVEN a contributor reading the crossmodel or trained-roofline source
- WHEN they want to understand where coefficients come from
- THEN the comments explain the derivation method and point to `defaults.yaml` + git history

### C) Risks & Mitigations

None — comment-only changes carry zero runtime risk.

### D) Test Strategy

No tests needed. Verification: `grep -r 'training/' sim/latency/ defaults.yaml` returns only `hypotheses/` and `docs/plans/` (archival, not modified).

---

## Part 2: Executable Tasks

### Task 1: Update crossmodel.go comments

**Files:** `sim/latency/crossmodel.go`

1. Line 16: Replace `(from training/ledger.md Iter 3)` with `(globally-fitted via OLS from 13 vLLM experiments; see defaults.yaml crossmodel_defaults)`
2. Line 51: Replace `See training/ledger.md Iter 3 and training/problem.md Section 2a for the physics rationale.` with `Decode is memory-bandwidth-bound on H100: each token reads accumulated KV from HBM. Prefill KV write cost is absorbed into β₀ where it overlaps with compute.`
3. Verify: `go build ./sim/latency/...`

### Task 2: Update trained_roofline.go comments

**Files:** `sim/latency/trained_roofline.go`

1. Line 13: Replace `(from training/DESIGN.md)` with `(see defaults.yaml trained_roofline_defaults)`
2. Line 22: Replace `from training/output/fit/coefficients.json, fitted` with `from defaults.yaml trained_roofline_defaults, fitted`
3. Verify: `go build ./sim/latency/...`

### Task 3: Update defaults.yaml comments

**Files:** `defaults.yaml`

1. Line 1746: Replace `from training/ledger.md Iter 3` with `globally-fitted via OLS from 13 vLLM experiments across 4 architectures`
2. Line 1753: Replace `from training/output/fit/coefficients.json` with `fitted via 3-phase NNLS from 13 experiments (4 models, 137K requests)`
3. Verify: `go build ./...` and `go test ./sim/latency/... -count=1`

### Task 4: Verify no remaining stale references in production code

Run: `grep -rn 'training/' sim/ defaults.yaml --include='*.go' --include='*.yaml'`
Expected: zero matches.

---

## Appendix

### Files Modified

| File | Change |
|------|--------|
| `sim/latency/crossmodel.go` | 2 comment updates (lines 16, 51) |
| `sim/latency/trained_roofline.go` | 2 comment updates (lines 13, 22) |
| `defaults.yaml` | 2 comment updates (lines 1746, 1753) |

### Out of Scope

- `hypotheses/` and `docs/plans/` references are archival (they document the state when `training/` existed) — not modified.
