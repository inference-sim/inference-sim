# Tasks: Per-Tenant Jain Fairness Index

**Input**: Design documents from `/specs/005-tenant-jain-fairness/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Organization**: Tasks grouped by user story — US1 (`blis run`) and US2 (`blis replay`) — with BDD/TDD as mandated by Constitution Principle IV (NON-NEGOTIABLE).

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no shared state)
- **[US1]/[US2]**: Which user story this task belongs to
- Tests MUST be written and verified RED before the corresponding implementation task

---

## Phase 1: Setup

**Purpose**: Orient to the implementation target. No structural changes needed — this feature slots into existing files.

- [x] T001 Read `ComputePerModelMetrics` and `printPerModelMetrics` patterns in `sim/cluster/metrics.go` (lines 503–570) and `cmd/root.go` (lines 1610–1633) to confirm the implementation template before writing any code

**Checkpoint**: Pattern understood — ready to write behavioral contracts.

---

## Phase 2: Foundational — `ComputePerTenantMetrics`

**Purpose**: The core aggregation function in `sim/cluster/metrics.go`. Both user stories depend on this. Must be complete before US1 or US2 can be implemented.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

### Tests (write first — must be RED before T003)

- [x] T002 Write behavioral contract tests BC-T1 through BC-T5 in `sim/cluster/metrics_tenant_test.go` (new file). Tests must be RED (compilation errors or failures) before T003:

  **BC-T1** — Absent section when no TenantIDs:
  ```
  GIVEN aggregated metrics where all RequestMetrics have empty TenantID
  WHEN ComputePerTenantMetrics is called
  THEN it returns nil
  ```

  **BC-T2** — Balanced two-tenant workload → Jain ≥ 0.99:
  ```
  GIVEN two tenants each with 50 completed requests and equal NumDecodeTokens
  WHEN ComputePerTenantMetrics is called
  THEN JainFairnessIndex over the token map is ≥ 0.99
  ```

  **BC-T3** — Skewed 10× workload → Jain < 0.70:
  ```
  GIVEN tenant "alice" with 10× the output tokens of tenant "bob"
  WHEN ComputePerTenantMetrics is called
  THEN JainFairnessIndex over the token map is < 0.70
  ```

  **BC-T4** — Single tenant → Jain = 1.0:
  ```
  GIVEN exactly one tenant with completed requests
  WHEN ComputePerTenantMetrics is called
  THEN the returned map has one entry AND JainFairnessIndex returns 1.0
  ```

  **BC-T5** — Requests without TenantID are excluded:
  ```
  GIVEN a mix of tenanted requests (TenantID="alice") and untenanted requests (TenantID="")
  WHEN ComputePerTenantMetrics is called
  THEN only "alice" appears in the result map (no "" entry)
  ```

- [x] T002-VERIFY Run `go test ./sim/cluster/... -run TestComputePerTenant -v` and confirm ALL BC-T1 through BC-T5 tests FAIL (RED). Do not proceed to T003 until confirmed RED.

### Implementation

- [x] T003 Implement `TenantMetrics` struct and `ComputePerTenantMetrics` function in `sim/cluster/metrics.go` immediately after the `ComputePerModelMetrics` block (after line 570):
  - Add `TenantMetrics` struct with fields: `TenantID string`, `CompletedRequests int`, `TotalTokensServed int`
  - Implement `ComputePerTenantMetrics(aggregated *sim.Metrics) map[string]*TenantMetrics`:
    - Iterate `aggregated.RequestE2Es` (completed request IDs only — see research.md Decision 3)
    - Skip requests missing from `aggregated.Requests` or with empty `TenantID`
    - Accumulate `CompletedRequests` and `TotalTokensServed` per tenant
    - Return `nil` if no entries (zero-value safe)
  - Run `go test ./sim/cluster/... -run TestComputePerTenant -v` — all 5 tests MUST be GREEN before proceeding

**Checkpoint**: `ComputePerTenantMetrics` is GREEN, `golangci-lint run ./sim/cluster/...` clean. Foundational phase complete.

---

## Phase 3: User Story 1 — Inspect Tenant Fairness After `blis run` (Priority: P1) 🎯 MVP

**Goal**: Operators running `blis run` with a multi-tenant workload see a `=== Per-Tenant Metrics ===` section in stdout showing per-tenant request counts, token totals, and Jain fairness index.

**Independent Test**: Run `./blis run --model qwen/qwen3-14b --workload chatbot --rate 10 --num-requests 100` with a workload spec containing two `tenant_id` values — output must include the `=== Per-Tenant Metrics ===` section.

### Tests (write first — must be RED before T005/T006)

- [x] T004 [US1] Write behavioral tests BC-T6 and BC-T7 in `cmd/kv_metrics_output_test.go` (extend existing file). Tests must be RED before T005:

  **BC-T6** — No-op when map is nil:
  ```
  GIVEN printPerTenantMetrics called with nil map
  WHEN output is captured
  THEN nothing is written to the writer
  ```

  **BC-T7** — Correct output format and ordering:
  ```
  GIVEN a map with two tenants "alice" (requests=50, tokens=12500) and "bob" (requests=50, tokens=12480)
  WHEN printPerTenantMetrics is called
  THEN output contains "=== Per-Tenant Metrics ===" header
  AND "alice" line appears before "bob" line (lexicographic order, R2)
  AND each line contains the request count and token total
  AND a "Jain Fairness Index:" line appears after the per-tenant lines
  AND the Jain value is ≥ 0.99 for this near-equal distribution
  ```

- [x] T004-VERIFY Run `go test ./cmd/... -run TestPrintPerTenant -v` and confirm BC-T6 and BC-T7 FAIL (RED). Do not proceed to T005 until confirmed RED.

### Implementation

- [x] T005 [US1] Implement `printPerTenantMetrics(w io.Writer, perTenantMetrics map[string]*cluster.TenantMetrics)` in `cmd/root.go` immediately after the `printPerModelMetrics` function (after line 1633):
  - Guard: return immediately if `len(perTenantMetrics) == 0`
  - Print `=== Per-Tenant Metrics ===` header
  - Sort keys (R2/INV-6), print each tenant: `  <id>: requests=<n>, tokens=<n>`
  - Build `map[string]float64` of TenantID → `float64(TotalTokensServed)`
  - Compute `jain := cluster.JainFairnessIndex(tokenMap)`
  - Print `  Jain Fairness Index: %.4f`
  - Run `go test ./cmd/... -run TestPrintPerTenant -v` — BC-T6 and BC-T7 MUST be GREEN

- [x] T006 [US1] Wire `printPerTenantMetrics` into the `blis run` output pipeline in `cmd/root.go`:
  - After the `printPerModelMetrics(os.Stdout, perModelMetrics)` call (around line 1544), add:
    ```go
    perTenantMetrics := cluster.ComputePerTenantMetrics(cs.AggregatedMetrics())
    printPerTenantMetrics(os.Stdout, perTenantMetrics)
    ```
  - Run `go test ./... -count=1` — full suite must pass

**Checkpoint**: `blis run` with a multi-tenant workload prints `=== Per-Tenant Metrics ===`. User Story 1 independently complete.

---

## Phase 4: User Story 2 — Inspect Tenant Fairness After `blis replay` (Priority: P2)

**Goal**: Operators replaying a TraceV2 trace with tenant labels see the same `=== Per-Tenant Metrics ===` section in `blis replay` output.

**Independent Test**: Run `./blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b` with a trace containing two tenant IDs — output must include the `=== Per-Tenant Metrics ===` section with the same structure as `blis run`.

### Implementation

- [x] T007 [US2] Wire `printPerTenantMetrics` into the `blis replay` output pipeline in `cmd/replay.go`:
  - Locate the `printPerModelMetrics(os.Stdout, perModelMetrics)` call in `cmd/replay.go` (around line 211)
  - Add immediately after:
    ```go
    perTenantMetrics := cluster.ComputePerTenantMetrics(cs.AggregatedMetrics())
    printPerTenantMetrics(os.Stdout, perTenantMetrics)
    ```
  - Run `go test ./... -count=1` — full suite must pass

**Checkpoint**: Both `blis run` and `blis replay` produce per-tenant metrics sections. User Story 2 complete.

---

## Phase 5: Polish & Cross-Cutting Concerns

- [x] T008 [P] Run `golangci-lint run ./...` and fix any warnings (zero tolerance per constitution)
- [x] T009 [P] Run `go test ./... -count=1` and confirm all packages pass, including `sim/cluster/` and `cmd/`
- [x] T010 [P] Verify determinism: run `./blis run --model qwen/qwen3-14b --workload chatbot --rate 10 --num-requests 20 --seed 42` twice and confirm per-tenant output is byte-identical (INV-6)
- [x] T011 Verify backward-compat: run `./blis run --model qwen/qwen3-14b` (no tenant labels) and confirm `=== Per-Tenant Metrics ===` section is absent from stdout

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 (needs `ComputePerTenantMetrics`)
- **US2 (Phase 4)**: Depends on Phase 3 (reuses `printPerTenantMetrics` defined in T005)
- **Polish (Phase 5)**: Depends on Phases 3 and 4

### Within-Phase Dependencies

```
T001 → T002 → T002-VERIFY → T003
                              ↓
              T004 → T004-VERIFY → T005 → T006 → T007
                                                    ↓
                                               T008/T009/T010/T011
```

T002 and T004 can be written in parallel (different files — T002 in `sim/cluster/`, T004 in `cmd/`).

### Parallel Opportunities

```bash
# Phase 2 + Phase 3 test writing can overlap:
Task: "T002 — Write BC-T1–BC-T5 in sim/cluster/metrics_tenant_test.go"
Task: "T004 — Write BC-T6–BC-T7 in cmd/kv_metrics_output_test.go"  # [P] — different package

# Phase 5 checks run in parallel:
Task: "T008 — golangci-lint run ./..."
Task: "T009 — go test ./... -count=1"
Task: "T010 — determinism verification"
```

---

## Implementation Strategy

### MVP (User Story 1 only)

1. T001 — read pattern
2. T002 + T002-VERIFY — tests RED
3. T003 — ComputePerTenantMetrics GREEN
4. T004 + T004-VERIFY — print tests RED
5. T005 — printPerTenantMetrics GREEN
6. T006 — wire into root.go
7. **VALIDATE**: `blis run` with two-tenant workload shows per-tenant section

### Full Delivery (US1 + US2)

8. T007 — wire into replay.go
9. T008–T011 — polish and verify
10. **VALIDATE**: both `blis run` and `blis replay` show per-tenant section with Jain index

---

## Notes

- BDD/TDD is NON-NEGOTIABLE (Constitution Principle IV): tests written and confirmed RED before implementation
- R2 applies in T005: sort tenant keys before printing to stdout
- R11 applies in T003: `JainFairnessIndex` already guards zero denominator — no additional guard needed in `ComputePerTenantMetrics`
- R4: `TenantMetrics` is a new struct with no construction sites elsewhere — no grep needed, but confirm after adding
- The function `printPerTenantMetrics` is defined in `cmd/root.go` and called from both `root.go` and `replay.go` — both are in package `cmd`, so no export needed
