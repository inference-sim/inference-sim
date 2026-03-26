# Contract: TenantTracker

**Package**: `sim/cluster`
**File**: `sim/cluster/tenant.go`

## Methods

### `NewTenantTracker(budgets map[string]float64, totalCapacity int) *TenantTracker`
- `budgets` may be nil → returns a tracker that never reports over-budget
- `totalCapacity` ≥ 1
- No side effects on inputs

### `IsOverBudget(tenantID string) bool`
- Returns `false` when `tenantID == ""`
- Returns `false` when no budget configured for `tenantID`
- Returns `true` when `inFlight[tenantID] > budgets[tenantID] * totalCapacity`
- Pure query — no side effects, no state mutation

### `OnStart(tenantID string)`
- Increments `inFlight[tenantID]` by 1
- No-op when `tenantID == ""`

### `OnComplete(tenantID string)`
- Decrements `inFlight[tenantID]` by 1, floor 0
- No-op when `tenantID == ""`

## Invariants

- `inFlight[t] ≥ 0` for all tenants at all times
- `OnStart`/`OnComplete` calls must be balanced: each dispatched request calls `OnStart` exactly once and `OnComplete` exactly once at terminal state (completed, dropped, timed-out)
- `IsOverBudget` never reads `req.OutputTokens` (INV-9)

## Zero-value safety

`TenantTracker` constructed with `nil` budgets never affects admission — `IsOverBudget` always returns `false`. Simulations without `TenantBudgets` configured are unaffected.
