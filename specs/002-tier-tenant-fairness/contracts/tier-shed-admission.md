# Contract: TierShedAdmission

**Interface**: `sim.AdmissionPolicy`
**Method**: `Admit(req *Request, state *RouterState) (admitted bool, reason string)`

## Behavioral Contract

### Preconditions
- `req` is non-nil with `SLOClass` and `TenantID` already set (arrival-time metadata)
- `state.Snapshots` may be empty (zero instances) — must not panic
- `req.OutputTokens` MUST NOT be read (INV-9: oracle knowledge boundary)

### Postconditions
- Returns `(true, "")` when tier priority ≥ `MinAdmitPriority` OR max effective load ≤ `OverloadThreshold`
- Returns `(false, reason)` when tier priority < `MinAdmitPriority` AND overloaded
- Batch (`"batch"`) and Background (`"background"`) always return `(true, "")` — deferred queue handles them
- Unknown or empty `SLOClass` is treated as Standard (priority 3) — never shed below Standard

### Monotonicity invariant
If `Admit(req_A)` returns `false` and `sloTierPriority(req_B.SLOClass) ≥ sloTierPriority(req_A.SLOClass)`, then `Admit(req_B)` returns `true` (assuming identical state).

### Side effects
None — stateless. No mutation of `req` or `state`.

## Configuration

| YAML key | Go field | Default | Constraint |
|----------|----------|---------|-----------|
| `tier_shed_threshold` | `OverloadThreshold int` | `0` | ≥ 0 |
| `tier_shed_min_priority` | `MinAdmitPriority int` | `3` | 0–4 |

`OverloadThreshold=0` means: shed qualifying tiers whenever any instance has effective load > 0.

## Registration

Policy name: `"tier-shed"` in `validAdmissionPolicies` map (`sim/bundle.go`).
