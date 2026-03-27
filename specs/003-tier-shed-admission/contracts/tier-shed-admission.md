# Contract: TierShedAdmission

**Package**: `sim`
**File**: `sim/admission.go`
**Interface**: `sim.AdmissionPolicy`
**Method**: `Admit(req *Request, state *RouterState) (admitted bool, reason string)`

## Behavioral Contract

### Preconditions
- `req` is non-nil with `SLOClass` already set (arrival-time metadata)
- `state.Snapshots` may be empty (zero instances) — must not panic; all requests admitted when no instances
- `req.OutputTokens` MUST NOT be read (INV-9: oracle knowledge boundary)

### Postconditions
- Returns `(true, "")` when `SLOTierPriority(req.SLOClass)` ≥ `MinAdmitPriority` OR `maxEffectiveLoad` ≤ `OverloadThreshold`
- Returns `(false, reason)` when `SLOTierPriority(req.SLOClass)` < `MinAdmitPriority` AND `maxEffectiveLoad` > `OverloadThreshold`
- `reason` is a human-readable string identifying the tier and threshold when rejecting (never empty on rejection)
- `"batch"` and `"background"` requests always return `(true, "")` regardless of load — deferred queue PR handles them
- Unknown or empty `SLOClass` is treated as Standard (priority 3) — never shed below Standard
- No mutation of `req`, `state`, or any field on the receiver

### Monotonicity Invariant
If `Admit(req_A, state)` returns `false` and `SLOTierPriority(req_B.SLOClass)` ≥ `SLOTierPriority(req_A.SLOClass)`, then `Admit(req_B, state)` returns `true` (assuming identical state).

### Side Effects
None — stateless. No mutation of `req` or `state`. Multiple calls with identical arguments return identical results.

## SLOTierPriority Helper

**Signature**: `func SLOTierPriority(class string) int`

- Pure function of the class string — no side effects, no state
- Exported so `sim/cluster/` can call it without circular import
- Never reads `Request.OutputTokens` (structurally: receives a string, not a Request)

## Configuration

| YAML key | Go field on DeploymentConfig | Default | Constraint |
|----------|------------------------------|---------|-----------|
| `tier_shed_threshold` | `TierShedThreshold int` | `0` | ≥ 0 |
| `tier_shed_min_priority` | `TierShedMinPriority int` | `0` (Go zero value) | 0–4 |

`OverloadThreshold=0` means: shed qualifying tiers whenever any instance has effective load > 0.

**Zero-value footgun**: `MinAdmitPriority=0` (the Go zero value) admits all tiers under overload — equivalent to `AlwaysAdmit`. This is almost certainly unintended. Callers must explicitly set `tier_shed_min_priority: 3` for Standard-and-above protection. A `logrus.Warn` is emitted at construction time when `MinAdmitPriority=0` is used with `tier-shed`.

## Registration

Policy name `"tier-shed"` is added to `validAdmissionPolicies` in `sim/bundle.go` for YAML validation via `IsValidAdmissionPolicy()`.

`TierShedAdmission` is NOT constructed via `NewAdmissionPolicy()` — the factory signature is float64-parameterized and cannot carry int fields. `NewClusterSimulator()` in `sim/cluster/cluster.go` detects `config.AdmissionPolicy == "tier-shed"` and constructs the struct directly before passing to the `ClusterSimulator` struct literal.
