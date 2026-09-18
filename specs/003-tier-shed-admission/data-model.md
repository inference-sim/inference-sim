# Data Model: Phase 1B-1a — Tier-Ordered Admission Shedding

## Entities

### SLO Tier Priority (new helper, `sim/admission.go`)

A pure function mapping `Request.SLOClass` string to an integer priority rank. Exported so `sim/cluster/` can call it without a circular import.

| SLOClass string | Priority int | Admission behavior |
|----------------|-------------|-------------------|
| `"critical"`   | 4 | Never shed by tier policy |
| `"standard"`   | 3 | Admitted at default threshold (MinAdmitPriority=3 → equal, not shed) |
| `"sheddable"`  | 2 | Shed first when overload threshold exceeded |
| `"batch"`      | 1 | Bypasses tier-shed entirely (deferred queue PR handles) |
| `"background"` | 0 | Bypasses tier-shed entirely (deferred queue PR handles) |
| `""` (empty)   | 3 | Treated as Standard — backward compatibility for untagged requests |
| unknown string | 3 | Treated as Standard — defensive default |

**Invariants:**
- `SLOTierPriority()` is a pure function: same input always returns same output, no side effects
- `SLOTierPriority()` MUST NOT read `req.OutputTokens` (INV-9 — this function receives only a string, not a Request, so this is structurally enforced)

---

### TierShedAdmission (new struct, `sim/admission.go`)

An `AdmissionPolicy` implementation that gates requests by tier priority under overload.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `OverloadThreshold` | `int` | `0` | Max per-instance effective load before shedding activates; 0 = shed when any instance has effective load > 0 |
| `MinAdmitPriority` | `int` | `3` | Minimum tier priority admitted under overload (3 = Standard and above pass) |

**Zero-value footgun**: `MinAdmitPriority=0` admits all tiers under overload — equivalent to `AlwaysAdmit`. This is almost certainly unintended. The YAML default must be 3; callers must explicitly opt into lower thresholds.

**State transitions**: Stateless — all decisions computed from `RouterState` snapshots at call time. `TierShedAdmission` holds no mutable state.

**Configuration via `DeploymentConfig`** (placed alongside existing token-bucket fields):
```
TierShedThreshold   int  // yaml:"tier_shed_threshold"   default: 0
TierShedMinPriority int  // yaml:"tier_shed_min_priority" default: 3
```

**Factory bypass**: `NewAdmissionPolicy()` cannot carry int parameters; `TierShedAdmission` is constructed directly in `NewClusterSimulator()` when `config.AdmissionPolicy == "tier-shed"`. The name `"tier-shed"` is still registered in `validAdmissionPolicies` for YAML validation.

---

### shedByTier Counter (new field on `ClusterSimulator`, `sim/cluster/cluster.go`)

An unexported map tracking per-tier rejection counts for capacity-planning visibility.

| Property | Value |
|----------|-------|
| Type | `map[string]int` |
| Keys | SLOClass strings: `"critical"`, `"standard"`, `"sheddable"`, `"batch"`, `"background"`, `""` |
| Initialized | `make(map[string]int)` in `NewClusterSimulator()` |
| Updated | `AdmissionDecisionEvent.Execute()` after `cs.rejectedRequests++` |
| Zero-value safety | Empty string key used when `SLOClass == ""`; does not panic |

**Note**: This is a runtime-mutated metrics map, not a validation map. R8 (unexported validation maps) does not prohibit it, but the field is correctly unexported to prevent external mutation.
