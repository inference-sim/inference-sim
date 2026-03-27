# Research: Phase 1B-1a — Tier-Ordered Admission Shedding

**Status**: Complete — all decisions resolved by reading the codebase directly.

## What Already Exists (No Work Needed)

| Thing | Location | Note |
|-------|----------|------|
| `Request.SLOClass string` | `sim/request.go` | "critical"/"standard"/"sheddable"/"batch"/"background"/"" |
| `AdmissionPolicy` interface | `sim/admission.go:11` | `Admit(req *Request, state *RouterState) (admitted bool, reason string)` |
| `RoutingSnapshot.EffectiveLoad()` | `sim/routing.go:28` | `QueueDepth + BatchSize + InFlightRequests` — already the correct overload signal |
| `RouterState.Snapshots` | `sim/routing.go` | Passed to `Admit()` — overload signal available without new wiring |
| `validAdmissionPolicies` | `sim/bundle.go:61` | Unexported map, exposed via `IsValidAdmissionPolicy()` — correct pattern (R8) |
| `cs.rejectedRequests` | `sim/cluster/cluster.go:32` | Admission rejection counter already exists; we add a per-tier companion |
| `cs.inFlightRequests map[string]int` | `sim/cluster/cluster.go:36` | Instance ID → in-flight count; already populated by `buildRouterState()` |

## Design Decisions

**D-1: Overload signal for TierShedAdmission**
- Decision: Max `EffectiveLoad()` across all `RouterState.Snapshots`, compared against configurable `OverloadThreshold int`.
- Rationale: `RouterState` is already passed to `Admit()`. `EffectiveLoad()` = `QueueDepth + BatchSize + InFlightRequests` already exists on `RoutingSnapshot`. No new wiring needed.
- Alternative rejected: KV utilization — lags queue depth, harder to configure meaningfully.

**D-2: Factory bypass for TierShedAdmission**
- Decision: Do NOT add `case "tier-shed":` to `NewAdmissionPolicy()`. Instead, detect `config.AdmissionPolicy == "tier-shed"` before the struct literal in `NewClusterSimulator()` and construct `&sim.TierShedAdmission{...}` directly.
- Rationale: `NewAdmissionPolicy(name string, capacity, refillRate float64)` is float64-parameterized. `TierShedAdmission` requires int fields (`OverloadThreshold`, `MinAdmitPriority`). Extending the factory signature would break all callers. Bypassing is the minimal change.
- Still register `"tier-shed"` in `validAdmissionPolicies` so `IsValidAdmissionPolicy()` returns true for YAML bundle validation.

**D-3: Config field placement — DeploymentConfig vs AdmissionConfig**
- Decision: Add `TierShedThreshold int` and `TierShedMinPriority int` to `DeploymentConfig` (alongside existing `TokenBucketCapacity float64`, `TokenBucketRefillRate float64`).
- Rationale: `NewClusterSimulator()` receives `DeploymentConfig`, not `AdmissionConfig`. Putting the fields there avoids a config-merging path. Consistent with how `TokenBucketCapacity` is also duplicated in `DeploymentConfig`.

**D-4: SLOTierPriority export requirement**
- Decision: `SLOTierPriority(class string) int` must be exported (capital S).
- Rationale: Called from `sim/cluster/cluster_event.go` (`sim/cluster` package). `sim/admission.go` is in `sim` package. Go package boundary requires export for cross-package calls. `sim/cluster/` can import `sim/` without circular dependency (correct direction per constitution I).

**D-5: Per-tier shed counter**
- Decision: Add `shedByTier map[string]int` (unexported) to `ClusterSimulator`. Initialize to `make(map[string]int)` in `NewClusterSimulator()`. Increment in `AdmissionDecisionEvent.Execute()` after `cs.rejectedRequests++`.
- Rationale: Enables per-tier visibility for capacity planners (FR-007, SC-004). Consistent with existing `rejectedRequests` counter pattern.
- Note: This is an unexported field mutated at runtime — not a validation map. R8 applies to validation maps only. No violation.

**D-6: Batch and Background pass-through**
- Decision: `TierShedAdmission.Admit()` returns `(true, "")` for any request with `SLOClass == "batch"` or `"background"`, regardless of load.
- Rationale: Deferred queue PR (#810) intercepts Batch/Background in `AdmissionDecisionEvent.Execute()` before calling `admissionPolicy.Admit()`. This PR must not shed them, or deferred queue will never see them. The two PRs compose without conflict.
