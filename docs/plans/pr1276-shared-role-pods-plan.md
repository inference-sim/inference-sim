# PR1276: Shared-Role Pods (GAP-5)

**Source:** [#1276](https://github.com/inference-sim/inference-sim/issues/1276) — feat(cluster): support shared-role pods (instances serving in multiple pools) (GAP-5)
**Parent:** [#1260](https://github.com/inference-sim/inference-sim/issues/1260)
**Closes:** Fixes #1276
**Tier:** Medium (5 production files, refined type semantics, no new interfaces)

---

## Part 1: Design Validation

### A) Executive Summary

llm-d-inference-scheduler's `bylabel` filter assigns each pod one of seven role labels (decode, prefill, encode, prefill-decode, encode-prefill, encode-prefill-decode, both). Decode and prefill filters both accept pods labelled `prefill-decode` (and the legacy `both`), so a single mixed-role pod is returned by both filters — the authoritative behavioural contract for shared-role pods.

BLIS today models `PoolRole` as a scalar enum with two values (`PoolRolePrefill`, `PoolRoleDecode`). `BuildPoolMembershipFromIndices` partitions instance indices into two disjoint halves; `FilterSnapshotsByPool` and `collectPoolThroughput` use exact equality (`==`). No instance can belong to both pools.

This PR converts `PoolRole` to a bitmask so an instance's role can be the union of prefill and decode (matching llm-d `prefill-decode` / legacy `both`). The call sites that branch on `poolMembership[id]` are updated to use set-membership (`role.Has(PoolRolePrefill)`). A new `--prefill-decode-instances` CLI flag declares the count of shared-role pods; `ValidatePoolTopology` is widened to accept `prefill + decode + shared ≤ total`.

Encode-containing shared roles (`encode-prefill`, `encode-prefill-decode`) are deferred to GAP-4 (#1264) where `PoolRoleEncode` will be added.

### B) Behavioral Contracts

**BC-1: Shared-role instance is returned by both pool filters.**
- GIVEN a cluster with one `PoolRolePrefillDecode` instance and one `PoolRolePrefill`-only instance
- WHEN `FilterSnapshotsByPool(snaps, membership, PoolRolePrefill)` is called
- THEN the shared instance AND the prefill-only instance both appear in the result
- AND `FilterSnapshotsByPool(..., PoolRoleDecode)` returns only the shared instance.

**BC-2: `PoolRole.Has` implements set-membership over role bits.**
- GIVEN `r := PoolRolePrefill | PoolRoleDecode` (i.e., `PoolRolePrefillDecode`)
- THEN `r.Has(PoolRolePrefill)` is true; `r.Has(PoolRoleDecode)` is true.
- GIVEN `r := PoolRolePrefill`
- THEN `r.Has(PoolRoleDecode)` is false.
- GIVEN `r := PoolRole(0)` (unassigned)
- THEN `r.Has(PoolRolePrefill)` is false AND `r.Has(PoolRoleDecode)` is false.

**BC-3: Unassigned (`PoolRole(0)`) behaviour is unchanged.**
- GIVEN no pool flags are set (disaggregation disabled)
- THEN `poolMembership` is nil; `poolsConfigured()` returns false; the PD completion-detection branch in `cluster.go` is not entered — preserving pre-PR behaviour byte-identically.

**BC-4: `ValidatePoolTopology` permits overlap via explicit shared count.**
- GIVEN `prefill=2, decode=2, shared=1, total=4` — the shared pod overlaps prefill and decode but the three disjoint groups fit: 2 prefill-only + 2 decode-only would exceed total, but because `shared` is an overlap pool, the correct check is `prefill + decode + shared ≤ total` where prefill and decode here mean "prefill-only" and "decode-only".
- THEN validation passes.
- GIVEN `prefill=2, decode=2, shared=0, total=3`
- THEN validation fails (sum exceeds total, no overlap declared).

**BC-5: PD completion detection fires both hooks for a shared-role pod.**
- GIVEN `poolMembership[instanceID]` has the `PoolRolePrefill` AND `PoolRoleDecode` bits set
- WHEN the cluster event loop reaches the `poolsConfigured()` branch in `runCluster` (cluster.go:664-669)
- THEN both `detectPrefillCompletions(inst)` and `detectDecodeCompletions(inst)` run for that instance.

**BC-6: `collectPoolThroughput` attributes a shared-role pod's completed requests to both pools.**
- GIVEN a shared-role instance with `m.CompletedRequests == N`
- WHEN `collectPoolThroughput` tallies prefill and decode completions
- THEN `prefillCompleted` is incremented by `N` AND `decodeCompleted` is incremented by `N`.

**BC-7: `resolveConfigForRole` is deterministic for shared-role instances — decode wins.**
- GIVEN a shared-role instance and a `DeploymentConfig` with `PrefillOverrides != DecodeOverrides`
- WHEN `resolveConfigForRole(PoolRolePrefillDecode)` is called
- THEN the function returns `ResolvePoolConfig(d.SimConfig, d.DecodeOverrides)` — decode wins, matching llm-d's `allowsNoLabel=true` on the decode filter (Permalink 2 in the issue).

**BC-8: CLI `--prefill-decode-instances` declares the shared-role count.**
- GIVEN `--prefill-instances=2 --decode-instances=2 --prefill-decode-instances=1 --num-instances=5`
- THEN two instances are prefill-only, two are decode-only, one is shared (`PoolRolePrefillDecode`), and zero are unassigned.
- GIVEN `--prefill-instances=0 --decode-instances=0 --prefill-decode-instances=1`
- THEN validation fails — shared pods require at least one of prefill/decode to be set, or alternatively the shared count is treated as both (see D below).

**BC-9: INV-6 determinism preserved.**
- GIVEN any two runs with identical seeds and pool topology (including shared-role)
- THEN stdout is byte-identical.

### C) Deviation Log

| ID | Source | Deviation | Reason |
|----|--------|-----------|--------|
| D-1 | Permalink 2 (llm-d NewDecodeRole passes `allowsNoLabel=true`) | BLIS does not replicate the "unlabeled defaults to decode" rule for routing eligibility. `PoolRole(0)` remains "unassigned"; such instances do not participate in PD routing. | CLARIFICATION. BLIS does not have a Kubernetes label surface; pool membership is always explicit. The llm-d rule's purpose is to let operators default all pods into the decode pool without labelling every one; BLIS has explicit CLI flags for this and the legacy no-PD path must remain byte-identical (INV-6). |
| D-2 | Issue "Per-pool config resolution" bullet (three options: strict reject, precedence, explicit `SharedRoleConfig`) | We pick option (b) — **decode wins** — when a shared-role instance is constructed with `PrefillOverrides != DecodeOverrides`. | CLARIFICATION. Simplest deterministic rule, matches the spirit of llm-d's `allowsNoLabel=true` decode default. Option (a) would reject many valid deployments; option (c) bloats the config surface for a rare case. Document the rule; if users need prefill-favoured overrides on a shared-role pod, they should express it via decode overrides. |
| D-3 | Issue "Suggested implementation sketch" | We pick shape 1 (bitmask `PoolRole`). | CLARIFICATION. Fewer touch-point changes: most existing `== PoolRolePrefill` sites become `.Has(PoolRolePrefill)` — mechanical search-and-replace. The `PoolRolePrefillDecode = PoolRolePrefill | PoolRoleDecode` constant expresses shared-role with zero additional type surface. |
| D-4 | Issue "Encode-containing shared roles" bullet | `PoolRoleEncode`, `PoolRoleEncodePrefill`, `PoolRoleEncodePrefillDecode` are not added. | INTENTIONAL DEFERRAL to GAP-4 (#1264). Called out in the issue itself. |
| D-5 | Issue hint "legacy `both` label" | We do not add a `PoolRoleBoth` alias/constant distinct from `PoolRolePrefillDecode`. | CLARIFICATION. BLIS has no label-string → role mapping (D-1); the constant is named for its semantics, not the wire label. `PoolRolePrefillDecode == PoolRolePrefill | PoolRoleDecode` is the single source of truth. |

### D) Component Interaction

- `sim/cluster/pool.go` — owner of `PoolRole` type and pool helpers.
- `sim/cluster/cluster.go` — consumer: pool-filtered snapshots, PD completion dispatch, unified construction loop.
- `sim/cluster/pd_metrics.go` — consumer: pool throughput attribution.
- `sim/cluster/deployment.go` — `DeploymentConfig.SharedInstances` field; `resolveConfigForRole` precedence for shared role.
- `cmd/root.go` — `--prefill-decode-instances` flag, validation.

No new interface; `PoolRole` goes from "scalar enum" to "bitmask-with-helper-method". Filter semantics widen from `==` to `.Has(role)`. All consumers are in-package (`cluster`) or in `cmd/`. No public API break outside these two packages.

### E) Risks

1. **Regression in the no-PD path (INV-6).** If the bitmask change subtly alters any `== PoolRolePrefill` comparison for `PoolRole(0)` (the default zero value), unassigned instances could be mis-classified.
   - Mitigation: `PoolRole(0).Has(X)` is `0 & X != 0` = `0 != 0` = false. Same truth value as `0 == X` for any nonzero `X`. The existing "pool disabled" branch is gated on `poolsConfigured()` (nil map check), not on role values — unchanged.
2. **Double-counting in INV-1.** A shared-role pod's `CompletedRequests` is added to both `prefillCompleted` and `decodeCompleted` in `collectPoolThroughput`. This is BC-6 by design (matches llm-d's semantic: one pod, serving in both capacities). The aggregated counter `CompletedRequests` in INV-1 conservation is computed from per-instance metrics, not from `prefillCompleted + decodeCompleted`, so INV-1 is unaffected.
3. **Construction-time precedence surprise.** If an operator sets `PrefillOverrides.TP = 8` and `DecodeOverrides.TP = 4`, a shared-role pod silently becomes TP=4 (D-2). Mitigation: emit an INFO log in `NewClusterSimulator` when `SharedInstances > 0` and the two override blocks differ.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Six tasks, TDD order: tests first (failing), then implementation, then CLI wiring.

### G) Tasks

#### Task 1: Add `PoolRole.Has` and `PoolRolePrefillDecode`; keep bitmask backward-compatible with existing constants

Convert `PoolRole` from `int` with `iota+1` to bit-flag constants, add `Has(role)` and update `String()`.

**Test first:** Add `TestPoolRole_HasAndCompose` in `sim/cluster/pool_test.go` covering BC-2 (Has across all combinations) and the updated `String` output for composite roles.

**Edit `sim/cluster/pool.go`:** replace the existing constants with
```go
const (
    PoolRolePrefill PoolRole = 1 << iota // 0b01
    PoolRoleDecode                       // 0b10
    // PoolRolePrefillDecode is the shared-role marker for pods that serve both stages.
    // Matches llm-d role labels "prefill-decode" and legacy "both".
    PoolRolePrefillDecode = PoolRolePrefill | PoolRoleDecode // 0b11
)
```
Add `func (r PoolRole) Has(other PoolRole) bool { return other != 0 && r&other == other }`.
Update `String()` to emit `"prefill"`, `"decode"`, `"prefill-decode"`, `"PoolRole(0)"`, or `"PoolRole(<int>)"` for unknowns.

**Commit:** `refactor(cluster): convert PoolRole to bitmask with Has() helper (#1276)`

#### Task 2: Widen `FilterSnapshotsByPool` and `BuildPoolMembershipFromIndices` to shared-role semantics (BC-1, BC-4)

**Test first:** Add `TestFilterSnapshotsByPool_SharedRole` asserting that a shared-role instance appears in both the `PoolRolePrefill` and `PoolRoleDecode` result lists (mirrors llm-d `roles_test.go`). Add `TestBuildPoolMembershipFromIndices_WithShared` asserting correct index ranges for `(total=5, prefill=2, decode=2, shared=1)` → indices 0,1 prefill-only; 2,3 decode-only; 4 shared.

**Edit `sim/cluster/pool.go`:**
- `FilterSnapshotsByPool`: change `membership[snap.ID] == role` to `membership[snap.ID].Has(role)`.
- `BuildPoolMembershipFromIndices(total, prefill, decode, shared int)`: emit `PoolRolePrefill` for `[0, prefill)`, `PoolRoleDecode` for `[prefill, prefill+decode)`, `PoolRolePrefillDecode` for `[prefill+decode, prefill+decode+shared)`.
- `ValidatePoolTopology(prefill, decode, shared, total int)`:
    - reject if any of the three is negative;
    - `prefill == 0 && decode == 0 && shared == 0` → nil (disabled);
    - when disaggregation enabled (any one nonzero), `prefill + decode + shared <= total`;
    - when shared == 0, preserve today's behaviour exactly (both prefill and decode must be nonzero);
    - when shared > 0, allow prefill == 0 or decode == 0 (a pure-shared-role cluster is legitimate).

Update the four existing `_test.go` call sites for the widened signature.

**Commit:** `feat(cluster): widen pool helpers for shared-role instances (#1276)`

#### Task 3: Update the two `poolMembership[instID] == PoolRole…` comparisons in `cluster.go:664-669` (BC-5)

**Test first:** Extend `sim/cluster/disaggregation_test.go` (or add a new focused test) with `TestSharedRolePodFiresBothDetectors`: build a minimal cluster where one instance is in `PoolRolePrefillDecode`, inject a PD sub-request pair, advance the sim, and assert both `detectPrefillCompletions` and `detectDecodeCompletions` paths observed the completion (observable via the existing parent-request bookkeeping).

**Edit `sim/cluster/cluster.go`:**
- Lines 664–669: replace `==` with `.Has(...)`.
- Line 264–268 (role assignment): unchanged — the per-instance role comes directly from `prePoolMembership[string(id)]`, which `BuildPoolMembershipFromIndices` now emits as `PoolRolePrefillDecode` for shared indices.
- Line 228 gate (`config.PrefillInstances > 0 || config.DecodeInstances > 0`): extend to `|| config.SharedInstances > 0` so a pure-shared cluster still enables PD.

**Commit:** `fix(cluster): fire both PD detectors for shared-role pods (#1276)`

#### Task 4: Update `collectPoolThroughput` to attribute shared-role completions to both pools (BC-6)

**Test first:** Extend `sim/cluster/pd_metrics_test.go` with `TestCollectPoolThroughput_SharedRole`: synthesize a membership with one `PoolRolePrefill`, one `PoolRoleDecode`, one `PoolRolePrefillDecode`, each with `CompletedRequests = 10`, and assert `prefillCompleted == 20` and `decodeCompleted == 20`.

**Edit `sim/cluster/pd_metrics.go:174-180`:** replace the `switch` with two independent `if .Has(...)` checks (one each for prefill and decode). Keep the `default`-style "instance not in pool membership" comment.

**Commit:** `fix(cluster): attribute shared-role throughput to both pools (#1276)`

#### Task 5: Add `SharedInstances` to `DeploymentConfig`; implement `resolveConfigForRole` decode-wins precedence (BC-7)

**Test first:** Extend `sim/cluster/deployment_test.go` (or add one) with `TestResolveConfigForRole_Shared`: set `PrefillOverrides.TP = intPtr(8)` and `DecodeOverrides.TP = intPtr(4)`, assert `resolveConfigForRole(PoolRolePrefillDecode)` returns `TP = 4`.

**Edit `sim/cluster/deployment.go`:**
- Add field: `SharedInstances int` (doc-commented: "Count of instances serving both prefill and decode. See issue #1276.").
- `resolveConfigForRole`: add `case PoolRolePrefillDecode: return ResolvePoolConfig(d.SimConfig, d.DecodeOverrides)` BEFORE the prefill/decode cases so the literal equality matches.
  - Because `PoolRolePrefillDecode == PoolRolePrefill | PoolRoleDecode`, the bare `case PoolRolePrefill:` would not match a shared role — correct without case-ordering tricks. Using `.Has` in a typed `switch` is clearer than bitwise ordering, but Go's `switch role` with equality is fine here because `PoolRolePrefillDecode != PoolRolePrefill`.

**Edit `sim/cluster/cluster.go`:** in `NewClusterSimulator`, after the `prePoolMembership` construction, if `config.SharedInstances > 0` and the two override structs are not DeepEqual, emit `logrus.Infof("[cluster] shared-role pods use DecodeOverrides (PrefillOverrides ignored per PR1276 D-2 decode-wins rule)")`.

**Commit:** `feat(cluster): DeploymentConfig.SharedInstances + decode-wins precedence (#1276)`

#### Task 6: CLI flag `--prefill-decode-instances`; update validation call site (BC-8)

**Test first:** Add a unit test (or extend an existing CLI test) invoking `blis run --num-instances 5 --prefill-instances 2 --decode-instances 2 --prefill-decode-instances 1 --dry-run` (or equivalent) to assert topology validates.

**Edit `cmd/root.go`:**
- Declare `var prefillDecodeInstances int` near `prefillInstances`, `decodeInstances`.
- Register the flag: `cmd.Flags().IntVar(&prefillDecodeInstances, "prefill-decode-instances", 0, "Number of instances serving both prefill and decode (shared-role pods, llm-d parity; 0 = disabled)")`.
- Validate (alongside the two existing `>= 0` checks): reject if `< 0`.
- Replace the `ValidatePoolTopology(prefillInstances, decodeInstances, config.NumInstances)` call with the 4-arg form.
- Set `DeploymentConfig.SharedInstances = prefillDecodeInstances`.

Register the flag on both `blis run` and `blis replay` to preserve cross-path parity (per the issue's "Cross-path parity" section: observe/convert/calibrate are not affected).

**Commit:** `feat(cmd): --prefill-decode-instances for shared-role pods (#1276)`

### H) Test Strategy

- **BC-1, BC-2:** `sim/cluster/pool_test.go` (pure unit tests).
- **BC-3, BC-9:** preserved by the existing test suite — no new tests required; `go test ./...` must pass.
- **BC-4:** `TestValidatePoolTopology` extended with shared-count scenarios (overflow, legitimate overlap, pure-shared cluster).
- **BC-5:** `sim/cluster/disaggregation_test.go` (integration).
- **BC-6:** `sim/cluster/pd_metrics_test.go`.
- **BC-7:** `sim/cluster/deployment_test.go`.
- **BC-8:** `cmd/` — existing CLI topology validation test, extended.

### I) Risks & Mitigations

(See Part 1 §E. Each risk has a named mitigation tied to an existing invariant test or a new test above.)

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No `logrus.Fatalf` in `sim/` library code (only the informational Infof).
- [x] No exported mutable maps (returned `PoolMembership()` map copy preserved).
- [x] No YAML zero-value ambiguity: `SharedInstances int` — zero means disabled, same as `PrefillInstances` / `DecodeInstances`.
- [x] No division-by-zero introduced (pool RPS computation unchanged).
- [x] Construction-site sweep for `PoolRole{...}` style literals: `rg -n "PoolRole\b" sim/ cmd/` — no struct has `PoolRole` as a field; constants are only used as values, so R4 does not apply here.
- [x] Deterministic iteration (INV-6): `collectPoolThroughput` sorts instance IDs before iterating (pd_metrics.go:164). `FilterSnapshotsByPool` preserves input slice order. No new map iteration introduced.
- [x] Error handling boundaries: all validation errors surface via `ValidatePoolTopology` return; CLI maps them to `logrus.Fatalf` at the boundary.
- [x] CLAUDE.md / principles.md / rules.md / invariants.md unaffected — no new invariants, no new rules, no new modules.

---

## Appendix: File-Level Details

**Files modified (production):** `sim/cluster/pool.go`, `sim/cluster/cluster.go`, `sim/cluster/pd_metrics.go`, `sim/cluster/deployment.go`, `cmd/root.go`.

**Files modified (tests):** `sim/cluster/pool_test.go`, `sim/cluster/pd_metrics_test.go`, `sim/cluster/disaggregation_test.go`, `sim/cluster/deployment_test.go` (if it exists; otherwise inline into pool_test.go).

**Rejected alternatives:** shape 2 (set-valued membership map) — widens every type signature unnecessarily; the bitmask preserves `map[string]PoolRole` shape.
