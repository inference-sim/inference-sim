# E/P/D Disaggregation Micro Plan

**Goal:** Add the encode pool role (`PoolRoleEncode`), an `EncodeDecider` interface, CLI flags, and a synchronous encode routing stage inside `executeDisaggregatedRouting`. Purely additive — when `--encode-instances 0` (default), behavior is byte-identical to pre-PR.

**Source:** issue #1264 (GAP-4) + design doc `2026-05-08-epd-disaggregation-design.md`.

**Closes:** #1264

**Tier:** Medium (≈6 files changed, one new interface/type, opt-in architectural addition). See [PR Size Tiers](../contributing/pr-workflow.md#pr-size-tiers).

---

## Deviation log

| Deviation | Reason | Type |
|-----------|--------|------|
| No new `EncodeRoutingEvent` type; encode stage is inlined into `executeDisaggregatedRouting`. | Issue's own Extension-friction note: "If GAP-1 (routing path unification) is resolved first, add the encode stage to `executeDisaggregatedRouting` rather than to `DisaggregationDecisionEvent.Execute`." GAP-1 is merged. Option A = zero-duration; wrapping in an event adds bookkeeping without modeling fidelity. | CLARIFICATION |
| No new `IsMultimodal` field on `sim.Request`; use a derived method `Request.IsMultimodal()` over existing `ImageTokenCount + AudioTokenCount + VideoTokenCount`. | Those fields already exist and already flow through spec → generator → trace v2 → replay. Adding a boolean duplicates the source of truth. | CLARIFICATION |
| No new `is_multimodal` column in TraceV2. | Derivable from existing `image_tokens`, `audio_tokens`, `video_tokens` columns. | CLARIFICATION |
| `EncodeInstanceID` only lives on `ParentRequest` (populated during `executeDisaggregatedRouting`) and only on the disagg path. | On the non-disagg path (BC-EPD-4), `encodeInstanceID` is captured as a local variable, recorded in the trace (`EncodeRoutingRecord`), and then discarded — no `ParentRequest` is created. This matches the current non-disagg behavior (no parent tracking); the encode instance is already observable via the trace record. | CLARIFICATION |
| `PoolRole` is a bitmask (per PR #1278), not `iota + 1`. | Issue was written against an older commit (before #1278). Use `PoolRoleEncode = 1 << 2` to extend the bitmask. | CLARIFICATION |
| Priority renumbering (shift `PrefillRoutingEvent` 4→5, `KVTransferStarted` 5→6) is **not applied** because we did not add a new event. | No new event type added; existing priorities retained. | CLARIFICATION |
| Introduce a `NeverEncode` decider in addition to `AlwaysEncode` and `MultimodalEncodeDecider`. | Useful for wiring tests; symmetric with `NeverDisaggregate`. | CLARIFICATION |

---

## Behavioral contracts

### BC-EPD-1 (opt-in invariance)
**GIVEN** `--encode-instances 0` (or unset)
**WHEN** the simulator runs any workload
**THEN** the result is byte-identical to the pre-PR simulator (determinism: INV-6).

### BC-EPD-2 (multimodal detection)
**GIVEN** a request with at least one non-zero per-modality token count (`ImageTokenCount` / `AudioTokenCount` / `VideoTokenCount`)
**WHEN** `MultimodalEncodeDecider.ShouldEncode(req, decodeInstanceID)` is called
**THEN** it returns `true`; otherwise it returns `false`.

### BC-EPD-3 (encode + disagg fires encode → prefill → decode)
**GIVEN** `--encode-instances > 0`, `--prefill-instances > 0`, `--decode-instances > 0`, decider = `multimodal`, and a multimodal request with `disaggDecision.Disaggregate == true`
**WHEN** the request reaches `executeDisaggregatedRouting`
**THEN** the trace records an `EncodeRoutingRecord` whose timestamp `≤` the corresponding `PrefillRoutingRecord` timestamp `≤` the corresponding `KVTransferRecord.TransferStartTime`, and `ParentRequest.EncodeInstanceID != ""`.

### BC-EPD-4 (encode + non-disagg)
**GIVEN** `--encode-instances > 0`, decider = `always`, and `disaggDecision.Disaggregate == false`
**WHEN** the request reaches `executeDisaggregatedRouting`
**THEN** the trace records an `EncodeRoutingRecord` for the request, no `PrefillRoutingRecord` for the request, and the request is injected directly to the pre-selected decode instance (standard `RoutingRecord` as in the current non-disagg fork). No `ParentRequest` is created (matches the current non-disagg behavior).

### BC-EPD-5 (empty encode pool rejects at config)
**GIVEN** a decider that would fire (`always` or `multimodal`) and `--encode-instances 0`
**WHEN** the CLI validates
**THEN** validation rejects with a clear error naming the missing flag. (Construction-time check: `EncodeDecider != nil` requires `EncodeInstances > 0`.)

### BC-EPD-6 (conservation)
**GIVEN** a run where some encode routings are rejected (all encode instances removed mid-run)
**WHEN** the end-of-run ledger is computed
**THEN** `encode_routing_rejections` appears as a term in the INV-1 equality and the ledger balances.

### BC-EPD-7 (INV-9 oracle boundary)
**STATEMENT:** `MultimodalEncodeDecider.ShouldEncode` (and `AlwaysEncode.ShouldEncode`, `NeverEncode.ShouldEncode`) must not reference `req.OutputTokens` at any callsite.
**VERIFICATION:** inspection + test: a request with `OutputTokens: nil` is accepted and correctly classified by `MultimodalEncodeDecider`.

### BC-EPD-8 (determinism with E/P/D enabled)
**GIVEN** the same seed and the same workload with `--encode-instances > 0`
**WHEN** the simulator runs twice
**THEN** stdout is byte-identical.

---

## Tasks (TDD)

### Task 1 — `PoolRoleEncode` constant + metadata
Add `PoolRoleEncode = 1 << 2` to `sim/cluster/pool.go` (bitmask extension — the existing `PoolRolePrefill = 1 << iota` continues via its `iota` increment; add `PoolRoleEncode` after `PoolRoleDecode`). Update `String()` to return `"encode"`. Update `ValidatePoolTopology` signature to accept `encode int`: add validation that `encode >= 0` and `prefill+decode+shared+encode <= total`. Update `BuildPoolMembershipFromIndices` to assign `PoolRoleEncode` to indices `[prefill+decode+shared, prefill+decode+shared+encode)`.

Update all existing call sites of `ValidatePoolTopology` and `BuildPoolMembershipFromIndices` to pass `0` for `encode` (no behavioral change).

**Tests:** extend `sim/cluster/pool_test.go` with:
- `TestPoolRoleEncodeHasBit` — `PoolRoleEncode.Has(PoolRoleEncode) == true`; prefill+decode bitmask does not match encode.
- `TestValidatePoolTopology_Encode_OK` — `(1,1,0,1, total=3)` and `(0,0,0,2, total=2)` (pure encode).
- `TestValidatePoolTopology_Encode_Exceeds` — `(1,1,0,1, total=2)` returns an error naming `encode-instances`.
- `TestBuildPoolMembershipFromIndices_Encode` — indices after `prefill+decode+shared` carry `PoolRoleEncode`.

**Commit:** `feat(cluster): add PoolRoleEncode bitmask, validation, and membership layout (GAP-4)`.

### Task 2 — `EncodeDecider` interface + implementations
Add to `sim/disaggregation.go`:

```go
type EncodeDecider interface {
    ShouldEncode(req *Request, decodeInstanceID string) bool
}

type AlwaysEncode struct{}
type NeverEncode struct{}
type MultimodalEncodeDecider struct{}

var (
    _ EncodeDecider = (*AlwaysEncode)(nil)
    _ EncodeDecider = (*NeverEncode)(nil)
    _ EncodeDecider = (*MultimodalEncodeDecider)(nil)
)
```

`MultimodalEncodeDecider.ShouldEncode` returns `req.IsMultimodal()` (uses the new method added in Task 3). Bundle registration (add to `sim/bundle.go`): `validEncodeDeciders = map[string]bool{"": true, "never": true, "always": true, "multimodal": true}` with `IsValidEncodeDecider` / `ValidEncodeDeciderNames` helpers, matching the disaggregation decider pattern. `NewEncodeDecider(name string) EncodeDecider` factory.

**Tests:** `sim/disaggregation_test.go`:
- `TestAlwaysEncode` — returns true for any request.
- `TestNeverEncode` — returns false for any request.
- `TestMultimodalEncodeDecider_True` — request with `ImageTokenCount: 10` → true.
- `TestMultimodalEncodeDecider_False` — request with all modality counts zero → false.
- `TestMultimodalEncodeDecider_IgnoresOutputTokens` — request with `OutputTokens: nil` + `ImageTokenCount: 5` → true (BC-EPD-7).
- `TestIsValidEncodeDecider` — table test for valid/invalid names.
- `TestNewEncodeDecider_Panics` — panics on unknown name.

**Commit:** `feat(sim): add EncodeDecider interface with always/never/multimodal implementations (GAP-4)`.

### Task 3 — `Request.IsMultimodal()` method
Add to `sim/request.go`:

```go
// IsMultimodal reports whether the request carries any non-text modality
// (image, audio, or video) based on per-modality token counts. Derived from
// existing workload-spec + TraceV2 fields; no separate source of truth.
func (r *Request) IsMultimodal() bool {
    return r.ImageTokenCount > 0 || r.AudioTokenCount > 0 || r.VideoTokenCount > 0
}
```

**Tests:** `sim/request_test.go`:
- `TestRequestIsMultimodal` — table test: text-only=false; image>0=true; audio>0=true; video>0=true; all three=true.

**Commit:** `feat(sim): add Request.IsMultimodal() derived from existing per-modality token counts (GAP-4)`.

### Task 4 — `ParentRequest.EncodeInstanceID`
Add `EncodeInstanceID InstanceID` to `sim/cluster/parent_request.go`. Zero-value (empty) means encode did not fire for this parent. (Encode timestamps deferred to follow-up with non-zero encode latency.)

**Tests:** `sim/cluster/parent_request_test.go` (may need creating) — `TestNewParentRequest_EncodeIDZero` — newly constructed `ParentRequest.EncodeInstanceID == ""`.

**Commit:** `feat(cluster): add ParentRequest.EncodeInstanceID field (GAP-4)`.

### Task 5 — `EncodeRoutingRecord` in trace + cluster counter
Add a new trace record type:

```go
// sim/trace/record.go
type EncodeRoutingRecord struct {
    ParentRequestID string
    Clock           int64
    ChosenInstance  string
    Scores          map[string]float64
    // Candidates / Regret fields for counterfactual (match PrefillRoutingRecord shape)
    Candidates []string
    Regret     float64
}
```

Add a `RecordEncodeRouting` method on `SimulationTrace` and an `EncodeRoutings []EncodeRoutingRecord` slice on the trace struct.

Add counter `encodeRoutingRejections int64` on `ClusterSimulator`. Expose via `EncodeRoutingRejections() int64`.

Add `encode_routing_rejections` to the INV-1 ledger in `sim/cluster/cluster_test.go` conservation checks and to any JSON output site that mirrors the ledger. Update `docs/contributing/standards/invariants.md` INV-1 text to list the new term.

**Tests:**
- `sim/cluster/cluster_test.go` — new `TestINV1_EncodePool_Conservation`: small run with `--encode-instances > 0` and `--encode-decider always`, verify `encode_routing_rejections` is zero and ledger still balances. (Forcing rejections synthetically is covered in Task 6 once the routing stage exists.)

**Commit:** `feat(cluster): add EncodeRoutingRecord trace + encode_routing_rejections conservation term (GAP-4)`.

### Task 6 — wire encode stage into `executeDisaggregatedRouting`
Modify `sim/cluster/cluster.go` `executeDisaggregatedRouting`:

After the decode decision is finalized (line ~1748, after `DecodePodOverride` handling and trace recording) and before the `if !disaggDecision.Disaggregate { ... }` fork, insert the encode stage:

```go
// Encode stage (GAP-4): if an encode decider is configured and it approves,
// route to the encode pool before proceeding to prefill/decode.
// Under option A, encode is zero-duration: routing decision + trace record
// are made synchronously at the current tick; no sub-request is injected.
if cs.encodeDecider != nil && cs.encodeDecider.ShouldEncode(req, decodeDecision.TargetInstance) {
    encodeSnapshots := cs.buildPoolFilteredSnapshots(PoolRoleEncode)
    if len(encodeSnapshots) == 0 {
        logrus.Warnf("[cluster] req %s: no routable instances in encode pool — request rejected at encode routing", req.ID)
        cs.encodeRoutingRejections++
        return
    }
    encodeState := &sim.RouterState{Snapshots: encodeSnapshots, Clock: cs.clock}
    encodePolicy := cs.encodeRoutingPolicy
    if encodePolicy == nil {
        encodePolicy = cs.routingPolicy
    }
    encodeDecision := encodePolicy.Route(req, encodeState)

    // Record encode routing (trace-compat with counterfactual).
    if cs.trace != nil {
        record := trace.EncodeRoutingRecord{
            ParentRequestID: req.ID,
            Clock:           cs.clock,
            ChosenInstance:  encodeDecision.TargetInstance,
            Scores:          copyScores(encodeDecision.Scores),
        }
        if cs.trace.Config.CounterfactualK > 0 {
            record.Candidates, record.Regret = computeCounterfactual(
                encodeDecision.TargetInstance, encodeDecision.Scores,
                encodeSnapshots, cs.trace.Config.CounterfactualK,
            )
        }
        cs.trace.RecordEncodeRouting(record)
    }

    encodeInstanceID = encodeDecision.TargetInstance // captured for parent population below
}
```

On the disagg path only, set `parent.EncodeInstanceID = InstanceID(encodeInstanceID)` after the `ParentRequest` is constructed. On the non-disagg path, no `ParentRequest` is created (matching the existing non-disagg behavior) — the encode instance is already recorded in the trace via `EncodeRoutingRecord`, which is the sole observable for BC-EPD-4 assertions.

Add corresponding fields to `ClusterSimulator`:
- `encodeDecider sim.EncodeDecider`
- `encodeRoutingPolicy sim.RoutingPolicy`
- `encodeRoutingRejections int64`

Initialize in `NewClusterSimulator` only when `config.EncodeInstances > 0`. Construct via `sim.NewEncodeDecider(config.EncodeDecider)`.

**Tests:** `sim/cluster/disaggregation_test.go` (new subtests or new file `epd_test.go`):
- `TestEPD_EncodeFires_WithDisagg_RecordsEncodeThenPrefill` (BC-EPD-3).
- `TestEPD_EncodeFires_WithoutDisagg_DirectDecode` (BC-EPD-4).
- `TestEPD_EncodeNever_IsPDUnchanged` — guard ensuring encode pool configured but decider="never" leaves PD behavior untouched.
- `TestEPD_EncodePoolEmpty_RoutingRejection` — configure encode decider but ensure no encode instances routable → `encodeRoutingRejections` increments; INV-1 holds.
- `TestEPD_DeciderReadsDecodeInstanceID` — custom spy decider records the passed `decodeInstanceID` argument; verify it matches the pre-selected decode pod.

**Commit:** `feat(cluster): add encode routing stage to executeDisaggregatedRouting (GAP-4)`.

### Task 7 — `DeploymentConfig` fields + CLI flags
Add to `sim/cluster/deployment.go`:

```go
EncodeInstances int    // Number of instances dedicated to encode (0 = disabled)
EncodeDecider   string // "", "never", "always", "multimodal"
```

Update `BuildPoolMembershipFromIndices` call sites (Task 1 already updated the signature). Update `resolveConfigForRole` to handle `PoolRoleEncode` — return the global `SimConfig` unchanged (no per-pool overrides in this PR).

Add CLI flags in `cmd/root.go` on **both** `run` and `replay` (following the `PD` pattern — but note `replay` currently warns on PD flags; keep that pattern and either support encode on replay or explicitly warn). **Decision:** register encode flags on both subcommands and fully support them on `run`. For `replay`, if `--encode-instances > 0` is set, accept it and honor it (replay does not disable PD — only `--prefill-decode-instances` is warned against. See cmd/replay.go:188). Treat encode symmetrically with prefill/decode.

```
--encode-instances int      Number of instances dedicated to encoding (0 = encode pool disabled)
--encode-decider   string   Encode decider: "never" (default), "always", "multimodal"
```

Config validation in `cmd/root.go`:
1. `encode-instances >= 0`.
2. If `encode-instances > 0`:
   - `prefill-instances + decode-instances + prefill-decode-instances > 0` (encode requires a decode-capable pool to serve the request).
   - `encode-decider` must be a valid name.
3. If `encode-decider` is set to a non-`never` value but `encode-instances == 0`: `logrus.Fatalf` with a clear message.

Update `ValidatePoolTopology` callers in `cmd/root.go` and `sim/cluster/cluster.go` to pass `encodeInstances`.

**Tests:** `cmd/root_test.go` if one exists, else defer to integration tests exercising end-to-end CLI flag parsing through an e2e test in `sim/cluster/`. Minimum: a unit test in `sim/cluster/` that calls `ValidatePoolTopology` with encode counts.

**Commit:** `feat(cli): add --encode-instances and --encode-decider flags on run and replay (GAP-4)`.

### Task 8 — Invariants documentation update
Update `docs/contributing/standards/invariants.md` INV-1 section to add the `encode_routing_rejections` term and a one-sentence rationale. Keep the CLAUDE.md mirror consistent with the canonical source.

**Commit:** `docs(standards): extend INV-1 with encode_routing_rejections term (GAP-4)`.

### Task 9 — Determinism + opt-in invariance guard
Add an explicit test `TestEPDDisabled_PDUnchanged` in `sim/cluster/epd_test.go` that runs a PD workload with `--encode-instances 0` and compares against a pre-computed golden — asserting INV-6 byte-identity (in practice: compare completed-request counts, routing trace lengths, and ledger to ensure no accidental divergence). Plus `TestEPDDeterminism_WithEncode` — same seed twice, assert equality across all tracked invariants.

**Commit:** `test(cluster): EPD opt-in invariance + determinism guards (GAP-4)`.

### Task 10 — End-to-end verification gate
Run:
```
go build ./...
go test ./... -count=1
golangci-lint run ./...
```
All three must be green. Report exit codes and counts.

---

## Sanity checklist

- [ ] `--encode-instances 0` leaves every PD test byte-identical.
- [ ] `MultimodalEncodeDecider` never references `req.OutputTokens`.
- [ ] `ParentRequest` canonical constructor `NewParentRequest` initializes `EncodeInstanceID = ""`.
- [ ] No new exported mutable map.
- [ ] `encode_routing_rejections` is included in INV-1 ledger and invariants.md.
- [ ] All `logrus.Fatalf` calls only live in `cmd/` (library code uses `panic` or `error`).
- [ ] No `--no-verify` used when committing.
- [ ] Encode trace record is only emitted when encode actually fired (not when decider returned false).

---

## Appendix — affected files

- `sim/cluster/pool.go` (new `PoolRoleEncode`, signature of `ValidatePoolTopology` and `BuildPoolMembershipFromIndices` extended).
- `sim/cluster/pool_test.go`
- `sim/cluster/cluster.go` (inline encode stage in `executeDisaggregatedRouting`; new `encodeDecider` / `encodeRoutingPolicy` / `encodeRoutingRejections` fields).
- `sim/cluster/cluster_event.go` / `cluster_test.go` (ledger check).
- `sim/cluster/deployment.go` (new `EncodeInstances`, `EncodeDecider` fields; `resolveConfigForRole` extension).
- `sim/cluster/parent_request.go` (new `EncodeInstanceID` field).
- `sim/cluster/epd_test.go` (new file; all BC-EPD-* tests).
- `sim/disaggregation.go` (new `EncodeDecider` interface + three implementations + factory).
- `sim/disaggregation_test.go`
- `sim/request.go` (new `IsMultimodal()` method).
- `sim/request_test.go`
- `sim/bundle.go` (valid encode decider names).
- `sim/trace/record.go` (new `EncodeRoutingRecord`).
- `sim/trace/trace.go` (new `RecordEncodeRouting`, `EncodeRoutings` field).
- `cmd/root.go` + `cmd/replay.go` (new flags).
- `docs/contributing/standards/invariants.md` (INV-1 update).
- `docs/plans/2026-05-08-epd-disaggregation-design.md` (this design doc).
- `docs/plans/2026-05-08-epd-disaggregation-plan.md` (this plan).
