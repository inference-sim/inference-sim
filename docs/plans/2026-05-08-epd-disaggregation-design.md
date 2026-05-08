# E/P/D Disaggregation (Encode Pool) — Design Doc

**Status:** Draft
**Date:** 2026-05-08
**Closes:** #1264
**Parent:** #1260 (cross-repo parity tracking)

---

## Context

llm-d-inference-scheduler supports a third pool role beyond prefill and decode: an **encode pool**. For multimodal models (image / video / audio inputs), `Handler.Pick()` has an optional Stage 2 that routes to an encode-dedicated endpoint if the encode decider approves, passing the pre-selected decode endpoint to the decider (single endpoint, not a snapshot). BLIS does not model this today.

This design adds the encode pool role, an encode decider interface, CLI flags, and conservation accounting. The change is **purely additive**: when `--encode-instances 0` (default), behavior is byte-identical to the pre-PR simulator.

## Module contract

**Encode routing stage** (sim/cluster/cluster.go, inline inside `executeDisaggregatedRouting`).

- **Observes:** `sim.Request.IsMultimodal`, pre-selected decode instance ID, encode-pool RouterState, encode decider.
- **Controls:** whether an encode routing decision is made and recorded.
- **State owned:** `ParentRequest.EncodeInstanceID` (new); cluster-level counter `encodeRoutingRejections` (new).
- **Invariants maintained:**
  - INV-1 extended with `encode_routing_rejections` term (see "INV-1 update").
  - INV-9 upheld: `EncodeDecider` implementations must not read `req.OutputTokens`.
- **Events produced/consumed:** none — encode is inlined into `executeDisaggregatedRouting` synchronously. No new event type in this PR. Rationale: encode latency is zero (option A); event wrapping would be pure ceremony. The inline stage is trivially refactorable into an `EncodeRoutingEvent` when non-zero latency is added (option B/C).
- **Extension friction:** adding a new encode decider requires ~15 lines in `sim/disaggregation.go` (new struct + `ShouldEncode` method + bundle registration + a switch case in `NewEncodeDecider`). Matches existing `DisaggregationDecider` pattern.

## Key design decisions

### D1. Stage placement: inline synchronous, not a new event

**Choice:** encode routing is performed synchronously inside `executeDisaggregatedRouting`, after the decode decision and before the prefill-routing-vs-direct-injection fork.

**Reasoning:** The issue's "Extension friction check" says: *"If GAP-1 (routing path unification) is resolved first, add the encode stage to `executeDisaggregatedRouting` rather than to `DisaggregationDecisionEvent.Execute`."* GAP-1 is resolved. Under option A (zero-duration encode), adding a dedicated event adds bookkeeping without modeling fidelity. If/when non-zero encode latency is introduced, the stage lifts cleanly into a `EncodeRoutingEvent` with no interface churn (the decider signature is unchanged; only the call site moves).

**Deviation from llm-d:** llm-d's `Handler.Pick()` wraps stages in `SchedulerProfile` invocations. BLIS's inline form is structurally equivalent for option-A semantics.

### D2. Encode latency = 0 (option A from the issue)

**Choice:** Encode completes at the same simulation tick that routing fires. No encode sub-request is injected into an instance; the encode pool's capacity and load are observed by routing but not modified.

**Reasoning:** BLIS has no multimodal workload calibration data today. Any non-zero constant latency would be arbitrary. Listed as a known deviation; follow-up issue will add `--encode-latency-ms` (option B).

**Consequence:** the encode routing policy still makes a real routing decision (which encode instance would have served the request). Load-imbalance tests and counterfactual tracing work normally.

### D3. `IsMultimodal` is derived, not a new request field

**Choice:** `Request.IsMultimodal()` is a method that returns `ImageTokenCount + AudioTokenCount + VideoTokenCount > 0`. These fields already exist on `sim.Request` and already flow through spec → generator → trace v2 → replay.

**Reasoning:** The issue's Step 2a proposes adding a new `IsMultimodal bool` field plus a new `is_multimodal` TraceV2 column. Both are redundant given the existing per-modality token counts. A derived method:

- avoids TraceV2 schema churn (backward-compat concern eliminated),
- avoids a second source-of-truth for multimodality (cannot desynchronize from the token counts),
- keeps the `blis observe` follow-up simpler (it only needs to populate token counts, not also set a boolean).

### D4. Encode requires disaggregated mode when `--encode-instances > 0` in this PR

**Choice:** Encode routing fires on the disaggregated path (`executeDisaggregatedRouting`) only. The non-disaggregated path (single-instance or `poolsConfigured()==false`) does not run the encode stage.

**Reasoning:** Cleanly scopes this PR. Encode-without-decode-pools is not a realistic production topology. Validation rejects `--encode-instances > 0 && (--prefill-instances == 0 && --decode-instances == 0 && --prefill-decode-instances == 0)`.

**Sub-case preserved:** encode + `disaggDecision.Disaggregate == false` (decode pool serves both prefill and decode for this request) IS supported — encode records its decision and the request then injects directly to the pre-selected decode pod (no prefill routing). This matches issue Step 8.3.

### D5. Encode decider defaults

**Choice:**
- When `--encode-instances == 0` (default): encode decider is `nil`. No encode stage runs. Backward-compatible.
- When `--encode-instances > 0` and `--encode-decider` unset: default to `"multimodal"` (per the issue's Step 7).
- `"always"`: always encode (for tests / wiring validation).
- `"never"`: never encode (disables the stage while keeping the pool reservable).

**Deviation from Step 2 of the issue:** `NeverEncode` is a new test-only decider not mentioned in the issue. Rationale: symmetric with `NeverDisaggregate`; useful for testing "flag-enabled but decider off" wiring.

## INV-1 update

New conservation term at cluster level:

```
injected_requests == completed_requests + still_queued + still_running
                   + dropped_unservable + timed_out + routing_rejections
                   + gateway_queue_depth + gateway_queue_shed + gateway_queue_rejected
                   + encode_routing_rejections  // NEW (GAP-4)
```

`encode_routing_rejections` counts requests rejected at the encode routing stage because the encode pool has zero routable instances. Encode sub-requests do not count separately in `injected_requests` (no separate injection occurs under option A; they are bookkeeping artifacts of the parent request).

Single-instance simulations have no encode pool; the term is always zero.

## Workload schema

No spec or trace schema changes. `IsMultimodal` is derived from the existing `image_tokens`, `audio_tokens`, `video_tokens` columns in TraceV2 (present since PR #420) and from the existing `MultimodalSpec` in the v2 workload spec.

## Tests (behavioral contracts)

See micro plan `2026-05-08-epd-disaggregation-plan.md` for full test list. Summary of contracts:

- **BC-EPD-1** (opt-in invariance): with `--encode-instances 0`, behavior is byte-identical to pre-PR across all existing PD and standard tests (INV-6).
- **BC-EPD-2** (encode decider fires for multimodal): `MultimodalEncodeDecider.ShouldEncode` returns true iff the request has non-zero image/audio/video tokens.
- **BC-EPD-3** (encode + disagg): when encode fires and `disaggDecision.Disaggregate == true`, the pipeline records encode routing, then prefill routing, then decode; `ParentRequest.EncodeInstanceID` is set.
- **BC-EPD-4** (encode + non-disagg): when encode fires and `disaggDecision.Disaggregate == false`, the pipeline records encode routing, then injects directly to the pre-selected decode instance — no prefill routing.
- **BC-EPD-5** (empty encode pool): `--encode-instances 0` with `--encode-decider multimodal` and a multimodal request → config validation rejects at CLI.
- **BC-EPD-6** (conservation): `encode_routing_rejections` is included in the INV-1 ledger.
- **BC-EPD-7** (INV-9 oracle boundary): `MultimodalEncodeDecider` does not reference `req.OutputTokens`.
- **BC-EPD-8** (determinism): same seed + same workload + E/P/D enabled → byte-identical stdout.

## Follow-ups (not in this PR)

- Non-zero encode latency (`--encode-latency-ms`, option B).
- Encode pool hardware overrides (analogous to `PrefillOverrides` / `DecodeOverrides`).
- `blis observe` multimodal detection (chat-completions body parsing → populate `image_tokens`/`audio_tokens`/`video_tokens` into TraceV2).
- Encode-specific routing scorers and per-pool scorer configs.
