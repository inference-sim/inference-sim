# Contract: Routing Snapshot & LoRA-Aware Scorer

**Feature**: `007-lora-control-plane`
**Extends**: `sim.RoutingSnapshot` (add field) + `sim/routing_scorers.go` (add scorerFunc).
NO new interface (R13) — reuses `scorerFunc func(req *Request, snapshots []RoutingSnapshot) map[string]float64`.

## `RoutingSnapshot.ResidentAdapters` (PR6)

- **Field**: set of adapter ids resident on the instance.
- **Freshness (R17, INV-7)**: **Periodic** — populated by `buildRouterState()` at
  snapshot-build time (default 50ms); Immediate when `--snapshot-refresh-interval 0`.
- **Zero value**: nil/empty ⇒ scorer neutral (no-op default).

## `lora-affinity` scorer (PR6, Policy Template ≤3 files)

Registered in `validScorerNames` (R8) and the `newScorerWithObserver` switch. Stateless
(no observer). Reads `req.Adapter` and `snapshot.ResidentAdapters`.

```
raw(instance) = 1.0 if req.Adapter ∈ instance.ResidentAdapters else 0.0
score(instance) = minMaxNormalize(raw over candidate instances)   # llm-d parity
```

### Contract (GIVEN/WHEN/THEN)

- **GIVEN** instance A holds `req.Adapter` and instance B does not, all else equal **WHEN** scored **THEN** `score(A) > score(B)` (US4 scenario 1).
- **GIVEN** `req.Adapter == ""` **WHEN** scored **THEN** all instances score equally (neutral; base-model requests unaffected).
- **GIVEN** the scorer is not in the weighted profile (`--lora-scorer-weight 0` / unset) **WHEN** routing runs **THEN** routing decisions are unchanged from today (US4 scenario 3; INV-6).
- **GIVEN** a skewed adapter-popularity workload **WHEN** run with vs without the scorer **THEN** total adapter loads/evictions are lower with it (US4 scenario 2; SC-005 target ≥30% fewer loads).
- **INV-9**: the scorer reads `Adapter`/`ResidentAdapters` only — never `OutputTokens`.
- **Doc requirement (R17)**: the scorer's doc comment MUST list `ResidentAdapters` and its Periodic freshness tier.
