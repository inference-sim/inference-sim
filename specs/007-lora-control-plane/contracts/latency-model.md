# Contract: Latency Model (adapter deltas)

**Feature**: `007-lora-control-plane`
**Extends**: `sim.LatencyModel` — NO new interface (R13). `StepTime(batch []*Request)`
already receives the batch, so it can read each request's `Adapter`.

## Compute-overhead factor (PR4)

The adapter delta is a **multiplicative factor ≥ 1.0** applied to the calibrated base
step time, computed in shared code so both backends (`roofline`, `trained-physics`)
apply it identically (R23):

```
r_max  = max adapter rank in batch      # selects the fitted-coefficient tier
A_B    = count of DISTINCT non-empty Adapter ids in batch
factor(batch) = (K7(r_max) + K6(r_max)·A_B) / K7(r_max) = 1 + (K6(r_max)/K7(r_max))·A_B
              # normalized to the no-adapter baseline: = 1.0 at A_B=0 by construction, for ANY fitted K7
              # rank enters via the max-rank tier (K6/K7 are per-rank, not flat) — a higher r_max ⇒ longer step (FR-009)
StepTime_with_adapters = StepTime_base(batch) · factor(batch)
```

Rank (and hence the max-rank contribution) is resolved via an **adapter-cost
accessor supplied at `LatencyModel` construction** — NOT carried on `Request`
(requests hold the adapter id only). An absent/nil accessor ⇒ `factor = 1.0`
(no-op default, INV-6). The same accessor is passed to both backends (R13/R23).

### Contract (GIVEN/WHEN/THEN)

- **GIVEN** a batch where every request has `Adapter == ""` **WHEN** `StepTime(batch)` **THEN** the factor is exactly `1.0` and the result equals the pre-feature `StepTime` bit-for-bit (INV-6, SC-001). *Holds by construction for ANY fitted `K7` (the normalized factor = 1.0 at `A_B=0`), not only the default `K7=1.0` — a non-unit `K7` must still yield exactly 1.0 here.*
- **GIVEN** two batches identical except batch B has strictly more distinct adapters (or a higher max rank) than batch A **WHEN** each is timed **THEN** `StepTime(B) ≥ StepTime(A)` (monotonicity; US3 scenario 2).
- **GIVEN** the same adapter on multiple requests in one batch **WHEN** `A_B` is computed **THEN** it is counted once (spec edge case).
- **GIVEN** backend ∈ {roofline, trained-physics} **WHEN** the same adapter batch is timed **THEN** both apply the identical factor (R23 — diff paths explicitly).
- **Postcondition preserved**: `StepTime ≥ 1` for all inputs (INV-3; interface contract).

## Cold-load latency (PR3)

Not part of `StepTime`. Charged by the pre-admission gate (see
[routing-snapshot.md](./routing-snapshot.md) / research R1) as:

```
LoadLatency(id) = load_base_latency_us + ceil(FootprintBytes(rank) / load_bandwidth_bytes_us)
```

- **GIVEN** a cold adapter **WHEN** admitted **THEN** `LoadLatency ≥ 0`, charged once, reflected in the request's TTFT (SC-004).
- **GIVEN** a warm adapter **WHEN** admitted **THEN** zero load latency (US3 scenario 1).
- **Determinism**: pure function of rank + config; no RNG (R7).
