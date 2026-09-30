# Contract — Metrics / Provenance Output

## Effective-triple provenance (B-7, D6/D8)
- **Field**: a run-level entry in `MetricsOutput` recording the resolved {routing, eviction, creation}
  policy triple actually used (post-bundle-expansion, post-override).
- **Computation**: once at policy resolution, before the event loop — not accumulated per-event
  (keeps it out of any state-mutation path; state/statistics separation).
- **Omission (INV-6)**: **absent** from stdout whenever every seam is at baseline (all-baseline /
  adapter-blind run), mirroring the existing adapter-metrics / HBM-reservation omit-when-inert pattern.
  Present whenever any non-baseline policy is active.
- **Reproducibility (SC-006)**: a reviewer can reconstruct the exact policy configuration from the
  recorded triple alone, without the original command line.

## Existing adapter metrics (unchanged)
- Per-adapter load/eviction counts and TTFT/throughput (from the merged control-plane subsystem) are
  unchanged. Pre-placed adapters show **zero load-count** (INV-L3 / D4) — the headline SC-002 signal.

## Determinism & parity
- Same seed + same policy selection ⇒ identical routes, victims, seedings, and identical `MetricsOutput`
  across two runs (INV-6).
- Export → replay with identical policy selection ⇒ identical per-request metrics (INV-13); the triple
  round-trips to an equivalent run configuration.

## Stdout/stderr separation
- The provenance field goes to **stdout** (deterministic results). All diagnostics/warnings (e.g., an
  unknown-name fatal, or a positional-tie-break advisory) go to **stderr** (Principle II).
