# H13: Determinism Invariant

**Status**: Confirmed
**Date**: 2026-02-20

## Hypothesis

> Same seed must produce byte-identical stdout across runs. BLIS uses PartitionedRNG for deterministic simulation -- running the same configuration with the same seed twice should produce identical output. This is critical for reproducible research.

**Refuted if:** Any configuration produces non-identical stdout between two runs with the same seed.
