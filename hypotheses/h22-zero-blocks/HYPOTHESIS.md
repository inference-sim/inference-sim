# H22: Zero KV Blocks -- CLI Validation Boundary

**Status**: Confirmed
**Date**: 2026-02-22

## Hypothesis

> Running with `--total-kv-blocks 0` (or other zero/negative KV configurations) should produce a clean CLI error (`logrus.Fatalf`), not a panic or stack trace from `sim/`.

**Refuted if:** Any zero or negative KV block configuration produces a panic, stack trace, or exit code 0.
