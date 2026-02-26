# Design: Seed Unification for CLI and Workload-Spec

**Date:** 2026-02-20
**Status:** Implemented
**Species:** Decision Record
**Closes:** #284

---

## Motivation

When using `--workload-spec`, the CLI `--seed` flag does not control workload generation. The workload-spec YAML has its own `seed:` field that independently seeds the workload RNG via a separate `PartitionedRNG` instance. Running `--seed 42`, `--seed 123`, `--seed 456` with the same workload-spec produces **identical workloads** — only simulation-level randomness (routing, tie-breaking) varies.

Users expect `--seed` to be the master seed controlling all randomness. A user running "3 seeds for statistical rigor" gets 3 identical runs if using workload-spec, violating the principle of least surprise (R18: CLI flag precedence).

## Scope

**In scope:**
- CLI `--seed` override of workload-spec YAML seed
- Documentation of seed interaction semantics
- Backward compatibility for existing workload-spec YAMLs
- Update to ED-4 experiment standard

**Out of scope:**
- Changes to `PartitionedRNG` architecture (subsystem isolation is sound)
- Changes to distribution-based or CSV workload generation (these already use the CLI seed correctly)
- Changes to `sim/workload/` library API (it correctly takes seed from the spec — the bug is at the CLI integration point)

**Deferred:**
- A `--workload-seed` flag for explicitly overriding only the workload seed (not needed until a user requests it)

## Modeling Decisions

| Aspect | Decision | Justification |
|--------|----------|---------------|
| Seed interaction model | CLI overrides YAML | Users expect `--seed` to control all randomness (R18) |
| When YAML seed is used | When `--seed` is not explicitly passed by the user | Preserves "same YAML, same workload" reproducibility for default runs |
| Override detection | `cmd.Flags().Changed("seed")` | Same pattern already used for `--horizon` and `--num-requests` |
| Library API change | None — `GenerateRequests` already uses `spec.Seed` | The fix is at the CLI layer (`cmd/root.go`), not the library |

## Invariants

**INV-6 (Determinism) — strengthened:**
- Same `--seed` + same `--workload-spec` MUST produce byte-identical stdout
- Different `--seed` values with the same `--workload-spec` MUST produce different workloads (arrival times, token counts)
- When `--seed` is not explicitly passed, the YAML seed governs workload generation (backward compatible)

**INV-6a (Seed supremacy):**
- `--seed` is the master seed. When explicitly provided, it controls all randomness: simulation RNG, routing RNG, and workload generation RNG. No configuration file may silently override it.

## Decision 1: CLI `--seed` Overrides YAML `seed:` When Explicitly Passed

**Problem:** Two independent seed paths exist with no interaction. The CLI seed controls simulation RNG; the YAML seed controls workload RNG. Users who pass `--seed` expect it to control everything.

**Decision:** When `cmd.Flags().Changed("seed")` is true and `--workload-spec` is in use, set `spec.Seed = seed` before calling `GenerateRequests`. When `--seed` is not explicitly passed, the YAML seed is used as-is.

**Rationale:**
- Follows the existing precedent: `--horizon` and `--num-requests` already use `Changed()` to override YAML defaults
- Satisfies R18 (CLI flag precedence): the CLI flag is the user's explicit intent
- Preserves backward compatibility: `./blis run --workload-spec w.yaml` without `--seed` produces the same output as before
- Minimal code change: 3 lines in `cmd/root.go`, no library API change
- Preserves the "shareable workload" use case: the YAML seed remains the default for users who don't specify `--seed`

**Alternatives considered:**

| Alternative | Why rejected |
|-------------|-------------|
| A. Document only | Doesn't fix the surprise. Users will continue hitting the trap. ED-4 is a workaround, not a solution. |
| B. Derived seed: `hash(cli_seed, yaml_seed)` | Breaks backward compatibility (existing YAML+seed combinations produce different output). More complex to reason about. Users can't predict the effective seed. "What seed produced this output?" has no simple answer. |
| C. Always use CLI seed, ignore YAML seed | Breaks the "shareable workload" use case. Users who distribute YAML files expect `seed: 42` to produce the same workload everywhere. Also changes default behavior (breaking). |
| D. Add `--workload-seed` flag | Over-engineering for the current need. Can be added later if a user requests explicit workload-seed control independent of `--seed`. |

## Decision 2: Log the Effective Seed

**Problem:** After the override, the user should know which seed controlled workload generation.

**Decision:** Log an info message when the CLI seed overrides the YAML seed:
`"CLI --seed %d overrides workload-spec seed %d"`. Also log the effective seed when no override occurs: `"Using workload-spec seed %d (CLI --seed not specified)"`.

**Rationale:**
- Transparency: the user always knows which seed was used
- Debuggability: when results differ between runs, the log reveals why
- Low cost: two logrus.Infof calls, no behavioral change

## Decision 3: Update ED-4 After Implementation

**Problem:** ED-4 in experiment standards documents the workaround for the current behavior. After the fix, the workaround becomes unnecessary and the standard should reflect the new behavior.

**Decision:** After implementation, update ED-4 to document the new behavior: `--seed` overrides YAML seed when explicitly passed. Remove the "generate per-seed YAML copies" recommendation. Keep the note about YAML seed being used when `--seed` is not specified.

## Extension Points

- **Future `--workload-seed` flag:** If users need independent control of simulation and workload seeds, add a dedicated flag. The current design doesn't preclude this — `Changed("workload-seed")` would take precedence over `Changed("seed")` which takes precedence over YAML seed.
- **Seed logging in JSON output:** The effective seed could be included in the JSON results output for full experiment reproducibility audit trails.

## Validation Strategy

**Correctness (INV-6, INV-6a):**
1. Run with `--workload-spec w.yaml --seed 42` twice → identical output (determinism)
2. Run with `--workload-spec w.yaml --seed 42` vs `--seed 123` → different workloads (seed supremacy)
3. Run with `--workload-spec w.yaml` (no `--seed`) twice → identical output (YAML seed default preserved)
4. Run with `--workload-spec w.yaml` (no `--seed`) → same output as before this change (backward compatibility)

**Fidelity:** N/A — this is a correctness fix, not a modeling change.

## DES Checklist

| Question | Answer |
|----------|--------|
| What analysis questions does this design help answer? | "Does varying `--seed` produce statistically independent runs?" — yes, after this fix |
| What is modeled, simplified, and deliberately omitted? | No modeling change. The RNG architecture is unchanged. |
| What events are introduced or modified? | None |
| How do new events interact with existing tie-breaking rules? | N/A |
| What new state is introduced? Who owns it? | None. The spec seed is mutated in-place before passing to `GenerateRequests`. |
| What new metrics are derived? | None |
| How will correctness be verified? | INV-6 determinism tests + INV-6a seed supremacy test |
| How will fidelity be validated? | N/A |
| Does this introduce new randomness? | No. Existing RNG paths are reused. The change is which seed value enters the workload RNG path. |
| What is the simplest version that answers the same questions? | This is already the simplest: 3 lines of code + logging + documentation |
