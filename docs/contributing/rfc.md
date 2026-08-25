# RFC Template for Large Features

**Status:** Active (v1.0)

Use this template when filing a tracking issue for a feature that introduces new package boundaries or modifies the architecture. Bug fixes and small features (new policy behind existing interface) do not need an RFC.

---

## Tracking Issue Structure

The tracking issue serves as both the feature proposal AND the RFC. It has two sections:

### Section 1: Feature Description (plain English)

Write for humans who have never seen archon or the codebase internals:

- **What:** one paragraph describing the capability being added
- **Why:** what problem does this solve? what can't users do today?
- **How it behaves:** expected user-visible behavior (CLI flags, output changes, config surface)
- **Scope:** what's in / what's explicitly out

### Section 2: Holes (architectural intent)

For each new package or component:

| Field | Description |
|-------|-------------|
| **Name** | Package path (e.g., `sim/kv/tierchain`) |
| **Responsibility** | One sentence — what this package does |
| **Surface** | What it exports (function signatures in plain English) |
| **Allowed imports** | What it may depend on (whitelist) |
| **Contracts** | Behavioral guarantees (GIVEN/WHEN/THEN or plain statements) |
| **Evidence type** | How each contract is verified (property_test, differential_test, metamorphic_test) |

### Section 3: Delivery Order

List which holes depend on which — this determines PR ordering:

```
H1 (tierchain) — no dependencies, can start first
H2 (transfer) — no dependencies, can parallel with H1
H3 (deferral) — depends on H1
H4 (blockkey) — no dependencies
H5 (config) — no dependencies, but should land early
```

---

## Example

See [inference-sim#1585](https://github.com/inference-sim/inference-sim/issues/1585) for a real example of this pattern applied to the multi-tier KV-offload feature (5 holes, 8 arrows, delivered across 6 PRs).

---

## After Agreement

The next step (encoding into `.archon` plan + creating sub-issues) is driven by the user — see `docs/templates/rfc-to-plan.md` when ready.
