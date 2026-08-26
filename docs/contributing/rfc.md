# RFC Template for Large Features

**Status:** Active (v1.0)

Use this template when filing a tracking issue for a feature that introduces new package boundaries or modifies the architecture. Bug fixes and small features (new policy behind existing interface) do not need an RFC.

---

## When an RFC is Needed

- New subsystem modules (new interface + integration)
- Backend swaps (alternative implementations requiring interface extraction)
- Architecture changes affecting module boundaries
- Multi-PR features requiring decomposition

**Not needed for:** Bug fixes, new policy templates behind existing interfaces, documentation changes, single-file refactors.

---

## Tracking Issue Structure

The tracking issue serves as both the feature proposal AND the design document. It has four sections:

### Section 1: Motivation & Scope

- **What:** one paragraph describing the capability being added
- **Why:** what problem does this solve? what can't users do today?
- **How it behaves:** expected user-visible behavior (CLI flags, output changes, config surface)
- **Scope in:** what this RFC covers
- **Scope out:** what is explicitly deferred or excluded
- **Modeling decisions** (if applicable): what is modeled / simplified / omitted, with justification for each simplification

### Section 2: Holes (architectural intent)

For each new package or component, provide the **module contract**:

| Field | Description |
|-------|-------------|
| **Name** | Package path (e.g., `sim/kv/tierchain`) |
| **Responsibility** | One sentence — what this package does |
| **Surface** | What it exports (function signatures in plain English) |
| **Allowed imports** | What it may depend on (whitelist — anything else is denied) |
| **Contracts** | Behavioral guarantees (GIVEN/WHEN/THEN or plain statements) |
| **Evidence type** | How each contract is verified (property_test, differential_test, metamorphic_test) |
| **Invariants** | Which existing invariants (INV-N) this hole must preserve or extend |
| **Extension type** | policy template / subsystem module / backend swap / tier composition |
| **No-op default** | Behavior when feature is not configured (must be byte-identical to before) |

### Section 3: Trade-offs & Decisions

For every non-obvious architectural decision:
- What alternatives were considered?
- Why was this approach chosen?
- What breaks if this decision is wrong?

### Section 4: Delivery Order

List which holes depend on which — this determines PR ordering:

```
H1 (tierchain) — no dependencies, can start first
H2 (transfer) — no dependencies, can parallel with H1
H3 (deferral) — depends on H1
H4 (blockkey) — no dependencies
H5 (config) — no dependencies, but should land early
```

---

## Design Review (before agreement)

The team reviews the RFC from these 8 perspectives before approving:

1. **Motivation & Scoping** — Are goals clear? Is the modeling decisions table complete? Does every simplification state what real-system behavior is lost?
2. **Module Contract Completeness** — Does every hole have all contract fields? Are invariants named and cross-referenced?
3. **Extension Framework Fit** — Is the extension type correct? Is the no-op default specified? Is parallel development possible?
4. **Trade-off Quality** — Does every non-obvious decision have alternatives with rationale? What breaks if it's wrong?
5. **Validation Strategy** — How will correctness be verified (which invariants)? How will fidelity be validated (against what)?
6. **Staleness Resistance** — Is content described behaviorally (what crosses a boundary and why) not structurally (how)?
7. **Domain Expertise** — DES: are new events classified (exogenous/endogenous)? vLLM: does this match real serving behavior? Platform: scaling assumptions valid?
8. **Prohibited Content** — No Go struct definitions, no method implementations, no file:line references. Describe behavior, not implementation.

---

## Quality Gates (before agreement)

- [ ] Every hole has a complete module contract (all fields filled)
- [ ] Every non-obvious decision has alternatives + rationale
- [ ] No-op default specified (existing behavior unchanged when feature absent)
- [ ] Validation strategy specified (which invariants? what evidence?)
- [ ] No implementation details (Go code, struct definitions, file paths)
- [ ] Invariants cross-referenced (INV-N)
- [ ] Extension type identified per hole

---

## Example

See [inference-sim#1585](https://github.com/inference-sim/inference-sim/issues/1585) for a real example of this pattern applied to the multi-tier KV-offload feature (5 holes, 8 arrows, delivered across 6 PRs).

---

## After Agreement

The next step (encoding into `.archon` plan + creating sub-issues) is driven by the user — see `docs/templates/rfc-to-plan.md` when ready.
