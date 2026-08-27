# Archon Issue Examples (for Claude)

Examples of how tracking issues and sub-issues should be structured for large features using the archon-based planning flow. Claude follows these when creating issues via `rfc.md` and `rfc-to-plan.md`.

This is NOT a GitHub issue template — it's a reference for Claude to produce consistent, well-structured issues that `blis-pr-review` can later verify against.

---

## Tracking Issue (RFC)

```markdown
# [Feature name]

## Feature Description

**What:** One paragraph — the capability being added.
**Why:** What problem this solves. What users can't do today.
**How it behaves:** CLI flags, output changes, config surface.
**Scope in:** [bullet list]
**Scope out:** [bullet list]
**Modeling decisions:** What is modeled / simplified / omitted, with justification.

## Holes

### H1: [package path]
- **Responsibility:** [one sentence]
- **Surface:** [exported functions in plain English]
- **Allowed imports:** [whitelist]
- **Contracts:** (naming: `BC-<hole-id>-<N>`)
  - BC-H1-1: [statement] [evidenced: property_test]
  - BC-H1-2: [statement] [evidenced: differential_test]
- **Invariants:** [which INV-N this preserves]
- **Extension type:** policy template / subsystem module / backend swap / tier composition
- **No-op default:** [behavior when feature is absent — must be byte-identical]

### H2: [package path]
[same structure]

## Trade-offs

| Decision | Alternatives | Why this approach | What breaks if wrong |
|----------|-------------|-------------------|---------------------|
| [decision] | [alt 1, alt 2] | [rationale] | [cost] |

## Delivery Order

H1 — no dependencies, can start first
H2 — no dependencies, parallel with H1
H3 — depends on H1
```

---

## Sub-issue 0 (create feature branch + persist plan)

```markdown
# Encode .archon plan + persist

Parent: #[tracking issue number]

## What to do

1. Create feature branch: `feature/<name>` off main
2. Commit plan files to `specs/NNN-feature/`:
   - design.md (RFC prose from tracking issue)
   - feature.archon (machine-checkable plan)
   - feature.plan.json (compiled graph)
3. Push the feature branch. No PR to main.

All subsequent PRs (sub-issues 1–N) target this feature branch.

archon-plan: specs/NNN-feature/feature.plan.json
```

---

## Sub-issue 1–N (per hole)

Each sub-issue is a self-contained work order. PRs target the feature branch. Developer self-reviews (`@claude /blis-pr-review` + `/archon-pr-review`). `blis-pr-review` reads this body to verify contracts are delivered.

```markdown
# [Hole name]: [short description]

Parent: #[tracking issue number]

## What to build

[1-2 sentences: what this hole does, why it matters]

## Surface

- FuncName(args) ReturnType
- FuncName2(args) ReturnType

## Allowed imports

- [package path 1]
- [package path 2]

## Contracts

Naming convention: `BC-<hole-id>-<N>` (e.g., hole H1's contracts are BC-H1-1, BC-H1-2; hole H2's are BC-H2-1, BC-H2-2).

- BC-H1-1: [statement] [evidenced: property_test]
- BC-H1-2: [statement] [evidenced: differential_test]
- BC-H1-3: [statement] [evidenced: metamorphic_test]

## Invariants

- INV-N: [which invariants this hole must preserve]

## No-op default

[Behavior when feature is not configured — must be byte-identical to before]

## Expected dist reduction

dist [before] → [after] (fills [hole] + [N] arrows)

## Dependencies

[Which sub-issues must land first, or "None — can start immediately"]

## Target branch

`feature/<name>` (PR against the feature branch, NOT main)

## Review

Self-reviewed by developer: `@claude /blis-pr-review` + `/archon-pr-review`

---

archon-plan: specs/[NNN-feature]/[feature].plan.json
```

---

## Final PR (feature branch → main)

```markdown
# [Feature name]: merge to main

Closes: #[tracking issue number]

## Summary

[Plain-English: what this feature adds, why, how it behaves — for maintainers who haven't followed the sub-issues]

## What was delivered

- H1: [hole name] — [one sentence]
- H2: [hole name] — [one sentence]
- ...

## Archon verification

- dist: 0 (all holes filled, all arrows established)
- All contracts evidenced (tests pass)
- No-op default confirmed (byte-identical when feature absent)

## Target branch

`feature/<name>` → `main`

## Review

Maintainer reviews this PR (the whole feature in one diff against main).
```

---

## Why the sub-issue includes full hole content

`blis-pr-review` reads the sub-issue body during review to:
1. Check if promised contracts have matching tests
2. Verify evidence types match (property_test → a property test exists)
3. Confirm the surface was implemented as declared
4. Verify no-op default holds (byte-identical when feature absent)

This avoids blis-pr-review needing to read `.archon` files or run archon. The sub-issue IS the contract between planning and implementation.
