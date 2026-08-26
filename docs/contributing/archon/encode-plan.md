# Archon Plan Encoding

After the team agrees on an RFC, encode the design into a machine-checkable `.archon` plan. This is done by the proposer (or Claude on their behalf) — not by hand.

## Steps

1. **Encode** — convert agreed holes/surfaces/contracts into `.archon` syntax:

```
invariant <name> {
  statement: <what must always hold>
  evidence: <test type>
}

box <existing-package-path>

hole <new-package-path> {
  surface:
    FuncName(args) ReturnType
  allow:
    import <package-path>
  contract:
    BC-X1 <statement>  [evidenced: <test_type>]
  cites:
    invariant <name>
}

arrow <from> -> <to> : import
```

2. **Compile** — validate syntax and count clauses:
```bash
archon-go plan compile --stats feature.archon > feature.plan.json
```

3. **Measure baseline distance** — how far is current code from the plan:
```bash
archon-go plan dist feature.plan.json $REPO
```
Output: `dist(P,G) = N` with breakdown (unfilled holes, absent boxes, absent arrows, disallowed arrows).

4. **Slice** — extract each hole as a work order:
```bash
archon-go plan slice feature.plan.json <hole-path>
```
Output: surface, allowed imports, contracts for that one hole.

## Sub-issue structure

- **Sub-issue 0:** Encode `.archon` + compile + persist to `specs/NNN-feature/` on feature branch. Expected dist: baseline (no reduction yet — this just commits the plan).
- **Sub-issues 1–N:** One per hole, ordered by dependency arrows. Each sub-issue states:
  - Which hole it fills
  - Expected dist reduction (e.g., "dist 13 → 10, fills H2 + 3 arrows")
  - Surface to implement
  - Contracts to satisfy
  - Allowed imports (whitelist)

## Where to persist

```
specs/NNN-feature-name/
  design.md            ← RFC prose (from tracking issue)
  feature.archon       ← machine-checkable plan
  feature.plan.json    ← compiled graph
```

After PR0 merges to the feature branch, CI has the plan to track against.

## Reference

For full plan syntax: see [archon plan-syntax.md](https://github.com/AI-native-Systems-Research/archon/blob/main/docs/plan-syntax.md).
For a real end-to-end example: see [archon demo](https://github.com/AI-native-Systems-Research/archon/blob/main/README.md#flow-2-design-phase--pr-review-declare-intent-before-coding).
