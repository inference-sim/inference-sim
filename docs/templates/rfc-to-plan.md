# RFC to Plan: Claude Prompt Template

Use this prompt after the team agrees on an RFC (tracking issue with holes/surfaces/contracts). Give it to Claude to encode the design into a machine-checkable plan and create sub-issues.

---

## Prompt

```
Here is the agreed tracking issue: #<NUMBER> [paste link or content]

Do the following:

1. BASELINE: Run `archon-go health $REPO` and `archon-go impact $REPO <target-package>`
   to understand current architecture and blast radius of the area being modified.

2. ENCODE: Read the archon plan syntax (reference below). Encode the agreed holes,
   surfaces, contracts, and allowed imports into a `.archon` file. Use the repo's
   actual module path. For each hole:
   - Declare surface (exported functions/types)
   - Declare allow list (only permitted imports)
   - Declare contracts with evidence type: [evidenced: property_test],
     [evidenced: differential_test], [evidenced: metamorphic_test]
   - Cite invariants where applicable

3. COMPILE: Run:
   archon-go plan compile --stats feature.archon > feature.plan.json
   Report any errors. Report clause count.

4. DISTANCE: Run:
   archon-go plan dist feature.plan.json $REPO
   Report baseline distance (how far current code is from the plan).

5. SLICE: For each hole, run:
   archon-go plan slice feature.plan.json <hole-path>
   This produces a work order for that hole.

6. SUB-ISSUES: Create sub-issues under the tracking issue:
   - Sub-issue 0: "Encode .archon plan + persist to specs/" — delivers PR0
     (the plan files committed to the feature branch, no implementation)
   - Sub-issues 1–N: one per hole, ordered by dependency arrows.
     Each sub-issue body contains the sliced work order (surface, contracts,
     allowed imports) in plain English.

7. Report: summary of baseline dist, number of holes, delivery order.
```

---

## Archon Plan Syntax (quick reference)

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

For full syntax: see `archon/docs/plan-syntax.md` in the archon repository.

---

## Output

After running this prompt, you will have:
- `specs/NNN-feature/feature.archon` — machine-checkable plan
- `specs/NNN-feature/feature.plan.json` — compiled graph
- Sub-issues created under the tracking issue with delivery order
- Baseline distance reported

The developer resolving sub-issue 0 commits these files to the feature branch (PR0). From PR1 onward, CI tracks dist against the plan.
