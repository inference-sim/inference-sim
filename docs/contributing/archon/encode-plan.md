# Archon Plan Encoding

After the team agrees on an RFC, encode the design into a machine-checkable `.archon` plan.

**Requires:** archon v0.2.0+. Build and usage: see [archon README](https://github.com/AI-native-Systems-Research/archon#quick-start).

## What to run

```bash
archon-go plan compile --stats feature.archon > feature.plan.json   # encode + validate
archon-go plan dist feature.plan.json $REPO                         # baseline distance
archon-go plan slice feature.plan.json <hole-path>                  # work order per hole
```

## Sub-issue structure

- **Sub-issue 0:** Persist `.archon` + `.plan.json` to `specs/NNN-feature/` on feature branch. Expected dist: baseline (no reduction — just commits the plan).
- **Sub-issues 1–N:** One per hole, ordered by dependency arrows. Each states: which hole, expected dist reduction, surface, contracts, allowed imports, dependencies.

## Where to persist

```
specs/NNN-feature-name/
  design.md            ← RFC prose (from tracking issue)
  feature.archon       ← machine-checkable plan
  feature.plan.json    ← compiled graph
```

## Full reference

- Plan syntax (holes, boxes, arrows, invariants): [archon README — Flow 2](https://github.com/AI-native-Systems-Research/archon#flow-2-design-phase--pr-review-declare-intent-before-coding)
- Plan syntax detailed reference: [docs/plan-syntax.md](https://github.com/AI-native-Systems-Research/archon/blob/main/docs/plan-syntax.md)
- Real end-to-end tracking across PRs: [demo/flow3-blis-design](https://github.com/AI-native-Systems-Research/archon/blob/main/demo/flow3-blis-design/README.md)
