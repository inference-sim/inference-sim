# Archon Plan Encoding

After the team agrees on an RFC, encode the design into a machine-checkable `.archon` plan.

**Requires:** archon v0.2.0+. See [archon README](https://github.com/AI-native-Systems-Research/archon) for plan syntax, compile/dist/slice commands, and full examples.

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
