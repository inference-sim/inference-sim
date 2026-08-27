# Archon Plan Encoding

After the team agrees on an RFC, encode the design into a machine-checkable `.archon` plan.

**Requires:** latest stable archon release — check [releases](https://github.com/AI-native-Systems-Research/archon/releases). See [archon README](https://github.com/AI-native-Systems-Research/archon) for plan syntax, compile/dist/slice commands, and full examples.

## What to run

```bash
archon-go plan compile --stats feature.archon > feature.plan.json   # encode + validate
archon-go plan dist feature.plan.json $REPO                         # baseline distance
archon-go plan slice feature.plan.json <hole-path>                  # work order per hole
```

## Sub-issue structure

- **Sub-issue 0:** Create feature branch (`feature/<name>`) off main. Persist `.archon` + `.plan.json` to `specs/NNN-feature/`. Push. No PR to main — this is the feature branch's first commit.
- **Sub-issues 1–N:** One per hole, ordered by dependency arrows. PRs target the feature branch. Each states: which hole, expected dist reduction, surface, contracts, allowed imports, dependencies.
- **Final PR:** Feature branch → main. Merges when dist=0 + all tests pass.

Every sub-issue body must include this line so `/archon-pr-review` can find the plan:
```
archon-plan: specs/NNN-feature/feature.plan.json
```

## Where to persist

```
specs/NNN-feature-name/
  design.md            ← RFC prose (from tracking issue)
  feature.archon       ← machine-checkable plan
  feature.plan.json    ← compiled graph
```

These live on the feature branch from PR0 onward. They merge to main with the final PR.
