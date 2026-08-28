# Archon Plan Encoding

After the team agrees on an RFC, encode the design into a machine-checkable `.archon` plan.

**Getting archon:** Run `scripts/archon-build.sh` from the repo root. It builds the version pinned in `.archon-version` and prints the binary path. Use that binary for the commands below. See [archon README](https://github.com/AI-native-Systems-Research/archon) for plan syntax and full examples.

## What to run

```bash
archon-go plan compile --stats feature.archon > feature.plan.json   # encode + validate
archon-go plan dist feature.plan.json $REPO                         # baseline distance
archon-go plan slice feature.plan.json <hole-path>                  # work order per hole
```

## Sub-issue structure

- **Sub-issue 0 (PR0):** Create feature branch (`feature/<name>`) off main. Persist `.archon` + `.plan.json` to `specs/NNN-feature/`. Push. No PR to main — this is the feature branch's first commit.
- **Sub-issues 1–N (PR1–PRN):** One per hole, ordered by dependency arrows. PRs target the feature branch. Self-reviewed by the developer (`@claude /blis-pr-review` + `/archon-pr-review`). Each states: which hole, expected dist reduction, surface, contracts, allowed imports, dependencies, target branch.
- **Final PR (PRN+1):** Feature branch → main. Reviewed by maintainer. Description must include: what tracking issue it closes, plain-English summary of the feature, and the archon design (holes filled, dist=0 confirmed). Merges when dist=0 + all tests pass + maintainer approves.

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
