# Archon PR Review Context

How to incorporate archon's PR review output during code review.

**Important:** `blis-pr-review` does NOT run archon. It just reads archon output from a previous PR comment and incorporates the verdict.

**Getting archon (for manual runs):** Run `scripts/archon-build.sh` from the repo root. It builds the version pinned in `.archon-version` and prints the binary path. See [archon README](https://github.com/AI-native-Systems-Research/archon) for verdict definitions and full examples.

## When it runs

`/archon-pr-review` runs when triggered via a comment on the PR (manually by a contributor or maintainer). It is NOT automatic — someone must type `/archon-pr-review` on the PR. It runs the standard delta review (boundary moves, surface changes, edge deltas). Additionally, if the PR body (or its closing issue body) contains an `archon-plan:` line with a path to a `.plan.json`, it also runs `--plan` for dist tracking.

```
# When triggered (pseudocode):
archon-go pr-review . $BASE $HEAD --out .archon           # always: delta view
if archon-plan path found AND file exists on base branch:
  archon-go pr-review . $BASE $HEAD --plan <path> --out .archon  # additionally: dist tracking
```

Both outputs are posted. No failure if the plan path is missing or wrong — falls back to delta-only.

## Convention: `archon-plan:` in sub-issues

When sub-issues are created via `rfc-to-plan.md`, each includes a standard line:
```
archon-plan: specs/NNN-feature/feature.plan.json
```
CI greps for this. If absent, dist tracking is skipped (safe default).

## Verdicts

| Verdict | Meaning | Reviewer action |
|---------|---------|-----------------|
| **FAST-TRACK** | No boundary moved | Skip architecture concerns — focus on correctness |
| **REALIZES** | Hole filled, dist decreased | Confirm it matches the sub-issue's expected outcome |
| **EXCEEDS** | Adds structure the plan didn't mention | Ask: intentional? Plan update needed? |
| **CONFLICTS** | Dist went up or disallowed dependency | Block. Code is wrong or plan needs update. |

## How to incorporate during review

1. Check if archon output exists in a previous PR comment (CI or manual)
2. **FAST-TRACK** → no architectural concern, proceed with normal review
3. **REALIZES** → verify dist reduction matches sub-issue expectation
4. **EXCEEDS/CONFLICTS** → flag to maintainer, may need plan-update PR
