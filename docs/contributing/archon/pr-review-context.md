# Archon PR Review Context

How to incorporate archon's PR review output during code review.

**Important:** `blis-pr-review` does NOT run archon. CI runs archon separately and posts the output as a PR comment. `blis-pr-review` just reads that comment and incorporates the verdict.

For archon CLI details: see [archon README](https://github.com/AI-native-Systems-Research/archon). Use latest stable release — check [releases](https://github.com/AI-native-Systems-Research/archon/releases).

## When it runs

`/archon-pr-review` runs automatically in CI on every PR. It always runs the standard delta review (boundary moves, surface changes, edge deltas). Additionally, if the PR body (or its closing issue body) contains an `archon-plan:` line with a path to a `.plan.json`, CI also runs `--plan` for dist tracking.

```
# CI logic (pseudocode):
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
