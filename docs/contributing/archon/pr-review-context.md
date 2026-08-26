# Archon PR Review Context

How to incorporate archon's PR review output during code review.

**Requires:** archon v0.2.0+. See [archon README](https://github.com/AI-native-Systems-Research/archon) for verdict definitions, dist tracking, and full examples.

## When it runs

`/archon-pr-review` runs automatically in CI on every PR. It may also be triggered manually. It produces a structural architecture review — what boundaries moved, what edges changed.

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
