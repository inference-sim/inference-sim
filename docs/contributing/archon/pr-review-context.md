# Archon PR Review Context

How to incorporate archon's PR review output during code review.

## When archon-pr-review runs

`/archon-pr-review` runs automatically in CI on every PR. It may also be triggered manually (`/archon-pr-review` comment on the PR). It produces a structural architecture review — what boundaries moved, what edges changed.

## Verdicts

| Verdict | Meaning | Reviewer action |
|---------|---------|-----------------|
| **FAST-TRACK** | No boundary moved. Internal-only change. | Skip architecture concerns — focus on correctness and behavior. |
| **REALIZES** | Hole filled, dist decreased. Progress toward the plan. | Confirm the filled hole matches the sub-issue's expected outcome. |
| **EXCEEDS** | Adds structure the plan didn't mention. | Ask: is this intentional? Should the plan be updated? |
| **CONFLICTS** | Dist went up or a disallowed dependency was introduced. | Block. Either the code is wrong or the plan needs a plan-update PR. |

## How to incorporate during review

1. **Check if archon output exists** — look for a previous comment from the archon CI action or a manual `/archon-pr-review` invocation.
2. **If FAST-TRACK** — no architectural concern. Proceed with normal code review.
3. **If REALIZES** — verify the sub-issue expected this hole to be filled. Check dist reduction matches expectation.
4. **If EXCEEDS** — flag to maintainer. May need a plan-update PR.
5. **If CONFLICTS** — block the PR. The implementation introduced a dependency the plan forbids.

## With --plan flag

When a `.plan.json` exists in `specs/*/`, archon runs with `--plan` and reports:
- `dist(P,G): before → after` (should decrease or stay)
- Plan verdict: REALIZES / EXCEEDS / CONFLICTS / UNRELATED
- Which specific holes were filled, which arrows appeared

Without a plan, archon just reports boundary moves (FAST-TRACK vs ARCHITECTURAL_CHANGE).

## Reference

For full archon pr-review documentation: see the [archon repository](https://github.com/AI-native-Systems-Research/archon).
