# Archon PR Review Context

How to incorporate archon's PR review output during code review.

**Important:** `blis-pr-review` does NOT run archon. It just reads archon output from a previous PR comment and incorporates the verdict.

**Getting archon (for manual runs):** Run `scripts/archon-build.sh` from the repo root. It builds the version pinned in `.archon-version` and prints the binary path. See [archon README](https://github.com/AI-native-Systems-Research/archon) for verdict definitions and full examples.

## What the comment contains

One comment per `/archon-pr-review`, in one of two shapes:

- **three views** — component, witness delta, interface-contract
- **three views plus plan checks** — the same, with `### G5 — Plan distance ratchet` and a plan verdict appended

The second shape appears when the PR belongs to a planned multi-PR feature. Nothing here needs to know how CI decides that; if a section is present, read it.

Two lines may precede the review:

- a note naming the plan the checks ran against, and the commit it came from. When that commit is the PR's own head it says the plan is not independently verified — expected on the final feature-branch-to-`main` PR, since `main` carries no plan.
- a `> [!WARNING]` saying the plan check was skipped. The PR asked to be checked against a plan and was not, so there is no ratchet or verdict in that comment — treat it as missing evidence, not as a passing gate.

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
5. **Plan check skipped** (the warning above) → the dist gate did not run. Say so; do not read the comment as evidence the plan was satisfied.
