# Archon PR Review Context

How to incorporate archon's PR review output during code review.

**Important:** `blis-pr-review` does NOT run archon. It just reads archon output from a previous PR comment and incorporates the verdict.

**Getting archon (for manual runs):** Run `scripts/archon-build.sh` from the repo root. It builds the version pinned in `.archon-version` and prints the binary path. See [archon README](https://github.com/AI-native-Systems-Research/archon) for verdict definitions and full examples.

## When it runs

`/archon-pr-review` runs when triggered via a comment on the PR (manually by a contributor or maintainer). It is NOT automatic — someone must type `/archon-pr-review` on the PR. It always runs the standard delta review (boundary moves, surface changes, edge deltas). If a plan is declared and readable, `--plan` is added to that same invocation, which appends `### G5 — Plan distance ratchet` for dist tracking and a plan verdict.

```
# When triggered (pseudocode):
plan = first `archon-plan: <path>` line in the PR body, else in a closing issue body
if <path> is a committed .json file at the base branch tip, else at the PR head:
  archon-go pr-review . $BASE $HEAD --plan <extracted> --out .archon   # 3 views + dist
else:
  archon-go pr-review . $BASE $HEAD --out .archon                      # 3 views
```

**One invocation, one comment, in every case.** The `--plan` `review.md` is the delta one plus the ratchet section, so posting both would duplicate every diagram.

**Where the plan is read from.** The base branch tip is preferred over the PR head, so a hole PR cannot be graded against a plan it rewrote. The final feature-branch-to-`main` PR has the plan only on its head; that copy is used, and the comment labels it as not independently verified, since a fork PR controls its own head. Either way the comment names the plan path and the commit it came from.

**When it does not run.** A PR that declares no plan is reviewed delta-only with no note and no warning — the normal case for a bug fix. A PR that *declares* a plan CI cannot read gets a visible warning saying so, because a silent delta-only review would be indistinguishable from a PR with no plan and would quietly remove dist as the merge gate.

## Convention: `archon-plan:` in sub-issues

When sub-issues are created via `rfc-to-plan.md`, each includes a standard line:
```
archon-plan: specs/NNN-feature/feature.plan.json
```
Requirements: the path must end in `.json`, be repository-relative, and be **committed** to the branch — CI reads it out of git, not the working tree, since the workflow runs with the default branch checked out. If the line is absent, dist tracking is skipped silently (safe default). If it is present but unreadable, the review says so.

Implementation: `scripts/archon-plan-resolve.sh` (detection and extraction) and `scripts/archon-review.sh` (the review itself), both driven by tests in `scripts/`.

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
