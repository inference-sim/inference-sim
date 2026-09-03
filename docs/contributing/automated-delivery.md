# Automated Delivery (L1)

One comment delivers a sub-issue. Agents implement, verify, correct, and re-verify until the PR is either ready for a human to merge or stopped with a reason. **The loop never merges** — it labels, and a human merges.

This is the automated counterpart to [PR Workflow](pr-workflow.md), which remains the manual path and the source of the rules the implement phase follows.

## The command

Comment on the sub-issue you want delivered:

```
/approve-issue-for-pr-delivery
```

**No issue number.** The target is the issue you commented on. An argument would be redundant — you are already on the issue — and a hazard: commenting on #100 with `#200` would deliver something other than what you are reading. A `#N` that *agrees* with the current issue is tolerated, since the earlier documented form used one; a disagreeing one is refused with an explanation rather than silently ignored.

Restricted to repository collaborators, same as `/archon-pr-review` and `@claude`.

## What can be delivered

**Any single deliverable issue.** It does not have to be a sub-issue of a planned feature:

| | Works | Notes |
|---|---|---|
| A **sub-issue** of an archon-planned feature | yes | the surface, contracts, allow list, target branch and `archon-plan:` are all honoured |
| A **standalone** issue — bug, enhancement, hardening | yes | no plan, no declared surface, targets the default branch. Small issues here legitimately skip the RFC and plan; the issue's own acceptance criteria are the contract |
| A **tracking issue** | **refused** | see below |

**Archon is optional, not required.** With a plan there is a deterministic number that must not move the wrong way (`PLAN_GATE=pass`); with no plan at all the signal is `absent`, which delivers exactly as `pass` does, and the gate is CI plus the review verdict. The honest cost of running plan-less is that the exit condition becomes entirely judgement — no worse than what a human reviewer works from, but proportionally less mechanically checkable.

**A tracking issue is refused, by design.** It is an umbrella over several holes, so delivering it would mean one PR attempting the whole feature — the thing one-hole-per-PR exists to avoid — and it would spend an agent run producing something unreviewable. Two signals detect it, because this repository uses both conventions: the issue has linked native GitHub sub-issues, or its title begins with `Tracking` / `Epic`. Either one refuses with a comment pointing you at a specific sub-issue. If an issue really is a single deliverable unit, rename it and unlink its children.

Then leave. Every phase posts a comment, so the whole delivery history is readable on the PR page without opening a single Actions log.

## What happens

```
/approve-issue-for-pr-delivery
  │
  ├─ Deliver — Implement    branch deliver/issue-N, tests, PR, self-review
  │      ↓
  ├─ Deliver — Verify       build/test/lint → archon review → methodology review
  │      │                  ready-for-merge → STOP
  │      │                  needs-human    → STOP
  │      ↓ correct
  ├─ Deliver — Correct      fixes the named findings, pushes
  │      ↓
  └─ back to Verify              (at most 3 correction rounds)
```

Phases chain with `workflow_dispatch`, passing the PR and sub-issue numbers as inputs. **No PAT and no GitHub App are needed** — `workflow_dispatch` and `repository_dispatch` are the two events that always create workflow runs even when triggered with `GITHUB_TOKEN`.

`workflow_run` is deliberately *not* used, for two documented reasons:

- **It caps at three levels.** "You can't use `workflow_run` to chain together more than three levels of workflows." implement → verify → correct → verify exhausts the budget, so the fourth link never fires and the loop dies after a single correction round — silently, since nothing runs to report it.
- **It cannot get a CI verdict anyway.** When a workflow using `GITHUB_TOKEN` opens or updates a PR, "the resulting `pull_request` event creates workflow runs in an **approval-required** state". The delivery's own CI runs would sit waiting for a human to click *Approve and run*, so polling for a conclusion would time out every round.

## How it ends

| Outcome | Meaning | What you do |
|---|---|---|
| `ready-for-merge` | CI green, archon plan not regressed, review returned GREEN | Read the history, merge |
| `needs-human` | Signals disagreed, evidence was missing, or 3 rounds did not close the findings | Read the last comment — it names the phase and the reason |

There is no third outcome. Every unrecognised or contradictory signal resolves to `needs-human`; the gate is closed by default.

## The gate

The decision is not the reviewing agent's to make. `deliver-verify.yml` collects three machine-readable signals and hands them to `scripts/deliver-gate.sh`, which is unit-tested (`scripts/deliver_gate_test.go`):

| Signal | Source |
|---|---|
| `CI_STATUS` | verify dispatches the repository's own `ci.yml` against the delivery branch and waits for it. Anything other than a `success` conclusion is `failure` |
| `PLAN_GATE` | `.archon/review.json` — `planRatchet.ok` and `planClassify.verdict`. `absent` (the PR never claimed a plan) delivers exactly as a satisfied plan does; `unverified` (the PR declares an `archon-plan:` but the check did not run) **blocks** |
| `AGENT_VERDICT` | the `DELIVER-VERDICT: GREEN` / `NOT-GREEN` marker, required to be the last line of a comment posted by the automation itself |

**Why verify dispatches `ci.yml` rather than running the checks itself.** `main` requires seven status contexts (`build`, `lint`, and five `test (...)` groups) before a PR can merge, and those must be present **on the PR's head commit**. Two things make that awkward, and an earlier version of this feature got both wrong by running the commands inline:

- a bot-opened PR has its `pull_request` runs held in an **approval-required** state, so the loop cannot simply wait for the runs GitHub would normally create;
- a `workflow_dispatch` run's check runs attach to the **dispatch ref, not the PR head** — so checks executed inside the verify job satisfy *none* of the required contexts. The loop would label a PR `ready-for-merge` while GitHub still showed zero checks and refused to merge it.

Dispatching `ci.yml` on the delivery branch solves both: `workflow_dispatch` always starts even under `GITHUB_TOKEN`, and the run produces `ci.yml`'s own job names against the branch head — exactly what the ruleset matches. It also removes any obligation to keep package groups, per-group timeouts or the Go version in step with `ci.yml`, because `ci.yml` is what runs.

The delivery still verifies the commit it pinned: the dispatched run's `head_sha` is compared to it, and a mismatch stops the delivery rather than letting CI vouch for different code.

**`unverified` is the subtle one.** `archon-review.sh` exits 0 and falls back to a plan-less delta review when plan resolution fails, so an absent `planRatchet` does *not* by itself mean "no plan" — it can equally mean "a plan was declared and never checked". Treating those alike would let a PR whose dist ratchet silently did not run reach `ready-for-merge` on a GREEN review.

Two properties hold structurally rather than by prompt adherence:

- **A GREEN review cannot override red CI or a plan regression.** That combination returns `needs-human` with the disagreement named — the loop does not get to resolve a contradiction between a judgment and an objective signal.
- **`ready-for-merge` is never applied to unverified code.** The PR tree is checked out at a pinned SHA, and that SHA is re-confirmed as the branch head before the label goes on. A push landing during the checks or the review downgrades the outcome to `needs-human`, because what passed is no longer what a human would merge.
- **A phase that fails or times out still reports.** Every phase has a reporter guarded on `failure() || cancelled()` — `cancelled()` because a job timeout cancels rather than fails, which would otherwise skip the comment entirely and leave the PR silent.

Archon is optional throughout: with a plan there is a deterministic number that must not move the wrong way, without one the gate is CI plus the review verdict.

## Controlling a delivery in flight

**Pause.** Add the `deliver:paused` label to the PR (or, before a PR exists, to the sub-issue). Every phase checks it as its first step and exits without invoking an agent or moving a label. Remove the label and re-issue the command to resume. You are never racing the loop.

**Round count.** The `deliver:round-N` label is the only record of how many corrections have been spent, and — since `workflow_dispatch` has no chain-depth cap — the only thing bounding the loop. It is advanced *before* the correction agent runs, so a crashed or timed-out round still consumes its budget rather than being retried forever. A missing round label is therefore fatal to the correct phase, unlike other label failures, which only warn.

## Configuration

Repository variables, all optional:

| Variable | Default | Effect |
|---|---|---|
| `DELIVER_IMPLEMENT_MODEL` | `claude-opus-4-8` | model for the implement phase |
| `DELIVER_CORRECT_MODEL` | `claude-opus-4-8` | model for the correct phase |
| `DELIVER_VERIFY_MODEL` | `claude-sonnet-4-6` | model for the review phase |
| `DELIVER_MAX_ROUNDS` | `3` | correction rounds before `needs-human`, per PR |

The verify model is deliberately *not* the implement model. Two instances of one model reviewing each other's work is closer to an agent grading its own homework; different models give real separation.

## Setup

**The labels must exist before the workflows are used.** A workflow applying a label that does not exist fails at the API call, which strands a delivery mid-loop:

```bash
gh label create 'deliver:round-1' --color ededed --description 'L1 delivery: correction round 1'
gh label create 'deliver:round-2' --color ededed --description 'L1 delivery: correction round 2'
gh label create 'deliver:round-3' --color ededed --description 'L1 delivery: correction round 3'
gh label create 'deliver:paused'  --color b60205 --description 'L1 delivery: halted; every phase exits on this'
gh label create 'ready-for-merge' --color 0e8a16 --description 'L1 delivery: verified green, awaiting human merge'
gh label create 'needs-human'     --color d93f0b --description 'L1 delivery: stopped, a human must look'
```

Raising `DELIVER_MAX_ROUNDS` above 3 needs matching `deliver:round-N` labels.

## Scope

L1 delivers **one sub-issue at a time**, and sequencing is yours: approve one, merge it, approve the next. The implement phase reads a `Depends on: #M` line and refuses when #M's PR is unmerged, so a mis-ordered approval costs a comment rather than a failed hour-long run.

Not yet automated, each its own follow-up: sequencing sub-issues `0..N` and opening the final PR (L2), RFC-to-merge (L3), flaky-test re-runs, stall detection, and the no-progress detector.

**Known limitation.** The correct phase may *dismiss* a finding rather than fix it, and the review phase is instructed to explicitly accept or re-raise each dismissal — but that is prompt adherence, not enforced state. A dismissal silently treated as resolved would not be caught by the gate. The round cap bounds it, and both the dismissal and its acceptance are comments a human reads before merging.

## Security

**The workflow's own steps run trusted code.** The repository root checkout is the default branch, so `scripts/` and `.archon-version` always come from `main` — matching `archon.yml`. The archon review and `deliver-gate.sh` are executed from that trusted tree, never from the PR.

**The PR's code never runs on the self-hosted runner.** The verify phase does not check the PR out at all — it dispatches `ci.yml`, which runs the PR's build and tests on ephemeral `ubuntu-latest` runners, exactly as it does for any other PR. An earlier version did check the PR out and run its test suite on the self-hosted runner, which meant executing PR code (including `deliver_gate_test.go`, which shells out to the PR's own `.sh` files) on persistent infrastructure. Dispatching removes that exposure rather than guarding it.

The correct phase does still check out and push to the delivery branch, so it requires the target PR to be same-repository, open, and on `deliver/issue-<N>` — the branch this loop owns.

`workflow_dispatch` is itself restricted to users with write access, and a refused dispatch stops before any checkout.

**The delivery target comes from the event, not from comment text.** The issue delivered is `github.event.issue.number` — where the command was typed. An optional `#N` is accepted only when it agrees with that issue and refused when it disagrees, so no untrusted string ever selects the target.

**The verdict marker is read only from bot-authored comments, and only as a comment's last line** — otherwise any human could set a delivery's verdict by quoting it.
