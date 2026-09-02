# Automated Delivery (L1)

One comment delivers a sub-issue. Agents implement, verify, correct, and re-verify until the PR is either ready for a human to merge or stopped with a reason. **The loop never merges** — it labels, and a human merges.

This is the automated counterpart to [PR Workflow](pr-workflow.md), which remains the manual path and the source of the rules the implement phase follows.

## The command

Comment on the sub-issue:

```
/approve-issue-for-pr-delivery #1234
```

Restricted to repository collaborators, same as `/archon-pr-review` and `@claude`.

Then leave. Every phase posts a comment, so the whole delivery history is readable on the PR page without opening a single Actions log.

## What happens

```
/approve-issue-for-pr-delivery #N
  │
  ├─ Deliver — Implement    branch deliver/issue-N, tests, PR, self-review
  │      ↓
  ├─ Deliver — Verify       waits for CI → archon review → methodology review
  │      │                  ready-for-merge → STOP
  │      │                  needs-human    → STOP
  │      ↓ correct
  ├─ Deliver — Correct      fixes the named findings, pushes
  │      ↓
  └─ back to Verify              (at most 3 correction rounds)
```

Phases chain with `workflow_run`, which is exempt from the rule that stops `GITHUB_TOKEN`-created events from triggering workflows — so **no PAT and no GitHub App are needed**. The PR number is carried between phases in a `deliver-context` artifact, because a `workflow_run` event chained off an `issue_comment` workflow reports the default branch's SHA rather than the PR's.

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
| `CI_STATUS` | every `pull_request`-triggered Actions run on the PR head; any conclusion other than `success` counts as failure |
| `PLAN_GATE` | `.archon/review.json` — `planRatchet.ok` and `planClassify.verdict`. Absent keys mean no plan, which delivers exactly as a satisfied plan does |
| `AGENT_VERDICT` | the `DELIVER-VERDICT: GREEN` / `NOT-GREEN` marker ending the review comment |

Two properties hold structurally rather than by prompt adherence:

- **A GREEN review cannot override red CI or a plan regression.** That combination returns `needs-human` with the disagreement named — the loop does not get to resolve a contradiction between a judgment and an objective signal.
- **Verify never reads CI before CI has run.** It polls until every PR-triggered workflow run on the head commit has completed. A timeout or an empty run list is `unknown`, which stops the delivery rather than passing it.

Archon is optional throughout: with a plan there is a deterministic number that must not move the wrong way, without one the gate is CI plus the review verdict.

## Controlling a delivery in flight

**Pause.** Add the `deliver:paused` label to the PR (or, before a PR exists, to the sub-issue). Every phase checks it as its first step and exits without invoking an agent or moving a label. Remove the label and re-issue the command to resume. You are never racing the loop.

**Round count.** The `deliver:round-N` label is the only record of how many corrections have been spent — nothing else survives between `workflow_run` invocations.

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

The verify phase checks out the **default branch** and only fetches the PR head into the object store, matching `archon.yml`. A pull request therefore never gets its own copy of `scripts/` executed on the self-hosted runner; the review agent reads the diff through `gh`, which needs no checkout. Adding a `ref:` to that checkout would turn any pull request into arbitrary code execution.

The sub-issue number is parsed out of an untrusted comment body, validated as digits, and only ever used as a number.
