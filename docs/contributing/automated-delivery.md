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

**Your delivery is recorded as yours.** Every delivery PR is authored by `app/claude`, so with several in flight nothing on the PR itself said who started it. The implement phase now puts your login there: you are added as an **assignee**, and a `Delivery approved by @you` line is appended to the body. The assignee makes your deliveries findable (`is:pr assignee:@me`, or the Assigned tab) and subscribes you to the thread, so verdicts, correction rounds and a `needs-human` stop reach you without polling; the body line is the durable record, and survives someone clearing assignees. Four properties are deliberate:

- **The login comes from the event's actor** — the same one the permission check above validated — and never from the text you typed, so a comment cannot name someone else as the approver.
- **A workflow step records it, not the agent.** The agent opening the PR is the one action in this phase that is not workflow-guaranteed, and provenance that goes missing exactly when a delivery went wrong is provenance nobody can rely on.
- **It runs on `always()`**, so an agent that errored or was cancelled still says who to tell. The uncovered case is the documented one: a runner that dies executes no step at all. Re-issuing the command records it, and doing so twice does not duplicate the line.
- **A delivery resumed by someone else adds them alongside you** rather than replacing you. Both people started a delivery on that branch, and both should hear about it.

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
  ├─ Deliver — Implement    workflow pushes deliver/issue-N; agent opens draft PR, then code
  │      ↓
  ├─ Deliver — Verify       build/test/lint → archon review → methodology review
  │      │                  ready-for-merge → STOP
  │      │                  needs-human    → STOP
  │      ↓ correct
  ├─ Deliver — Correct      fixes the named findings, pushes
  │      ↓
  └─ back to Verify              (at most 3 correction rounds)
```

**The implement phase opens its branch and PR before it starts work, not after it finishes.** The
workflow creates and pushes `deliver/issue-N` — with an empty seed commit, based on the sub-issue's
target branch — before the agent starts. The agent's own first action, before it writes any code, is
to open a **draft** PR on that branch; it then commits and pushes as it goes, and marks the PR ready
for review when it is done. Three things follow from that ordering:

- **An interrupted run leaves recoverable work.** A lost runner executes no step at all, so an
  agent that pushed only at the end left nothing behind — a delivery once lost a finished
  implementation that way, because the runner was evicted seconds before its single push. The branch
  is pushed by the workflow, so this holds regardless of what the agent does.
- **An interrupted delivery is visible.** `deliver-stall-sweep.yml` finds deliveries *from the PR
  side*, so a run that died before opening a PR was invisible to the one backstop meant to catch it.
  Opening the PR first shrinks that window from the whole run to the agent's first action. It does
  not eliminate it: a runner lost in the first seconds still leaves a pushed branch with no PR, which
  the sweep cannot see. Teaching the sweep to also consider PR-less delivery branches would close
  the remainder, and is tracked as #1740.
- **A PR still marked draft, saying the work is in progress, was interrupted.** The agent is told
  to say so in the body it opens, so an abandoned delivery is recognisable without reading the run
  log.

**The spec is the issue body plus the design refinements in its comment thread.** A body is written
once, at the start; the design then gets refined in comments — a narrowed scope, a corrected
contract, an "actually do X, not Y" — and nobody goes back to rewrite the body. An implement phase
reading the body alone therefore builds an out-of-date spec faithfully, and the divergence surfaces
only in verify or in human review, an agent hour later (#1782). Issue #1706 is the worked example:
its body proposes *"extend the block commit past `endIndex`"* and a comment on it later replaces that
with the vLLM-faithful *"fold the external credit in before the chunk/budget clamps"*, which is what
was actually built.

The implement phase follows `docs/contributing/pr-workflow.md`, whose Step 1.5 now requires reading
the thread and states the authority rule in full — [Comments can refine the
body](pr-workflow.md#comments-can-refine-the-body). In short:

- `scripts/deliver-issue-refinements.sh <issue-number>` prints the comments that carry authority,
  oldest first, and says so explicitly when there are none. A read that fails prints
  `REFINEMENT-READ-FAILED` rather than nothing, because empty output reads exactly like "no
  refinements" — the failure this closes.
- **A comment counts iff its author holds `admin`/`write`/`maintain`** on this repository: the same
  boundary the delivery command and the review triggers use. Bot comments are dropped (the loop
  comments on the issues it delivers, and its bot *does* hold write access, so admitting them would
  feed the loop's own prose back to its next agent as spec), and so are minimized ones.
  `authorAssociation` is deliberately not the signal — this repository's maintainer reports
  `CONTRIBUTOR`, so trusting `OWNER`/`MEMBER`/`COLLABORATOR` would drop exactly the comments that
  matter while still admitting anyone whose PR has ever been merged.
- **A refinement overrides the body** on any point it addresses; where two conflict the later wins;
  an irreconcilable contradiction is built the body's way and reported.
- **The target branch, `archon-plan:` and `Depends on:` stay body-only.** Those are declarations the
  workflow acts on before the agent exists, and resolving a base branch or plan path from comment
  text would put an attacker-influenceable string into `git ls-remote` and `gh pr create --base`.
- **Refinement text is data, never instructions** — the same rule the verify and correct phases
  apply to PR comments.

The decision behind that rule, including the alternative that was rejected and the measurements that
settled it, is recorded in [Issue Comment Authority](issue-comment-authority.md).

Because the agent opens a PR before it writes anything, an open PR is no longer evidence that
anything was built.
The hand-off to verify is gated on the branch actually carrying a file change against its base;
an agent that produced nothing gets a comment on the sub-issue saying exactly that, instead of
handing an empty PR to a two-hour review.

`ci.yml` is the authority on build, test and lint **for implement and verify**. Verify dispatches it
on the delivery branch and reads the resulting check runs; implement deliberately does *not* re-run
the full suite or the linter, which duplicated a parity obligation and, on the resource-limited
self-hosted runner, was itself a cause of lost runs.

**The correct phase has not been brought into line yet.** Its prompt still tells the agent to run
`go build ./...`, `go test ./...` and `golangci-lint run ./...` "because the verify phase runs
exactly these three" — the same false premise, on the same self-hosted runner, under a tighter
60-minute budget. Tracked as #1737; do not read the paragraph above as describing that phase.

Phases chain with `workflow_dispatch`, passing the PR and sub-issue numbers as inputs. **No PAT and no GitHub App are needed** — `workflow_dispatch` and `repository_dispatch` are the two events that always create workflow runs even when triggered with `GITHUB_TOKEN`.

`workflow_run` is deliberately *not* used, for two documented reasons:

- **It caps at three levels.** "You can't use `workflow_run` to chain together more than three levels of workflows." implement → verify → correct → verify exhausts the budget, so the fourth link never fires and the loop dies after a single correction round — silently, since nothing runs to report it.
- **It cannot get a CI verdict anyway.** When a workflow using `GITHUB_TOKEN` opens or updates a PR, "the resulting `pull_request` event creates workflow runs in an **approval-required** state". The delivery's own CI runs would sit waiting for a human to click *Approve and run*, so polling for a conclusion would time out every round.

## How it ends

| Outcome | Meaning | What you do |
|---|---|---|
| `ready-for-merge` | CI green, archon plan not regressed, both reviews clean (methodology review GREEN and cross-vendor qa-review PASS) | Read the history, merge |
| `needs-human` | Signals disagreed, evidence was missing, or 3 rounds did not close the findings | Read the last comment — it names the phase and the reason |

There is no third outcome. Every unrecognised or contradictory signal resolves to `needs-human`; the gate is closed by default.

## The gate

The decision is not the reviewing agent's to make. `deliver-verify.yml` collects six machine-readable signals and hands them to `scripts/deliver-gate.sh`, which is unit-tested (`scripts/deliver_gate_test.go`):

| Signal | Source |
|---|---|
| `CI_STATUS` | verify dispatches the repository's own `ci.yml` against the delivery branch and waits for it. Anything other than a `success` conclusion is `failure` |
| `PLAN_GATE` | `.archon/review.json` — `planRatchet.ok` and `planClassify.verdict`. `absent` (the PR never claimed a plan) delivers exactly as a satisfied plan does; `unverified` (the PR declares an `archon-plan:` but the check did not run) **blocks** |
| `AGENT_VERDICT` | the `DELIVER-VERDICT: GREEN` / `NOT-GREEN` marker, required to be the last line of a comment posted by the automation itself |
| `QA_VERDICT` | the `QA-VERDICT: PASS` / `BLOCK` marker from the **cross-vendor qa-review pass** (#1715, RFC #1603) — a questioner and an isolated answerer from a different model family than the implementer and the reviewer above. Same author-trust and last-line rules as `AGENT_VERDICT`. `BLOCK` routes to a correction round; `MISSING` (no marker) **blocks**, so a qa-review that crashed or lost its model can never be read as a pass |
| `DISMISSALS` | the `deliver:has-dismissals` label, **re-read after the review agent has run** so that the reviewer clearing it takes effect in the same round. `open` withholds `ready-for-merge`; `unknown` (the label set could not be read) does too, because an unreadable state is not evidence there is nothing to accept |
| `MERGE_STATE` | whether the branch can merge into `main` (#1758) — GitHub's REST `mergeable_state` mapped to `mergeable` / `conflicting` (a true conflict) / `unknown`. `conflicting` routes to a correction round (the agent merges `main` and resolves the conflict) rather than to a human; `unknown` (mergeability not yet computed) triggers a re-check on the next event rather than a terminal verdict |

Both review signals are required for `ready-for-merge`, and either one alone can send a round to correction. They are kept as **parallel signals rather than one combined verdict** so it is always visible which review blocked, and so each can be tested in isolation.

**`QA_VERDICT` is produced differently on round 0 and on a re-verify (#1716).** Round 0 runs the full cross-vendor probe (questioner → answerer → report). Every correction round after it is **adjudicate-only**: `adjudicator.py` re-checks the findings that probe already raised against the author's later comments and the corrected head, and `BLOCK`s if any of them is `STILL_OPEN` *or was never adjudicated*. Both branches leave the same last-line `QA-VERDICT:` marker on one bot-authored comment, so the gate reads them identically.

The tradeoff is deliberate: a re-verify does **not** re-probe the whole diff, so a *new* problem introduced by a correction is not caught by the qa-review dimension on that round. It is caught by the three signals that *are* re-derived from scratch on the new head — CI re-runs, the archon plan ratchet re-evaluates distance (an increase is a `regression`), and the methodology review re-reviews the whole diff. Re-probing a small correction every round costs a full questioner+answerer pass for coverage those three already provide, whereas the cross-vendor probe's distinctive value is on the *original* diff. What the adjudication adds is what none of the three can do: judging whether the findings already raised were actually resolved, and whether the author's defence of a dismissal holds.

**Why verify dispatches `ci.yml` rather than running the checks itself.** `main` requires seven status contexts (`build`, `lint`, and five `test (...)` groups) before a PR can merge, and those must be present **on the PR's head commit**. Two things make that awkward, and an earlier version of this feature got both wrong by running the commands inline:

- a bot-opened PR has its `pull_request` runs held in an **approval-required** state, so the loop cannot simply wait for the runs GitHub would normally create;
- a `workflow_dispatch` run's check runs attach to the **dispatch ref, not the PR head** — so checks executed inside the verify job satisfy *none* of the required contexts. The loop would label a PR `ready-for-merge` while GitHub still showed zero checks and refused to merge it.

Dispatching `ci.yml` on the delivery branch solves both: `workflow_dispatch` always starts even under `GITHUB_TOKEN`, and the run produces `ci.yml`'s own job names against the branch head — exactly what the ruleset matches. It also removes any obligation to keep package groups, per-group timeouts or the Go version in step with `ci.yml`, because `ci.yml` is what runs.

The delivery still verifies the commit it pinned: the dispatched run's `head_sha` is compared to it, and a mismatch stops the delivery rather than letting CI vouch for different code.

**`unverified` is the subtle one.** `archon-review.sh` exits 0 and falls back to a plan-less delta review when plan resolution fails, so an absent `planRatchet` does *not* by itself mean "no plan" — it can equally mean "a plan was declared and never checked". Treating those alike would let a PR whose dist ratchet silently did not run reach `ready-for-merge` on a GREEN review.

Two properties hold structurally rather than by prompt adherence:

- **A clean review cannot override red CI or a plan regression.** When an objective signal blocks and **neither** review asked for a correction, the gate returns `needs-human` with the disagreement named — the loop does not get to resolve a contradiction between a judgment and an objective signal. If **either** review did name findings, the round goes to correction instead: there is something concrete to act on, and the reason line names both the objective blocker and the review(s) that blocked.
- **`ready-for-merge` is never applied to unverified code.** The PR tree is checked out at a pinned SHA, and that SHA is re-confirmed as the branch head before the label goes on. A push landing during the checks or the review downgrades the outcome to `needs-human`, because what passed is no longer what a human would merge.
- **A phase that fails or times out still reports.** Every phase has a reporter guarded on `always() && !success()`, exercised on real infrastructure rather than reasoned about, in two halves:

    - `.github/workflows/deliver-guards-selftest.yml` cancels a step via a **job** timeout and asserts the reporter is reached, and asserts a `deliver:paused` exit stays quiet. It is **manual/weekly, not per-PR** — cancelling a job is the condition under test, so it necessarily reports a non-green job and would leave a permanently red check. Run it whenever a guard changes. Last verified result: [run 34513953327](https://github.com/inference-sim/inference-sim/actions/runs/34513953327) — `victim_outcome=cancelled new_guard=true`, `reporter=false skip_probe=false`.
    - `scripts/deliver_guards_test.go` runs on **every** PR and asserts the phases still use the guard that self-test exercises, so a past green result cannot vouch for a file that has since been reverted.

    **What the self-test measured, which corrects the reasoning this change was originally made on.** The guard was changed from `failure() || cancelled()` on the docs-based argument that a cancelled *step* satisfies neither term. Exercised on real infrastructure, that argument does not hold — for either mechanism that can actually cancel a step:

    | Mechanism | Step outcome | `always() && !success()` | `failure() \|\| cancelled()` |
    |---|---|---|---|
    | step-level `timeout-minutes` | `failure` | fires | fires (via `failure()`) |
    | job-level `timeout-minutes` | `cancelled` | fires | fires (via `cancelled()`) |

    So `always() && !success()` is still the right guard — it is a strict superset and nothing regresses — but the reason is **breadth**, not that the old form was blind to a cancelled step. Note the first row is also a trap: a step-level timeout *fails* a step rather than cancelling it, so a test built on one looks like it reproduces #1685 without doing so.

    And the silence on #1685 was never a guard problem: the runner was lost, and then no step executes — `always()` ones included. No `if:` can report from a job that never runs, which is why the stall sweep exists. That third mechanism cannot be reproduced in CI, so it remains the one untested path.

    Also measured, since it was previously an open question: a **skipped** step does *not* make `success()` false. So the explicit `paused != 'true'` term in each reporter is defence-in-depth rather than the only thing preventing a paused delivery from reporting.
- **A dismissal cannot become a resolution by omission.** The correct phase reports a `DELIVER-DISMISSALS: <n>` count as the last line of its comment and a workflow step derives `deliver:has-dismissals` from it; a missing or unreadable count is treated as outstanding. The label is not applied by the agent, so forgetting to apply it is not a way past the gate.

Archon is optional throughout: with a plan there is a deterministic number that must not move the wrong way, without one the gate is CI plus the review verdict.

## Controlling a delivery in flight

**Pause.** Add the `deliver:paused` label to the PR (or, before a PR exists, to the sub-issue). Every phase checks it as its first step and exits without invoking an agent or moving a label. Remove the label and re-issue the command to resume. You are never racing the loop.

**Dismissals.** If the correct phase dismisses a finding rather than fixing it, it says so in a `DELIVER-DISMISSALS: <n>` line and a workflow step applies `deliver:has-dismissals` when that count is above zero — or when the count is missing, since an unreported state is not a report of zero. The gate refuses `ready-for-merge` while the label is present, so a dismissal cannot quietly become a resolution. Only the review phase removes it, and only when it has explicitly accepted each one; a reviewer that forgets costs a human glance rather than passing a waved-away finding.

**Round count.** The `deliver:round-N` label is the only record of how many corrections have been spent, and — since `workflow_dispatch` has no chain-depth cap — the only thing bounding the loop. It is advanced *before* the correction agent runs, so a crashed or timed-out round still consumes its budget rather than being retried forever. A missing round label is therefore fatal to the correct phase, unlike most label failures, which only warn.

**Two label writes are fatal, and for the same reason: both are the mechanism that bounds or blocks the loop.** Failing to advance `deliver:round-N` loses the loop's only bound. Failing to apply `deliver:has-dismissals` when a dismissal was reported loses the gate's only record of it — the next verify would read `none` and could reach `ready-for-merge` with a dismissal nobody accepted. In both cases the phase stops and a human picks it up, rather than continuing with the safeguard silently absent.

## When a delivery goes quiet, and how to resume one

**Every phase reports its own failure**, including a step that was cancelled, and applies `needs-human`.

**A dead runner cannot report itself.** If the runner is lost mid-job, no step executes — not `always()` ones, not even the action's own post-steps. This has happened twice: the first time the delivery left no branch, no PR, no comment and no label, which is indistinguishable from nobody having run the command; the second time it also lost a finished implementation that had never been pushed. Two *different* mechanisms close those two holes, and it is worth keeping them apart: **opening the branch and PR up front makes the delivery visible** — to the sweep, and to a human reading the issue — while **pushing the branch and then pushing as you go is what makes the work survive**. Visibility alone would leave an empty PR and still lose everything since the last push, so a phase that pushed only at the end would be findable but no less destroyed. `deliver-stall-sweep.yml` covers the visibility half: it runs on a schedule and flags any open PR on a `deliver/issue-*` branch that carries neither terminal label, is not paused, and has had no activity for 180 minutes. Activity means comments, reviews *and* review comments — the loop reacts to all three, so counting only comments would flag a delivery that was in fact responding to a review.

180 minutes is not arbitrary: the threshold has to exceed the longest *legitimate* silence, which is one phase's own budget (implement and verify are both capped at 120 minutes, and each reports at the end), plus slack for a busy runner. `scripts/deliver_guards_test.go` enforces that relationship, so a phase budget cannot be raised past the window without a failing test.

The sweep stands down while a delivery phase has **started recently**, because a legitimate verify is silent for long stretches while it waits on CI and then on a review. Recently, not merely "at all": a job that never got a runner sits queued for up to 24 hours, so counting queued runs of any age let an offline runner — the very scenario the sweep is for — keep it silent for a day. A run older than the quiet window is the symptom, not a sign of life.

The stand-down is repository-wide, so one recent phase run suppresses flagging for every delivery. That is fine while L1 delivers one sub-issue at a time and becomes wrong once deliveries run in parallel; it cannot be narrowed from run state, because a `workflow_dispatch` run cannot be attributed to a PR at all. For the same reason the sweep cannot say *which* phase stopped, so it links the run list instead.

**To resume a stopped delivery without repeating the implementation**, dispatch **Deliver — Verify** directly with the PR and issue numbers, from the Actions tab or:

```bash
gh workflow run deliver-verify.yml -f pr_number=<PR> -f issue_number=<N>
```

The correction round count lives on the PR's `deliver:round-N` label, so resuming this way keeps it. Dispatching verify directly remains the right move when the **implementation is already complete** and only the verdict is missing — nothing about the code needed redoing.

**Closing the delivery PR does not stop the branch being reused.** The phase looks for an *open* PR
on `deliver/issue-N`; if you close one, a re-issued command leaves the branch alone and the agent
opens a fresh PR on it. To stop a delivery, use the `deliver:paused` label rather than closing its PR.

**Re-issuing `/approve-issue-for-pr-delivery` resumes the implement phase rather than restarting it.** If `deliver/issue-N` already exists, the phase checks it out instead of creating it, reuses the open PR instead of opening a second one, and the agent is told to read the commits already there and continue from them. That is the right move when the implementation was left **part-finished** — a runner lost mid-flight. It is not a way to get a second opinion on finished work: the agent continues the existing branch, it does not start over.

**Any human push to a delivery branch re-verifies it**, so a verdict always describes the current head and a terminal label never outlives the commit it was granted to. Both terminal labels are cleared as soon as verification *starts*, not when it finishes — otherwise a push to a PR already marked `ready-for-merge` would keep advertising merge-readiness, on an unverified commit, for the whole re-verification.

An automation push does not double-trigger, and this does not rest on the token rule. A `GITHUB_TOKEN` push creates no workflow run, but the correct phase does not push from a workflow step — its agent pushes from inside `claude-code-action`, which performs an OIDC exchange and may hold an App installation token. The push-triggered path therefore refuses a **Bot-sent** push outright, so the property holds whichever credential is in play.

**A new review or review comment on the PR also re-verifies**, so a finding that arrives after the loop has converged is acted on rather than ignored. Because this repository is public and submitting a review needs only read access, the review-triggered path is gated on the same collaborator permission check the `/approve-issue-for-pr-delivery` command uses — without it, any GitHub user could start an agent run holding this workflow's secrets. An `approved` review is ignored (it carries no finding to act on); a `commented` review is not, since that is how most real findings have arrived.

## Configuration

Repository variables, all optional:

| Variable | Default | Effect |
|---|---|---|
| `DELIVER_IMPLEMENT_MODEL` | `claude-opus-4-8` | model for the implement phase |
| `DELIVER_CORRECT_MODEL` | `claude-opus-4-8` | model for the correct phase |
| `DELIVER_VERIFY_MODEL` | `claude-sonnet-4-6` | model for the review phase |
| `DELIVER_MAX_ROUNDS` | `3` | correction rounds before `needs-human`, per PR |
| `QA_QUESTIONER_MODEL` | `gcp/gemini-3.6-flash` | questioner model for the cross-vendor qa-review pass |
| `QA_ANSWERER_MODEL` | `azure/gpt-5.6-sol` | answerer model for the cross-vendor qa-review pass |
| `QA_ADJUDICATOR_MODEL` | `azure/gpt-5.6-sol` | adjudicator model for the adjudicate-only re-verify |

The verify model is deliberately *not* the implement model. Two instances of one model reviewing each other's work is closer to an agent grading its own homework; different models give real separation. The qa-review defaults go further and leave the vendor entirely: a decorrelated second opinion is the point (RFC #1603), so a failure mode shared by every Claude model is exactly what it exists to catch.

## Setup

**"Allow GitHub Actions to create and approve pull requests" is deliberately NOT required, and
should stay disabled.** This constrains how the loop is allowed to work, so it is worth knowing why.
GitHub exposes creating and approving pull requests as a **single** toggle
(`can_approve_pull_request_reviews`) that grants both to *every* workflow in the repository — there
is no way to take only creation. This loop must never approve anything: it labels, and a human
merges. So no phase creates a pull request from a workflow step, which would use `GITHUB_TOKEN` and
need that toggle; the agent opens the PR instead, with the App installation token
`claude-code-action` obtains via OIDC, which the setting does not govern.
`scripts/deliver_guards_test.go` fails if a workflow step reintroduces PR creation, so this cannot
regress quietly.

**The labels must exist before the workflows are used.** A workflow applying a label that does not exist fails at the API call, which strands a delivery mid-loop:

```bash
gh label create 'deliver:round-1' --color ededed --description 'L1 delivery: correction round 1'
gh label create 'deliver:round-2' --color ededed --description 'L1 delivery: correction round 2'
gh label create 'deliver:round-3' --color ededed --description 'L1 delivery: correction round 3'
gh label create 'deliver:paused'  --color b60205 --description 'L1 delivery: halted; every phase exits on this'
gh label create 'deliver:has-dismissals' --color fbca04 --description 'L1 delivery: a correction dismissed a finding the review has not accepted'
gh label create 'ready-for-merge' --color 0e8a16 --description 'L1 delivery: verified green, awaiting human merge'
gh label create 'needs-human'     --color d93f0b --description 'L1 delivery: stopped, a human must look'
```

Raising `DELIVER_MAX_ROUNDS` above 3 needs matching `deliver:round-N` labels.

## Scope

L1 delivers **one sub-issue at a time**, and sequencing is yours: approve one, merge it, approve the next. The implement phase reads a `Depends on: #M` line and refuses when #M's PR is unmerged, so a mis-ordered approval costs a comment rather than a failed hour-long run.

Not yet automated, each its own follow-up: sequencing sub-issues `0..N` and opening the final PR (L2), RFC-to-merge (L3), flaky-test re-runs, and the no-progress detector.

**Known limitations.**

- **The reviewer's *acceptance* of a dismissal is still prose.** The correct phase's dismissal count is now machine-readable and the gate enforces it (see *Dismissals* above), so a dismissal can no longer be treated as resolved by silence. What remains prompt-dependent is the other end: the review phase decides when to clear `deliver:has-dismissals`, and a reviewer that clears it without genuinely accepting each dismissal is not caught. Failing closed means the cost of *forgetting* is a human glance; the cost of clearing it wrongly is a human reading the comments, which they do before merging anyway.
- **The stall sweep's stand-down is repository-wide.** One recently started phase suppresses flagging for every delivery. Correct while L1 delivers one sub-issue at a time; a blocker for parallel deliveries.
- **Workflow expressions are not unit-tested.** `scripts/deliver-gate.sh` and the sweep's selection filter have tests; the trigger guards, step conditions and concurrency keys are covered by `actionlint` plus a live delivery, because nothing in this repository can evaluate a GitHub Actions expression.
- **Verify now weighs body + refinements, matching implement.** Verify's prompt runs `scripts/deliver-issue-refinements.sh` and treats the contracts as body + authoritative comment refinements ([Step 1.5](pr-workflow.md#comments-can-refine-the-body)), so it no longer reports an implementation that correctly followed an overriding refinement as a contract divergence. This closes the divergence the earlier revision documented as open; it was applied in a human-run correction round because the automated loop cannot push workflow files (next bullet). The *implement* phase remains prose-routed through `pr-workflow.md` for the same token reason — nothing mechanically verifies the implement agent read the digest.
- **The loop cannot deliver a change to its own workflow files.** `GITHUB_TOKEN` has no `workflows` permission — there is no such permission to grant in a `permissions:` block — so a push touching `.github/workflows/*` is rejected with *"refusing to allow a GitHub App to create or update workflow … without `workflows` permission"*. Measured, not inferred, while delivering #1782: a docs-and-scripts commit pushes, the same commit with a workflow hunk does not. Consequences worth knowing before approving such an issue: a delivery whose scope is a workflow file will get everything *except* that file, and behaviour meant for the delivery agents is best placed where the agents already read it (`docs/contributing/pr-workflow.md`, which the implement prompt points at, and `scripts/`) rather than inlined into a prompt. Applying a workflow hunk stays a human step.

## Security

**The workflow's own steps run trusted code.** The repository root checkout is the default branch, so `scripts/` and `.archon-version` always come from `main` — matching `archon.yml`. The archon review and `deliver-gate.sh` are executed from that trusted tree, never from the PR.

**The PR's code never runs on the self-hosted runner.** The verify phase never checks the PR out into its workspace or runs its build and tests there — it dispatches `ci.yml`, which runs the PR's build and tests on ephemeral `ubuntu-latest` runners, exactly as it does for any other PR. (The one place the PR head is materialised on the runner is the ephemeral, read-only `--detach` worktree that qa-review *reads* but never executes — see "qa-review reads the PR's files but never executes them" below.) An earlier version did check the PR out and run its test suite on the self-hosted runner, which meant executing PR code (including `deliver_gate_test.go`, which shells out to the PR's own `.sh` files) on persistent infrastructure. Dispatching removes that exposure rather than guarding it.

The correct phase does still check out and push to the delivery branch, so it requires the target PR to be same-repository, open, and on `deliver/issue-<N>` — the branch this loop owns.

**Every trigger requires write access.** `workflow_dispatch` is restricted to users with write access by GitHub, and pushing to a branch in this repository requires it too; a refused dispatch stops before any checkout. The two review triggers do **not** carry that property for free — this repository is public, and submitting a review or a review comment needs only *read* access, while both events run in the base-repository context with access to secrets. A review-triggered verify is therefore gated on an explicit collaborator permission check (`admin`/`write`/`maintain`), the same one the `/approve-issue-for-pr-delivery` command uses. Without it, any GitHub user could repeatedly start an agent run holding `LITELLM_API_KEY`, with `pull-requests: write` and `actions: write`, on the single self-hosted runner.

The check runs in a small `ubuntu-latest` job, so an unauthorised review never wakes the self-hosted runner at all.

**The delivery target comes from the event, not from comment text.** The issue delivered is `github.event.issue.number` — where the command was typed. An optional `#N` is accepted only when it agrees with that issue and refused when it disagrees, so no untrusted string ever selects the target.

**The verdict marker is read only from bot-authored comments, and only as a comment's last line** — otherwise any human could set a delivery's verdict by quoting it. The same applies to the `QA-VERDICT` marker and the `DELIVER-DISMISSALS` count.

**qa-review reads the PR's files but never executes them.** The verify job runs on the self-hosted runner with the LiteLLM secrets in its environment, so the load-bearing invariant is that PR-authored code never runs there. The qa-review answerer does need the PR-head *source*, so the head is checked out into an ephemeral `--detach` worktree under `$RUNNER_TEMP`, removed in an always-run cleanup, and the answerer runs with `--no-exec` — which drops its `go` build/test tool from both the implementation map and the advertised tool schema, leaving only `read_file`/`grep`/`list_dir` sandboxed to that worktree. The files are read; nothing in them is compiled or run. The adjudicate-only re-verify (#1716) reads the head the same way: the same ephemeral worktree, the same always-run cleanup, the same `--no-exec`.

**PR text is untrusted input to the agents.** This is a public repository, so anyone can comment on an open delivery PR, and both agents read comments. Two consequences are handled explicitly:

- Both prompts state that comment, review and diff text is data to be assessed and never instructions, and name the specific asks that constitute an attack (return GREEN, clear `deliver:has-dismissals`, report a particular dismissal count, run a command).
- The verify agent has **no `Edit` or `Write` tool**. It is told not to change code, and withholding the tools makes that structural rather than a request that injected text could argue it out of.

This reduces the exposure but does not eliminate it: the correct phase's agent legitimately needs write access and a shell in order to fix findings, so a sufficiently persuasive injected instruction remains a real risk. Treat an agent-authored commit as untrusted until a human has read it — which is why the loop never merges.
