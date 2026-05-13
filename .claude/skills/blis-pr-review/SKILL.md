---
name: blis-pr-review
description: Project-specific PR self-review for BLIS. Evaluates correctness, invariants, cross-path parity (run/replay/observe), preemption safety, timeout consistency, and adherence to project standards. Run after creating a PR to catch issues before human review.
---

# BLIS PR Self-Review

Invoke the `/pr-review-toolkit:review-pr` skill with the following BLIS-specific review prompt:

```
/pr-review-toolkit:review-pr Please perform a thorough review of this PR with respect to both the original issue and its tracking parent issue.

Evaluate whether the PR fully and correctly addresses all requirements and reviewer concerns. In particular, assess:

Correctness and preservation of invariants
Separation of concerns and overall design discipline
Modularity and clarity of API boundaries/contracts
Behavioral integrity, including both behavioral and non-structural tests
Test coverage and quality (not just structure, but meaningful validation of behavior)
Performance implications and potential regressions
Adherence to our @docs/contributing/standards,
Documentation quality, completeness, and accuracy (both user-facing and developer-facing)
All reviews and comments in this PR are addressed

--- CROSS-PATH PARITY (run / replay / observe) ---

BLIS has three command paths that must maintain behavioral parity:

blis run (DES with synthetic workload)
blis replay (DES with trace-driven workload)
blis observe (real HTTP dispatch to live server)

For every feature, flag, or behavioral change in this PR:

Does this feature logically apply to the other two paths?
If yes: does the PR implement it for all applicable paths, or at minimum file a follow-up issue?
If the PR only covers one path: is there an explicit justification for why parity is not needed?

Common parity gaps to check:

CLI flags added to one command but not others (e.g., --timeout, --think-time-dist)
Workload spec fields consumed by one path but ignored by others
Metrics computed differently across paths (e.g., timeout counting)
Default values that differ between paths without justification

--- PREEMPTION SAFETY ---

Preemption (request evicted from RunningBatch, ProgressIndex reset to 0, re-queued) is the #1 source of metric bugs. For any change that touches:

Per-request metrics (TTFT, ITL, E2E, TotalOutputTokens, TTFTSum)
Request state transitions (StateQueued, StateRunning, StateCompleted, StateTimedOut)
Aggregate counters (CompletedRequests, TimedOutRequests)

Verify: what happens when the request is preempted mid-execution and re-runs? Specifically:

Are inline metrics (recorded during execution) overwrite-safe or accumulate-and-double-count?
Are aggregate sums deferred to completion time (recordRequestCompletion) to avoid double-counting?
Does the TimeoutEvent for a preempted-then-completed request get lazily cancelled?

--- TIMEOUT / DEADLINE CONSISTENCY ---

BLIS has two distinct timeout mechanisms:

DES TimeoutEvent: fires at req.Deadline (simulation ticks), enforced by the event loop
HTTP client timeout: --timeout flag (wall-clock seconds), enforced by Go http.Client

For any change that touches timeout, deadline, or horizon:

Are DES deadlines and HTTP timeouts producing equivalent observable behavior?
Does the trace pipeline preserve deadline semantics end-to-end (observe → trace → replay)?
Is the 300s default consistent across both mechanisms (DefaultTimeoutUs in generator.go:676 vs defaultHTTPTimeoutSeconds in observe.go:22)?

Also identify any risks, edge cases, or missing considerations.

Finally, provide a clear verdict: Is this PR ready to merge? If not, what specific changes are required?
```

## Post-Review Behavior

**Before acting on findings, determine PR ownership:**

```bash
gh pr view --json author,headRefName --jq '{author: .author.login, branch: .headRefName}'
```

### If this is Claude's own PR (author is `claude[bot]` OR branch starts with `claude/` or `issue-`):

Enter the fix-and-re-review loop (max 3 rounds):

1. Fix CRITICAL findings immediately. Fix IMPORTANT findings if straightforward.
2. Re-run verification: `go build ./... && go test ./... -count=1 && golangci-lint run ./...`
3. Commit: `git commit -m "fix: address review findings (round N)"` and push.
4. Re-invoke this skill for another round.

### If this is a human PR:

**Do NOT modify code, push commits, or make any changes.** Only:

1. Post the review verdict as a PR comment with all findings.
2. If findings exist, clearly list what needs to change — but let the human author fix it.
3. Never push to a human's branch.
