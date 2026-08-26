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
Adherence to our @docs/contributing/standards
Documentation quality, completeness, and accuracy (both user-facing and developer-facing)
All reviews and comments in this PR are addressed

--- 10 CODE REVIEW PERSPECTIVES ---

Review from the 10 code review perspectives defined in @docs/contributing/perspectives.md (Code Review Perspectives section). Each catches issues the others miss: substance, code quality, test quality, UX, automated reviewer sim, DES expert, vLLM expert, distributed platform, performance, security.

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

--- Q/A PHASE (probing questions) ---

After completing the checklist above, perform a Q/A review:

1. Spawn a haiku subagent to generate 10-15 probing questions about this PR. Questions should cover:
   - Does this PR fully implement the issue it closes, without unnecessary additions?
   - Is all documentation updated (CLAUDE.md, README, guides)?
   - Are there stale comments left by this change?
   - Are there edge cases the tests don't cover?
   - Additional questions generated from the diff + standards docs

2. Spawn an opus subagent to investigate the code and answer each question with file:line evidence. Tag each answer:
   - CONFIDENT — verified correct with evidence
   - FLAW_FOUND — concrete issue with file:line reference
   - CANNOT_ANSWER — insufficient evidence to determine

Only FLAW_FOUND with concrete evidence (file:line or reproducible scenario) blocks the PR.

--- CONTRACT VERIFICATION (if .archon plan exists) ---

If a `.archon` plan exists in `specs/*/` for the feature this PR belongs to:

1. Read the plan's contract clauses (lines with `[evidenced: ...]`)
2. For each contract, verify a corresponding test exists
3. Check that the test actually tests what the contract claims (not just structural)
4. Report any contracts without matching evidence

---

Finally, provide a clear verdict: Is this PR ready to merge? If not, what specific changes are required?
```

## Output

Post the review verdict as a PR comment. This skill is READ-ONLY:

- Do NOT modify any code
- Do NOT run `git commit` or `git push`
- Do NOT use the Edit or Write tools
- Do NOT fix findings — only report them

The caller is responsible for acting on findings.
