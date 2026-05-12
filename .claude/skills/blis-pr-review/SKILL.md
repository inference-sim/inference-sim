---
name: blis-pr-review
description: Project-specific PR self-review for BLIS. Evaluates correctness, invariants, cross-path parity (run/replay/observe), preemption safety, timeout consistency, and adherence to project standards. Run after creating a PR to catch issues before human review.
---

# BLIS PR Self-Review

Perform a thorough review of the current PR with respect to both the original issue and its tracking parent issue.

---

## Step 0 — Identify the PR

Resolve the PR to review:

1. **Current branch**: if on a feature branch with an open PR, use that.
2. **Skill argument**: if a PR number or URL was passed, use that.
3. **GitHub Actions context**: if running inside `claude-code-action` on a PR trigger, use the current PR.

Fetch PR context:

```bash
gh pr view --json number,title,body,baseRefName,headRefName,url
gh pr diff
```

If the PR body references an issue (`Closes #N`), fetch the issue:

```bash
gh issue view <N> --json title,body,labels
```

If the issue references a parent/tracking issue, fetch that too.

---

## Step 1 — Standards Compliance

Read and check against all project standards:

```bash
cat docs/contributing/standards/rules.md
cat docs/contributing/standards/invariants.md
cat docs/contributing/standards/principles.md
```

For every file in the diff, verify:

- **R1-R23 antipattern rules**: check each applicable rule against the changed code
- **INV-1 through INV-12**: verify no invariant is violated by the change
- **Engineering principles**: separation of concerns, interface design, factory validation, config grouping

---

## Step 2 — Correctness and Design

Evaluate:

1. **Correctness**: Does the implementation correctly solve the stated problem?
2. **Preservation of invariants**: Are all affected invariants maintained with evidence?
3. **Separation of concerns**: Does `sim/` remain a library? Do cluster policies only see `*RouterState`? Instance policies only local data?
4. **Modularity and API boundaries**: Are interfaces single-method? Are factory signatures narrow?
5. **Behavioral integrity**: Do tests verify observable behavior, not internal structure?

---

## Step 3 — Test Quality

For each test added or modified:

1. **Refactor survival**: Would this test still pass if the implementation were completely rewritten but behavior preserved?
2. **Table-driven**: Are related scenarios in table-driven format?
3. **Laws not just values**: Do golden tests have companion invariant tests?
4. **THEN clauses**: Do assertions describe observable behavior, not internal state?
5. **Coverage**: Are edge cases covered (empty input, boundary values, error paths)?

---

## Step 4 — Cross-Path Parity

BLIS has three command paths that must maintain behavioral parity:

- `blis run` (DES with synthetic workload)
- `blis replay` (DES with trace-driven workload)
- `blis observe` (real HTTP dispatch to live server)

For every feature, flag, or behavioral change in this PR:

| Check | Question |
|-------|----------|
| Applicability | Does this feature logically apply to the other two paths? |
| Implementation | If yes: does the PR implement it for all applicable paths, or at minimum file a follow-up issue? |
| Justification | If only one path: is there an explicit justification for why parity is not needed? |

**Common parity gaps to check:**

- CLI flags added to one command but not others (e.g., `--timeout`, `--think-time-dist`)
- Workload spec fields consumed by one path but ignored by others
- Metrics computed differently across paths (e.g., timeout counting)
- Default values that differ between paths without justification

---

## Step 5 — Preemption Safety

Preemption (request evicted from RunningBatch, ProgressIndex reset to 0, re-queued) is the #1 source of metric bugs.

For any change that touches:
- Per-request metrics (TTFT, ITL, E2E, TotalOutputTokens, TTFTSum)
- Request state transitions (StateQueued, StateRunning, StateCompleted, StateTimedOut)
- Aggregate counters (CompletedRequests, TimedOutRequests)

Verify: what happens when the request is preempted mid-execution and re-runs?

| Check | Question |
|-------|----------|
| Overwrite safety | Are inline metrics (recorded during execution) overwrite-safe or accumulate-and-double-count? |
| Deferred aggregation | Are aggregate sums deferred to completion time (`recordRequestCompletion`) to avoid double-counting? |
| Timeout cancellation | Does the `TimeoutEvent` for a preempted-then-completed request get lazily cancelled? |

---

## Step 6 — Timeout / Deadline Consistency

BLIS has two distinct timeout mechanisms:

- **DES TimeoutEvent**: fires at `req.Deadline` (simulation ticks), enforced by the event loop
- **HTTP client timeout**: `--timeout` flag (wall-clock seconds), enforced by Go `http.Client`

For any change that touches timeout, deadline, or horizon:

| Check | Question |
|-------|----------|
| Equivalence | Are DES deadlines and HTTP timeouts producing equivalent observable behavior? |
| Trace pipeline | Does the trace pipeline preserve deadline semantics end-to-end (observe -> trace -> replay)? |
| Default consistency | Is the 300s default consistent across both mechanisms (`DefaultTimeoutUs` in generator.go vs `defaultHTTPTimeoutSeconds` in observe.go)? |

---

## Step 7 — Performance and Documentation

1. **Performance implications**: Are there O(n^2) loops, unbounded allocations, or hot-path lock contention introduced?
2. **Documentation**: Are user-facing docs updated if behavior changes? Are developer-facing comments accurate?
3. **All prior review comments addressed**: If this is a revision, are all previous reviewer concerns resolved?

---

## Step 8 — Verdict

Produce a structured verdict:

```markdown
## Self-Review Verdict

**Status**: READY TO MERGE | CHANGES REQUIRED | NEEDS DISCUSSION

### Findings

#### Critical (must fix before merge)
- <finding with file:line reference>

#### Important (should fix before merge)
- <finding with file:line reference>

#### Suggestions (optional improvements)
- <finding with file:line reference>

### Cross-Path Parity Summary
| Path | Applies | Implemented | Gap |
|------|---------|-------------|-----|
| run | ... | ... | ... |
| replay | ... | ... | ... |
| observe | ... | ... | ... |

### Preemption Safety: PASS / FAIL / N/A
### Timeout Consistency: PASS / FAIL / N/A
### Invariants Preserved: <list affected INVs with status>
```

---

## Step 9 — Act on Findings

- **CRITICAL findings**: Fix immediately. Re-run tests. Amend or add a fixup commit.
- **IMPORTANT findings**: Fix if straightforward. If complex, add a TODO comment referencing the issue.
- **After fixes**: Re-run verification (`go test ./... && golangci-lint run ./...`) and update the PR.

If the PR is READY TO MERGE after addressing findings, post the verdict as a PR comment:

```bash
gh pr comment --body "<verdict markdown>"
```
